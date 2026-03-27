import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from feature_extractor import FeatureExtractor
from ddi_classifier import DDIClassifier, load_feature_metadata
from gnn_encoder import GNNEncoder
from predict import DDIPredictor
from mol_graph import smiles_to_graph
from rdkit import Chem

DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
FEATURE_METADATA_PATH = os.path.join(DATA_DIR, 'feature_metadata.json')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDIDataset(Dataset):
    def __init__(self, df, graph_cache, feature_metadata):
        self.graph_cache = graph_cache
        frame = df.reset_index(drop=True)
        self.drug_1_ids = frame['drug_1_id'].tolist()
        self.drug_2_ids = frame['drug_2_id'].tolist()
        self.labels = torch.tensor(
            pd.to_numeric(frame['label'], errors='coerce').fillna(0).astype(np.int64).to_numpy(),
            dtype=torch.long,
        )

        feature_columns = []
        for column in feature_metadata['feature_order']:
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
            else:
                values = np.zeros(len(frame), dtype=np.float32)

            cap = float(feature_metadata['feature_caps'].get(column, 1.0) or 1.0)
            values = np.clip(values, 0.0, cap)
            if cap > 0:
                values = values / cap
            feature_columns.append(values)

        extras = np.stack(feature_columns, axis=1) if feature_columns else np.zeros((len(frame), 0), dtype=np.float32)
        self.extras = torch.from_numpy(extras)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        graph_a = self.graph_cache.get(self.drug_1_ids[idx])
        graph_b = self.graph_cache.get(self.drug_2_ids[idx])
        if graph_a is None or graph_b is None:
            return None
        return graph_a, graph_b, self.extras[idx], self.labels[idx]

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    graphs_a, graphs_b, extras, labels = zip(*batch)
    return Batch.from_data_list(graphs_a), Batch.from_data_list(graphs_b), torch.stack(extras), torch.stack(labels)


def main():
    print("Initializing predictor to get ML model logic...")
    predictor = DDIPredictor()
    predictor.gnn.eval()
    predictor.classifier.eval()
    
    print("Preparing validation split...")
    feature_metadata = load_feature_metadata(FEATURE_METADATA_PATH)
    interactions_path = os.path.join(DATA_DIR, 'drugbank_interactions_enriched.csv')
    if not os.path.exists(interactions_path):
        interactions_path = os.path.join(DATA_DIR, 'drugbank_interactions_enriched.csv.gz')
    
    interactions = pd.read_csv(interactions_path, low_memory=False)
    interactions = interactions[
        (interactions['label'] == 1) &
        (interactions['direct_drugbank_hit'] == 1) &
        (interactions['drug_1_id'].isin(predictor.smiles_dict)) &
        (interactions['drug_2_id'].isin(predictor.smiles_dict))
    ].copy()
    
    all_drugs = sorted(set(interactions['drug_1_id']) | set(interactions['drug_2_id']))
    _, val_drugs = train_test_split(all_drugs, test_size=0.2, random_state=42)
    val_drugs = set(val_drugs)
    
    val_pos = interactions[
        interactions['drug_1_id'].isin(val_drugs) &
        interactions['drug_2_id'].isin(val_drugs)
    ].copy()
    
    positive_pairs = {
        (row.drug_1_id, row.drug_2_id)
        for row in interactions.itertuples(index=False)
    }
    
    print("Sampling validation negatives...")
    val_neg = predictor.feature_extractor.sample_hard_negatives(
        val_drugs,
        positive_pairs,
        n=len(val_pos),
        seed=43,
        candidate_multiplier=10,
        hard_fraction=0.7,
    )
    val_df = pd.concat([val_pos, val_neg], ignore_index=True)
    
    # Precompute graphs
    print("Caching graphs...")
    graph_cache = {}
    for d_id in set(val_df['drug_1_id']) | set(val_df['drug_2_id']):
        s = predictor.smiles_dict.get(d_id)
        if s:
            g = smiles_to_graph(s)
            if g is not None:
                graph_cache[d_id] = g

    # Filter val_df to ensure both graphs exist
    val_df = val_df[val_df['drug_1_id'].isin(graph_cache) & val_df['drug_2_id'].isin(graph_cache)].copy().reset_index(drop=True)
    print(f"Total valid validation pairs for calibration: {len(val_df)}")
    
    dataset = DDIDataset(val_df, graph_cache, feature_metadata)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    y_true = []
    y_prob = []
    
    print("Running Batched Inference...")
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            if batch is None: continue
            graphs_a, graphs_b, extra, labels = batch
            graphs_a = graphs_a.to(DEVICE)
            graphs_b = graphs_b.to(DEVICE)
            extra = extra.to(DEVICE)
            
            logits, _ = predictor.classifier(predictor.gnn(graphs_a), predictor.gnn(graphs_b), extra)
            probs = torch.sigmoid(logits).view(-1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            if step % 20 == 0:
                print(f"Batch {step}/{len(loader)}")

    print("Computing vectorized heuristic scores...")
    df = val_df.copy()
    assert len(df) == len(y_prob), "Mismatch in lengths"
    
    rule_score = np.zeros(len(df))
    c1 = df['shared_enzyme_count'] > 0
    rule_score[c1] += np.minimum(0.32, 0.16 + 0.04 * df.loc[c1, 'shared_enzyme_count'])
    c2 = df['shared_target_count'] > 0
    rule_score[c2] += np.minimum(0.22, 0.10 + 0.03 * df.loc[c2, 'shared_target_count'])
    c3 = df['shared_transporter_count'] > 0
    rule_score[c3] += np.minimum(0.10, 0.04 + 0.02 * df.loc[c3, 'shared_transporter_count'])
    c4 = df['shared_carrier_count'] > 0
    rule_score[c4] += np.minimum(0.08, 0.03 + 0.02 * df.loc[c4, 'shared_carrier_count'])
    c5 = df['shared_pathway_count'] > 0
    rule_score[c5] += np.minimum(0.08, 0.02 + 0.01 * df.loc[c5, 'shared_pathway_count'])
    c6 = df['shared_major_cyp_count'] > 0
    rule_score[c6] += np.minimum(0.18, 0.10 + 0.04 * df.loc[c6, 'shared_major_cyp_count'])
    rule_score = np.minimum(rule_score, 0.9)

    prr_cap = 245.05
    prr_comp = np.minimum(df['twosides_max_prr'].fillna(0), prr_cap) / prr_cap
    sig_cap = 1882
    sig_comp = np.minimum(df['twosides_num_signals'].fillna(0), sig_cap) / sig_cap
    twosides_score = 0.7 * prr_comp + 0.3 * sig_comp

    X = np.stack([rule_score, np.array(y_prob), twosides_score], axis=1)
    y = np.array(y_true)

    print("Fitting Logistic Regression (Platt Scaling)...")
    clf = LogisticRegression(random_state=42, class_weight='balanced')
    clf.fit(X, y)
    
    weights = {
        'coef': clf.coef_[0].tolist(),
        'intercept': clf.intercept_[0]
    }
    
    out_dir = os.path.join(ROOT_DIR, 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fusion_weights.json')
    with open(out_path, 'w') as f:
        json.dump(weights, f, indent=4)
        
    print(f"Fusion weights saved to {out_path}")
    print(f"Coefficients (Rule, ML, Twosides): {weights['coef']}")
    print(f"Intercept: {weights['intercept']}")

if __name__ == '__main__':
    main()
