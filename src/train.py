import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

try:
    from .ddi_classifier import DDIClassifier, load_feature_metadata
    from .feature_extractor import FeatureExtractor
    from .gnn_encoder import GNNEncoder
    from .mol_graph import smiles_to_graph
except ImportError:
    from ddi_classifier import DDIClassifier, load_feature_metadata
    from feature_extractor import FeatureExtractor
    from gnn_encoder import GNNEncoder
    from mol_graph import smiles_to_graph


RDLogger.DisableLog('rdApp.*')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'ddi_model_reprocessed.pt')
FEATURE_METADATA_PATH = os.path.join(DATA_DIR, 'feature_metadata.json')

os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)


def resolve_interactions_path():
    candidates = [
        os.path.join(DATA_DIR, 'drugbank_interactions_enriched.csv'),
        os.path.join(DATA_DIR, 'drugbank_interactions_enriched.csv.gz'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        'Could not find drugbank_interactions_enriched.csv or '
        'drugbank_interactions_enriched.csv.gz in data/processed.'
    )

def resolve_device():
    requested = os.getenv('DRUGINSIGHT_DEVICE', 'auto').strip().lower()
    if requested == 'cpu':
        return torch.device('cpu'), 'forced by DRUGINSIGHT_DEVICE=cpu'

    if requested.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(requested), f'forced by DRUGINSIGHT_DEVICE={requested}'
        return torch.device('cpu'), (
            f"requested '{requested}' but CUDA is unavailable "
            f"(torch.version.cuda={torch.version.cuda})"
        )

    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            return torch.device('cuda'), (
                f"auto-selected CUDA ({torch.cuda.get_device_name(0)})"
            )
        except Exception as exc:
            return torch.device('cpu'), f'CUDA detected but unusable ({exc})'

    return torch.device('cpu'), f'CUDA unavailable (torch.version.cuda={torch.version.cuda})'


DEVICE, DEVICE_REASON = resolve_device()
print(f'Using device: {DEVICE} ({DEVICE_REASON})')
print(f'Output checkpoint: {MODEL_PATH}')


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
    if not batch:
        return None

    graphs_a, graphs_b, extras, labels = zip(*batch)
    return (
        Batch.from_data_list(graphs_a),
        Batch.from_data_list(graphs_b),
        torch.stack(extras),
        torch.stack(labels),
    )


def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(str(smiles).strip()) is not None


print('Loading feature metadata...')
feature_metadata = load_feature_metadata(FEATURE_METADATA_PATH)
print(f"Feature dim: {feature_metadata['extra_dim']}")

print('Loading validated SMILES...')
smiles_df = pd.read_csv(os.path.join(DATA_DIR, 'drugbank_smiles_filtered.csv'))
smiles_df = smiles_df[smiles_df['smiles'].apply(is_valid_smiles)]
smiles_dict = dict(zip(smiles_df['drugbank_id'], smiles_df['smiles']))
print(f'Valid drugs with SMILES: {len(smiles_dict)}')

print('Pre-computing molecular graphs...')
graph_cache = {}
for drugbank_id, smiles in smiles_dict.items():
    graph = smiles_to_graph(smiles)
    if graph is not None:
        graph_cache[drugbank_id] = graph
print(f'Cached {len(graph_cache)} graphs')

print('Loading canonical pair data...')
feature_extractor = FeatureExtractor(DATA_DIR)
interactions_path = resolve_interactions_path()
print(f'Using interactions file: {interactions_path}')
interactions = pd.read_csv(interactions_path, low_memory=False)
interactions = interactions[
    (interactions['label'] == 1) &
    (interactions['direct_drugbank_hit'] == 1) &
    (interactions['drug_1_id'].isin(graph_cache)) &
    (interactions['drug_2_id'].isin(graph_cache))
].copy()
print(f'Direct DrugBank positives: {len(interactions)}')

all_drugs = sorted(set(interactions['drug_1_id']) | set(interactions['drug_2_id']))
train_drugs, val_drugs = train_test_split(all_drugs, test_size=0.2, random_state=42)
train_drugs = set(train_drugs)
val_drugs = set(val_drugs)

train_pos = interactions[
    interactions['drug_1_id'].isin(train_drugs) &
    interactions['drug_2_id'].isin(train_drugs)
].copy()
val_pos = interactions[
    interactions['drug_1_id'].isin(val_drugs) &
    interactions['drug_2_id'].isin(val_drugs)
].copy()
print(f'Train positives: {len(train_pos)} | Val positives: {len(val_pos)}')

positive_pairs = {
    (row.drug_1_id, row.drug_2_id)
    for row in interactions.itertuples(index=False)
}
train_neg = feature_extractor.sample_hard_negatives(
    train_drugs,
    positive_pairs,
    n=len(train_pos),
    seed=42,
    candidate_multiplier=10,
    hard_fraction=0.7,
)
val_neg = feature_extractor.sample_hard_negatives(
    val_drugs,
    positive_pairs,
    n=len(val_pos),
    seed=43,
    candidate_multiplier=10,
    hard_fraction=0.7,
)
print(f'Train negatives: {len(train_neg)} | Val negatives: {len(val_neg)}')

train_df = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1.0, random_state=42)
val_df = pd.concat([val_pos, val_neg], ignore_index=True).sample(frac=1.0, random_state=42)

train_ids = set(train_df['drug_1_id']) | set(train_df['drug_2_id'])
val_ids = set(val_df['drug_1_id']) | set(val_df['drug_2_id'])
print(f'Drug overlap between splits: {len(train_ids & val_ids)}')
print(f'Train rows: {len(train_df)} | Val rows: {len(val_df)}')

train_loader = DataLoader(
    DDIDataset(train_df, graph_cache, feature_metadata),
    batch_size=64,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == 'cuda'),
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    DDIDataset(val_df, graph_cache, feature_metadata),
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == 'cuda'),
    collate_fn=collate_fn,
)
print(f'Train batches per epoch: {len(train_loader)} | Val batches: {len(val_loader)}')

gnn = GNNEncoder().to(DEVICE)
classifier = DDIClassifier(feature_metadata_path=FEATURE_METADATA_PATH, dropout=0.5).to(DEVICE)

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

optimizer = torch.optim.Adam(
    [
        {'params': gnn.parameters(), 'lr': 3e-5},
        {'params': classifier.parameters(), 'lr': 1e-4},
    ],
    weight_decay=5e-4,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3,
)
criterion = nn.BCEWithLogitsLoss()

gnn.eval()
with torch.no_grad():
    sample_batch = next(iter(train_loader))
    if sample_batch is not None:
        graphs_a, _, extra, _ = sample_batch
        embeddings = gnn(graphs_a.to(DEVICE))
        print(f'Embedding mean: {embeddings.mean().item():.4f}')
        print(f'Embedding std:  {embeddings.std().item():.4f}')
        print(f'Any NaN:        {torch.isnan(embeddings).any()}')
        print(f'Extra sample:   {extra[0]}')


def train_epoch(loader):
    gnn.train()
    classifier.train()
    total_loss = 0.0
    total = 0
    correct = 0
    log_every = max(1, len(loader) // 10)

    for step, batch in enumerate(loader, start=1):
        if batch is None:
            continue

        graphs_a, graphs_b, extra, labels = batch
        graphs_a = graphs_a.to(DEVICE, non_blocking=(DEVICE.type == 'cuda'))
        graphs_b = graphs_b.to(DEVICE, non_blocking=(DEVICE.type == 'cuda'))
        extra = extra.to(DEVICE, non_blocking=(DEVICE.type == 'cuda'))
        labels = labels.float().to(DEVICE, non_blocking=(DEVICE.type == 'cuda'))

        embed_a = gnn(graphs_a)
        embed_b = gnn(graphs_b)
        if torch.rand(1).item() > 0.5:
            embed_a, embed_b = embed_b, embed_a

        logits, _ = classifier(embed_a, embed_b, extra)
        logits = logits.view(-1)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(gnn.parameters()) + list(classifier.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
        total_loss += loss.item()

        if step % log_every == 0 or step == len(loader):
            running_acc = correct / total if total else 0.0
            print(
                f'  Train step {step}/{len(loader)} | '
                f'Loss: {loss.item():.4f} | Running Acc: {running_acc:.4f}'
            )

    return total_loss / max(len(loader), 1), (correct / total if total else 0.0)


def eval_epoch(loader):
    gnn.eval()
    classifier.eval()
    total = 0
    correct = 0
    y_true = []
    y_prob = []
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            if batch is None:
                continue

            graphs_a, graphs_b, extra, labels = batch
            graphs_a = graphs_a.to(DEVICE, non_blocking=(DEVICE.type == 'cuda'))
            graphs_b = graphs_b.to(DEVICE, non_blocking=(DEVICE.type == 'cuda'))
            extra = extra.to(DEVICE, non_blocking=(DEVICE.type == 'cuda'))
            labels = labels.to(DEVICE, non_blocking=(DEVICE.type == 'cuda'))

            logits, _ = classifier(gnn(graphs_a), gnn(graphs_b), extra)
            probs = torch.sigmoid(logits).view(-1)
            preds = (probs > 0.5).long()

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_prob.extend(probs.detach().cpu().numpy().tolist())

            for pred, label in zip(preds, labels):
                if label == 1 and pred == 1:
                    tp += 1
                elif label == 0 and pred == 0:
                    tn += 1
                elif label == 0 and pred == 1:
                    fp += 1
                else:
                    fn += 1

            if step == len(loader):
                print(f'  Eval step {step}/{len(loader)}')

    print(f'  TP:{tp} TN:{tn} FP:{fp} FN:{fn}')
    accuracy = correct / total if total else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float('nan')
        ap = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float('nan')
    except Exception:
        auc = float('nan')
        ap = float('nan')
    return accuracy, auc, ap


EPOCHS = 10
best_auc = -1.0
patience = 6
patience_left = patience

for epoch in range(1, EPOCHS + 1):
    loss, train_acc = train_epoch(train_loader)
    val_acc, val_auc, val_ap = eval_epoch(val_loader)
    scheduler.step(val_auc if not np.isnan(val_auc) else val_acc)

    print(
        f'Epoch {epoch:02d} | Loss: {loss:.4f} | '
        f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | '
        f'Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}'
    )

    improved = (not np.isnan(val_auc) and val_auc > best_auc + 1e-4)
    if improved:
        best_auc = val_auc
        patience_left = patience
        torch.save({'gnn': gnn.state_dict(), 'classifier': classifier.state_dict()}, MODEL_PATH)
        print(f'  Saved best model (val_auc={best_auc:.4f})')
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f'Early stopping at epoch {epoch:02d} (best_val_auc={best_auc:.4f})')
            break

print(f'\nTraining complete. Best val AUC: {best_auc:.4f}')
