import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

try:
    from .predict import DDIPredictor
    from .feature_extractor import canonical_pair_ids
except ImportError:
    from predict import DDIPredictor
    from feature_extractor import canonical_pair_ids


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')


def resolve_interactions_path():
    candidates = [
        os.path.join(DATA_DIR, 'drugbank_interactions_enriched.csv'),
        os.path.join(DATA_DIR, 'drugbank_interactions_enriched.csv.gz'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError('Could not find drugbank_interactions_enriched.csv')


def create_test_set(n_samples=500, seed=42):
    """
    Creates a balanced robust test set by sampling positive pairs and generating
    hard negative pairs, ensuring they represent a variety of tiers.
    """
    print('Loading interactions to create test set...')
    interactions_path = resolve_interactions_path()
    interactions = pd.read_csv(interactions_path, low_memory=False)
    
    # Filter valid pairs
    valid_positives = interactions[
        (interactions['label'] == 1) & 
        (interactions['direct_drugbank_hit'] == 1)
    ].copy()
    
    # Only keep those with SMILES available to ensure ML fallback works
    smiles_df = pd.read_csv(os.path.join(DATA_DIR, 'drugbank_smiles_filtered.csv'))
    valid_smiles_ids = set(smiles_df['drugbank_id'])
    
    valid_positives = valid_positives[
        valid_positives['drug_1_id'].isin(valid_smiles_ids) & 
        valid_positives['drug_2_id'].isin(valid_smiles_ids)
    ]
    
    print(f'Found {len(valid_positives)} valid positive pairs with SMILES.')
    
    # Sample positives
    rng = np.random.default_rng(seed)
    n_pos = min(n_samples // 2, len(valid_positives))
    pos_sample = valid_positives.sample(n=n_pos, random_state=seed)
    
    # Generate negatives
    print('Generating hard negatives...')
    predictor = DDIPredictor()
    all_drugs = set(valid_positives['drug_1_id']) | set(valid_positives['drug_2_id'])
    positive_pair_keys = set()
    for row in valid_positives.itertuples(index=False):
        id_a, id_b = canonical_pair_ids(row.drug_1_id, row.drug_2_id)
        positive_pair_keys.add(f'{id_a}||{id_b}')
        
    neg_sample = predictor.feature_extractor.sample_hard_negatives(
        drug_pool=list(all_drugs),
        positive_pairs=[k.split('||') for k in positive_pair_keys],
        n=n_pos,
        seed=seed
    )
    
    test_df = pd.concat([pos_sample, neg_sample], ignore_index=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f'Test set created with {len(test_df)} pairs ({n_pos} positive, {len(neg_sample)} negative).')
    
    return test_df


def evaluate_model(test_df):
    """Evaluates the DDIPredictor against the test dataframe."""
    predictor = DDIPredictor()
    
    results = []
    y_true = []
    y_pred = []
    y_prob = []
    tiers = []
    
    print(f'Starting evaluation on {len(test_df)} pairs...')
    
    for i, row in test_df.iterrows():
        if i % 50 == 0 and i > 0:
            print(f'  Processed {i}/{len(test_df)} pairs...')
            
        drug_a = row['drug_1_id']
        drug_b = row['drug_2_id']
        true_label = int(row['label'])
        
        try:
            prediction = predictor.predict(drug_a, drug_b)
            
            if 'error' in prediction:
                print(f"Error predicting {drug_a} vs {drug_b}: {prediction['error']}")
                continue
                
            pred_prob = prediction['probability']
            pred_label = 1 if prediction['interaction'] else 0
            tier = prediction['evidence_tier']
            
            y_true.append(true_label)
            y_prob.append(pred_prob)
            y_pred.append(pred_label)
            tiers.append(tier)
            
            results.append({
                'drug_a': drug_a,
                'drug_b': drug_b,
                'true_label': true_label,
                'pred_label': pred_label,
                'pred_prob': pred_prob,
                'tier': tier,
                'severity': prediction['severity']
            })
            
        except Exception as e:
            print(f"Exception for {drug_a} vs {drug_b}: {e}")
            continue

    print('\nEvaluation Complete. Calculating metrics...\n')
    
    # Global Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
        ap = float('nan')
        
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0,0,0,0)
    
    print(f"{'='*50}")
    print(f" OVERALL PERFORMANCE (N={len(y_true)})")
    print(f"{'='*50}")
    print(f" Accuracy  : {acc:.4f}")
    print(f" Precision : {prec:.4f}")
    print(f" Recall    : {rec:.4f}")
    print(f" F1-Score  : {f1:.4f}")
    print(f" ROC-AUC   : {auc:.4f}")
    print(f" PR-AUC    : {ap:.4f}")
    print(f" Confusion : TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    # Tier-based Metrics
    print(f"\n{'-'*50}")
    print(f" PERFORMANCE BY EVIDENCE TIER")
    print(f"{'-'*50}")
    
    results_df = pd.DataFrame(results)
    for tier in sorted(results_df['tier'].unique()):
        tier_df = results_df[results_df['tier'] == tier]
        t_true = tier_df['true_label']
        t_pred = tier_df['pred_label']
        t_prob = tier_df['pred_prob']
        
        t_acc = accuracy_score(t_true, t_pred)
        try:
            t_auc = roc_auc_score(t_true, t_prob) if len(set(t_true)) > 1 else float('nan')
            t_ap = average_precision_score(t_true, t_prob) if len(set(t_true)) > 1 else float('nan')
        except ValueError:
            t_auc, t_ap = float('nan'), float('nan')
            
        print(f" {tier} (N={len(tier_df)}):")
        print(f"   Acc={t_acc:.4f} | AUC={t_auc:.4f} | AP={t_ap:.4f}")

    print(f"{'='*50}\n")
    

def main():
    # Evaluate a sample of 200 pairs to keep execution fast but statistically relevant
    test_df = create_test_set(n_samples=200, seed=42)
    evaluate_model(test_df)


if __name__ == '__main__':
    main()
