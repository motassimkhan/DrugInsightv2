import pandas as pd
import numpy as np

# ── Load ───────────────────────────────────────────────────────────────────────
print("# Load input files")
interactions   = pd.read_csv('data/processed/drugbank_interactions_filtered.csv.gz')
drugs          = pd.read_csv('data/processed/drugbank_drugs.csv')
enzymes        = pd.read_csv('data/processed/drugbank_enzymes.csv')
targets        = pd.read_csv('data/processed/drugbank_targets.csv')
transporters   = pd.read_csv('data/processed/drugbank_transporters.csv')
carriers       = pd.read_csv('data/processed/drugbank_carriers.csv')
pathways       = pd.read_csv('data/processed/drugbank_pathways.csv')
twosides       = pd.read_csv('data/processed/twosides_features_filtered.csv')
rxnorm         = pd.read_csv('data/processed/rxnorm_bridge.csv')

print(f"Transporter cols: {transporters.columns.tolist()}")
print(f"Carrier cols:     {carriers.columns.tolist()}")

# ── Build lookup sets ──────────────────────────────────────────────────────────
print("Building lookup sets...")
drug_enzymes  = enzymes.groupby('drugbank_id')['enzyme_id'].apply(set).to_dict()
drug_targets  = targets.groupby('drugbank_id')['target_id'].apply(set).to_dict()
drug_pathways = pathways.groupby('drugbank_id')['smpdb_id'].apply(set).to_dict()

# Transporters
t_id_col = 'transporter_id' if 'transporter_id' in transporters.columns else transporters.columns[2]
drug_transporters = transporters.groupby('drugbank_id')[t_id_col].apply(set).to_dict()

# Carriers
c_id_col = 'carrier_id' if 'carrier_id' in carriers.columns else carriers.columns[2]
drug_carriers = carriers.groupby('drugbank_id')[c_id_col].apply(set).to_dict()

# ── Shared count function ──────────────────────────────────────────────────────
def compute_shared(df, lookup):
    d1 = df['drug_1_id'].values
    d2 = df['drug_2_id'].values
    return [len(lookup.get(a, set()) & lookup.get(b, set())) for a, b in zip(d1, d2)]

# ── Compute all shared counts ──────────────────────────────────────────────────
print("Computing shared counts...")
interactions['shared_enzyme_count']      = compute_shared(interactions, drug_enzymes)
interactions['shared_target_count']      = compute_shared(interactions, drug_targets)
interactions['shared_transporter_count'] = compute_shared(interactions, drug_transporters)
interactions['shared_carrier_count']     = compute_shared(interactions, drug_carriers)
interactions['shared_pathway_count']     = compute_shared(interactions, drug_pathways)

print(f"  Enzyme count > 0:      {(interactions['shared_enzyme_count'] > 0).sum()}")
print(f"  Target count > 0:      {(interactions['shared_target_count'] > 0).sum()}")
print(f"  Transporter count > 0: {(interactions['shared_transporter_count'] > 0).sum()}")
print(f"  Carrier count > 0:     {(interactions['shared_carrier_count'] > 0).sum()}")
print(f"  Pathway count > 0:     {(interactions['shared_pathway_count'] > 0).sum()}")

# ── Twosides join via RxNorm bridge ───────────────────────────────────────────
print("Joining twosides PRR...")

db_to_rx = dict(zip(rxnorm['drugbank_id'], rxnorm['rxnorm_id'].astype(str)))
interactions['rx_1'] = interactions['drug_1_id'].map(db_to_rx)
interactions['rx_2'] = interactions['drug_2_id'].map(db_to_rx)

twosides['drug_1_rxnorn_id'] = twosides['drug_1_rxnorn_id'].astype(str)
twosides['drug_2_rxnorm_id'] = twosides['drug_2_rxnorm_id'].astype(str)

ts_max = (twosides
    .groupby(['drug_1_rxnorn_id', 'drug_2_rxnorm_id'])['PRR']
    .max()
    .reset_index()
    .rename(columns={
        'drug_1_rxnorn_id': 'rx_1',
        'drug_2_rxnorm_id': 'rx_2',
        'PRR': 'max_PRR'
    })
)

ts_rev  = ts_max.rename(columns={'rx_1': 'rx_2', 'rx_2': 'rx_1'})
ts_both = pd.concat([ts_max, ts_rev], ignore_index=True).drop_duplicates()

interactions = interactions.merge(ts_both, on=['rx_1', 'rx_2'], how='left')
interactions['max_PRR']        = interactions['max_PRR'].fillna(0.0)
interactions['twosides_found'] = (interactions['max_PRR'] > 0).astype(int)

print(f"  Twosides found > 0: {(interactions['twosides_found'] > 0).sum()}")

# ── Apply fixes from data analysis ─────────────────────────────────────────────
print("Applying data quality fixes...")

# Cap PRR at 99th percentile
prr_99th = interactions['max_PRR'].quantile(0.99)
print(f"  Capping max_PRR at 99th percentile: {prr_99th:.2f}")
interactions['max_PRR'] = np.clip(interactions['max_PRR'], a_min=None, a_max=prr_99th)

# Remove duplicate drug pairs keeping highest-confidence (max_PRR) record
pair_ids = [tuple(sorted([str(a), str(b)])) for a, b in zip(interactions['drug_1_id'], interactions['drug_2_id'])]
interactions['pair_id'] = pair_ids
interactions = interactions.sort_values(by='max_PRR', ascending=False)
initial_len = len(interactions)
interactions = interactions.drop_duplicates(subset=['pair_id'], keep='first')
print(f"  Removed {initial_len - len(interactions)} duplicate pairs.")

# ── Cleanup & save ─────────────────────────────────────────────────────────────
enriched = interactions.drop(columns=['rx_1', 'rx_2', 'pair_id'])

print("\nFinal feature stats:")
print(enriched[['shared_enzyme_count', 'shared_target_count',
                     'shared_transporter_count', 'shared_carrier_count',
                     'shared_pathway_count',
                     'max_PRR', 'twosides_found']].describe())

# --- 3. Save resulting dataset ---
out_path = 'data/processed/drugbank_interactions_enriched.csv.gz'
enriched.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print(f"Shape: {enriched.shape}")
