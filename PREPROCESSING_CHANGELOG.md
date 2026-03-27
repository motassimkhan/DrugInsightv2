# Preprocessing Change Log

This document summarizes the major and minor changes made since the preprocessing rebuild started.

## Scope

The preprocessing pipeline was rebuilt to generate cleaner DrugBank/TWOSIDES training and inference inputs without relying on `rxnorm_bridge.csv`. The rebuild also aligned downstream feature extraction, prediction routing, and model training around the same pair-level artifacts.

## Major Changes

### 1. Preprocessing now starts from DrugBank-native CSVs plus raw TWOSIDES

- Removed the old dependency on `rxnorm_bridge.csv`.
- Rebuilt preprocessing around the available DrugBank CSV exports and the raw `TWOSIDES.csv` file.
- Centralized the pipeline in `src/preprocess_data.py` so processed artifacts can be regenerated in a consistent order.

### 2. Canonical DrugBank drug registry was added

- Created `data/processed/drug_catalog.csv`.
- Combined DrugBank drug records, names, synonyms, SMILES availability, and external identifier fallback data into one canonical lookup table.
- Added normalized naming so downstream pair mapping is less brittle.

### 3. SMILES preprocessing was rebuilt and validated

- Rebuilt `data/processed/drugbank_smiles_filtered.csv` from DrugBank source data.
- Validated SMILES with RDKit before using them for graph creation.
- Preserved the distinction between valid structure data and missing/invalid structure data instead of silently treating all missing cases the same.

### 4. TWOSIDES mapping was redesigned

- Added a new mapped TWOSIDES artifact: `data/processed/twosides_mapped.csv`.
- Rebuilt mapping in ordered fallback stages:
- exact normalized DrugBank name
- DrugBank synonym
- DrugBank `RxCUI` from external identifiers
- manual alias override
- Logged unresolved and ambiguous TWOSIDES mappings into the preprocessing manifest instead of letting joins fail silently.

### 5. Pair-level enrichment was rebuilt from the drug-level evidence tables

- Rebuilt `data/processed/drugbank_interactions_enriched.csv.gz`.
- Added a decompressed companion file: `data/processed/drugbank_interactions_enriched.csv`.
- Pair enrichment now derives overlap features directly from DrugBank IDs rather than depending on fragile late-stage joins.
- Shared evidence fields now include:
- `shared_enzyme_count`
- `shared_target_count`
- `shared_transporter_count`
- `shared_carrier_count`
- `shared_pathway_count`
- Major CYP evidence was promoted to first-class features:
- `shared_major_cyp_count`
- `cyp3a4_shared`
- `cyp2d6_shared`
- `cyp2c9_shared`
- TWOSIDES aggregate evidence was attached at pair level, including:
- `twosides_found`
- `twosides_max_prr`
- `twosides_mean_prr`
- `twosides_num_signals`
- `twosides_total_coreports`
- `twosides_mean_report_freq`
- `twosides_top_condition`

### 6. Training and inference now use the same feature contract

- Added `data/processed/feature_metadata.json`.
- Standardized the extra model feature vector to 12 inputs:
- `shared_enzyme_count`
- `shared_target_count`
- `shared_transporter_count`
- `shared_carrier_count`
- `shared_pathway_count`
- `shared_major_cyp_count`
- `cyp3a4_shared`
- `cyp2d6_shared`
- `cyp2c9_shared`
- `twosides_max_prr`
- `twosides_num_signals`
- `twosides_found`
- Moved feature scaling caps into metadata so training and prediction use the same normalization rules.

### 7. Prediction routing now follows the tiered evidence design

- Tier 1: known direct DrugBank pair returns direct evidence and bypasses ML/fusion.
- Tier 2: unknown pair with structured evidence uses model plus evidence fusion.
- Tier 3: unknown pair with no structured evidence uses structure-only ML and is explicitly low confidence.
- Prediction outputs now expose routing metadata such as `evidence_tier`, `decision_source`, and `severity_source`.

## Minor Changes

### 1. Better preprocessing observability

- Added `data/processed/preprocess_manifest.json`.
- The manifest records row counts, mapping statistics, feature metadata, and source paths used for the rebuild.

### 2. Duplicate pair handling was cleaned up

- Canonical unordered pair keys were used so `(drug_a, drug_b)` and `(drug_b, drug_a)` collapse to the same pair.
- The enriched interaction dataset was checked to ensure duplicate unordered pairs were removed.

### 3. Feature extraction now exposes pair context directly

- `src/feature_extractor.py` now exposes a callable extraction path that returns enriched pair context for debugging and sanity checks.
- This made it possible to directly inspect cases like `Warfarin` + `Aspirin` and verify non-zero overlap features.

### 4. Negative sampling became stricter

- Negative sampling now excludes:
- direct DrugBank positive pairs
- mapped TWOSIDES-signaled pairs
- This reduces the chance of sampling plausible positives as negatives during training.

### 5. Training compatibility and retraining support were updated

- `src/train.py` saves retrained weights to a separate checkpoint path: `models/ddi_model_reprocessed.pt`.
- The training script now resolves either `drugbank_interactions_enriched.csv` or `drugbank_interactions_enriched.csv.gz`.
- The dataset path also now includes a decompressed `.csv` copy in `data/processed` while keeping the original `.csv.gz`.

### 6. Training loop startup visibility was improved

- `src/train.py` reports the selected device, chosen interactions file, and batch counts per epoch.
- Progress logging was added inside training and evaluation loops so long epochs no longer appear frozen.
- Feature tensors are precomputed once per dataset instead of being rebuilt sample-by-sample from pandas rows, reducing loader overhead.

## Files Added or Regenerated

- `src/preprocess_data.py`
- `data/processed/drug_catalog.csv`
- `data/processed/drugbank_smiles_filtered.csv`
- `data/processed/twosides_mapped.csv`
- `data/processed/drugbank_interactions_enriched.csv.gz`
- `data/processed/drugbank_interactions_enriched.csv`
- `data/processed/feature_metadata.json`
- `data/processed/preprocess_manifest.json`

## Files Updated to Consume the New Pipeline

- `src/feature_extractor.py`
- `src/predict.py`
- `src/ddi_classifier.py`
- `src/train.py`

## Practical Outcome

The main effect of this rebuild is that the model is no longer operating on mostly-empty enrichment features. The system now has:

- a deterministic preprocessing path
- usable shared enzyme/target/transporter/carrier/pathway features
- mapped TWOSIDES evidence
- explicit CYP mechanism features
- one shared feature contract across training and prediction
- tiered evidence-aware prediction behavior
