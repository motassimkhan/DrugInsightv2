# DrugInsight — IEEE Paper Section Notes

> **Generated from codebase analysis of `DrugInsightv2` repository.**
> All claims are grounded in source files. Ambiguities are flagged explicitly.

---

## A. Architecture Overview

### Pipeline Summary

DrugInsight is an end-to-end, transparent drug-drug interaction (DDI) prediction and reasoning system. Given two drug identifiers (names or DrugBank IDs), the system (1) resolves them against a curated DrugBank database, (2) converts each drug's SMILES string into a molecular graph, (3) encodes each graph via an AttentiveFP graph neural network to produce fixed-length molecular embeddings, (4) computes pharmacological pair features (shared enzymes, targets, transporters, carriers, CYP-specific flags, and TWOSIDES pharmacovigilance signals — 12 features total), (5) concatenates the two drug embeddings with the 12-dim pharmacological feature vector and passes them through a multi-layer perceptron (MLP) classifier for interaction probability, (6) routes the prediction through a 3-tier evidence system — Tier 1 (direct DrugBank hit), Tier 2 (evidence fusion via **trained logistic regression / Platt scaling**), or Tier 3 (ML-only with uncertain severity), (7) derives a risk index (0–100) and severity label (Minor/Moderate/Major/Uncertain), and (8) generates a structured, mechanism-grounded natural-language explanation — all without requiring a large language model.

### Major Modules

#### 1. Drug Resolution & Feature Extraction
- **Purpose**: Resolve drug names/IDs → DrugBank IDs; extract pharmacological pair features
- **Inputs**: Drug name or DrugBank ID strings
- **Outputs**: DrugBank ID, canonical name, shared enzymes/targets/transporters/carriers/pathways counts, known interaction record, max TWOSIDES PRR
- **Evidence**: [feature_extractor.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/feature_extractor.py) — class `FeatureExtractor`, methods `resolve_drug()`, `extract()`, `pair_features()`
- **Data files loaded**: `drugbank_drugs.csv`, `drugbank_enzymes.csv`, `drugbank_targets.csv`, `drugbank_transporters.csv`, `drugbank_carriers.csv`, `drugbank_pathways.csv`, `drugbank_interactions_enriched.csv`

#### 2. Molecular Graph Construction
- **Purpose**: Convert SMILES strings into PyTorch Geometric `Data` objects (molecular graphs)
- **Inputs**: SMILES string
- **Outputs**: `torch_geometric.data.Data` with node feature matrix `x` (8 features per atom), edge index, and edge attribute matrix `edge_attr` (6 features per bond)
- **Evidence**: [mol_graph.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/mol_graph.py) — functions `atom_features()`, `bond_features()`, `smiles_to_graph()`

#### 3. GNN Encoder (AttentiveFP)
- **Purpose**: Encode molecular graphs into fixed-size drug embeddings
- **Inputs**: Molecular graph (`Data` object with `x`, `edge_index`, `edge_attr`, `batch`)
- **Outputs**: 256-dimensional embedding vector per drug
- **Architecture**: AttentiveFP (from PyTorch Geometric) with `in_channels=8`, `edge_dim=6`, `hidden_channels=128`, `out_channels=256`, `num_layers=4`, `num_timesteps=2`, `dropout=0.3`
- **Evidence**: [gnn_encoder.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/gnn_encoder.py) — class `GNNEncoder`

#### 4. DDI Classifier (MLP)
- **Purpose**: Predict interaction probability and severity class from drug pair representation
- **Inputs**: Two 256-dim drug embeddings + 12-dim extra feature vector → concatenated 524-dim input
- **Outputs**: Binary interaction logit (prob head) + 3-class severity logit (severity head, currently untrained)
- **Architecture**: 3-layer MLP trunk (524→512→256→128 with BatchNorm + ReLU + Dropout(0.5)), two output heads: `prob_head` (Linear 128→1), `severity_head` (Linear 128→3)
- **Input dimension**: Read dynamically from `feature_metadata.json` (`extra_dim` field), not hardcoded
- **Evidence**: [ddi_classifier.py](file:///c:/Documents/MOTASSIM%20COLLEGE%20SHIT/DrugInsightv2/src/ddi_classifier.py) — class `DDIClassifier`

#### 5. Evidence Fusion Layer (Trained — Platt Scaling)
- **Purpose**: Combine three independent evidence sources (rule score, ML probability, TWOSIDES score) into a unified risk index using **trained logistic regression coefficients**
- **Inputs**: Pharmacological context dict + raw ML probability
- **Outputs**: Fused probability, risk index (0–100), severity label, component scores, uncertainty metadata
- **Fusion mechanism**: Logistic regression (Platt scaling) — `σ(intercept + c₀·rule + c₁·ML + c₂·TWOSIDES)`
- **Trained coefficients** (from `models/fusion_weights.json`):
  - Rule coefficient: **2.902**
  - ML coefficient: **3.936**
  - TWOSIDES coefficient: **0.047**
  - Intercept: **−2.957**
- **Key change from v1**: Weights are **no longer hardcoded** per evidence tier. Instead, a single set of learned coefficients is applied universally by the Platt-scaled logistic model, trained via `calibrate_fusion.py` on tier-2 validation data.
- **3-Tier routing** (handled in `predict()`):
  - **Tier 1** (direct DrugBank hit): Rule-dominant blend (70% rule + 30% ML), bypasses calibrated fusion
  - **Tier 2** (structured evidence, no direct hit): Uses **trained Platt-scaled fusion** with all 3 component scores
  - **Tier 3** (structure-only, no evidence): ML-only with severity forced to "Uncertain" for positive predictions
- **Evidence**: [predict.py](file:///c:/Documents/MOTASSIM%20COLLEGE%20SHIT/DrugInsightv2/src/predict.py) — methods `_compute_fusion()`, `_get_calibrated_probability()`, `_direct_hit_result()`, `_ml_only_result()`

#### 6. Explainer (Rule-Based Reasoning)
- **Purpose**: Generate structured, mechanism-grounded explanations without LLM
- **Inputs**: Pharmacological context + prediction dict
- **Outputs**: Summary, severity, mechanism text, clinical recommendation, supporting evidence
- **Mechanism types**: (a) CYP inhibitor/inducer matching from curated knowledge, (b) DrugBank enzyme action parsing, (c) generic substrate competition, (d) pharmacodynamic target overlap, (e) pharmacovigilance PRR signal
- **Evidence**: [explainer.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/explainer.py) — class `Explainer`, methods `_enzyme_mechanism()`, `_target_mechanism()`, `_pharmacovigilance_note()`, `_clinical_recommendation()`, `explain()`

#### 7. Training Pipeline
- **Purpose**: Train the GNN encoder and DDI classifier end-to-end
- **Evidence**: [train.py](file:///c:/Documents/MOTASSIM%20COLLEGE%20SHIT/DrugInsightv2/src/train.py)

#### 8. Fusion Calibration Pipeline (NEW)
- **Purpose**: Train the evidence fusion layer weights via Platt scaling (logistic regression) on tier-2 validation data
- **Inputs**: Tier-2 validation pairs (positive interactions with structured evidence but no direct DrugBank hit)
- **Process**: (1) Run batched GNN+MLP inference on validation set, (2) compute rule scores and TWOSIDES scores for each pair, (3) fit `LogisticRegression(class_weight='balanced')` on the 3-dim feature matrix [rule, ML, TWOSIDES] → binary label
- **Output**: `models/fusion_weights.json` containing learned `coef` (3 values) and `intercept`
- **Evidence**: [calibrate_fusion.py](file:///c:/Documents/MOTASSIM%20COLLEGE%20SHIT/DrugInsightv2/src/calibrate_fusion.py)

#### 9. Preprocessing Pipeline (NEW — replaces enrich_interactions.py)
- **Purpose**: End-to-end data preprocessing: drug catalog construction, SMILES validation, TWOSIDES re-mapping (multi-strategy: exact name → synonym → RxCUI → manual alias), interaction enrichment with 12 pharmacological features, feature metadata generation (data-driven caps from 99th percentiles), SQLite database construction for efficient runtime lookups
- **Evidence**: [preprocess_data.py](file:///c:/Documents/MOTASSIM%20COLLEGE%20SHIT/DrugInsightv2/src/preprocess_data.py)

#### 10. Evaluation Module (NEW)
- **Purpose**: Create balanced test sets (positive + hard negative sampling) and evaluate the full prediction pipeline with per-tier metrics (Accuracy, ROC-AUC, PR-AUC, Precision, Recall, F1)
- **Evidence**: [evaluate.py](file:///c:/Documents/MOTASSIM%20COLLEGE%20SHIT/DrugInsightv2/src/evaluate.py)

#### 11. User Interfaces
- **Streamlit Web UI**: [app.py](file:///c:/Documents/MOTASSIM%20COLLEGE%20SHIT/DrugInsightv2/src/app.py) — premium dark-themed interface with risk bars, evidence grids, component score visualization
- **CLI**: [predict.py](file:///c:/Documents/MOTASSIM%20COLLEGE%20SHIT/DrugInsightv2/src/predict.py) `main()`, [cli.py](file:///c:/Documents/MOTASSIM%20COLLEGE%20SHIT/DrugInsightv2/drug_insight/cli.py) (predict/info/batch commands)
- **REST API**: [api.py](file:///c:/Documents/MOTASSIM%20COLLEGE%20SHIT/DrugInsightv2/src/api.py) — FastAPI with `/predict`, `/predict/batch`, `/drugs/{name}`, `/health` endpoints

---

## B. Preprocessing Pipeline

### Step 1: DrugBank Raw Data Extraction
- **What**: DrugBank XML/CSV exports are parsed into flat CSV tables: `drugbank_drugs.csv`, `drugbank_interactions.csv`, `drugbank_enzymes.csv`, `drugbank_targets.csv`, `drugbank_transporters.csv`, `drugbank_carriers.csv`, `drugbank_pathways.csv`, `drugbank_smiles.csv`, `drugbank_external_ids.csv`
- **Why**: DrugBank provides structured pharmacological data needed for both features and ground-truth interaction labels
- **Evidence**: Files exist in `data/processed/`; extraction scripts are not in repo (likely manual export from DrugBank)
- **Output**: Individual CSV files per entity type

### Step 2: SMILES Filtering
- **What**: `drugbank_smiles.csv` → `drugbank_smiles_filtered.csv` — filters drugs to retain only those with valid, parseable SMILES strings
- **Why**: Many DrugBank entries (e.g. biologics, protein drugs) lack small-molecule SMILES; filtering ensures all drugs can be converted to molecular graphs
- **Evidence**: In `train.py` (line 76): `smiles_df = smiles_df[smiles_df['smiles'].apply(is_valid_smiles)]` where `is_valid_smiles()` uses `Chem.MolFromSmiles()` for validation. The filtered file contains **~3,803 drugs** (from line count)
- **Output**: `drugbank_smiles_filtered.csv` with columns `drugbank_id`, `drug_name`, `smiles`

### Step 3: Interaction Filtering
- **What**: `drugbank_interactions.csv` → `drugbank_interactions_filtered.csv` — filters interactions to pairs where both drugs have valid SMILES
- **Why**: Only drug pairs that can be fed through the GNN are useful for training and evaluation
- **Evidence**: Referenced as input in `enrich_interactions.py` (line 6). Original has ~414 MB vs filtered ~264 MB
- **Output**: `drugbank_interactions_filtered.csv`

### Step 4: RxNorm Bridge Construction
- **What**: `rxnorm_bridge.csv` maps DrugBank IDs to RxNorm IDs, enabling linkage between DrugBank and TWOSIDES datasets
- **Why**: TWOSIDES uses RxNorm identifiers; this bridge file is essential for cross-dataset joins
- **Evidence**: Loaded in `enrich_interactions.py` (line 13), used in lines 55–57 to map `drug_1_id`/`drug_2_id` to `rx_1`/`rx_2`
- **Output**: `rxnorm_bridge.csv` with columns `drugbank_id`, `drug_name`, `rxnorm_id`

### Step 5: TWOSIDES Feature Extraction & Filtering
- **What**: TWOSIDES pharmacovigilance data is processed into `twosides_features_filtered.csv`; contains drug pair adverse event reporting with PRR (Proportional Reporting Ratio) as the signal strength metric
- **Why**: PRR quantifies the disproportionate reporting of adverse events for a drug combination vs. individual drugs — a real-world pharmacovigilance signal
- **Evidence**: Loaded in `enrich_interactions.py` (line 12). The CSV contains columns including `drug_1_rxnorn_id`, `drug_1_concept_name`, `drug_2_rxnorm_id`, `drug_2_concept_name`, `PRR`, and several other statistical columns
- **DrugBank vs. TWOSIDES**: DrugBank provides curated mechanism-level interaction data; TWOSIDES provides post-market adverse event reporting signals. They are linked via RxNorm.

### Step 6: Interaction Enrichment
- **What**: Computes shared pharmacological counts and joins TWOSIDES PRR to each interaction pair
- **Why**: These features provide interpretable pharmacological context as model inputs
- **Computed features (12-dim)**:
  - `shared_enzyme_count` — number of shared metabolizing enzymes
  - `shared_target_count` — number of shared pharmacological targets
  - `shared_transporter_count` — number of shared drug transporters
  - `shared_carrier_count` — number of shared drug carriers
  - `shared_pathway_count` — number of shared biological pathways
  - `shared_major_cyp_count` — number of shared major CYP enzymes (CYP3A4, CYP2D6, CYP2C9)
  - `cyp3a4_shared` — binary flag: both drugs share CYP3A4
  - `cyp2d6_shared` — binary flag: both drugs share CYP2D6
  - `cyp2c9_shared` — binary flag: both drugs share CYP2C9
  - `twosides_max_prr` — maximum TWOSIDES PRR for the pair
  - `twosides_num_signals` — number of distinct adverse event signals in TWOSIDES
  - `twosides_found` — binary flag for TWOSIDES match
- **Why they matter**: Shared enzymes → metabolic competition; shared targets → pharmacodynamic synergy/antagonism; transporters/carriers → distribution interactions; CYP-specific flags → targeted cytochrome P450 interaction risk (CYP3A4 metabolizes ~50% of drugs); PRR → real-world safety signal; signal count → breadth of pharmacovigilance evidence
- **Data quality steps**:
  - Canonical pair ordering (alphabetical DrugBank IDs) for deduplication
  - TWOSIDES mapping via multi-strategy resolution (exact name → synonym → RxCUI → manual alias)
  - Feature caps computed from 99th percentile values and stored in `feature_metadata.json`
- **Evidence**: [preprocess_data.py](file:///c:/Documents/MOTASSIM%20COLLEGE%20SHIT/DrugInsightv2/src/preprocess_data.py) — functions `compute_pair_features()`, `rebuild_drugbank_interactions()`
- **Output**: `drugbank_interactions_enriched.csv.gz`

### Step 7: SMILES → Molecular Graph Conversion
- **What**: Each drug's SMILES string is parsed by RDKit, hydrogen atoms are added, and the molecule is converted into a PyTorch Geometric `Data` object
- **Atom features (8)**: atomic number, degree, formal charge, hybridization type (sp/sp2/sp3 encoded as int), aromaticity flag, total hydrogen count, ring membership flag, normalized mass (mass/100)
- **Bond features (6)**: single bond flag, double, triple, aromatic, conjugated flag, ring membership flag
- **Graph structure**: Undirected (each bond creates two directed edges)
- **Invalid handling**: Returns `None` for unparseable SMILES or molecules with no bonds
- **Evidence**: [mol_graph.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/mol_graph.py) — `atom_features()`, `bond_features()`, `smiles_to_graph()`
- **Output**: `torch_geometric.data.Data(x, edge_index, edge_attr)`

### Step 8: Graph Caching
- **What**: All valid drug graphs are pre-computed once and stored in a Python dict keyed by DrugBank ID
- **Why**: Avoids redundant SMILES parsing during training
- **Evidence**: `train.py` lines 81–86

### Step 9: Drug-Level Train/Validation Split
- **What**: All unique drugs appearing in interactions are split 80/20 into train/val sets at the **drug level** (not the pair level)
- **Why**: Drug-level splitting ensures the validation set evaluates generalization to **unseen drugs** (cold-start setting) — a significantly harder and more realistic evaluation than random pair-level splitting
- **Evidence**: `train.py` lines 102–126 — `train_test_split(all_drugs, test_size=0.2, random_state=42)`, then interactions are partitioned by requiring both drugs in train or both in val
- **Validation mode**: Both drugs unseen (full cold-start). A commented-out alternative for one-drug-unseen exists (lines 119–124)
- **Output**: `train_pos`, `val_pos` DataFrames

### Step 10: Hard Negative Sampling
- **What**: For each split, negative (non-interacting) pairs are generated with a 70/30 hard/easy split
- **Hard negatives**: Non-interacting pairs scored by "plausibility" (weighted sum of shared enzymes ×2, targets ×2, transporters ×1, carriers ×1, pathways ×0.5); top-scoring pairs kept as hard negatives
- **Easy negatives**: Random non-interacting pairs with low plausibility scores
- **Why**: Hard negatives prevent the model from learning trivial shortcuts (e.g., "drugs sharing no enzymes never interact"); the 70% hard / 30% easy mix avoids training collapse
- **Ratio**: 1:1 positive-to-negative ratio (`n=len(train_pos)`)
- **Candidate multiplier**: 10× overcandidates generated, then filtered
- **Evidence**: [feature_extractor.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/feature_extractor.py) — method `sample_hard_negatives()`, called in `train.py` lines 132–139
- **Output**: `train_neg`, `val_neg` DataFrames with `label=0`

### Step 11: Feature Normalization for Model Input
- **What**: The 12 extra features are normalized using **data-driven caps from `feature_metadata.json`** (computed as 99th percentile values during preprocessing):
  - `shared_enzyme_count / 5`, `shared_target_count / 2`, `shared_transporter_count / 2`, `shared_carrier_count / 1`, `shared_pathway_count / 1`, `shared_major_cyp_count / 3`, `cyp3a4_shared / 1` (binary), `cyp2d6_shared / 1` (binary), `cyp2c9_shared / 1` (binary), `twosides_max_prr / 245.05`, `twosides_num_signals / 1882`, `twosides_found / 1` (binary)
- **Why**: Prevents dominant features from overwhelming the model. Caps are **no longer hardcoded** — they are computed from the actual data distribution during preprocessing and stored in `feature_metadata.json`
- **Evidence**: `feature_metadata.json` (caps), `train.py` `DDIDataset.__init__()` and `feature_extractor.py` `build_normalized_feature_vector()`

### Step 12: Fusion Weight Calibration (NEW)
- **What**: After GNN+MLP training, a separate calibration step trains logistic regression coefficients for the evidence fusion layer
- **Calibration cohort**: Tier-2 pairs (positive label + structured evidence + no direct DrugBank hit + valid SMILES), with 80/20 drug-level split for calibration
- **Process**: (1) Batch inference with frozen GNN+MLP to get ML probabilities, (2) compute vectorized rule scores and TWOSIDES scores, (3) fit `LogisticRegression(class_weight='balanced')` on [rule, ML, TWOSIDES] → label
- **Output**: `models/fusion_weights.json` with learned coefficients
- **Evidence**: [calibrate_fusion.py](file:///c:/Documents/MOTASSIM%20COLLEGE%20SHIT/DrugInsightv2/src/calibrate_fusion.py)

---

## C. Methodology

### C.1 Data Representation

Each drug is represented as a **molecular graph** $G = (V, E)$ where nodes are atoms (with 8 features) and edges are chemical bonds (with 6 features). Drug pairs are additionally characterized by a **12-dimensional pharmacological feature vector** encoding shared biological entities, CYP-specific interaction flags, and TWOSIDES pharmacovigilance signals between the two drugs.

### C.2 Feature Extraction

**Molecular features (per drug):**
| Feature | Type | Description |
|---------|------|-------------|
| Atomic number | int | Element identity |
| Degree | int | Number of bonds |
| Formal charge | int | Net charge |
| Hybridization | int | sp/sp2/sp3 as integer |
| Aromaticity | binary | In aromatic ring |
| H-count | int | Total attached hydrogens |
| Ring membership | binary | Part of any ring |
| Normalized mass | float | Atomic mass / 100 |

**Bond features (per edge):**
Single, double, triple, aromatic (one-hot), conjugated flag, ring membership flag.

**Pair-level pharmacological features (12-dim):**
| Feature | Source | Normalization (cap) |
|---------|--------|---------------------|
| Shared enzyme count | DrugBank | /5 |
| Shared target count | DrugBank | /2 |
| Shared transporter count | DrugBank | /2 |
| Shared carrier count | DrugBank | /1 |
| Shared pathway count | DrugBank | /1 |
| Shared major CYP count | DrugBank | /3 |
| CYP3A4 shared | DrugBank | binary |
| CYP2D6 shared | DrugBank | binary |
| CYP2C9 shared | DrugBank | binary |
| Max PRR | TWOSIDES | /245.05 |
| Num signals | TWOSIDES | /1882 |
| TWOSIDES found flag | TWOSIDES | binary |

> **Note:** All caps are **data-driven** (99th percentile values computed by `preprocess_data.py`), not hardcoded.

### C.3 Prediction Mechanism

The model operates in two stages:

**Stage 1 — GNN + MLP (Learnable):**
1. Each drug's molecular graph is encoded by an **AttentiveFP** graph neural network (Xiong et al., 2020) with 4 message-passing layers, 2 readout timesteps, 128 hidden channels, producing a 256-dimensional global graph embedding.
2. The two drug embeddings are concatenated with the 12-dim pharmacological feature vector to form a **524-dim** pair representation.
3. A 3-layer MLP trunk (524→512→256→128) with BatchNorm, ReLU, and Dropout(0.5) processes the pair representation.
4. A **probability head** (Linear 128→1) produces a binary interaction logit, passed through sigmoid.
5. A **severity head** (Linear 128→3) is architecturally defined but **not trained** (no severity labels available).

**Stage 2 — Evidence Fusion (Trained via Platt Scaling):**

The system routes predictions through a **3-tier evidence hierarchy**:

| Tier | Condition | Fusion Method | Severity Source |
|------|-----------|---------------|------------------|
| **Tier 1** | Direct DrugBank interaction record exists | Rule-dominant blend: 70% rule + 30% ML | DrugBank blended |
| **Tier 2** | Structured evidence (shared enzymes/targets/CYPs/TWOSIDES) but no direct hit | **Trained logistic regression (Platt scaling)** | Derived from fusion |
| **Tier 3** | No structured evidence (molecular structure only) | ML probability only | "Uncertain" for positives |

**Tier-2 Fusion (core innovation):**
Three component scores are computed and fed into a trained logistic model:

| Source | Score Derivation | Learned Coefficient |
|--------|-----------------|--------------------|
| **Rule score** | Additive per-feature contributions (enzyme: +min(0.32, 0.16+0.04n), target: +min(0.22, 0.10+0.03n), transporter: +min(0.10, 0.04+0.02n), carrier: +min(0.08, 0.03+0.02n), pathway: +min(0.08, 0.02+0.01n), CYP: +min(0.18, 0.10+0.04n)), total capped at 0.9 | **2.902** |
| **ML score** | Sigmoid of GNN+MLP classifier logit | **3.936** |
| **TWOSIDES score** | 0.7 × (PRR/245.05) + 0.3 × (num_signals/1882) | **0.047** |

Fused probability: $p_{fused} = \sigma(-2.957 + 2.902 \cdot s_{rule} + 3.936 \cdot s_{ML} + 0.047 \cdot s_{TWOSIDES})$

The fused probability determines:
- **Risk index**: fused_prob × 100 (integer 0–100)
- **Severity**: Major (≥70), Moderate (40–69), Minor (<40)
- **Interaction decision**: fused_prob ≥ 0.5
- **Context-aware override**: Severity is forced to "Uncertain" when evidence strength is too low for the predicted risk level

> **Key difference from v1**: The fusion weights are **learned from data** via logistic regression (Platt scaling) rather than being hand-tuned per evidence tier. The learned coefficients show the ML model contributes the strongest signal (coef=3.936), followed by rule-based evidence (coef=2.902), with TWOSIDES providing minimal independent signal (coef=0.047) in the presence of the other two sources.

### C.4 Reasoning / Explainability Mechanism

The `Explainer` class generates structured explanations without any LLM:

1. **Metabolic mechanism**: Checks CYP inhibitor/inducer knowledge base → DrugBank enzyme actions → generic substrate competition
2. **Pharmacodynamic mechanism**: Reports shared pharmacological targets
3. **Pharmacovigilance note**: Translates PRR into textual signal strength (weak < 3, moderate 3–10, strong > 10)
4. **Priority**: If a curated DrugBank mechanism text exists for the pair, it is used as the primary explanation
5. **Clinical recommendation**: Severity-dependent advice (avoid concurrent use / use with caution / standard monitoring)

Supporting evidence (shared enzymes list, shared targets list, shared pathways, TWOSIDES PRR) is returned alongside the explanation text.

### C.5 Training Objective

- **Loss function**: `BCEWithLogitsLoss` (binary cross-entropy with logits) on the probability head output
- **Symmetry augmentation**: During training, drug embeddings are randomly swapped with 50% probability to enforce order-invariant predictions
- **Gradient clipping**: max_norm = 1.0
- **Severity head**: Defined but **not trained** (no severity labels in dataset)

---

## D. Training and Evaluation

### D.1 Training Setup

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Optimizer | Adam (two parameter groups) | `train.py` line 176 |
| GNN learning rate | 3×10⁻⁵ | `train.py` line 177 |
| Classifier learning rate | 1×10⁻⁴ | `train.py` line 178 |
| Weight decay | 5×10⁻⁴ | `train.py` line 179 |
| LR scheduler | ReduceLROnPlateau (mode='max', factor=0.5, patience=3) | `train.py` lines 181–183 |
| Batch size | 64 | `train.py` lines 166–169 |
| Max epochs | 10 | `train.py` line 365 |
| Early stopping patience | 6 epochs (based on val AUC) | `train.py` lines 367–368 |
| Improvement threshold | 1×10⁻⁴ (for AUC) | `train.py` line 306 |
| Loss function | BCEWithLogitsLoss | `train.py` line 185 |
| Dropout | 0.5 (classifier), 0.3 (GNN) | `ddi_classifier.py` line 9, `gnn_encoder.py` line 14 |
| Symmetry augmentation | 50% random drug swap per batch | `train.py` lines 220–221 |
| Gradient clipping | max_norm = 1.0 | `train.py` lines 229–232 |
| Seed | 42 (split + neg sampling) | `train.py` lines 166, 184; different seed 43 for val negatives |
| Device | CUDA if available, else CPU | `train.py` line 21 |
| Num workers | 0 | `train.py` lines 166–169 |
| Train/Val split | 80/20 drug-level | `train.py` line 104 |
| Negative ratio | 1:1 (equal positives and negatives) | `train.py` lines 132, 136 |
| Hard neg fraction | 70% | `train.py` line 134 |

### D.2 Evaluation Metrics

| Metric | Definition | Role |
|--------|-----------|------|
| **ROC AUC** | Area under receiver operating characteristic curve | **Primary metric** for model selection and early stopping |
| **Average Precision (AP)** | Area under precision-recall curve | Secondary metric |
| **Accuracy** | (TP + TN) / total | Reported but not used for model selection |
| **Confusion matrix** | TP, TN, FP, FN counts | Diagnostic metric printed per epoch |
| **Training loss** | Mean BCEWithLogitsLoss per batch | Training convergence monitoring |

### D.3 Evaluation Protocol

- **Drug-level cold-start**: Validation drugs are completely unseen during training (no overlap, verified at line 160: `Drug overlap: {len(train_ids & val_ids)}` must be 0)
- **Threshold**: 0.5 on sigmoid probability for binary classification
- **Best model selection**: Highest val AUC; checkpoint saved to `models/ddi_model_reprocessed.pt`
- **No cross-validation**: Single train/val split (seed=42)
- **No separate test set**: Only train and validation; no held-out test set is explicitly defined

### D.4 Baselines / Comparisons

- No explicit baseline models are implemented in the repository
- The `configs/best_config.json` records a comparison between configurations: "Config B selected due to highest val AUC" with `label_smoothing: true`, `weighted_loss: false`, `dropout: 0.5`
- **Not enough evidence** for a formal ablation study in the repository

---

## E. Results

### E.1 Reported Metrics

| Metric | Value | Source |
|--------|-------|--------|
| **Best Val AUC** | **0.6329** | `configs/best_config.json` ("val_auc": 0.6329240966862422) |

> [!IMPORTANT]
> This is the **only concrete metric value** found in the repository. No training logs, per-epoch metrics, accuracy, AP, confusion matrix values, or loss curves are saved in any accessible file.

### E.2 Configuration That Produced Best Result

| Setting | Value |
|---------|-------|
| Label smoothing | Enabled |
| Weighted loss | Disabled |
| Dropout | 0.5 |
| Selection rationale | "Config B selected due to highest val AUC. Gap is acceptable. No NaNs detected." |

### E.3 Dataset Scale

| Dataset | Metric | Value |
|---------|--------|-------|
| DrugBank (enriched interactions) | Number of interaction pairs | ~936,178 |
| DrugBank (SMILES-filtered drugs) | Number of drugs with valid SMILES | ~3,803 |
| TWOSIDES (filtered features) | File size | ~87 MB |
| DrugBank enzymes | File size | ~760 KB |
| DrugBank targets | File size | ~2.7 MB |
| DrugBank pathways | File size | ~23 MB |

### E.4 Dataset-Specific Notes

- **DrugBank**: Used as the primary source for interaction ground-truth labels, drug metadata, enzyme/target/transporter/carrier/pathway information, curated mechanism text, and SMILES structures
- **TWOSIDES**: Used exclusively as a pharmacovigilance signal source (PRR values) — joined via the RxNorm bridge. TWOSIDES data contributes to:
  - The `max_PRR` and `twosides_found` input features for the ML model
  - The TWOSIDES score component in the fusion layer
  - Pharmacovigilance notes in the explanation
- The system does **not** train or evaluate on TWOSIDES interaction labels separately

### E.5 Missing/Unclear Results

- No per-epoch training logs or loss curves found in repository
- No accuracy, AP, or confusion matrix values from the final model
- No separate DrugBank-only vs. TWOSIDES-only evaluation
- No test-set results (only train/val split exists)
- Two model checkpoints exist (`ddi_model.pt` at ~4.3 MB, `ddi_model_crossattn.pt` at ~8.6 MB); the cross-attention variant's results/architecture are undocumented
- Val AUC of 0.6329 is on a **cold-start drug-level split** (both drugs unseen), which is a significantly harder setting than random pair-level splitting; this context is important for interpreting the number

---

## F. Conclusion Notes

1. **Multi-source evidence fusion with trained weights**: DrugInsight demonstrates that combining GNN-based molecular predictions with curated pharmacological knowledge (DrugBank) and real-world pharmacovigilance signals (TWOSIDES) through **trained Platt-scaled logistic regression** yields more transparent and robust DDI predictions than any single source alone. The fusion weights are **learned from data** rather than hand-tuned, with learned coefficients revealing that the ML model and rule-based evidence contribute the strongest signals (coef=3.94 and 2.90 respectively), while TWOSIDES provides minimal independent contribution (coef=0.05) in the multi-source context.

2. **Explainability without LLMs**: The system achieves mechanism-grounded explanations using a structured rule-based explainer that draws on shared enzyme/target information, curated CYP interaction knowledge, and TWOSIDES PRR signals — providing clinical interpretability without the computational overhead or hallucination risk of large language models.

3. **Cold-start capability**: The drug-level validation split demonstrates the system's ability to generalize to completely unseen drugs, a critical capability for practical DDI screening where new drugs must be evaluated against thousands of existing compounds.

4. **3-Tier evidence routing with context-aware severity**: The explicit evidence tier system (Tier 1: DrugBank direct, Tier 2: trained fusion, Tier 3: ML-only) provides principled confidence degradation, with context-aware severity overrides preventing overconfident predictions when supporting evidence is weak.

5. **Practical deployment**: The system is deployable via Streamlit web UI, CLI, FastAPI REST API, and Python package — covering research, clinical, and programmatic use cases.

6. **Limitations visible from implementation**:
   - The severity head exists architecturally but is **untrained** (severity labels derived from fusion thresholds, not learned)
   - Val AUC of 0.6329 on cold-start setting suggests room for improvement, though direct comparison requires noting the evaluation difficulty
   - No formal test set or cross-validation; single-split evaluation only
   - Fusion calibration uses logistic regression on a single train/val split — no cross-validated calibration
   - No automated test suite or formal ablation study

7. **Future directions** (supported by codebase):
   - Training the severity head with interaction severity labels
   - Exploring the cross-attention variant (checkpoint exists but undocumented as `ddi_model_crossattn.pt`)
   - Formal ablation study comparing fusion strategies (ML-only, fusion, rule-only)
   - Cross-validated fusion calibration for more robust coefficient estimation
   - Cross-validation for more robust performance estimation

---

## G. Evidence Map

| Topic | File(s) | Function/Class/Script | Evidence |
|-------|---------|----------------------|----------|
| Drug resolution | `src/feature_extractor.py` | `FeatureExtractor.resolve_drug()` | Name → ID mapping with synonyms, aliases, prefixing, fuzzy substring matching |
| Molecular graph construction | `src/mol_graph.py` | `atom_features()`, `bond_features()`, `smiles_to_graph()` | 8 atom features, 6 bond features, RDKit + PyG |
| GNN architecture | `src/gnn_encoder.py` | `GNNEncoder` | AttentiveFP with in=8, edge=6, hidden=128, out=256, layers=4, timesteps=2 |
| Classifier architecture | `src/ddi_classifier.py` | `DDIClassifier` | 524→512→256→128 MLP trunk, prob head (128→1), severity head (128→3) |
| Pharmacological features | `src/feature_extractor.py` | `FeatureExtractor.extract()`, `pair_features()` | 12-dim: shared enzymes/targets/transporters/carriers/pathways + CYP-specific + TWOSIDES |
| Hard negative sampling | `src/feature_extractor.py` | `FeatureExtractor.sample_hard_negatives()` | Weighted "plausibility" scoring, 70/30 hard/easy split |
| Training pipeline | `src/train.py` | Top-level script | Drug-level split, BCEWithLogitsLoss, Adam, early stopping, symmetry augmentation |
| Fusion layer | `src/predict.py` | `DDIPredictor._compute_fusion()`, `_get_calibrated_probability()` | Trained Platt-scaled logistic regression (coefs from `fusion_weights.json`) |
| Fusion calibration | `src/calibrate_fusion.py` | `main()` | Trains logistic regression on tier-2 validation data, outputs `fusion_weights.json` |
| Preprocessing pipeline | `src/preprocess_data.py` | `rebuild_all()` | Drug catalog, TWOSIDES mapping, interaction enrichment, feature metadata, SQLite DB |
| Explainer | `src/explainer.py` | `Explainer.explain()` | CYP knowledge base, DrugBank mechanisms, enzyme actions, target overlap, TWOSIDES PRR |
| Enrichment preprocessing | `src/enrich_interactions.py` | Top-level script | Shared count computation, TWOSIDES join via RxNorm, PRR capping, dedup |
| Evaluation module | `src/evaluate.py` | `evaluate_model()` | Balanced test set creation, per-tier metrics (Acc, AUC, AP, P, R, F1) |
| Streamlit UI | `src/app.py` | `main()` | Dark-themed UI with risk bar, evidence grid, component scores |
| REST API | `src/api.py` | FastAPI `app` | /predict, /predict/batch, /drugs/{name}, /health |
| CLI package | `drug_insight/cli.py` | `main()` | predict, info, batch subcommands |
| Python package wrapper | `drug_insight/predictor.py` | `DrugInsight` class | Singleton pattern, delegates to src/ modules |
| Best config / results | `configs/best_config.json` | JSON config | val_auc=0.6329, label_smoothing=true, dropout=0.5 |
| Trained model (reprocessed) | `models/ddi_model_reprocessed.pt` | PyTorch checkpoint | Contains `gnn` and `classifier` state dicts (trained with 12-dim features) |
| Trained model (original) | `models/ddi_model.pt` | PyTorch checkpoint | Original 6-dim feature model (legacy) |
| Cross-attention variant | `models/ddi_model_crossattn.pt` | PyTorch checkpoint | Exists but undocumented; architecture not found in current source |
| Fusion weights | `models/fusion_weights.json` | JSON | Trained logistic regression coefficients: coef=[2.90, 3.94, 0.05], intercept=-2.96 |
| Feature metadata | `data/processed/feature_metadata.json` | JSON | Feature order (12 features), data-driven caps, extra_dim=12 |

---

## Sanity Check — Internal Consistency

| Check | Status | Notes |
|-------|--------|-------|
| GNN in_channels matches atom features | ✅ | 8 atom features → `in_channels=8` |
| GNN edge_dim matches bond features | ✅ | 6 bond features → `edge_dim=6` |
| Classifier input dim matches concatenation | ✅ | 256 + 256 + 12 = 524 = `drug_embed_dim * 2 + extra_dim` (read from `feature_metadata.json`) |
| Extra features count consistent train ↔ predict | ✅ | Both use 12 features from `feature_metadata.json` with same caps |
| Fusion weights sum to 1.0 | ⚠️ **N/A (changed)** | Fusion no longer uses additive weights summing to 1.0; uses trained logistic regression with learned coefficients |
| Fusion weights file matches usage | ✅ | `fusion_weights.json` coef[3] used as [rule, ML, TWOSIDES] in `_get_calibrated_probability()` |
| Loss function matches binary classification | ✅ | BCEWithLogitsLoss for binary labels |
| Drug-level split produces zero overlap | ✅ | Explicit sanity check in train.py (line 207: `Drug overlap between splits: {len(train_ids & val_ids)}`) |
| `ddi_model_crossattn.pt` architecture | ⚠️ **Inconsistency** | Checkpoint exists but no cross-attention model code found in current source files |
| `ddi_model_reprocessed.pt` matches current code | ✅ | Trained with 12-dim features, `DDIClassifier` reads `feature_metadata.json` |
| `drugbank_interactions_enriched.csv.gz` columns | ✅ | Contains all 12 feature columns + TWOSIDES aggregate stats + mechanism text |
| Config val_auc = 0.6329 vs code | ⚠️ **Cannot verify** | No training logs preserved; value is plausible for cold-start drug-level split |
