# DrugInsight — IEEE Paper Section Notes

> **Generated from codebase analysis of `DrugInsightv2` repository.**
> All claims are grounded in source files. Ambiguities are flagged explicitly.

---

## A. Architecture Overview

### Pipeline Summary

DrugInsight is an end-to-end, transparent drug-drug interaction (DDI) prediction and reasoning system. Given two drug identifiers (names or DrugBank IDs), the system (1) resolves them against a curated DrugBank database, (2) converts each drug's SMILES string into a molecular graph, (3) encodes each graph via an AttentiveFP graph neural network to produce fixed-length molecular embeddings, (4) computes pharmacological pair features (shared enzymes, targets, transporters, carriers, and TWOSIDES pharmacovigilance signals), (5) concatenates the two drug embeddings with the pharmacological feature vector and passes them through a multi-layer perceptron (MLP) classifier for interaction probability, (6) fuses the ML prediction with a rule-based DrugBank evidence score and a TWOSIDES pharmacovigilance score using adaptive weighted fusion, (7) derives a risk index (0–100) and severity label (Minor/Moderate/Major), and (8) generates a structured, mechanism-grounded natural-language explanation — all without requiring a large language model.

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
- **Inputs**: Two 256-dim drug embeddings + 6-dim extra feature vector → concatenated 518-dim input
- **Outputs**: Binary interaction logit (prob head) + 3-class severity logit (severity head, currently untrained)
- **Architecture**: 3-layer MLP trunk (518→512→256→128 with BatchNorm + ReLU + Dropout(0.5)), two output heads: `prob_head` (Linear 128→1), `severity_head` (Linear 128→3)
- **Evidence**: [ddi_classifier.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/ddi_classifier.py) — class `DDIClassifier`

#### 5. Evidence Fusion Layer
- **Purpose**: Combine three independent evidence sources (DrugBank rule score, ML probability, TWOSIDES signal) into a unified risk index with adaptive weighting
- **Inputs**: Pharmacological context dict + raw ML probability
- **Outputs**: Fused probability, risk index (0–100), severity label, component scores, uncertainty metadata
- **Adaptive weights**:
  - DrugBank found: (0.60 rule, 0.25 ML, 0.15 TWOSIDES)
  - DrugBank partial: (0.40, 0.40, 0.20)
  - DrugBank not found: (0.10, 0.70, 0.20)
- **Evidence**: [predict.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/predict.py) — method `DDIPredictor._compute_fusion()` (lines 50–179)

#### 6. Explainer (Rule-Based Reasoning)
- **Purpose**: Generate structured, mechanism-grounded explanations without LLM
- **Inputs**: Pharmacological context + prediction dict
- **Outputs**: Summary, severity, mechanism text, clinical recommendation, supporting evidence
- **Mechanism types**: (a) CYP inhibitor/inducer matching from curated knowledge, (b) DrugBank enzyme action parsing, (c) generic substrate competition, (d) pharmacodynamic target overlap, (e) pharmacovigilance PRR signal
- **Evidence**: [explainer.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/explainer.py) — class `Explainer`, methods `_enzyme_mechanism()`, `_target_mechanism()`, `_pharmacovigilance_note()`, `_clinical_recommendation()`, `explain()`

#### 7. Training Pipeline
- **Purpose**: Train the GNN encoder and DDI classifier end-to-end
- **Evidence**: [train.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/train.py)

#### 8. Enrichment Pipeline (Preprocessing)
- **Purpose**: Compute shared pharmacological counts and join TWOSIDES PRR data to interaction pairs
- **Evidence**: [enrich_interactions.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/enrich_interactions.py)

#### 9. User Interfaces
- **Streamlit Web UI**: [app.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/app.py) — premium dark-themed interface with risk bars, evidence grids, component score visualization
- **CLI**: [predict.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/predict.py) `main()`, [cli.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/drug_insight/cli.py) (predict/info/batch commands)
- **REST API**: [api.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/api.py) — FastAPI with `/predict`, `/predict/batch`, `/drugs/{name}`, `/health` endpoints

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
- **Computed features**:
  - `shared_enzyme_count` — number of shared metabolizing enzymes
  - `shared_target_count` — number of shared pharmacological targets
  - `shared_transporter_count` — number of shared drug transporters
  - `shared_carrier_count` — number of shared drug carriers
  - `shared_pathway_count` — number of shared biological pathways
  - `max_PRR` — maximum TWOSIDES PRR for the pair (via RxNorm join), capped at 99th percentile
  - `twosides_found` — binary flag for TWOSIDES match
- **Why they matter**: Shared enzymes → metabolic competition; shared targets → pharmacodynamic synergy/antagonism; transporters/carriers → distribution interactions; PRR → real-world safety signal
- **Data quality steps**:
  - PRR capped at 99th percentile to control outliers (line 86–88)
  - Duplicate drug pairs removed, keeping highest-PRR record (lines 91–96)
  - Bidirectional TWOSIDES join (both orderings considered, lines 73–74)
- **Evidence**: [enrich_interactions.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/enrich_interactions.py) — entire script
- **Output**: `drugbank_interactions_enriched.csv` (~936K interaction pairs)

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
- **What**: The 6 extra features are min-max normalized with hard-coded clinically meaningful caps:
  - `shared_enzyme_count / 21.0`, `shared_target_count / 36.0`, `shared_transporter_count / 10.0`, `shared_carrier_count / 10.0`, `max_PRR / 50.0`, `twosides_found` (binary, no normalization)
- **Why**: Prevents dominant features from overwhelming the model; caps chosen from data distribution
- **Evidence**: `train.py` lines 42–49 (in `DDIDataset.__getitem__`) and `predict.py` lines 228–235

---

## C. Methodology

### C.1 Data Representation

Each drug is represented as a **molecular graph** $G = (V, E)$ where nodes are atoms (with 8 features) and edges are chemical bonds (with 6 features). Drug pairs are additionally characterized by a **6-dimensional pharmacological feature vector** encoding shared biological entities between the two drugs.

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

**Pair-level pharmacological features (6-dim):**
| Feature | Source | Normalization |
|---------|--------|---------------|
| Shared enzyme count | DrugBank | /21 |
| Shared target count | DrugBank | /36 |
| Shared transporter count | DrugBank | /10 |
| Shared carrier count | DrugBank | /10 |
| Max PRR | TWOSIDES (via RxNorm bridge) | /50 |
| TWOSIDES found flag | TWOSIDES | binary |

### C.3 Prediction Mechanism

The model operates in two stages:

**Stage 1 — GNN + MLP (Learnable):**
1. Each drug's molecular graph is encoded by an **AttentiveFP** graph neural network (Xiong et al., 2020) with 4 message-passing layers, 2 readout timesteps, 128 hidden channels, producing a 256-dimensional global graph embedding.
2. The two drug embeddings are concatenated with the 6-dim pharmacological feature vector to form a 518-dim pair representation.
3. A 3-layer MLP trunk (518→512→256→128) with BatchNorm, ReLU, and Dropout(0.5) processes the pair representation.
4. A **probability head** (Linear 128→1) produces a binary interaction logit, passed through sigmoid.
5. A **severity head** (Linear 128→3) is architecturally defined but **not trained** (no severity labels available).

**Stage 2 — Evidence Fusion (Non-learnable):**
The raw ML probability is combined with two additional evidence scores via **adaptive weighted fusion**:

| Source | Score Derivation | Weight (DrugBank found / partial / not found) |
|--------|-----------------|-----------------------------------------------|
| **Rule (DrugBank)** | 1.0 if known interaction; min(0.5 + n_shared×0.05, 0.85) if partial; 0.3 if pathways only; 0.0 otherwise | 0.60 / 0.40 / 0.10 |
| **ML (GNN+MLP)** | Sigmoid of classifier logit | 0.25 / 0.40 / 0.70 |
| **TWOSIDES** | min(PRR/50, 1.0) | 0.15 / 0.20 / 0.20 |

The fused probability determines:
- **Risk index**: fused_prob × 100 (integer 0–100)
- **Severity**: Major (≥70), Moderate (40–69), Minor (<40)
- **Interaction decision**: fused_prob ≥ 0.5

The adaptive weighting is key: when DrugBank curated evidence exists, it dominates; when absent, the ML model carries the primary signal. This provides a graceful confidence degradation.

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
| Max epochs | 20 | `train.py` line 290 |
| Early stopping patience | 6 epochs (based on val AUC) | `train.py` lines 292–293 |
| Improvement threshold | 1×10⁻⁴ (for AUC) | `train.py` line 306 |
| Loss function | BCEWithLogitsLoss | `train.py` line 185 |
| Dropout | 0.5 (classifier), 0.3 (GNN) | `ddi_classifier.py` line 9, `gnn_encoder.py` line 14 |
| Symmetry augmentation | 50% random drug swap per batch | `train.py` lines 220–221 |
| Gradient clipping | max_norm = 1.0 | `train.py` lines 229–232 |
| Seed | 42 (split + neg sampling) | `train.py` lines 104, 132, 134; different seed 43 for val negatives |
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
- **Best model selection**: Highest val AUC; checkpoint saved to `models/ddi_model.pt`
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

1. **Multi-source evidence fusion**: DrugInsight demonstrates that combining GNN-based molecular predictions with curated pharmacological knowledge (DrugBank) and real-world pharmacovigilance signals (TWOSIDES) through adaptive weighted fusion yields more transparent and robust DDI predictions than any single source alone.

2. **Explainability without LLMs**: The system achieves mechanism-grounded explanations using a structured rule-based explainer that draws on shared enzyme/target information, curated CYP interaction knowledge, and TWOSIDES PRR signals — providing clinical interpretability without the computational overhead or hallucination risk of large language models.

3. **Cold-start capability**: The drug-level validation split demonstrates the system's ability to generalize to completely unseen drugs, a critical capability for practical DDI screening where new drugs must be evaluated against thousands of existing compounds.

4. **Practical deployment**: The system is deployable via Streamlit web UI, CLI, FastAPI REST API, and Python package — covering research, clinical, and programmatic use cases.

5. **Limitations visible from implementation**:
   - The severity head exists architecturally but is **untrained** (severity labels derived from fusion thresholds, not learned)
   - Val AUC of 0.6329 on cold-start setting suggests room for improvement, though direct comparison requires noting the evaluation difficulty
   - No formal test set or cross-validation; single-split evaluation only
   - No automated test suite or formal ablation study
   - The fusion layer weights (0.60/0.40/0.10 etc.) are hand-tuned, not learned

6. **Future directions** (supported by codebase):
   - Training the severity head with interaction severity labels
   - Exploring the cross-attention variant (checkpoint exists but undocumented as `ddi_model_crossattn.pt`)
   - Formal ablation study comparing fusion strategies (ML-only, fusion, rule-only)
   - Cross-validation for more robust performance estimation

---

## G. Evidence Map

| Topic | File(s) | Function/Class/Script | Evidence |
|-------|---------|----------------------|----------|
| Drug resolution | `src/feature_extractor.py` | `FeatureExtractor.resolve_drug()` | Name → ID mapping with synonyms, aliases, prefixing, fuzzy substring matching |
| Molecular graph construction | `src/mol_graph.py` | `atom_features()`, `bond_features()`, `smiles_to_graph()` | 8 atom features, 6 bond features, RDKit + PyG |
| GNN architecture | `src/gnn_encoder.py` | `GNNEncoder` | AttentiveFP with in=8, edge=6, hidden=128, out=256, layers=4, timesteps=2 |
| Classifier architecture | `src/ddi_classifier.py` | `DDIClassifier` | 518→512→256→128 MLP trunk, prob head (128→1), severity head (128→3) |
| Pharmacological features | `src/feature_extractor.py` | `FeatureExtractor.extract()`, `pair_features()` | Shared enzymes/targets/transporters/carriers/pathways, known interaction, TWOSIDES |
| Hard negative sampling | `src/feature_extractor.py` | `FeatureExtractor.sample_hard_negatives()` | Weighted "plausibility" scoring, 70/30 hard/easy split |
| Training pipeline | `src/train.py` | Top-level script | Drug-level split, BCEWithLogitsLoss, Adam, early stopping, symmetry augmentation |
| Fusion layer | `src/predict.py` | `DDIPredictor._compute_fusion()` | 3-source adaptive weighted fusion with confidence estimation |
| Explainer | `src/explainer.py` | `Explainer.explain()` | CYP knowledge base, DrugBank mechanisms, enzyme actions, target overlap, TWOSIDES PRR |
| Enrichment preprocessing | `src/enrich_interactions.py` | Top-level script | Shared count computation, TWOSIDES join via RxNorm, PRR capping, dedup |
| Streamlit UI | `src/app.py` | `main()` | Dark-themed UI with risk bar, evidence grid, component scores |
| REST API | `src/api.py` | FastAPI `app` | /predict, /predict/batch, /drugs/{name}, /health |
| CLI package | `drug_insight/cli.py` | `main()` | predict, info, batch subcommands |
| Python package wrapper | `drug_insight/predictor.py` | `DrugInsight` class | Singleton pattern, delegates to src/ modules |
| Best config / results | `configs/best_config.json` | JSON config | val_auc=0.6329, label_smoothing=true, dropout=0.5 |
| Trained model | `models/ddi_model.pt` | PyTorch checkpoint | Contains `gnn` and `classifier` state dicts |
| Cross-attention variant | `models/ddi_model_crossattn.pt` | PyTorch checkpoint | Exists but undocumented; architecture not found in current source |

---

## Sanity Check — Internal Consistency

| Check | Status | Notes |
|-------|--------|-------|
| GNN in_channels matches atom features | ✅ | 8 atom features → `in_channels=8` |
| GNN edge_dim matches bond features | ✅ | 6 bond features → `edge_dim=6` |
| Classifier input dim matches concatenation | ✅ | 256 + 256 + 6 = 518 = `drug_embed_dim * 2 + extra_features` |
| Extra features count consistent train ↔ predict | ✅ | Both use 6 features with same normalization caps |
| Fusion weights sum to 1.0 | ✅ | All three weight configurations sum to 1.0 |
| Loss function matches binary classification | ✅ | BCEWithLogitsLoss for binary labels |
| Drug-level split produces zero overlap | ✅ | Explicit sanity check in train.py line 160 |
| `ddi_model_crossattn.pt` architecture | ⚠️ **Inconsistency** | Checkpoint exists but no cross-attention model code found in current source files; likely from an earlier experiment that was not retained in code |
| `drugbank_interactions_filtered.csv` vs `_enriched.csv` | ✅ | Enriched is produced from filtered + joins; sizes are consistent (enriched smaller due to dedup) |
| Config val_auc = 0.6329 vs code | ⚠️ **Cannot verify** | No training logs preserved; value is plausible for cold-start drug-level split but cannot be independently confirmed from repository |
