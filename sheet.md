## DrugInsight v2: ML + evidence, end to end

This document explains how DrugInsight v2 works from first principles (the ML bits) through training, inference, and deployment. It is written to match the current repo: Streamlit UI (`streamlit_app.py`), CLI predictor (`src/predict.py`), and FastAPI service (`src/api.py`).

The short version: DrugInsight predicts drug-drug interactions (DDIs) with a hybrid approach. A GNN learns from molecular structure when SMILES are available, but the final score is fused with curated DrugBank evidence, shared-biology overlap signals, and TWOSIDES pharmacovigilance associations. The output includes a risk index, severity, confidence, and an evidence-grounded explanation.

---

## 1) The ML concepts you actually need here

### 1.1 Supervised learning for pairs

Training examples look like:

- two drugs (A, B)
- features that describe the pair
- a label \(y \in \{0,1\}\) meaning "known interaction" vs "not labeled as an interaction"

The model produces a logit \(z\) for each pair. We train with `BCEWithLogitsLoss`, which is the standard way to learn a probability-like score for binary classification.

### 1.2 Why we use graphs

Molecules are graphs: atoms are nodes, bonds are edges. A graph neural network (GNN) does message passing across bonds and outputs a fixed-size embedding vector for the whole molecule. In DrugInsight, that embedding is the main representation of "what this drug looks like chemically."

---

## 2) Inputs and outputs at inference

### Inputs

Users can provide:

- drug names ("Warfarin") or DrugBank IDs ("DB00682")

The system resolves both inputs to canonical DrugBank IDs and checks whether each drug has a valid SMILES string for graph construction.

### Outputs

A prediction returns:

- `interaction` (bool)
- `probability` (float)
- `risk_index` (int 0-100)
- `severity` (Minor/Moderate/Major)
- `confidence` and an `uncertainty` payload describing which evidence sources were available
- an explanation: summary, mechanism, recommendation, plus evidence + component scores

---

## 3) Data sources and the "contract" between preprocessing and inference

DrugInsight mixes two evidence families:

- DrugBank-derived curated/biological context (IDs, names, known interactions, enzymes/targets/CYP, SMILES)
- TWOSIDES pharmacovigilance signals summarized into pair-level statistics

### 3.1 Canonical pair keys (order-invariant)

DDIs are treated as symmetric for feature lookup. The project uses a canonical `pair_key` formed from sorted DrugBank IDs:

```text
pair_key = f"{min(id_a, id_b)}||{max(id_a, id_b)}"
```

That single key is used across preprocessing outputs, training rows, and inference lookups. It prevents duplicated rows and reduces "why did A+B differ from B+A?" surprises.

### 3.2 Processed assets

Preprocessing produces and/or relies on tables under `data/processed` (for example: `drug_catalog.csv`, `drugbank_smiles_filtered.csv`, `twosides_mapped.csv`, and `drugbank_interactions_enriched.csv.gz`). It also writes `feature_metadata.json`, which defines:

- the feature order
- `extra_dim` (currently 12)
- caps used for clipping/normalization

That metadata is not just documentation. It is part of the model input definition.

---

## 4) Feature pipeline: what the model consumes

Each example (drug pair) becomes three inputs:

1. Graph for drug A (from SMILES)
2. Graph for drug B (from SMILES)
3. Auxiliary pair feature vector `extra` of length `extra_dim` (currently 12)

### 4.1 The 12 auxiliary features

The feature vector is built in `build_normalized_feature_vector()` in `src/feature_extractor.py`. Names and order come from `data/processed/feature_metadata.json`:

1. `shared_enzyme_count`
2. `shared_target_count`
3. `shared_transporter_count`
4. `shared_carrier_count`
5. `shared_pathway_count`
6. `shared_major_cyp_count`
7. `cyp3a4_shared`
8. `cyp2d6_shared`
9. `cyp2c9_shared`
10. `twosides_max_prr`
11. `twosides_num_signals`
12. `twosides_found`

Normalization rules:

- `*_shared` and `twosides_found` are binarized (0/1)
- other numeric features are clipped and scaled:

```text
min(raw_value, cap) / cap
```

If you change the caps or the ordering, you should assume you need to retrain or at least re-validate carefully. Old weights can still load and still output numbers while quietly consuming the wrong feature definition.

---

## 5) Molecular graph construction (SMILES -> PyG)

Graph building is implemented in `src/mol_graph.py`:

- parse SMILES with RDKit
- add explicit hydrogens (`Chem.AddHs`)
- build atom/node features (basic chemistry + flags)
- build bond/edge features (bond type, conjugation, ring membership)
- build an undirected edge index by adding both directions

This is why RDKit and Torch Geometric are core dependencies.

---

## 6) Model architecture

DrugInsight uses a two-stage structure model:

1. A shared GNN encoder that turns each drug graph into an embedding
2. An MLP classifier head that fuses both embeddings with the auxiliary feature vector

### 6.1 Two-tower (Siamese) pattern

Both drugs are embedded with the same GNN:

```text
embed_a = gnn(graph_a)
embed_b = gnn(graph_b)
logit, severity_logits = classifier(embed_a, embed_b, extra)
```

Sharing weights keeps both drugs in the same representation space and supports symmetry.

### 6.2 GNN encoder: AttentiveFP

The encoder is `torch_geometric.nn.AttentiveFP` (configured in `src/gnn_encoder.py`). AttentiveFP is designed for molecule-level property learning: attention-weighted message passing plus a graph-level readout.

### 6.3 Pairwise head: MLP fusion

`src/ddi_classifier.py` defines `DDIClassifier`. It concatenates `[embed_a, embed_b, extra]`, runs an MLP trunk, and outputs:

- a probability head (one logit for interaction)
- a severity head (three logits)

The current training path optimizes the interaction head with `BCEWithLogitsLoss`. The severity head exists but is not trained with a separate loss in the current `src/train.py` pipeline.

---

## 7) Training strategy (and why it is set up this way)

### 7.1 Objective and metrics

Training is supervised binary classification with `BCEWithLogitsLoss`. Evaluation reports accuracy, ROC AUC, and average precision (AP). AUC/AP are useful because DDI data tends to be imbalanced and you care about ranking quality.

### 7.2 Drug-level splitting (leakage control)

Instead of splitting random rows, the training pipeline splits by drug IDs and then forms pairs within those drug sets. The goal is to reduce leakage where the model sees the same drug in both train and validation and learns drug identity shortcuts.

### 7.3 Hard negative sampling

Random negative pairs are often too easy. DrugInsight samples hard negatives: negative pairs that share biological overlap with positives (enzymes/targets/major CYP), while excluding:

- pairs that appear in known curated positives
- pairs that appear in TWOSIDES-mapped data (to reduce label contamination)

This happens in `FeatureExtractor.sample_hard_negatives()`.

### 7.4 Symmetry augmentation

Even if the features are symmetric, concatenating `[embed_a, embed_b]` gives the head an ordering. During training, the code randomly swaps the two embeddings for some batches to reduce order artifacts.

### 7.5 Optimization

The training loop uses Adam with different learning rates for the encoder vs classifier head, a `ReduceLROnPlateau` scheduler keyed to validation AUC, and gradient clipping.

---

## 8) Inference-time system: tiers + deterministic fusion

DrugInsight does not always run ML, and it does not always trust ML equally. It routes by evidence availability and fuses three components.

### 8.1 Evidence tiers

`FeatureExtractor.determine_evidence_tier()` assigns:

- tier 1: direct curated DrugBank hit
- tier 2: some structured evidence exists (shared biology overlap and/or TWOSIDES signal)
- tier 3: no structured evidence, so rely on structure-only ML if possible

### 8.2 SMILES gating

The GNN path requires SMILES for both drugs. In `DDIPredictor.predict()` (`src/predict.py`):

- if one or both drugs lack SMILES and the request is not tier 1, prediction cannot run and returns an error
- tier 1 can still return a response because it can rely on curated evidence

### 8.3 Fusion formula

Fusion blends:

- a rule-based shared-biology score (`rule_score`)
- the ML model probability (`ml_prob`)
- a TWOSIDES score derived from normalized PRR and signal counts (`twosides_score`)

The fused probability is computed as:

```text
fused_prob = w_rule * rule_score + w_ml * ml_prob + w_twosides * twosides_score
```

Weights depend on which evidence signals exist for the pair. The final probability is mapped to:

- `risk_index = round(fused_prob * 100)`
- severity bins (Minor/Moderate/Major) via fixed thresholds

The fusion is deterministic (not trained end-to-end). That makes it easier to explain and audit, especially in a safety-adjacent domain.

---

## 9) Explainability (grounded, not generative)

`src/explainer.py` produces structured explanations from evidence fields:

- shared enzymes/targets and CYP-related heuristics for mechanism hints
- TWOSIDES notes when signals exist
- recommendation strings

The explanation uses the same evidence that drove the tiering and fusion. It is not an LLM.

---

## 10) How the project ships: UI, API, and artifacts

### 10.1 Streamlit UI

The Streamlit app is `streamlit_app.py`. It loads the predictor once (cached), accepts two drug selections, runs prediction, and renders the result plus component scores and evidence summaries.

Run locally:

```bash
streamlit run streamlit_app.py
```

### 10.2 CLI

CLI prediction is in `src/predict.py`:

```bash
python src/predict.py Warfarin Aspirin
python src/predict.py Warfarin Aspirin --json
```

### 10.3 REST API (FastAPI)

The API is `src/api.py`. The module header includes a dev run command:

```bash
uvicorn src.api:app --reload
```

Key endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- `GET /drugs/{name}`
- `GET /drugs/{name}/interactions`

---

## 11) Deployment checklist (what breaks first)

DrugInsight is a Python deployment with a few required artifacts:

- model checkpoint: `models/ddi_model.pt`
- processed data under `data/processed` (especially `feature_metadata.json`)
- a working RDKit + PyTorch + Torch Geometric install

### Streamlit deployment

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

If you clone from GitHub, install Git LFS first so large assets pull correctly.

### API deployment

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Put a reverse proxy in front if you expose it publicly, and keep the batch request cap (the API limits batches to 100 pairs).

### Smoke tests to run after deploy

- a known pair resolves to DrugBank IDs
- both graphs can be built when SMILES exists (RDKit is working)
- the model loads and returns a prediction
- `/health` reports the model as loaded

### Things to log for debugging

- evidence tier used (1/2/3)
- whether each drug had SMILES
- fusion weights and component scores
- request latency

---

## 12) Next improvements that are worth the time

- Train the severity head with a real loss if you have severity labels
- Learn fusion weights (or a small fusion model) and compare against deterministic fusion
- Calibrate probabilities (reliability curves, temperature scaling)
- Make symmetry exact with symmetric pair features (`embed_a + embed_b`, `|embed_a - embed_b|`)
- Add a small regression test set: a handful of pairs that should always resolve and run without throwing

## DrugInsight v2: from ML ideas to deployment

This is the "how it works" document for DrugInsight v2. It covers the pipeline end-to-end: how DrugBank/TWOSIDES data becomes features, what the GNN is doing with molecular graphs, how evidence fusion works at inference, and how the project is deployed as a Streamlit app and a REST API.

If you only read one paragraph, read this: DrugInsight is a hybrid DDI system. It uses a molecular-structure model (a GNN) when it can, but it does not treat the model output as the final word. It also pulls in curated DrugBank interaction evidence, shared biology overlap (enzymes/targets/CYP), and TWOSIDES pharmacovigilance signals, then blends those into a risk score and an explanation you can check.

---

## 0) Quick ML concepts (only what this project uses)

### Supervised learning for pairs

At training time, the model sees examples like:

- Input: drug A, drug B, plus a small engineered feature vector for the pair
- Output: a label (interaction vs no interaction)

The model learns a function \( f(A, B) \to p \), where \( p \) is a probability-like interaction score. In practice, that score is produced from a single logit trained with `BCEWithLogitsLoss`.

### Why molecules become graphs

Molecules are naturally graphs: atoms are nodes and bonds are edges. A graph neural network (GNN) does message passing along bonds, learns local chemical patterns, and produces an embedding vector for the whole molecule. In DrugInsight, that embedding is the main "structure signal" used for pair prediction.

---

## 1) What the system returns

Given two drug names or DrugBank IDs, DrugInsight returns:

- A predicted interaction decision (boolean)
- A fused probability and a `risk_index` scaled to 0-100
- A severity label (Minor / Moderate / Major)
- Evidence and component scores (what came from DrugBank rules vs ML vs TWOSIDES)
- A mechanistic explanation and recommendation that are tied to evidence fields

You can get that output through:

- Streamlit UI: `streamlit_app.py`
- CLI inference: `src/predict.py`
- REST API: `src/api.py`

---

## 2) Data and the basic "contract"

DrugInsight mixes two evidence families.

### 2.1 DrugBank-derived curated and biological context

DrugBank is used for canonical drug IDs and name resolution, curated interaction lookups, shared-biology context (enzymes/targets/CYP and related overlap), and SMILES structures when they exist.

### 2.2 TWOSIDES pharmacovigilance signals

TWOSIDES contributes post-market association signals, summarized into pair-level features like `twosides_max_prr`, `twosides_num_signals`, and `twosides_found`.

### 2.3 Canonical pair keys (order-invariant)

DDIs are symmetric for the purpose of feature lookup: (A, B) and (B, A) should map to the same row.

DrugInsight uses an order-invariant key based on sorted DrugBank IDs:

```text
pair_key = f"{min(id_a, id_b)}||{max(id_a, id_b)}"
```

This prevents duplicate training rows and keeps inference lookups consistent.

---

## 3) Feature pipeline: what the ML model consumes

Each pair becomes three inputs:

1. A molecular graph for drug A
2. A molecular graph for drug B
3. A fixed-length auxiliary feature vector (`extra_dim`, currently 12)

### 3.1 Auxiliary feature vector (12 features)

The vector is built in `build_normalized_feature_vector()` in `src/feature_extractor.py`. The names and order are defined in `data/processed/feature_metadata.json`:

1. `shared_enzyme_count`
2. `shared_target_count`
3. `shared_transporter_count`
4. `shared_carrier_count`
5. `shared_pathway_count`
6. `shared_major_cyp_count`
7. `cyp3a4_shared`
8. `cyp2d6_shared`
9. `cyp2c9_shared`
10. `twosides_max_prr`
11. `twosides_num_signals`
12. `twosides_found`

Normalization is part of the interface:

- `*_shared` features and `twosides_found` are binarized (0/1)
- other numeric features are clipped to a cap and scaled:

```text
min(raw_value, cap) / cap
```

### 3.2 Preprocessing outputs and scaling metadata

`src/preprocess_data.py` generates the processed assets under `data/processed` and writes `feature_metadata.json` so training and inference use the same caps and ordering.

The processed directory typically includes:

- `drug_catalog.csv`
- `drugbank_smiles_filtered.csv`
- `twosides_mapped.csv`
- `drugbank_interactions_enriched.csv.gz`
- `preprocess_manifest.json`

If you change feature order or caps, treat it as an interface change. Old weights can still load and still produce numbers, but those numbers may no longer mean what you think they mean.

---

## 4) Molecular graphs: SMILES to PyTorch Geometric

Graph construction happens in `src/mol_graph.py`:

- Parse SMILES with RDKit
- Add explicit hydrogens (`Chem.AddHs(mol)`)
- Build atom/node features (basic chemistry + a few flags)
- Build bond/edge features (bond type, conjugation, ring membership)
- Make the graph undirected by adding edges in both directions

This is why RDKit is a hard dependency in the environment.

---

## 5) The model: shared GNN encoder + pairwise classifier

The model is two-stage:

1. Encode each molecule into an embedding using a shared GNN encoder
2. Fuse the two embeddings plus auxiliary features in an MLP head that outputs an interaction logit

### 5.1 Two towers with shared weights (Siamese pattern)

Both drugs run through the same encoder:

```text
embed_a = gnn(graph_a)
embed_b = gnn(graph_b)
logit, severity_logits = classifier(embed_a, embed_b, extra)
```

Sharing the encoder weights forces both drugs into the same representation space.

### 5.2 GNN encoder: AttentiveFP

The encoder is `torch_geometric.nn.AttentiveFP` (configured in `src/gnn_encoder.py`). Conceptually:

- Message passing updates atom states using neighbors and bond features
- Attention weights which neighbors matter more
- A readout step produces a graph-level embedding vector

### 5.3 Pairwise classifier: MLP fusion head

`src/ddi_classifier.py` defines `DDIClassifier`, which:

- concatenates `[embed_a, embed_b, extra]`
- runs the result through an MLP trunk (Linear + BatchNorm + ReLU + Dropout)
- outputs:
  - a probability head (one logit)
  - a severity head (three logits)

The training loop in `src/train.py` uses `BCEWithLogitsLoss` for the interaction logit. The severity head exists, but it is not trained with its own loss in the current training path.

---

## 6) Training strategy: the parts that matter

### 6.1 Objective

Training is binary classification:

- Label: 0/1 for whether the pair is a curated positive
- Loss: `nn.BCEWithLogitsLoss()`

### 6.2 Drug-level split (leakage control)

Rather than splitting rows randomly, the training pipeline splits by drug IDs. The goal is to reduce leakage where the model sees the same drug in both training and validation and learns to "recognize" drugs instead of learning interactions.

### 6.3 Hard negative sampling (make negatives plausible)

Random negatives are often too easy in DDI datasets. DrugInsight uses hard negatives: negative pairs that share biological overlap with positives (enzymes/targets/CYP), while explicitly excluding:

- known curated positives
- TWOSIDES-mapped pairs (to reduce label contamination)

This happens in `FeatureExtractor.sample_hard_negatives()`, which scores candidates using the engineered overlap features and keeps a fraction of the hardest candidates.

### 6.4 Symmetry augmentation

Because the head sees `[embed_a, embed_b]` as ordered inputs, training randomly swaps the two embeddings for some batches. The auxiliary vector is symmetric already (it is built from order-invariant overlaps).

### 6.5 Optimization and evaluation

`src/train.py` uses:

- Adam with separate learning rates for encoder vs classifier
- ReduceLROnPlateau keyed off validation AUC
- gradient clipping

Reported metrics include accuracy, ROC AUC, and average precision.

---

## 7) Inference-time architecture: evidence tiers + deterministic fusion

This is where the project stops being "just a model" and becomes an evidence-aware predictor.

### 7.1 Evidence tier routing

`FeatureExtractor.determine_evidence_tier()` assigns:

- Tier 1: direct curated DrugBank interaction hit
- Tier 2: structured evidence exists (shared biology overlap and/or TWOSIDES signal)
- Tier 3: no structured evidence, so you rely on structure-only ML (if possible)

### 7.2 SMILES gating (cold start behavior)

The GNN path requires SMILES for both drugs to build graphs. In `DDIPredictor.predict()` (`src/predict.py`):

- if one or both drugs lack SMILES, ML inference cannot run
- tier 1 can still return a response because it can rely on curated evidence

### 7.3 Fusion formula (rule score + ML score + TWOSIDES score)

Fusion is deterministic, not learned end-to-end:

```text
fused_prob = w_rule * rule_score + w_ml * ml_prob + w_twosides * twosides_score
```

- `rule_score` comes from engineered shared-biology overlaps (capped)
- `ml_prob` is the GNN+MLP output (after sigmoid)
- `twosides_score` is derived from normalized PRR and signal counts
- weights are chosen based on which evidence sources exist for the pair

The system then maps `fused_prob` to:

- `risk_index = round(fused_prob * 100)`
- severity bins (Minor/Moderate/Major) using fixed thresholds

The upside is you can audit why a score moved. The downside is you're not optimizing fusion jointly with the model.

---

## 8) Explainability: grounded, not generative

`src/explainer.py` builds explanations from structured evidence fields:

- shared enzymes/targets (mechanistic hints)
- CYP-related heuristics for inhibition/induction/substrate patterns
- a TWOSIDES note when signals exist
- a recommendation string

The explanation is based on the same features used in fusion. It is not a language model inventing a story.

---

## 9) Interfaces: Streamlit, CLI, and REST

### 9.1 Streamlit UI

`streamlit_app.py` is the interactive UI. It:

- caches the predictor so the model loads once
- lets users pick drugs (with SMILES available in the list)
- displays severity, risk index, confidence, evidence summaries, and component scores

### 9.2 CLI inference

`src/predict.py` supports simple CLI usage:

```bash
python src/predict.py Warfarin Aspirin
python src/predict.py Warfarin Aspirin --json
```

### 9.3 REST API (FastAPI)

`src/api.py` defines a FastAPI app and loads the predictor at startup. The header comment includes the dev run command:

```bash
uvicorn src.api:app --reload
```

Endpoints include:

- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- `GET /drugs/{name}`
- `GET /drugs/{name}/interactions`

---

## 10) Deployment: what you ship, what breaks first

DrugInsight deploys as a Python application with a few practical requirements:

- Model checkpoint: `models/ddi_model.pt`
- Processed assets: `data/processed/*` (especially `feature_metadata.json`)
- A working RDKit + PyTorch + Torch Geometric environment

### 10.1 Streamlit deployment (simplest)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

If you clone the repo from GitHub, make sure Git LFS is installed so large assets pull correctly.

### 10.2 API deployment (service-style)

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

If you're exposing it beyond localhost, put a reverse proxy in front and add basic request limits. Batch prediction already enforces a maximum of 100 pairs per request.

### 10.3 Treat these as artifacts

These should be treated like versioned artifacts:

- `models/ddi_model.pt`
- `data/processed/*`

If preprocessing changes feature order/caps, you need to rebuild processed assets and typically retrain, otherwise inference can silently drift because the model is reading the wrong input definition.

### 10.4 Operational smoke tests

Before you call it deployed, verify:

- The server can parse SMILES and build graphs (RDKit works)
- The model loads on the target machine
- `feature_metadata.json` exists and matches the model's expected `extra_dim`
- `GET /health` returns `"model": "loaded"` if running the API

### 10.5 Common failure modes

The failures you will see most often:

- A drug has no SMILES (structure model can't run)
- Missing files under `data/processed` or mismatched filenames
- PyTorch / Torch Geometric mismatch on the target environment
- RDKit install issues

Log the evidence tier, SMILES availability, fusion weights, and request latency. That alone answers most "why did it do that?" debugging sessions.

---

## 11) Practical next improvements (if you keep working on it)

If you want the next steps that usually pay off:

- Train the severity head with a proper loss if you have severity labels
- Learn fusion weights (or a small fusion module) and compare against deterministic fusion
- Calibrate probabilities (reliability curves, temperature scaling) so scores behave like probabilities
- Make symmetry exact with symmetric pair features (e.g., `embed_a + embed_b`, `|embed_a - embed_b|`) instead of relying only on random swapping
- Add a tiny regression test set: a handful of pairs that should always resolve and run without errors

## DrugInsight v2 ML & Evidence Architecture Sheet

This document describes the core *mostly-ML* techniques used in DrugInsightv2: how data is turned into features and graphs, how a graph neural network (GNN) is trained for drug-drug interaction (DDI) prediction, and how predictions are fused with curated and pharmacovigilance evidence (DrugBank + TWOSIDES) in a tiered decision system.

The main idea is a **hybrid model**:
1. A **structure-based ML model** that learns from molecular graphs (GNN encoder) + pairwise auxiliary features.
2. A **knowledge-based evidence layer** that computes risk/interaction evidence from shared biology and TWOSIDES pharmacovigilance signals.
3. A **routing + fusion policy** that decides when to trust curated evidence, when to run ML, and how to blend them.

---

## 1) Data & Feature Pipeline (What/Why/How)

### 1.1 Canonical drug registry + canonical pair keys
**What:** The system relies on canonical DrugBank IDs and a **pair key** that is invariant to input order.

**Why:** DDI is (to a first approximation) symmetric: `(drug_a, drug_b)` should behave the same as `(drug_b, drug_a)`. Canonical pair keys prevent duplicated training rows and ensure feature lookups are consistent.

**How:** In `src/feature_extractor.py`, the key is formed by sorting IDs alphabetically:
`pair_key = f"{min(id_a,id_b)}||{max(id_a,id_b)}"`.

---

### 1.2 Feature extraction contract (What the ML model consumes)
**What:** For each drug pair, the ML model consumes:
1. Two learned graph embeddings (one per drug).
2. A fixed-length **auxiliary feature vector** of length `extra_dim` (currently 12).

**Why:** Pure graph embeddings may miss systematic pharmacology/ADME evidence patterns (e.g., shared CYP enzymes). Auxiliary features provide structured biochemical signals that complement learned representations.

**How:** The feature vector is built in `build_normalized_feature_vector()` in `src/feature_extractor.py`.

The feature names (and order) come from `data/processed/feature_metadata.json`:
1. `shared_enzyme_count`
2. `shared_target_count`
3. `shared_transporter_count`
4. `shared_carrier_count`
5. `shared_pathway_count`
6. `shared_major_cyp_count`
7. `cyp3a4_shared`
8. `cyp2d6_shared`
9. `cyp2c9_shared`
10. `twosides_max_prr`
11. `twosides_num_signals`
12. `twosides_found`

In `src/feature_extractor.py`:
- "*_shared" features and `twosides_found` are **binarized** (`> 0` => 1.0 else 0.0).
- Other numeric features are **clipped/normalized** by per-feature caps from `feature_metadata.json`:
  - `min(raw_value, cap) / cap`

---

### 1.3 Preprocessing and normalization metadata
**What:** The preprocessing step computes feature caps and writes them to `feature_metadata.json`.

**Why:** Training and inference must share the *same scaling contract*. Caps prevent extreme values (outliers) from dominating learning.

**How:** `src/preprocess_data.py`:
- Computes a 99th-percentile cap for many numeric features.
- Special-cases:
  - CYP presence-like binary features capped at 1.
  - `shared_major_cyp_count` capped conceptually at 3.
- Writes `feature_order`, `extra_dim`, and `feature_caps` into `feature_metadata.json`.

It also builds:
- `drug_catalog.csv` (names/synonyms, SMILES availability)
- `drugbank_smiles_filtered.csv` (RDKit-validated SMILES)
- `twosides_mapped.csv` (TWOSIDES aggregated to canonical pair key)
- `drugbank_interactions_enriched.csv.gz` (positive pairs with shared biology + TWOSIDES features)
- `preprocess_manifest.json` (row counts and mapping statistics)

---

### 1.4 Molecular graph construction
**What:** Each drug's SMILES string is converted into a PyTorch Geometric graph with:
- Atom/node features
- Bond/edge features
- Edge index structure

**Why:** Graph neural networks learn molecular representations directly from topology and chemistry rather than hand-coding global descriptors.

**How:** `src/mol_graph.py`:
1. `smiles_to_graph(smiles)` uses RDKit to parse SMILES.
2. It calls `Chem.AddHs(mol)` to include explicit hydrogens.
3. Node features (`atom_features`) include:
   - atomic number, degree, formal charge
   - hybridization (encoded as int)
   - aromatic flag, total H count, ring membership
   - normalized mass
4. Edge/bond features (`bond_features`) include:
   - bond type flags: single/double/triple/aromatic
   - conjugation flag
   - ring membership
5. The graph is made undirected by adding edges in both directions.

---

## 2) Structure-Based ML Model

DrugInsight's ML model is a **two-stage architecture**:
1. A **GNN encoder** that converts each molecular graph into a vector embedding.
2. A **pairwise classifier head** that fuses the two embeddings with auxiliary features to output a DDI logit.

---

### 2.1 Siamese (two-tower) pairing with a shared encoder
**What:** The same GNN encoder is applied independently to `drug_a` and `drug_b` to produce `embed_a` and `embed_b`.

**Why:** Weight sharing enforces that both drugs are embedded in the same representation space, supporting symmetric pair modeling.

**How:** In `src/train.py` and `src/predict.py`, the pipeline is:
```text
embed_a = gnn(graphs_a)
embed_b = gnn(graphs_b)
logit, severity_logits = classifier(embed_a, embed_b, extra)
```

This is a "two-tower" (Siamese) pattern with shared weights.

---

### 2.2 GNN encoder: AttentiveFP
**What:** The encoder is `torch_geometric.nn.AttentiveFP`.

**Why:** AttentiveFP is designed for graph-level molecular property learning. It uses attention mechanisms in message passing to weight informative neighboring atoms/bonds, and performs iterative refinement steps (`num_timesteps`).

**How:** `src/gnn_encoder.py` defines:
- `AttentiveFP(in_channels=8, edge_dim=6, hidden_channels=128, out_channels=256, num_layers=4, num_timesteps=2, dropout=0.3)`
- Forward call takes `(x, edge_index, edge_attr, batch)` and returns a **graph embedding** of size `out_channels` per molecule.

---

### 2.3 Pairwise classifier: embedding + auxiliary feature fusion (MLP)
**What:** `src/ddi_classifier.py` defines `DDIClassifier`, an MLP that:
- concatenates `embed_a`, `embed_b`, and the auxiliary feature vector `extra`
- transforms them through a deep trunk
- outputs two heads:
  - `prob_head` (1 logit for interaction)
  - `severity_head` (3 logits for severity classes)

**Why:** The classifier learns a non-linear decision boundary over both:
- learned structure embeddings (from the GNN)
- interpretable engineered signals (shared enzymes/targets/CYP + TWOSIDES)

**How:** In `DDIClassifier.forward()`:
- `x = torch.cat([embed_a, embed_b, extra], dim=-1)`
- `x` passes through a Sequential trunk of Linear + BatchNorm1d + ReLU + Dropout.
- `prob_head = nn.Linear(128, 1)` returns the interaction logit.

**Note on severity head:** In `src/train.py`, training uses `BCEWithLogitsLoss` with labels and only consumes the probability logit (`logits, _ = classifier(...)`). The severity head exists and is returned, but it is not currently trained with a separate loss in `train.py`.

---

## 3) Training Strategy (What/Why/How)

### 3.1 Supervised objective: binary DDI classification
**What:** Each training example is a pair `(drug_1, drug_2)` with a binary label `label in {0,1}`.

**Why:** DrugInsight frames DDI prediction as a classification problem.

**How:** In `src/train.py`:
- Loss: `nn.BCEWithLogitsLoss()`
- Model output: one logit per pair, then used directly in BCE-with-logits.
- Targets: labels converted to `float` and moved to device.

---

### 3.2 Hard negative sampling (contrastive selection without explicit metric learning)
**What:** Negatives are not random pairs. They are **hard negatives**: negatives that share biological overlap with positives (e.g., shared enzymes/CYP) but are labeled as non-interactions.

**Why:** Random negatives are often "too easy" and can lead to a model that learns dataset artifacts rather than robust interaction discrimination.

**How:** In `FeatureExtractor.sample_hard_negatives()`:
1. Candidate pairs are drawn by randomly sampling drug IDs from the split's drug pool.
2. Exclusions prevent label contamination:
   - any pair that appears in known DrugBank positives (`known_pair_keys`)
   - any pair that appears as a TWOSIDES-mapped pair (`twosides_pair_keys`)
3. A "hardness" score is computed from engineered shared-count features:
   - large weights for shared enzymes/targets/major CYP, smaller weights for transporters/pathways/carriers
4. The sampler keeps a proportion of hardest negatives (`hard_fraction`) and fills the remainder with easier negatives.

Training uses those sampled negatives to build `train_df`/`val_df`.

---

### 3.3 Split strategy: avoid leakage by splitting on drugs
**What:** Train/validation splits are created by splitting over sets of drug IDs, not random rows.

**Why:** If the same drug appears in both train and validation, the model may "memorize" drug embeddings rather than generalize to new drug combinations.

**How:** `src/train.py`:
- collects `all_drugs` from `drug_1_id` and `drug_2_id`
- splits `all_drugs` into `train_drugs` and `val_drugs`
- selects positives where both `drug_1_id` and `drug_2_id` are in the split's drug set
- samples negatives inside that split

---

### 3.4 Symmetry augmentation: random swap of drug order
**What:** During training, embeddings for `drug_a` and `drug_b` are sometimes swapped.

**Why:** Even with canonical features, the model input includes `embed_a` and `embed_b` concatenated; a neural network can learn order-specific artifacts. Random swapping enforces approximate invariance.

**How:** In `train_epoch()`:
```text
if torch.rand(1).item() > 0.5:
    embed_a, embed_b = embed_b, embed_a
```
The auxiliary feature vector is symmetric because it is built from set intersections and order-invariant pair features.

---

### 3.5 Optimization and scheduling
**What:** The model uses:
- Adam optimizer with different learning rates for encoder vs classifier
- ReduceLROnPlateau scheduler based on validation AUC

**Why:** Different learning rates stabilize optimization: the GNN encoder typically needs a smaller LR than the classifier head.

**How:** In `src/train.py`:
- optimizer groups:
  - `gnn.parameters()` with lr `3e-5`
  - `classifier.parameters()` with lr `1e-4`
- scheduler: `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)` stepped with `val_auc`
- gradient clipping: `clip_grad_norm_(..., max_norm=1.0)`

---

### 3.6 Evaluation metrics: AUC and average precision
**What:** The system reports:
- accuracy
- ROC AUC
- average precision (AP)

**Why:** DDI datasets are often imbalanced; AUC and AP capture ranking quality better than accuracy alone.

**How:** `eval_epoch()` computes:
- `probs = sigmoid(logits)`
- `roc_auc_score(y_true, y_prob)`
- `average_precision_score(y_true, y_prob)`

---

## 4) Inference-Time Architecture: Tiered Evidence Fusion

At inference, DrugInsight does not always run the ML model and does not always trust it equally. Instead it uses:
1. **Evidence tier routing**
2. **A fusion formula** that blends:
   - a rule-based "shared biology" score
   - the ML probability
   - a TWOSIDES-derived score

This creates an interpretable hybrid system.

---

### 4.1 Evidence tier determination (What/Why/How)
**What:** `FeatureExtractor.determine_evidence_tier()` assigns one of:
1. `tier_1_direct_drugbank` (curated direct hit)
2. `tier_2_evidence_fusion` (some structured evidence exists)
3. `tier_3_structure_only` (no structured evidence)

**Why:** It controls compute and prevents ML from being used when curated evidence should dominate.

**How:**
- Tier 1: `direct_drugbank_hit == 1`
- Tier 2: any "structured evidence" feature is > 0, including:
  - shared enzyme/target/transporter/carrier/pathway counts
  - shared major CYP count or CYP-specific flags
  - `twosides_found`
- Tier 3: none of the above

---

### 4.2 SMILES availability gating (cold-start behavior)
**What:** The ML model requires molecular graphs (SMILES). If one or both drugs lack SMILES, ML inference cannot run.

**Why:** Graph encoders need structure.

**How:** In `DDIPredictor.predict()` (`src/predict.py`):
- it checks `has_smiles_a` and `has_smiles_b`
- if missing SMILES and tier is not tier 1, it returns an error
- tier 1 can still produce a response without ML because it uses rule-based probability

---

### 4.3 Fusion formulas (rule score + ML score + TWOSIDES score)
DrugInsight's fusion layer is *not* learned end-to-end; it is a deterministic weighted blend.

#### 4.3.1 Rule-based "shared biology" score
**What:** `rule_score` is derived from engineered overlap counts.

**Why:** Shared enzymes/targets/CYPs are a strong mechanistic proxy for interactions (especially metabolic interactions).

**How:** In `_compute_fusion()`:
- each shared-count contributes a bounded term via `min(...)` with small linear scaling
- CYP overlap has the strongest per-feature contribution among these engineered signals
- `rule_score` is then capped to `[0, 0.9]` (before final weighted blending)

---

#### 4.3.3 TWOSIDES score
**What:** A TWOSIDES-based confidence score derived from PRR and signal counts.

**Why:** TWOSIDES provides pharmacovigilance evidence that can support real-world association.

**How:** In `_twosides_score()`:
- `twosides_max_prr` is normalized by a cap from `feature_metadata.json`
- `twosides_num_signals` is normalized by its cap
- the final score is a weighted sum:
  - 0.7 * PRR component + 0.3 * signal component

---

#### 4.3.4 Weighting and final fused probability
**What:** The final probability is computed as:
`fused_prob = w_rule * rule_score + w_ml * ml_prob + w_twosides * twosides_score`

**Why:** Different evidence types should dominate under different situations (e.g., if no structured overlap exists but TWOSIDES signal exists, TWOSIDES should matter more).

**How:** In `_compute_fusion()` the weights depend on:
- whether structured overlap exists (`has_structural`)
- whether TWOSIDES found a signal (`twosides_found`)

Then `fused_prob` is clipped to `[0, 1]`, mapped to:
- a `risk_index = round(fused_prob * 100)`
- severity bins:
  - >= 70 Major
  - >= 40 Moderate
  - else Minor

---

## 5) Explainability: Mechanism- grounded structured explanations

### 5.1 What the explainer does
**What:** `src/explainer.py` produces a structured explanation containing:
- an interaction summary (severity + confidence)
- metabolic mechanism (shared enzymes and CYP inhibition/induction logic)
- pharmacodynamic mechanism (shared targets)
- pharmacovigilance note if TWOSIDES exists
- a clinical recommendation string

**Why:** This avoids "black box" explanations and uses mechanistic and evidence-grounded heuristics.

**How:** It leverages `FeatureExtractor`'s structured context fields like:
- `shared_enzymes`, `shared_targets`
- `twosides_found`, `max_PRR`, etc.

It contains hard-coded CYP inhibitor/inducer/substrate keyword sets and falls back to generic substrate competition statements when specific matches are not found.

---

## 6) Key Architectural Concepts (Quick Reference)

1. **Graph representation learning**: SMILES -> RDKit molecule -> PyG graph -> embedding.
2. **AttentiveFP molecular encoder**: attention-guided message passing for chemistry-aware embeddings.
3. **Siamese two-tower design**: shared GNN encoder embeds both drugs into the same space.
4. **Feature fusion MLP**: concatenates embeddings + engineered normalized overlap features.
5. **Hard negative sampling**: negatives are chosen to be plausible (high overlap) while excluding known positives.
6. **Drug-level split**: avoids leakage by holding out drug IDs in validation.
7. **Symmetry augmentation**: random swapping of drug embeddings during training.
8. **Evidence-tier routing**: tier 1 bypasses ML (curated direct hit), tier 2 fuses ML with rule and TWOSIDES, tier 3 uses ML only.
9. **Deterministic fusion policy**: weights depend on evidence availability; easier to audit than an end-to-end trained fusion model.
10. **Mechanistic explainability**: rule-based explanation text generated from structured evidence context (no LLM).

---

## 7) Practical "How to Think About It"

- If you want to improve *pure ML performance*, focus on:
  - negative sampling quality
  - calibration of fusion weights or learning a fusion module
  - training the severity head explicitly (if severity labels exist)
  - augmenting symmetry invariance (e.g., explicit symmetric pooling instead of concatenation)

- If you want to improve *end-user trust/interpretability*, focus on:
  - validating the curated evidence fields (`direct_drugbank_hit`, shared counts, TWOSIDES mappings)
  - improving mapping quality and feature scaling caps to reduce brittle behavior
  - ensuring explanation text aligns with the actual evidence fields used in fusion

