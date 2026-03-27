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

