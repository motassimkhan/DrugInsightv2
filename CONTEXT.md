# DrugInsight v0.1 — Agent Context File

> **Last updated:** 2026-03-27 · **Maintained by:** Antigravity (Gemini)  
> **Purpose:** Give any agent an accurate picture of where the project is, what has been done, and what the known issues are. Update this file after every significant session.

---

## 1. Project Overview

DrugInsight is a **Drug-Drug Interaction (DDI) prediction system** built for research use. It fuses:

1. **GNN-based ML model** (graph neural network on molecular structures via SMILES)
2. **DrugBank structural rules** (shared enzymes, targets, transporters, carriers, pathways)
3. **TWOSIDES pharmacovigilance signals** (PRR-based post-market surveillance)

The output is a **risk index (0–100)**, a **severity label** (Minor / Moderate / Major / Uncertain), a **mechanism explanation**, and a **clinical recommendation** — all generated without an LLM.

**Live app:** Streamlit Community Cloud (repo: `AymanUzayr/DrugInsightv2`)

---

## 2. Repository Layout

```
DrugInsightv2/
├── src/
│   ├── app.py                  # Streamlit UI entry point
│   ├── predict.py              # Core DDIPredictor class ← most important file
│   ├── explainer.py            # Explainer class — generates mechanism text
│   ├── feature_extractor.py   # FeatureExtractor — builds context dict from DrugBank data
│   ├── ddi_classifier.py      # DDIClassifier (PyTorch MLP head)
│   ├── gnn_encoder.py         # GNNEncoder (PyG graph encoder)
│   ├── mol_graph.py           # SMILES → PyG graph conversion
│   ├── train.py               # Model training script
│   ├── calibrate_fusion.py    # Fits Platt-scaling LogReg for fusion weights
│   ├── evaluate.py            # Evaluation metrics (AUC, PR-AUC, accuracy)
│   ├── preprocess_data.py     # Full preprocessing pipeline from DrugBank XML
│   ├── enrich_interactions.py # Adds structural features to interaction CSV
│   ├── build_sqlite_db.py     # Migrates CSVs → SQLite for low-memory deployment
│   ├── api.py                 # FastAPI wrapper (optional)
│   └── test_predict.py        # Quick smoke-test script
│
├── models/
│   ├── ddi_model_reprocessed.pt  # ← active model (use this)
│   ├── ddi_model.pt              # older checkpoint
│   ├── ddi_model_crossattn.pt    # cross-attention variant (experimental)
│   └── fusion_weights.json       # Platt-scaling LR coefficients [rule, ml, twosides]
│
├── data/processed/
│   ├── druginsight.db             # SQLite DB (used by app.py at runtime)
│   ├── drugbank_drugs.csv         # Drug metadata incl. SMILES, toxicity, pharmacodynamics
│   ├── drugbank_smiles.csv        # DrugBank ID → SMILES mapping
│   ├── drugbank_enzymes.csv       # Drug-enzyme relationships
│   ├── drugbank_targets.csv       # Drug-target relationships
│   ├── drugbank_transporters.csv  # Drug-transporter relationships
│   ├── drugbank_carriers.csv      # Drug-carrier relationships
│   ├── drugbank_pathways.csv      # Drug-pathway relationships
│   ├── drugbank_interactions_enriched.csv  # Full training/eval data with structural features
│   ├── drug_catalog.csv           # Rich textual info: pharmacodynamics, MoA, toxicity
│   ├── twosides_features_filtered.csv     # Processed TWOSIDES signals per drug pair
│   ├── feature_metadata.json      # Feature order + normalization caps for the model
│   └── rxnorm_bridge.csv          # RxNorm → DrugBank ID mapping
│
├── PLAN.md                   # Original 4-stage knowledge graph plan
├── PREPROCESSING_CHANGELOG.md
├── training_log.md           # Epoch-by-epoch training metrics
├── requirements.txt
└── CONTEXT.md                # ← this file
```

---

## 3. Prediction Pipeline (`predict.py`)

### 3.1 Evidence Tiers

`FeatureExtractor.extract()` assigns one of three tiers to every drug pair:

| Tier | Condition | Handler in `predict.py` |
|------|-----------|------------------------|
| `tier_1_direct_drugbank` | DrugBank has a curated interaction record for this exact pair | `_direct_hit_result()` |
| `tier_2_evidence_fusion` | No direct hit but structural evidence exists (shared enzymes/targets/etc.) | `_compute_fusion()` |
| `tier_3_ml_only` | No structural evidence at all | `_ml_only_result()` |

### 3.2 Score Components

| Score | Description | Range |
|-------|-------------|-------|
| `rule_score` | Heuristic from shared biology (enzymes, targets, transporters, carriers, pathways, major CYPs) | 0–0.9 |
| `ml_score` | GNN + MLP classifier sigmoid output | 0–1 |
| `twosides_score` | Weighted PRR + signal count from TWOSIDES | 0–1 |

### 3.3 Fusion Weights (`fusion_weights.json`)

Platt-scaling **Logistic Regression** trained on the validation split of `drugbank_interactions_enriched.csv`. Coefficients:

```json
{
  "coef": [0.081, 4.941, 0.021],   // [rule_score, ml_score, twosides_score]
  "intercept": -2.490
}
```

Used only in **Tier-2** (`_compute_fusion`). Tier-1 and Tier-3 use fixed manual weights.

### 3.4 Severity Thresholds

```python
risk_index >= 70  →  Major
risk_index >= 40  →  Moderate
risk_index  < 40  →  Minor
```

---

## 4. Recent Changes & Bug Fixes

### Session: 2026-03-27

#### Fix 1 — `test_predict.py`: `context` was undefined (NameError)
- **File:** `src/test_predict.py`
- **Problem:** Line 27 called `explainer.explain(context, result)` but `context` is an internal variable inside `predictor.predict()` and is never returned.
- **Fix:** Removed the standalone `Explainer` call. The explanation is already embedded in `result` — fields `summary`, `mechanism`, `recommendation`, `full_explanation` are returned by `DDIPredictor.predict()` directly.

#### Fix 2 — `predict.py`: Tier-1 ignored ML entirely → wrong severity
- **File:** `src/predict.py`, method `_direct_hit_result()`
- **Problem (A):** When a DrugBank direct hit was found, `ml_score` was hardcoded to `0.0` and weights were `Rule 1.0 · ML 0.0`. The ML model output (often "Moderate") was completely discarded.
- **Problem (B):** Safety-net NTI override fired whenever *any* toxicity keyword appeared (`"life-threatening"`, `"fatal"`, etc.), regardless of structural evidence. For **Amoxicillin + Ibuprofen** (one shared enzyme, CYP2C8), this forced the result to **Major** at risk=75 — clinically incorrect.
- **Fix:**
  - Tier-1 now blends: **`blended_prob = 0.70 × rule_score + 0.30 × ml_prob`** when SMILES are available
  - Safety-net NTI override now requires **`rule_score >= 0.15`** (real structural evidence) before upgrading to Major
  - `component_scores.weights` now reflects actual weights used: `{rule: 0.70, ml: 0.30, twosides: 0.0}`
- **Expected result for Amoxicillin + Ibuprofen:**
  - rule_score ≈ 0.14 (one CYP2C8 enzyme) → below 0.15 threshold → override does NOT fire
  - blended = 0.70×0.14 + 0.30×0.811 ≈ **0.341 → Moderate** ✓

---

## 5. Key Classes & Their Responsibilities

### `DDIPredictor` (`predict.py`)
- Central orchestrator. Load with `DDIPredictor()` (auto-resolves model path).
- Main method: `predict(drug_a: str, drug_b: str) -> dict`
- Returns a rich dict with: `interaction`, `severity`, `risk_index`, `probability`, `mechanism`, `recommendation`, `component_scores`, `uncertainty`, `evidence`, `full_explanation`.

### `FeatureExtractor` (`feature_extractor.py`)
- Resolves drug names → DrugBank IDs, looks up shared biology, fetches TWOSIDES signals.
- Returns a `context` dict — the central data object passed to all downstream components.
- Key output keys: `drug_a`, `drug_b`, `shared_enzymes`, `shared_targets`, `shared_pathways`, `twosides_found`, `twosides_max_prr`, `evidence_tier`, `direct_drugbank_hit`.

### `Explainer` (`explainer.py`)
- `explain(context: dict, prediction: dict) -> dict`
- `context` comes from `FeatureExtractor.extract()`.
- `prediction` needs: `interaction` (bool), `probability` (float), `severity_idx` (int -1 to 2).
- Returns: `summary`, `mechanism`, `recommendation`, `full_text`, `supporting_evidence`, `severity`, `severity_color`, `confidence`.
- Priority order for mechanism text: DrugBank curated text → enzyme/CYP analysis → shared targets → TWOSIDES → drug catalog pharmacodynamics → generic fallback.

### `GNNEncoder` (`gnn_encoder.py`)
- PyTorch Geometric GNN. Input: molecular graph. Output: 128-dim embedding.

### `DDIClassifier` (`ddi_classifier.py`)
- MLP head. Input: two embeddings + normalized feature vector. Output: interaction logit.

---

## 6. Data Notes

- **Active model:** `models/ddi_model_reprocessed.pt` (trained on reprocessed DrugBank data, ~epoch 20+)
- **Feature vector** is built by `build_normalized_feature_vector()` in `feature_extractor.py` using the order and caps defined in `data/processed/feature_metadata.json`
- **SQLite DB** (`druginsight.db`) is used by `app.py` in production (Streamlit) to avoid loading large CSVs into memory. Built by `build_sqlite_db.py`.
- **TWOSIDES** raw data is in `TWOSIDES.csv` (root, 100 MB). Processed versions are in `data/processed/`.
- **DrugBank XML** (`full_database.xml`, 1.9 GB, at root) is only needed for re-preprocessing.

---

## 7. Known Issues / TODOs

- [ ] **Tier-1 TWOSIDES not used:** Even when TWOSIDES has a strong signal for a Tier-1 pair, it's excluded from the blended score. Consider adding it when `twosides_score > 0`.
- [ ] **`calibrate_fusion.py` was trained on Tier-1 pairs only** (line 84–88) — the Platt LR is Tier-2-specific. If it is ever re-run, Tier-2 pairs should be used instead to avoid distribution mismatch.
- [ ] **`_fusion_weight_dict()`** returns the raw LR coefficients as display weights. These are scale-dependent (ML coeff = 4.94) and look misleading in the UI. A normalization step would improve UX.
- [ ] **Streamlit memory:** On Streamlit Community Cloud the app is limited to ~1 GB RAM. Large CSVs are migrated to SQLite via `build_sqlite_db.py`. If memory errors recur, check that `app.py` is reading from `druginsight.db` and not any raw CSV.
- [ ] **`test_predict.py`** only tests Amoxicillin + Acetaminophen. Expand to cover: a Tier-3 pair (no DB hit, no SMILES), a Tier-2 pair, and a pair with TWOSIDES signal.
- [ ] **`severity_idx = -1` (Uncertain)** is generated by the feature extractor for pairs with missing data but the Platt LR does not produce this — it comes from the extractor's `evidence_tier` logic. Verify `SEVERITY_LABELS[-1]` paths are exercised in tests.

---

## 8. Running Locally

```bash
# From project root, activate venv first:
.\venv\Scripts\activate

# Run the smoke test:
python src/test_predict.py

# Run the Streamlit app:
streamlit run src/app.py

# Re-fit fusion weights (after model retraining):
python src/calibrate_fusion.py

# Evaluate model performance:
python src/evaluate.py
```

---

## 9. Conversation History Summary

| Date | Topic |
|------|-------|
| 2026-03-27 | Fixed Tier-1 severity bug (Amoxicillin+Ibuprofen → Moderate), fixed `context` NameError in `test_predict.py` |
| 2026-03-27 | Implemented Platt-scaling fusion meta-model (`calibrate_fusion.py`), clinical recommendations from drug catalog, safety-net NTI overrides |
| 2026-03-26 | Evaluated model performance (ROC-AUC, PR-AUC); added unit/integration tests |
| 2026-03-26 | Migrated large CSVs to SQLite (`build_sqlite_db.py`) to resolve Streamlit memory limits |
| 2026-03-25 | Analyzed `training_log.md` — model shows healthy learning with no overfitting |
| 2026-03-24 | Created `PLAN.md` for knowledge graph; resolved Git LFS deployment issues; deployed to Streamlit Cloud |
