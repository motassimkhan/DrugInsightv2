# DrugInsight v2 - Complete Project Documentation (Notion Ready)

## 1. Project overview

### 1.1 What DrugInsight does
DrugInsight is an explainable drug-drug interaction (DDI) prediction system. A user provides two drugs (common names or DrugBank IDs), and the system returns a structured risk assessment with mechanistic context.

Core outputs:
- interaction decision (`true` or `false`)
- fused probability score (`0.0` to `1.0`)
- risk index (`0` to `100`)
- severity tier (`Minor`, `Moderate`, `Major`)
- mechanistic explanation grounded in curated and inferred evidence
- evidence breakdown by source (DrugBank, TWOSIDES, neural model)

This is not a generic text explainer over static rules. The prediction path combines molecular graphs, engineered biological overlap features, curated interaction tables, and post-market pharmacovigilance data.

### 1.2 Why this project exists
Most practical DDI workflows run into one of two problems:
- pure rules are explainable but can miss novel or weakly documented combinations
- pure ML can detect patterns but often fails to explain why a pair was flagged

DrugInsight is designed as a hybrid system that keeps explanation first while still using learned molecular representations. The guiding idea is: if a model says risk is high, the user should be able to see what evidence supported that claim.

### 1.3 Scope and intended use
Current scope:
- research and pre-clinical decision support
- education and exploratory DDI analysis
- offline/local inference with transparent evidence outputs

Out of scope:
- autonomous clinical prescribing
- replacing physician/pharmacist review
- regulated medical-device claims

---

## 2. ML problem definition

### 2.1 Formal task
The core model solves a supervised binary classification task over drug pairs:
- input: `(drug_a, drug_b)`
- target: interaction label (`1` positive, `0` negative)
- output: interaction probability

Severity is not directly trained as a separate supervised target in the current training loop. It is derived downstream from fused risk thresholds.

### 2.2 Learning paradigm and supervision
Learning type:
- supervised deep learning for pair classification

Supervision source:
- positive pairs from direct DrugBank interaction evidence in enriched interaction tables
- negative pairs from controlled hard negative sampling (feature-aware, exclusion-aware)

This setup deliberately avoids naive random negatives, which often create easy examples and inflate metrics.

### 2.3 Generalization goal
The project is trying to generalize to unseen pair combinations, not just memorize known rows. That is why split logic is done at the drug level instead of random row-level splitting.

### 2.4 Decision layer design
The final user-facing decision is a fusion result, not raw neural output only:
- tiered evidence routing
- weighted fusion between rule score, ML score, and TWOSIDES score
- uncertainty labels attached to each evidence channel

---

## 3. Data sources and contracts

### 3.1 Data sources
Primary sources used by the pipeline:
- DrugBank structured exports (drugs, interactions, enzymes, targets, pathways, transporters, carriers, external IDs)
- DrugBank structural data (SMILES)
- TWOSIDES adverse event co-reporting dataset

Repository-level references:
- `TWOSIDES.csv` (or `data/raw/TWOSIDES.csv`)
- processed artifacts under `data/processed`

### 3.2 Core processed artifacts
The preprocessing pipeline builds and maintains these contract files:
- `data/processed/drug_catalog.csv`
- `data/processed/drugbank_smiles_filtered.csv`
- `data/processed/twosides_mapped.csv`
- `data/processed/drugbank_interactions_enriched.csv.gz`
- `data/processed/feature_metadata.json`
- `data/processed/preprocess_manifest.json`
- `data/processed/druginsight.db`

### 3.3 Snapshot counts (from current manifest)
Based on the latest checked-in manifest snapshot:
- drug catalog rows: `19,830`
- valid SMILES rows: `14,606`
- mapped TWOSIDES pairs: `4,700`
- enriched DrugBank interaction rows: `1,455,276`

These values are data-version dependent and expected to change as the source exports change.

### 3.4 Pair canonicalization contract
The system canonicalizes pair identity using sorted IDs:
- `pair_key = min(drug_id_a, drug_id_b) || max(drug_id_a, drug_id_b)`

Why this matters:
- prevents directional duplication (`A-B` and `B-A`)
- stabilizes joins across DrugBank and TWOSIDES
- enables indexed retrieval in SQLite

### 3.5 Mapping reliability notes
TWOSIDES concept mapping includes multiple matching strategies:
- exact normalized name
- synonym map
- RxCUI bridge
- manual alias overrides

Mapping outcomes are tracked (`mapped`, `ambiguous`, `unresolved`) and logged in manifest stats to avoid silent failures.

---

## 4. Feature engineering

### 4.1 Feature philosophy
The structured feature block is intended to summarize known pharmacological overlap and post-market signal strength in a form that can be fused with graph embeddings.

### 4.2 Current feature set (12 dimensions)
From `feature_metadata.json`, the active feature order is:
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

### 4.3 Biological interpretation of key features
- shared enzyme/target counts indicate overlap in metabolic or pharmacodynamic pathways
- major CYP overlap flags likely metabolic competition or modulation risk
- pathway overlap captures higher-level systems biology intersection
- TWOSIDES metrics provide real-world co-reporting signal support

### 4.4 Normalization and caps
Feature normalization is metadata-driven:
- values are clipped at predefined caps (`feature_caps`)
- normalized to `[0, 1]` by division with cap
- binary indicators remain binary

This keeps input scales stable and limits outlier distortion from long-tailed TWOSIDES values.

### 4.5 Hard negative sampling design
Negatives are sampled from candidate pairs while excluding:
- known positives
- known curated or TWOSIDES pair keys

Each candidate receives a hardness score based on overlap features. The sampler keeps a blend of hard and easier negatives (default hard fraction in code: `0.7`) to avoid collapse into trivial learning.

---

## 5. Model architecture

### 5.1 Molecular graph encoder
Module: `src/gnn_encoder.py`

Backbone:
- `AttentiveFP` from PyTorch Geometric

Default configuration in code:
- atom feature channels: `8`
- edge feature channels: `6`
- hidden channels: `128`
- embedding output: `256`
- message-passing layers: `4`
- readout timesteps: `2`
- dropout: `0.3`

The encoder transforms each drug molecule graph into a dense learned embedding.

### 5.2 Graph construction
Module: `src/mol_graph.py`

SMILES conversion pipeline:
- parse SMILES with RDKit
- add explicit hydrogens
- build atom-level feature vectors
- build bidirectional bond edges and edge attributes
- emit PyG `Data(x, edge_index, edge_attr)` object

Atom features include atomic number, degree, charge, hybridization, aromatic flag, hydrogen count, ring flag, and mass term. Bond features include bond type one-hot signals, conjugation, and ring participation.

### 5.3 Pair classifier
Module: `src/ddi_classifier.py`

Classifier input:
- embedding A (`256`)
- embedding B (`256`)
- extra feature vector (`12` via metadata contract)

MLP trunk:
- input -> `512` -> `256` -> `128`
- each hidden stage uses batch norm, ReLU, dropout

Output heads:
- `prob_head`: binary logit
- `severity_head`: 3-way logits (present in architecture, not fully used in current train objective)

### 5.4 Why this architecture is practical
This design balances:
- expressive structural representation (GNN)
- interpretable tabular evidence (engineered overlap features)
- manageable inference latency for local and API use

---

## 6. Training pipeline

### 6.1 Script and runtime
Primary script: `src/train.py`

Key stages:
1. resolve compute device (CUDA or CPU fallback)
2. load feature metadata
3. load and validate SMILES
4. precompute graph cache for valid molecules
5. load enriched positive pairs
6. split drugs into disjoint train/validation sets
7. generate negative samples per split
8. build dataloaders with custom graph collate
9. train with BCE + gradient clipping
10. evaluate with accuracy, AUC, AP
11. save best model by validation AUC

### 6.2 Optimization details
- optimizer: Adam
- learning rates:
  - GNN: `3e-5`
  - classifier: `1e-4`
- weight decay: `5e-4`
- scheduler: ReduceLROnPlateau on validation metric
- early stopping patience: implemented in loop

### 6.3 Batch construction notes
Each batch contains:
- graph batch for drug A
- graph batch for drug B
- aligned extra feature tensor
- binary labels

The training loop randomly swaps A/B embeddings half the time to reduce order sensitivity.

### 6.4 Current observed performance snapshot
From `training_log.md`, recent run highlights include:
- best validation AUC around `0.7065`
- early stopping triggered after plateau
- strong class-level confusion reporting per epoch

Interpretation:
- model learns usable signal, but there is room for calibration and external validation work before high-stakes deployment.

---

## 7. Inference and evidence fusion

### 7.1 Inference entry point
Primary inference logic: `src/predict.py` via `DDIPredictor`.

Prediction steps:
1. resolve input drugs to canonical IDs
2. fetch pair context and engineered features
3. build molecular graphs
4. run GNN + classifier for ML score
5. determine evidence tier
6. route through direct, fused, or ML-only decision logic
7. build structured response with explanation and uncertainty

### 7.2 Evidence tiers
- `tier_1_direct_drugbank`: direct curated interaction exists
- `tier_2_evidence_fusion`: no direct hit, but overlap/TWOSIDES evidence exists
- `tier_3_structure_only`: mostly molecular inference fallback

### 7.3 Fusion behavior
Fusion combines three channels:
- rule score (shared mechanisms)
- ML score (neural output)
- TWOSIDES score (pharmacovigilance signal)

Weights are conditional on available evidence. This prevents over-trusting one channel when another has stronger direct support.

### 7.4 Severity and risk translation
- fused probability -> risk index (`round(prob * 100)`)
- severity thresholds:
  - `>= 70`: Major
  - `>= 40`: Moderate
  - `< 40`: Minor

### 7.5 Uncertainty labels
The output includes confidence descriptors per channel plus an overall confidence summary. This is useful when surfacing predictions in UI or API clients that need to filter low-certainty results.

---

## 8. Explainability system

### 8.1 Design
Module: `src/explainer.py`

Explainer is rule- and context-driven, not LLM-driven. It composes explanation text from known context fields:
- shared enzymes
- shared targets
- known curated mechanism text
- TWOSIDES PRR strength
- severity level and recommendation templates

### 8.2 Mechanism composition priority
1. if DrugBank mechanism text is available, use it as primary mechanism source
2. otherwise infer mechanism from shared enzymes/targets and CYP patterns
3. append pharmacovigilance signal context when present
4. append severity-linked clinical consequence language

### 8.3 Recommendation generation
Recommendations are severity-conditioned and adjusted by evidence type (for example, shared enzyme evidence can trigger dose-monitoring language).

### 8.4 Why this matters
Users receive not just a score but a narrative trail that explains what evidence likely drove risk. This is important for trust, debugging, and downstream review.

---

## 9. Application architecture

### 9.1 Core code organization
Active stack (preferred):
- `src/preprocess_data.py`
- `src/feature_extractor.py`
- `src/mol_graph.py`
- `src/gnn_encoder.py`
- `src/ddi_classifier.py`
- `src/train.py`
- `src/predict.py`
- `src/explainer.py`
- `src/api.py`
- `src/app.py`

### 9.2 Data access pattern
The system avoids loading huge pair tables entirely into memory during serving. Instead, it creates a local SQLite database (`druginsight.db`) and indexes pair keys for fast lookup.

### 9.3 Contract dependencies
Inference assumes presence of:
- model checkpoint
- processed CSV artifacts
- feature metadata file aligned with classifier input contract

If these drift out of sync, inference adaptation logic may partially recover, but prediction quality can degrade.

---

## 10. Interfaces: CLI, API, and UI

### 10.1 CLI
Two CLI pathways exist in repository history, with the packaged command currently defined as:
- `druginsight=drug_insight.cli:main`

Supported command shapes include:
- single prediction
- batch CSV prediction
- drug profile lookup

### 10.2 FastAPI service
Module: `src/api.py`

Main endpoints:
- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- drug lookup routes

The app loads predictor at startup to avoid per-request model cold starts.

### 10.3 Streamlit app
Module: `src/app.py`

UI capabilities:
- searchable/selectable drugs with valid structures
- one-click prediction
- risk and confidence panels
- mechanistic text and recommendation rendering
- component score visualization for fusion channels

The interface is tuned for interactive analysis and demonstration, not high-throughput production workloads.

---

## 11. Deployment architecture

### 11.1 Current reliable deployment mode
The practical default is local or controlled-server deployment with prepared data and model artifacts.

### 11.2 Local setup commands
Install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Rebuild processed artifacts:
```bash
python src/preprocess_data.py --rebuild-all
```

Train:
```bash
python src/train.py
```

Run Streamlit:
```bash
streamlit run src/app.py
```

Run API:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 11.3 Production-grade deployment plan (recommended)
A stable production shape would include:
- containerized inference image with pinned Python and CUDA stack
- read-only mounted model and processed artifacts
- FastAPI behind reverse proxy/API gateway
- auth + rate limiting
- structured logging and metrics
- health/readiness probes

### 11.4 Artifact strategy for deployment
Treat these as immutable release artifacts per version:
- model checkpoint file
- `feature_metadata.json`
- `preprocess_manifest.json`
- commit hash and build metadata

---

## 12. Validation and evaluation

### 12.1 Existing tests
`tests/test_predict.py` currently covers:
- known high-risk interaction scenario
- low-risk scenario sanity check
- invalid input behavior
- output schema contracts

### 12.2 Existing training metrics
Current training loop reports:
- train/validation accuracy
- validation ROC AUC
- validation average precision
- confusion matrix counts

### 12.3 What is still missing
For stronger scientific confidence, add:
- external test cohort evaluation
- calibration metrics (ECE/Brier)
- subgroup performance by drug class
- temporal validation split
- ablation studies for fusion components

---

## 13. Data engineering and preprocessing internals

### 13.1 Preprocessing responsibilities
`src/preprocess_data.py` handles:
- catalog build and normalization
- SMILES validation/canonicalization
- TWOSIDES mapping and aggregation
- enriched interaction table creation
- feature cap metadata generation
- manifest generation

### 13.2 SQLite build path
`src/build_sqlite_db.py` constructs indexed tables from chunked CSV reads. This avoids large peak memory use and speeds pair-key retrieval during inference.

### 13.3 Manifest role
`preprocess_manifest.json` acts as a lightweight lineage record:
- row counts
- source paths
- feature metadata snapshot
- mapping stats

This should be extended with artifact hashes for stronger reproducibility guarantees.

---

## 14. Known gaps and technical risks

### 14.1 Dual predictor stack risk
There are two predictor paths in repository:
- `src/*` stack (newer 12-feature contract)
- `drug_insight/*` stack (older behavior, 6-feature constructor usage in parts)

Risk:
- different entrypoints may yield different outputs for same pair
- API and package behavior can drift over time

### 14.2 Contract drift risk
If checkpoint feature expectations, metadata, and runtime feature order diverge, model compatibility issues appear. Some adaptation logic exists, but it is a fallback, not a substitute for aligned training/inference contracts.

### 14.3 Service-level hardening gaps
- permissive CORS defaults
- no built-in auth/rate limiting
- limited request auditing
- limited incident monitoring hooks

### 14.4 Clinical communication risk
Even with explanations, output can be over-interpreted. Interfaces must keep explicit disclaimers and encourage pharmacist/physician confirmation.

---

## 15. Security, governance, and compliance

### 15.1 Data licensing and governance
Operational deployment should document:
- DrugBank usage/license constraints
- TWOSIDES dataset usage constraints
- retention and redistribution policy

### 15.2 API security baseline
Before exposing publicly:
- enforce authentication
- implement per-client rate limits
- redact sensitive logs where needed
- enforce payload and timeout controls

### 15.3 Responsible use policy
Each surface (CLI/UI/API docs) should state:
- research support intent
- no standalone clinical decision authority
- mandatory qualified-review recommendation

---

## 16. MLOps lifecycle plan

### 16.1 Versioning model
Release each model as a bundle:
- `model.bin` / checkpoint
- feature metadata contract
- preprocess manifest
- code version and environment lock

### 16.2 Monitoring strategy
Track in production:
- feature distribution drift
- score distribution drift
- confidence distribution drift
- disagreement between rule and ML channels
- error rates and latency

### 16.3 Retraining triggers
Candidate triggers:
- new DrugBank release
- significant TWOSIDES updates
- monitored quality degradation
- major mapping-quality shifts

### 16.4 Registry and rollback
Use a model registry with stage promotion and rollback support so bad releases can be reverted quickly without rebuilding ad hoc.

---

## 17. Suggested roadmap

### 17.1 Phase 1: unify and stabilize
- consolidate on one predictor implementation path
- align API/CLI imports with single source of truth
- freeze feature contract and publish compatibility table

### 17.2 Phase 2: production readiness
- add Docker build and CI checks
- add integration tests for API and batch scoring
- add auth, rate limiting, observability

### 17.3 Phase 3: scientific robustness
- external cohort validation
- calibration and threshold analysis
- evidence ablations and error analysis dashboard

### 17.4 Phase 4: next-model expansion
- evaluate alternative graph architectures
- integrate planned knowledge-graph embedding work from `PLAN.md`
- compare fixed fusion vs learned meta-fusion

---

## 18. Notion structure recommendation

If splitting into Notion subpages, use this structure:
1. Product and problem framing
2. Data contracts and preprocessing
3. Model architecture and training
4. Inference, fusion, and explainability
5. Interface layer (CLI/API/UI)
6. Deployment and operations
7. Risk register and governance
8. Roadmap and next experiments

This split keeps each page focused while preserving full traceability from data to final prediction.

---

## 19. Glossary

- DDI: drug-drug interaction
- GNN: graph neural network
- PRR: proportional reporting ratio
- CYP: cytochrome P450 family enzymes
- AUC: area under ROC curve
- AP: average precision
- Evidence tier: routing class indicating available certainty sources

---

## 20. Current source-of-truth recommendation

For ongoing development, treat these as authoritative until consolidation is complete:
- active modeling/inference logic: `src/`
- current feature contract: `data/processed/feature_metadata.json`
- preprocessing lineage: `data/processed/preprocess_manifest.json`
- training checkpoint target in current training script: `models/ddi_model_reprocessed.pt`

When introducing new features or retraining, update all four together and version them as one release unit.
