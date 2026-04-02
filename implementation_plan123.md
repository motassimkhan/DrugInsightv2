# DrugInsight Clinical Translation Layer: Implementation Plan

## 1. Objective
Build a data-driven, non-hardcoded clinical translation layer that converts mechanistic DDI signals into clinically useful outcomes, recommendations, and uncertainty decisions with source-backed evidence.

## 2. Scope
In scope:
- Retrieval-based evidence grounding
- Learned clinical outcome/recommendation translation
- Probability calibration
- Abstention/uncertainty handling
- Integration into existing prediction/API/UI flow

Out of scope (v1):
- Patient-specific dose personalization
- EHR integration
- Real-time external API calls at inference

## 3. Design Principles
- No hardcoded clinical verdict rules.
- Separate mechanistic plausibility from clinical actionability.
- Every clinical claim must be evidence-grounded (citation attached).
- Abstain when evidence is weak or conflicting.
- Keep modules independently testable and replaceable.
- Treat DrugBank as mechanistic support, not primary clinical recommendation evidence.

## 4. Target Architecture
### 4.1 New modules
- `src/clinical_retriever.py`
  - Embedding + vector retrieval over curated corpus.
  - Returns top-k evidence chunks with score and metadata.
- `src/clinical_translator.py`
  - Learns mapping from base features + retrieved evidence features to:
    - outcome probabilities
    - recommendation class
- `src/clinical_calibrator.py`
  - Calibrates translation probabilities (isotonic or Platt/temperature).
- `src/abstention.py`
  - Learns/derives uncertainty decision from calibration confidence + retrieval quality + disagreement.
- `src/clinical_pipeline.py`
  - Orchestrates retriever -> translator -> calibrator -> abstention.

### 4.2 Data and model artifacts
- `data/clinical/evidence_corpus.jsonl`
- `data/clinical/source_registry.csv`
- `data/clinical/source_documents/`
- `data/clinical/outcome_schema.json`
- `data/clinical/train_labels.csv`
- `data/clinical/splits/{train,val,test}.csv`
- `models/clinical/retriever_index/*`
- `models/clinical/translator.*`
- `models/clinical/calibrator.json`
- `models/clinical/abstention.*`

### 4.3 Integration points
- `src/predict.py`
  - Add optional clinical stage after existing mechanistic prediction.
  - Return both `mechanistic_assessment` and `clinical_assessment`.
- `src/app.py`
  - Split UI into two clear blocks:
    - Mechanistic Evidence
    - Clinical Recommendation (with citations + uncertainty badge)

### 4.4 Evidence source hierarchy
Primary clinical evidence:
- FDA-approved labeling and regulatory text
  - `Drugs@FDA` approval/label metadata
  - `DailyMed` Structured Product Labeling (SPL) sections
- High-quality pair-specific clinical references added to the corpus with citation metadata

Secondary supporting evidence:
- DrugBank direct interaction records
- TWOSIDES pharmacovigilance signals
- Existing DrugInsight structural/mechanistic features

Tertiary explanatory context only:
- DrugBank monograph text such as pharmacodynamics and mechanism-of-action
- Single-drug descriptive text used only to enrich explanations, never to drive recommendation class

### 4.5 Source acquisition pipeline
Source 1: `DailyMed`
- Use DailyMed web services to search SPLs by drug name/NDC and retrieve SPL XML by `setid`
- Parse only clinically relevant sections such as boxed warnings, contraindications, drug interactions, warnings/precautions, adverse reactions, and dosage/administration when interaction-specific
- Store raw XML metadata and normalized text chunks with `setid`, product name, label type, effective date, section code, and source URL

Source 2: `openFDA drug label`
- Use the official drug label API as a machine-friendly index over SPL-derived labeling JSON
- Use harmonized identifiers (`openfda` fields) to improve mapping between names, NDCs, application numbers, and label records
- Treat openFDA as a retrieval/discovery layer and retain source provenance to the underlying label record

Source 3: `Drugs@FDA`
- Pull approval-oriented metadata to anchor products to approved application records
- Use Drugs@FDA as the approval-status and recency check because DailyMed "in use" labeling may differ from the latest FDA-approved labeling

Source 4: `DrugBank`
- Retain only pair-level interaction presence, structured biology, and direct mechanism text as secondary features
- Exclude generic single-drug monograph text from the recommendation-training target generation path

### 4.6 Acquisition implementation files
- `src/fetch_dailymed_labels.py`
  - Discover SPLs, download XML, and normalize selected clinical sections
- `src/fetch_openfda_labels.py`
  - Query openFDA label endpoint for index/identifier reconciliation
- `src/fetch_drugsfda_metadata.py`
  - Pull approval metadata and effective-date anchors
- `src/build_evidence_corpus.py`
  - Merge normalized chunks into `evidence_corpus.jsonl`
- `src/validate_evidence_corpus.py`
  - Enforce provenance, deduplicate text, and reject weak/missing metadata

## 5. Canonical Output Contract (Clinical Layer)
`clinical_assessment` object:
- `outcomes`: list of `{name, probability, confidence}`
- `recommendation`: one of `avoid | monitor | dose_adjust | generally_safe | uncertain`
- `recommendation_confidence`: float [0,1]
- `uncertain`: bool
- `uncertainty_reason`: short string
- `evidence`: list of `{source_id, source_type, quote_snippet, score, date}`
- `model_version`: string
- `calibration_version`: string

## 6. Features for Translator Model
### 6.1 Base DrugInsight features
- `rule_score`, `ml_score`, `twosides_score`
- `evidence_tier`
- shared counts (enzymes/targets/transporters/carriers/pathways/major CYP)
- direct DrugBank hit flag
- direct DrugBank mechanism present flag

### 6.2 Retrieval-derived features
- top-k relevance scores
- aggregated evidence strength statistics (mean/max/top1 gap)
- source diversity (DailyMed, Drugs@FDA, pair reference)
- recency features (source age)
- contradiction signal (if available)
- section-type features (boxed warning, contraindication, interaction, adverse reaction)

### 6.3 Optional textual features (v1.1)
- reduced-dimension embedding summary of top-k chunks

## 7. Training Strategy
### 7.1 Dataset preparation
- Build labeled dataset in two phases:
  - MVP weak supervision:
    - bootstrap labels from pair-level DrugBank interaction severity where explicit
    - join retrieved FDA-label evidence and existing mechanistic features
  - Phase 2 curated supervision:
    - clinician-reviewed pair-level outcomes and recommendations
- Label quality gates:
  - source citation required
  - reviewer agreement tracking
  - timestamped evidence provenance
- Label governance:
  - double-review for high-risk (`avoid`) labels
  - adjudication workflow for disagreement
  - immutable label snapshots per training run

### 7.2 Model choice
Initial: Gradient boosting model (LightGBM/XGBoost) for fast iteration and interpretability.
Future: Lightweight transformer head if needed.

### 7.3 Calibration
- Train on train set, fit calibrator on validation set only.
- Report ECE, Brier, reliability curves per outcome/recommendation.

### 7.4 Abstention policy/model
- Inputs: calibrated confidence, retrieval quality, source disagreement, OOD score.
- Output: `uncertain` flag + reason.
- Optimize for reducing unsafe confident errors.

### 7.5 MVP constraints from current data reality
- Do not assume DrugBank monograph text is clinically reliable enough for recommendation training
- Do not require a large clinician-annotated dataset for MVP launch
- Recommendation model must be able to run in weakly supervised mode with conservative abstention defaults

## 8. Evaluation Plan
Primary metrics:
- Outcome AUROC/PR-AUC
- Recommendation macro-F1
- Major-risk PPV and false-major rate
- Calibration metrics (ECE/Brier)
- Coverage vs. risk for abstention

Clinical acceptance metrics:
- Clinician agreement rate
- Actionability score of recommendations
- Citation relevance score

## 9. Milestones
### Phase 0 (Audit + baseline)
- Freeze current baseline outputs on a fixed benchmark set
- Create side-by-side report: current layer vs new clinical layer
- Define acceptance thresholds before implementation rollout
- Inventory how many drugs in the current catalog can be mapped to DailyMed/openFDA/Drugs@FDA identifiers

### Sprint 1 (Foundation)
- Create data schema and outcome ontology
- Build source fetchers and evidence corpus loader
- Build retriever index
- Define `clinical_assessment` contract

### Sprint 2 (First model)
- Train baseline translator model on weak supervision
- Add calibrator
- Offline evaluation notebook/report

### Sprint 3 (Uncertainty + integration)
- Implement abstention module
- Integrate into `predict.py` and `app.py`
- Add unit/integration tests

### Sprint 4 (Clinical hardening)
- Clinician review loop
- Error analysis and label refinement
- Threshold tuning and release criteria

### Rollout and fallback
- Launch clinical layer behind feature flag (default OFF first)
- Run shadow mode in app/API logs for one cycle
- Switch default ON only after release gates pass
- Keep immediate rollback to mechanistic-only recommendation path

## 10. Testing Plan
- Unit tests:
  - retriever deterministic behavior on fixed corpus
  - translator input schema and output bounds
  - calibrator monotonicity and serialization
  - abstention decision logic
- Integration tests:
  - end-to-end clinical pipeline response shape
  - citation presence for non-uncertain recommendations
  - fallback to uncertain on missing evidence
- Regression tests:
  - tier-specific behavior stability
  - no crash on sparse/unknown pairs

## 11. Risk Register and Mitigations
- Overestimating DrugBank free text as clinical evidence
  - Mitigation: restrict DrugBank free text to explanation support and secondary features only.
- Source mismatch between DailyMed "in use" labeling and latest FDA-approved labeling
  - Mitigation: cross-check approval metadata with Drugs@FDA and preserve source/date in retrieval output.
- Data leakage between retrieval corpus and labels
  - Mitigation: split by pair and source document families.
- Overconfident wrong recommendations
  - Mitigation: strong calibration + abstention threshold.
- Citation mismatch/hallucinated evidence
  - Mitigation: enforce retrieval-only citations and snippet checks.
- Distribution drift
  - Mitigation: scheduled recalibration and drift monitoring dashboard.

## 12. Release Gates (Go/No-Go)
- ECE <= 0.08 on recommendation confidence
- False-major rate reduced by >= 25% vs current baseline
- Major-risk PPV >= baseline + 10% relative lift
- Citation relevance score >= 0.85 on reviewed sample
- Clinician agreement >= 0.75 on adjudicated validation set
- Uncertain rate between 10% and 35% (outside this requires review)

## 13. Concrete Repo Task List
1. Add new files:
   - `src/clinical_retriever.py`
   - `src/clinical_translator.py`
   - `src/clinical_calibrator.py`
   - `src/abstention.py`
   - `src/clinical_pipeline.py`
   - `src/fetch_dailymed_labels.py`
   - `src/fetch_openfda_labels.py`
   - `src/fetch_drugsfda_metadata.py`
   - `src/build_evidence_corpus.py`
   - `src/validate_evidence_corpus.py`
2. Add data directory:
   - `data/clinical/` with schema files
3. Add training script:
   - `src/train_clinical_translator.py`
4. Add evaluation script:
   - `src/evaluate_clinical_layer.py`
5. Update integration:
   - `src/predict.py` response payload
   - `src/app.py` UI sections
6. Add tests:
   - `tests/test_clinical_retriever.py`
   - `tests/test_clinical_translator.py`
   - `tests/test_clinical_pipeline.py`

## 14. Self-Review Log (5 Passes)
Pass 1:
- Checked plan completeness (data, model, eval, integration).
- Added output contract and release gates.

Pass 2:
- Added Phase 0 baseline/audit and rollout fallback strategy.
- Ensured safe launch path with feature-flag and shadow mode.

Pass 3:
- Converted release gates into concrete numeric targets.
- Added explicit uncertain-rate operating band.

Pass 4:
- Added label governance details (double review/adjudication/snapshots).
- Reinforced reproducibility requirements for clinical labels.

Pass 5:
- Reviewed for alignment with your current architecture (`predict.py`, `app.py`, tiered evidence).
- Checked consistency with non-hardcoded requirement end-to-end.

Post-review revision:
- Narrowed DrugBank to a secondary mechanistic role.
- Added concrete FDA/DailyMed/openFDA acquisition pipeline and source-validation files.
