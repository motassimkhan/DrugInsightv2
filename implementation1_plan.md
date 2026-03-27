# Dynamic Data-Driven Severity Calculation & Explainer Enhancements

This document outlines the strategy to completely eliminate hardcoded probabilities and arbitrary heuristic fusion weights from the DDI pipeline. It establishes a data-driven approach to severity prediction and enriches explanations using existing catalog domain knowledge.

## Proposed Changes

### 1. Learned Fusion & Platt Scaling (`src/calibrate_fusion.py`)
We will replace the manual `0.72` base probability and hardcoded fusion weights (`rule: 0.45, ml: 0.35, twosides: 0.20`) with a mathematically grounded **Meta-Model (Platt Scaling)**. 

*   **Create a new script (`calibrate_fusion.py`):** This script will iterate over the existing validation dataset.
*   **Extract Signals:** For every validation pair, it extracts the `rule_score` (structural overlap), the raw `ml_prob` (from the main GNN/DDI classifier), and the [twosides_score](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/predict.py#177-183).
*   **Fit Logistic Regression:** It will train a `sklearn.linear_model.LogisticRegression` using these 3 signals as features ($X$) against the true interaction labels ($y$). 
*   **Export Weights:** The mathematically optimal coefficients (the learned fusion weights) are serialized into `models/fusion_weights.json`. The output of this Logistic Regression is a fully calibrated probability natively between 0 and 1. 

### 2. Eliminating Hardcoding ([src/predict.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/predict.py))
*   **Load Weights:** [predict.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/predict.py) will load `fusion_weights.json` on startup.
*   **Dynamic Severity:** The [_direct_hit_result](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/predict.py#268-305) and [_compute_fusion](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/predict.py#306-400) functions will be simplified. Instead of forcing `probability = 0.72` for direct hits, or relying on ad-hoc baseline bonuses, they will pass the trio of `[rule_score, ml_prob, twosides_score]` into the learned Logistic Regression equation to compute the exact calibrated `probability`.
*   **Result:** A known interaction with confidently low ML and TWOSIDES evidence natively outputs a low calibrated probability (fitting the "Minor" Risk Index), fixing the Amoxicillin/Acetaminophen false positive.
*   **False Negative Safety Net:** To avoid dangerously under-classifying severe interactions when data is missing, we will implement two safeguards in [predict.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/predict.py) directly overriding the calibrated meta-model:
    1. **High-Risk Drug Override:** Using the newly extracted `toxicity` fields from [drug_catalog.csv](file:///c:/Users/Ayman/Documents/DrugInsightv2/data/processed/drug_catalog.csv), if either drug has a known narrow therapeutic index or high toxicity, the minimum severity defaults to **Major**, bypassing the model entirely.
    2. **Uncertainty Floor:** If a pair is a known DrugBank hit but lacks structural data to compute `ml_prob` AND lacks a TWOSIDES signal, the system assumes maximum uncertainty and sets a **Moderate** safety floor (Risk Index \ge 45), guaranteeing clinicians are warned to monitor the patient.

### 3. Explainer & Catalog Enhancements ([src/feature_extractor.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/feature_extractor.py), [src/explainer.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/explainer.py))
Since the severity string is not being parsed from text, we will enrich the explanation generation for clinicians using standard drug facts.
*   **Update [feature_extractor.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/feature_extractor.py):** Extract all clinically relevant textual columns from [drug_catalog.csv](file:///c:/Users/Ayman/Documents/DrugInsightv2/data/processed/drug_catalog.csv) (e.g., `indication`, `pharmacodynamics`, `mechanism_of_action`, `metabolism`, `absorption`, `half_life`, `toxicity`, `categories`, `description`) for each drug. Pass these new fields into the `context` dictionary under `drug_a_info` and `drug_b_info`.
*   **Update [explainer.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/explainer.py):** Use the dynamic, calibrated Risk Index to advise the clinician on severity. If a robust specific mechanism isn't found, the explainer seamlessly inserts the drug's catalog `pharmacodynamics` and `metabolism` profiles to provide high-quality context instead of generic placeholders.

## Verification Plan
*   **Validation Check:** Run `calibrate_fusion.py` to ensure the Logistic Regression converges and produces sensible weights that map to actual validation metrics.
*   **End-to-End Prediction Test:** Run [predict.py](file:///c:/Users/Ayman/Documents/DrugInsightv2/src/predict.py) from the terminal for Amoxicillin + Acetaminophen to verify the Risk Index calculates safely below $\ge 70$. 

## User Review Required
> [!IMPORTANT]
> The plan is finalized for **Platt Scaling & Learned Fusion Weights** along with Explainer drug catalog additions. If this plan looks correct, approve it and I will enact the code changes across the pipeline!
