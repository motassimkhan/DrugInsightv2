Project: DrugInsight — a drug-drug interaction prediction framework.

## Context
We are rebuilding the preprocessing and prediction pipeline from scratch. 
The current system has critical bugs: enrichment joins silently produce zeros 
for shared_enzyme_count, shared_target_count etc., and TWOSIDES mapping fails 
because the RxNorm bridge is fragile. The full implementation plan is below.

## Existing files you can READ but must NOT modify:
- data/raw/drugbank_full.xml
- data/raw/TWOSIDES.csv
- data/processed/drugbank_drugs.csv
- data/processed/drugbank_interactions.csv
- data/processed/drugbank_enzymes.csv
- data/processed/drugbank_targets.csv
- data/processed/drugbank_transporters.csv
- data/processed/drugbank_carriers.csv
- data/processed/drugbank_pathways.csv
- data/processed/drugbank_smiles.csv
- data/processed/drugbank_external_ids.csv
- src/mol_graph.py
- src/gnn_encoder.py
- models/best_model.pt

## Files you will CREATE or OVERWRITE:
- src/preprocess_data.py — full rebuild pipeline, runnable as `python -m src.preprocess_data --rebuild-all`
- src/train.py — updated for 12-feature vector and new enriched pair table
- src/feature_extractor.py — updated to use new canonical tables
- src/predict.py — updated with tier 1/2/3 routing
- data/processed/drug_catalog.csv
- data/processed/drugbank_smiles_filtered.csv — rebuilt with RDKit validation
- data/processed/drugbank_interactions_enriched.csv — canonical pair table
- data/processed/twosides_mapped.csv — TWOSIDES after name/synonym/RxCUI mapping
- data/processed/feature_metadata.json — normalization caps for training and inference
- The classifier input dim must read extra_dim from feature_metadata.json, 
  not be hardcoded. Update ddi_classifier.py accordingly.
## Implementation Plan
# Rebuild DrugBank/TWOSIDES Preprocessing and Tiered DDI Pipeline

## Summary
- Replace the current one-off enrichment flow with a deterministic preprocessing pipeline that starts from raw `TWOSIDES.csv` plus the DrugBank CSV exports already in `data/processed`.
- Remove all dependence on `rxnorm_bridge.csv`; TWOSIDES-to-DrugBank mapping will be rebuilt from DrugBank-owned sources: `drugbank_drugs.csv` names/synonyms first, then `drugbank_external_ids.csv` `RxCUI` values as fallback.
- Use `drugbank_drugs.csv`, `drugbank_smiles.csv`, `drugbank_smiles_filtered.csv`, `drugbank_enzymes.csv`, `drugbank_targets.csv`, `drugbank_transporters.csv`, `drugbank_carriers.csv`, and `drugbank_pathways.csv` to create a canonical drug registry, normalized per-drug evidence tables, and a model-ready pair table.
- Update runtime behavior to support three explicit tiers: direct DrugBank hit bypasses ML, partial-evidence unknown pairs use fusion, and no-evidence pairs use ML only with low-confidence labeling.

## Implementation Changes
- Add one preprocessing entrypoint that rebuilds all processed artifacts in a fixed order and emits a run manifest with row counts, mapping stats, and feature caps.
- Build a canonical `drug_catalog` from `drugbank_drugs.csv` plus SMILES and external IDs. Normalize names and synonyms to lowercase, trim punctuation/whitespace, keep a reviewed alias override file for residual unmapped TWOSIDES concepts, and add `has_smiles`, `canonical_smiles`, `drug_type`, `groups`, and `rxcui_ids`.
- Rebuild the SMILES table from `drugbank_smiles.csv`, validate each SMILES with RDKit, keep only valid structures in `drugbank_smiles_filtered.csv`, and mark invalid/missing structures in the catalog instead of silently dropping them from all downstream tables.
- Normalize DrugBank entity tables by `drugbank_id` and derive per-drug sets for enzymes, targets, transporters, carriers, and pathways. From enzyme `gene_name`, derive first-class CYP features: `shared_major_cyp_count`, `cyp3a4_shared`, `cyp2d6_shared`, and `cyp2c9_shared`.
- Rebuild direct DrugBank interaction pairs from `drugbank_interactions.csv` or `drugbank_interactions_filtered.csv` by canonicalizing unordered pair keys, deduplicating symmetric duplicates, preserving `mechanism_primary`, and optionally storing concatenated `mechanism_all` when multiple texts exist for the same pair.
- Rebuild TWOSIDES from raw `TWOSIDES.csv` by aggregating to unordered pair level after mapping each drug to DrugBank. Store `twosides_found`, `twosides_max_prr`, `twosides_mean_prr`, `twosides_num_signals`, `twosides_total_coreports`, `twosides_mean_report_freq`, `twosides_top_condition`, `twosides_mapping_source`, and `twosides_mapping_status`.
- Use this mapping order for each TWOSIDES drug concept: exact normalized DrugBank name, exact normalized DrugBank synonym, exact DrugBank `RxCUI` from `drugbank_external_ids.csv`, then reviewed manual alias override. Log and drop unresolved or ambiguous matches.
- Produce one canonical positive-pair table for training/inference, keeping backward-compatible naming where practical, with these required columns: `pair_key`, `drug_1_id`, `drug_2_id`, `drug_1_name`, `drug_2_name`, `label`, `direct_drugbank_hit`, `mechanism_primary`, `shared_enzyme_count`, `shared_target_count`, `shared_transporter_count`, `shared_carrier_count`, `shared_pathway_count`, `shared_major_cyp_count`, `cyp3a4_shared`, `cyp2d6_shared`, `cyp2c9_shared`, `twosides_found`, `twosides_max_prr`, `twosides_mean_prr`, `twosides_num_signals`, `twosides_total_coreports`, `twosides_mean_report_freq`, `twosides_top_condition`, and `both_have_smiles`.
- Update the feature extractor to stop treating TWOSIDES as a property of only known DrugBank rows. It should always compute structured overlap features from per-drug tables and always look up TWOSIDES from the mapped TWOSIDES pair table by canonical pair key.
- Expand the model feature vector to 12 inputs in a fixed order: shared enzyme count, shared target count, shared transporter count, shared carrier count, shared pathway count, shared major CYP count, `cyp3a4_shared`, `cyp2d6_shared`, `cyp2c9_shared`, clipped `twosides_max_prr`, clipped `twosides_num_signals`, and `twosides_found`.
- Move normalization caps out of hardcoded training/prediction code and into a preprocessing-generated metadata file so training and inference use the same feature scaling contract.
- Keep DrugBank direct pairs as supervised positives for model training, because they are the only curated pair labels available, but exclude both direct DrugBank pairs and mapped TWOSIDES-signaled pairs from negative sampling to avoid contaminating negatives with plausible positives.
- Update prediction routing to enforce tiers exactly:
- `Tier 1`: if `direct_drugbank_hit=1`, return the DrugBank interaction immediately with mechanism text and high confidence; skip graph construction, model inference, and fusion.
- `Tier 2`: if no direct hit but any structured evidence exists (`shared_* > 0`, CYP flag present, pathway overlap, or `twosides_found=1`), run model inference and evidence fusion.
- `Tier 3`: if no direct hit and no structured evidence exists, run ML only and label overall confidence as low unless the model margin is unusually large.
- Unify readers so training, feature extraction, and prediction all consume the same canonical enriched pair artifact; remove the current `.csv` versus `.csv.gz` split-brain.

## Public Interfaces
- Add a single preprocessing command contract, for example `python -m src.preprocess_data --rebuild-all`, that regenerates every derived data artifact and a manifest in one run.
- Extend extracted pair context with `direct_drugbank_hit`, `shared_pathway_count`, `shared_major_cyp_count`, `cyp3a4_shared`, `cyp2d6_shared`, `cyp2c9_shared`, full TWOSIDES aggregates, and `evidence_tier`.
- Extend prediction/API output with `evidence_tier` (`tier_1_direct_drugbank`, `tier_2_evidence_fusion`, `tier_3_structure_only`), `decision_source` (`drugbank_direct`, `fused`, `ml_only`), and `severity_source`.
- For tier 1 responses, allow `component_scores` to be null or omitted because no fusion ran. If the existing UI still requires a severity label, keep it explicitly marked as derived rather than implying it came from curated DrugBank severity data.

## Test Plan
- Preprocessing smoke test: one full rebuild completes without reading `rxnorm_bridge.csv` and emits a manifest with counts for exact-name matches, synonym matches, `RxCUI` fallback matches, unresolved TWOSIDES names, ambiguous matches, and dropped rows.
- Pair canonicalization test: `(A,B)` and `(B,A)` always collapse to the same `pair_key`, and the final pair table contains no duplicate unordered pairs.
- DrugBank direct-hit test: a known pair from the DrugBank interaction table returns `direct_drugbank_hit=1` and non-empty `mechanism_primary`.
- Shared-feature regression test: known overlapping pairs produce non-zero shared enzyme/target/transporter/carrier counts instead of the current mostly-zero failure mode.
- TWOSIDES regression test: a mapped TWOSIDES pair that is not a direct DrugBank interaction still returns non-zero TWOSIDES features through the feature extractor.
- Runtime routing test: tier 1 skips ML, tier 2 runs fusion, and tier 3 runs ML-only.
- Feature-contract test: training and prediction consume the same 12-feature order and the same normalization metadata.
- Negative-sampling test: sampled negatives never include direct DrugBank pairs or mapped TWOSIDES-signaled pairs.
- Compatibility test: drug resolution still works from DrugBank IDs, canonical names, and synonyms built from `drugbank_drugs.csv`.

## Assumptions and Defaults
- DrugBank interaction exports in this repo do not currently contain curated severity labels, so tier 1 will return direct interaction evidence plus mechanism and high confidence; any displayed severity remains derived by our own logic and must be marked as such.
- TWOSIDES is treated as supporting pharmacovigilance evidence, not as supervised ground-truth positives for the main classifier.
- Name/synonym mapping is the primary TWOSIDES bridge because it uses DrugBank-native data and already covers most TWOSIDES concept names; `RxCUI` from `drugbank_external_ids.csv` is only a fallback, not a separate bridge dependency.
- Unresolved or ambiguous TWOSIDES mappings are excluded from the final mapped TWOSIDES table and reported in the manifest rather than silently coerced.
- Backward-compatible output filenames should be preserved where possible so the app can be migrated incrementally instead of rewritten in one step.


## Key requirements to follow exactly:
1. TWOSIDES mapping order: exact normalized name → synonym → RxCUI fallback → manual alias override. Log unresolved and ambiguous matches, do not silently drop or coerce.
2. Canonical pair key: always sort (drug_1_id, drug_2_id) alphabetically so (A,B) and (B,A) collapse to the same key. No duplicate unordered pairs in the final table.
3. Feature vector is exactly 12 inputs in this fixed order: shared_enzyme_count, shared_target_count, shared_transporter_count, shared_carrier_count, shared_pathway_count, shared_major_cyp_count, cyp3a4_shared, cyp2d6_shared, cyp2c9_shared, twosides_max_prr (clipped), twosides_num_signals (clipped), twosides_found. Normalization caps come from feature_metadata.json, not hardcoded.
4. CYP features: derive from enzyme gene_name column. Major CYPs are CYP3A4, CYP2D6, CYP2C9. shared_major_cyp_count is the count of these three that are shared.
5. Tier routing in predict.py:
   - Tier 1: direct_drugbank_hit=1 → return immediately, skip graph construction and model inference entirely
   - Tier 2: no direct hit but any shared_* > 0 or CYP flag or twosides_found=1 → run model + fusion
   - Tier 3: no direct hit and no structured evidence → run ML only, label confidence as low
6. Prediction output must include: evidence_tier, decision_source, severity_source in every response.
7. Negative sampling must exclude all direct DrugBank pairs AND all TWOSIDES-mapped pairs.
8. Emit a run manifest at data/processed/preprocess_manifest.json with: timestamp, row counts per output file, TWOSIDES mapping stats (exact_name_matches, synonym_matches, rxcui_matches, unresolved, ambiguous, dropped).

## Before writing any code:
List every file you will create or modify and confirm none of them are in the DO NOT modify list above.

## After completing all files:
Run the preprocessing smoke test — execute `python -m src.preprocess_data --rebuild-all` and paste the manifest output so we can verify row counts and mapping stats before proceeding to training.
Also run this sanity check after the rebuild:
python -c "from src.feature_extractor import extract; print(extract('Warfarin', 'Aspirin'))"
And confirm shared_enzyme_count > 0 and the output has all 12 features.