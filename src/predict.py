import argparse
import json
import os
import re

import pandas as pd
import torch
from rdkit import RDLogger

try:
    from .ddi_classifier import DDIClassifier, load_feature_metadata
    from .explainer import Explainer
    from .feature_extractor import FeatureExtractor, build_normalized_feature_vector
    from .gnn_encoder import GNNEncoder
    from .mol_graph import smiles_to_graph
except ImportError:
    from ddi_classifier import DDIClassifier, load_feature_metadata
    from explainer import Explainer
    from feature_extractor import FeatureExtractor, build_normalized_feature_vector
    from gnn_encoder import GNNEncoder
    from mol_graph import smiles_to_graph


RDLogger.DisableLog('rdApp.*')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
FEATURE_METADATA_PATH = os.path.join(DATA_DIR, 'feature_metadata.json')
DEFAULT_MODEL_FILENAMES = (
    'ddi_model_reprocessed.pt',
    'ddi_model.pt',
)

FUSION_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'models', 'fusion_weights.json')

def load_fusion_weights(path=FUSION_WEIGHTS_PATH):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

FUSION_WEIGHTS = load_fusion_weights()

def resolve_model_path(model_path=None):
    candidate_paths = []

    if model_path:
        candidate_paths.append(os.path.abspath(model_path))

    env_model_path = os.getenv('DRUGINSIGHT_MODEL_PATH')
    if env_model_path:
        if os.path.isabs(env_model_path):
            candidate_paths.append(env_model_path)
        else:
            candidate_paths.append(os.path.join(ROOT_DIR, env_model_path))

    candidate_paths.extend(
        os.path.join(ROOT_DIR, 'models', filename)
        for filename in DEFAULT_MODEL_FILENAMES
    )

    seen = set()
    for candidate in candidate_paths:
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            return normalized

    return os.path.abspath(candidate_paths[0])


MODEL_PATH = resolve_model_path()


class DDIPredictor:
    def __init__(self, model_path=MODEL_PATH, data_dir=DATA_DIR, context_aware_severity=True):
        self.data_dir = data_dir
        self.model_path = resolve_model_path(model_path)
        self.context_aware_severity = bool(context_aware_severity)
        self.feature_metadata = load_feature_metadata(os.path.join(data_dir, 'feature_metadata.json'))

        self.feature_extractor = FeatureExtractor(data_dir)
        self.smiles_dict = self.feature_extractor.smiles_dict
        self.explainer = Explainer()

        self.gnn = GNNEncoder().to(DEVICE)
        self.classifier = DDIClassifier(
            feature_metadata_path=os.path.join(data_dir, 'feature_metadata.json')
        ).to(DEVICE)
        self._load_checkpoint(self.model_path)
        self.gnn.eval()
        self.classifier.eval()

    def _adapt_classifier_state(self, source_state):
        target_state = self.classifier.state_dict()
        adapted_state = {}
        adapted = False

        for key, target_tensor in target_state.items():
            source_tensor = source_state.get(key)
            if source_tensor is None:
                adapted_state[key] = target_tensor
                adapted = True
                continue

            if source_tensor.shape == target_tensor.shape:
                adapted_state[key] = source_tensor
                continue

            if (
                source_tensor.ndim == 2 and
                target_tensor.ndim == 2 and
                source_tensor.shape[0] == target_tensor.shape[0]
            ):
                merged = target_tensor.clone()
                merged.zero_()
                width = min(source_tensor.shape[1], target_tensor.shape[1])
                merged[:, :width] = source_tensor[:, :width]
                adapted_state[key] = merged
                adapted = True
                continue

            adapted_state[key] = target_tensor
            adapted = True

        return adapted_state, adapted

    def _load_checkpoint(self, model_path):
        if not os.path.exists(model_path):
            print(f'Checkpoint not found at {model_path}. Using randomly initialized weights.')
            return

        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
        self.gnn.load_state_dict(checkpoint['gnn'])

        classifier_state = checkpoint.get('classifier', {})
        try:
            self.classifier.load_state_dict(classifier_state)
        except RuntimeError:
            adapted_state, adapted = self._adapt_classifier_state(classifier_state)
            self.classifier.load_state_dict(adapted_state, strict=False)
            if adapted:
                print('Loaded classifier with input-layer adaptation for the new feature contract.')

        print(f'Model loaded from {model_path}')

    def _get_graph(self, drugbank_id):
        smiles = self.smiles_dict.get(drugbank_id)
        if not smiles:
            raise ValueError(f'No molecular structure available for {drugbank_id}')
        graph = smiles_to_graph(str(smiles).strip())
        if graph is None:
            raise ValueError(f'Could not parse SMILES for {drugbank_id}')
        return graph

    def _run_model(self, context):
        from torch_geometric.data import Batch

        graph_a = self._get_graph(context['drug_a']['id'])
        graph_b = self._get_graph(context['drug_b']['id'])
        extra = torch.tensor(
            [build_normalized_feature_vector(context, self.feature_metadata)],
            dtype=torch.float,
        ).to(DEVICE)

        batch_a = Batch.from_data_list([graph_a]).to(DEVICE)
        batch_b = Batch.from_data_list([graph_b]).to(DEVICE)

        with torch.no_grad():
            embed_a = self.gnn(batch_a)
            embed_b = self.gnn(batch_b)
            prob_logit, _ = self.classifier(embed_a, embed_b, extra)
            ml_prob = torch.sigmoid(prob_logit).item()

        return min(max(float(ml_prob), 0.0), 1.0)

    def _ml_confidence(self, ml_prob):
        margin = abs(float(ml_prob) - 0.5)
        if margin > 0.35:
            return 'high'
        if margin > 0.15:
            return 'moderate'
        return 'low'

    def _twosides_score(self, context):
        prr_cap = float(self.feature_metadata['feature_caps']['twosides_max_prr'])
        signal_cap = float(self.feature_metadata['feature_caps']['twosides_num_signals'])
        prr_component = min(float(context.get('twosides_max_prr', 0.0) or 0.0), prr_cap) / prr_cap
        signal_component = min(float(context.get('twosides_num_signals', 0) or 0), signal_cap) / signal_cap
        return round((0.7 * prr_component) + (0.3 * signal_component), 4)

    def _twosides_confidence(self, context):
        prr = float(context.get('twosides_max_prr', 0.0) or 0.0)
        if prr > 10:
            return 'high'
        if prr > 3:
            return 'moderate'
        if prr > 0:
            return 'weak'
        return 'no_signal'

    def _get_rule_score(self, context):
        rule_score = 0.0
        if int(context.get('shared_enzyme_count', 0) or 0) > 0:
            rule_score += min(0.32, 0.16 + (0.04 * int(context['shared_enzyme_count'])))
        if int(context.get('shared_target_count', 0) or 0) > 0:
            rule_score += min(0.22, 0.10 + (0.03 * int(context['shared_target_count'])))
        if int(context.get('shared_transporter_count', 0) or 0) > 0:
            rule_score += min(0.10, 0.04 + (0.02 * int(context['shared_transporter_count'])))
        if int(context.get('shared_carrier_count', 0) or 0) > 0:
            rule_score += min(0.08, 0.03 + (0.02 * int(context['shared_carrier_count'])))
        if int(context.get('shared_pathway_count', 0) or 0) > 0:
            rule_score += min(0.08, 0.02 + (0.01 * int(context['shared_pathway_count'])))
        if int(context.get('shared_major_cyp_count', 0) or 0) > 0:
            rule_score += min(0.18, 0.10 + (0.04 * int(context['shared_major_cyp_count'])))
        return round(min(rule_score, 0.9), 4)

    def _get_calibrated_probability(self, rule_score, ml_prob, twosides_score):
        if not FUSION_WEIGHTS:
            return min(max(float(ml_prob)*0.5 + rule_score*0.3 + twosides_score*0.2, 0.0), 1.0)
        import math
        c = FUSION_WEIGHTS['coef']
        i = FUSION_WEIGHTS['intercept']
        logit = i + c[0]*rule_score + c[1]*float(ml_prob) + c[2]*twosides_score
        return 1.0 / (1.0 + math.exp(-logit))

    def _severity_from_risk(self, risk_index):
        if risk_index >= 70:
            return 'Major', 2
        if risk_index >= 40:
            return 'Moderate', 1
        return 'Minor', 0

    def _evidence_strength(self, context):
        score = 0.0
        score += min(0.30, 0.08 * float(context.get('shared_enzyme_count', 0) or 0))
        score += min(0.20, 0.06 * float(context.get('shared_target_count', 0) or 0))
        score += min(0.08, 0.03 * float(context.get('shared_transporter_count', 0) or 0))
        score += min(0.06, 0.02 * float(context.get('shared_carrier_count', 0) or 0))
        score += min(0.06, 0.02 * float(context.get('shared_pathway_count', 0) or 0))
        score += min(0.15, 0.05 * float(context.get('shared_major_cyp_count', 0) or 0))

        if bool(context.get('direct_drugbank_hit')):
            score += 0.25
        if bool(context.get('twosides_found')):
            score += min(0.15, 0.04 * float(context.get('twosides_max_prr', 0.0) or 0.0))

        return round(min(score, 1.0), 4)

    def _apply_context_aware_severity(self, context, probability, risk_index, default_source, uncertainty):
        severity, severity_idx = self._severity_from_risk(risk_index)
        if not self.context_aware_severity:
            return severity, severity_idx, default_source, ''

        tier = context.get('evidence_tier', '')
        overall_conf = str(uncertainty.get('overall_confidence', 'low')).lower()
        evidence_strength = float(uncertainty.get('evidence_strength_score', 0.0) or 0.0)

        # Tier-3 ML-only positives are uncertain by policy.
        if tier == 'tier_3_structure_only' and float(probability) >= 0.5:
            return 'Uncertain', -1, 'tier3_ml_only_uncertain', 'ML-only positive without corroborating structure evidence'

        # High-risk predictions should require enough supporting evidence quality.
        if risk_index >= 70 and overall_conf == 'low' and evidence_strength < 0.40:
            return 'Uncertain', -1, 'context_aware_low_certainty_high_risk', 'High risk score but low corroborating evidence'

        # Moderate risk with very weak support is also uncertain.
        if 40 <= risk_index < 70 and overall_conf == 'low' and evidence_strength < 0.20:
            return 'Uncertain', -1, 'context_aware_low_certainty_moderate_risk', 'Moderate risk score with weak evidence support'

        return severity, severity_idx, default_source, ''

    def _fusion_weight_dict(self):
        """Return fusion weights in a UI-friendly dict shape."""
        if not FUSION_WEIGHTS:
            return {'rule': 0.3, 'ml': 0.5, 'twosides': 0.2, 'heuristic': True}

        coefs = FUSION_WEIGHTS.get('coef', [])
        if isinstance(coefs, (list, tuple)) and len(coefs) >= 3:
            return {'rule': float(coefs[0]), 'ml': float(coefs[1]), 'twosides': float(coefs[2])}

        # Defensive fallback if config exists but is malformed.
        return {'rule': 0.3, 'ml': 0.5, 'twosides': 0.2, 'heuristic': True}

    def _build_result(
        self,
        context,
        probability,
        interaction,
        risk_index,
        severity,
        severity_idx,
        decision_source,
        severity_source,
        component_scores,
        uncertainty,
    ):
        prediction_for_explainer = {
            'interaction': interaction,
            'probability': probability,
            'severity_idx': severity_idx,
        }
        explanation = self.explainer.explain(context, prediction_for_explainer)

        return {
            'drug_a': context['drug_a']['name'],
            'drug_b': context['drug_b']['name'],
            'drugbank_id_a': context['drug_a']['id'],
            'drugbank_id_b': context['drug_b']['id'],
            'interaction': interaction,
            'probability': round(probability, 4),
            'risk_index': int(risk_index),
            'severity': severity,
            'confidence': f'{probability * 100:.1f}%',
            'evidence_tier': context['evidence_tier'],
            'decision_source': decision_source,
            'severity_source': severity_source,
            'summary': explanation['summary'],
            'mechanism': explanation['mechanism'],
            'recommendation': explanation['recommendation'],
            'evidence': {
                'drugbank': {
                    'shared_enzymes': explanation['supporting_evidence']['shared_enzymes'],
                    'shared_targets': explanation['supporting_evidence']['shared_targets'],
                    'shared_pathways': context['shared_pathways'],
                    'known_interaction': bool(context.get('direct_drugbank_hit')),
                    'direct_drugbank_hit': bool(context.get('direct_drugbank_hit')),
                },
                'twosides': {
                    'signal_found': bool(context.get('twosides_found')),
                    'max_PRR': float(context.get('twosides_max_prr', 0.0) or 0.0),
                    'num_signals': int(context.get('twosides_num_signals', 0) or 0),
                    'top_condition': context.get('twosides_top_condition', ''),
                    'mapping_source': context.get('twosides_mapping_source', ''),
                    'confounding_flag': uncertainty['confounding_flag'],
                },
                'ml': {
                    'raw_probability': round(component_scores['ml_score'], 4),
                    'confidence': uncertainty['ml_confidence'],
                },
            },
            'component_scores': component_scores,
            'uncertainty': uncertainty,
            'model_features': {
                'names': context['model_feature_names'],
                'values': context['model_feature_values'],
                'vector': context['feature_vector'],
            },
            'full_explanation': explanation['full_text'],
        }

    def _direct_hit_result(self, context, ml_prob=None):
        rule_score = self._get_rule_score(context)

        # Tier-1 policy: blend rule score with ML when available (rule 70%, ML 30%).
        # If ML is unavailable, fall back to rule score alone.
        if ml_prob is not None:
            blended_prob = round(min(max(0.70 * rule_score + 0.30 * float(ml_prob), 0.0), 1.0), 4)
            weights_used = {'rule': 0.70, 'ml': 0.30, 'twosides': 0.0, 'policy': 'tier1_drugbank_blended'}
            ml_score_display = round(float(ml_prob), 4)
        else:
            blended_prob = round(min(max(float(rule_score), 0.0), 1.0), 4)
            weights_used = {'rule': 1.0, 'ml': 0.0, 'twosides': 0.0, 'policy': 'tier1_drugbank_only'}
            ml_score_display = 0.0

        risk_index = int(round(blended_prob * 100))
        severity_source = 'drugbank_blended' if ml_prob is not None else 'drugbank_rule_only'

        component_scores = {
            'rule_score': rule_score,
            'ml_score': ml_score_display,
            'twosides_score': 0.0,
            'weights': weights_used,
        }

        ml_conf = self._ml_confidence(float(ml_prob)) if ml_prob is not None else 'not_used'
        uncertainty = {
            'drugbank_confidence': 'found',
            'ml_confidence': ml_conf,
            'twosides_confidence': 'not_used',
            'overall_confidence': 'high' if ml_prob is not None else 'moderate',
            'evidence_strength_score': self._evidence_strength(context),
            'confounding_flag': float(context.get('twosides_max_prr', 0.0) or 0.0) > 100,
            'tier1_policy': 'tier1_drugbank_blended' if ml_prob is not None else 'tier1_drugbank_only',
        }
        severity, severity_idx, severity_source, policy_notes = self._apply_context_aware_severity(
            context=context,
            probability=blended_prob,
            risk_index=risk_index,
            default_source=severity_source,
            uncertainty=uncertainty,
        )
        if policy_notes:
            uncertainty['policy_notes'] = policy_notes

        return self._build_result(
            context=context,
            probability=blended_prob,
            interaction=True,
            risk_index=risk_index,
            severity=severity,
            severity_idx=severity_idx,
            decision_source='drugbank_direct',
            severity_source=severity_source,
            component_scores=component_scores,
            uncertainty=uncertainty,
        )

    def _compute_fusion(self, context, ml_prob):
        has_structural = any(
            float(context.get(name, 0) or 0) > 0
            for name in (
                'shared_enzyme_count',
                'shared_target_count',
                'shared_transporter_count',
                'shared_carrier_count',
                'shared_pathway_count',
                'shared_major_cyp_count',
                'cyp3a4_shared',
                'cyp2d6_shared',
                'cyp2c9_shared',
            )
        )

        rule_score = self._get_rule_score(context)
        twosides_score = self._twosides_score(context)
        
        fused_prob = self._get_calibrated_probability(rule_score, ml_prob, twosides_score)
        fused_prob = round(fused_prob, 4)
        risk_index = int(round(fused_prob * 100))
        severity, severity_idx = self._severity_from_risk(risk_index)

        component_scores = {
            'rule_score': rule_score,
            'ml_score': round(float(ml_prob), 4),
            'twosides_score': twosides_score,
            'weights': self._fusion_weight_dict(),
        }

        confidence_scores = {
            'found': 3,
            'inferred': 2,
            'not_found': 1,
            'high': 3,
            'moderate': 2,
            'weak': 1,
            'no_signal': 1,
            'low': 1,
        }
        drugbank_confidence = 'inferred' if has_structural else 'not_found'
        ml_confidence = self._ml_confidence(ml_prob)
        twosides_confidence = self._twosides_confidence(context)
        avg_confidence = (
            confidence_scores[drugbank_confidence] +
            confidence_scores[ml_confidence] +
            confidence_scores[twosides_confidence]
        ) / 3.0
        overall_confidence = 'high' if avg_confidence >= 2.5 else 'moderate' if avg_confidence >= 1.5 else 'low'
        uncertainty = {
            'drugbank_confidence': drugbank_confidence,
            'ml_confidence': ml_confidence,
            'twosides_confidence': twosides_confidence,
            'overall_confidence': overall_confidence,
            'evidence_strength_score': self._evidence_strength(context),
            'confounding_flag': float(context.get('twosides_max_prr', 0.0) or 0.0) > 100,
        }
        severity, severity_idx, severity_source, policy_notes = self._apply_context_aware_severity(
            context=context,
            probability=fused_prob,
            risk_index=risk_index,
            default_source='derived_from_fusion',
            uncertainty=uncertainty,
        )
        if policy_notes:
            uncertainty['policy_notes'] = policy_notes

        return self._build_result(
            context=context,
            probability=fused_prob,
            interaction=fused_prob >= 0.5,
            risk_index=risk_index,
            severity=severity,
            severity_idx=severity_idx,
            decision_source='fused',
            severity_source=severity_source,
            component_scores=component_scores,
            uncertainty=uncertainty,
        )

    def _ml_only_result(self, context, ml_prob):
        risk_index = int(round(float(ml_prob) * 100))
        # Tier-3 has no corroborating structured evidence (DrugBank/TWOSIDES),
        # so positive ML-only signals are surfaced as Uncertain severity.
        interaction = float(ml_prob) >= 0.5
        if interaction:
            severity, severity_idx = 'Uncertain', -1
            severity_source = 'tier3_ml_only_uncertain'
        else:
            severity, severity_idx = self._severity_from_risk(risk_index)
            severity_source = 'derived_from_ml'
        component_scores = {
            'rule_score': 0.0,
            'ml_score': round(float(ml_prob), 4),
            'twosides_score': 0.0,
            'weights': {'rule': 0.0, 'ml': 1.0, 'twosides': 0.0},
        }
        uncertainty = {
            'drugbank_confidence': 'not_found',
            'ml_confidence': self._ml_confidence(ml_prob),
            'twosides_confidence': 'no_signal',
            'overall_confidence': 'low',
            'evidence_strength_score': self._evidence_strength(context),
            'confounding_flag': False,
        }
        severity, severity_idx, severity_source_ctx, policy_notes = self._apply_context_aware_severity(
            context=context,
            probability=float(ml_prob),
            risk_index=risk_index,
            default_source=severity_source,
            uncertainty=uncertainty,
        )
        severity_source = severity_source_ctx
        if policy_notes:
            uncertainty['policy_notes'] = policy_notes
        return self._build_result(
            context=context,
            probability=float(ml_prob),
            interaction=interaction,
            risk_index=risk_index,
            severity=severity,
            severity_idx=severity_idx,
            decision_source='ml_only',
            severity_source=severity_source,
            component_scores=component_scores,
            uncertainty=uncertainty,
        )

    def predict(self, drug_a, drug_b):
        if str(drug_a).strip().lower() == str(drug_b).strip().lower():
            return {'error': f"Both inputs refer to the same drug: '{drug_a}'"}

        try:
            context = self.feature_extractor.extract(drug_a, drug_b)
        except ValueError as exc:
            return {'error': str(exc)}

        has_smiles_a = context['drug_a']['id'] in self.smiles_dict
        has_smiles_b = context['drug_b']['id'] in self.smiles_dict
        
        ml_prob = None
        if has_smiles_a and has_smiles_b:
            try:
                ml_prob = self._run_model(context)
            except Exception as exc:
                if context['evidence_tier'] != 'tier_1_direct_drugbank':
                    return {'error': f'Model inference failed: {exc}'}
        else:
            if context['evidence_tier'] != 'tier_1_direct_drugbank':
                if not has_smiles_a:
                    return {'error': f"No molecular structure available for {context['drug_a']['name']} ({context['drug_a']['id']})"}
                return {'error': f"No molecular structure available for {context['drug_b']['name']} ({context['drug_b']['id']})"}

        if context['evidence_tier'] == 'tier_1_direct_drugbank':
            return self._direct_hit_result(context, ml_prob)
        if context['evidence_tier'] == 'tier_2_evidence_fusion':
            return self._compute_fusion(context, ml_prob)
        return self._ml_only_result(context, ml_prob)

    def drug_names_with_smiles(self):
        catalog = self.feature_extractor.drug_catalog

        approved_ids = None
        if 'groups' in catalog.columns:
            groups = catalog['groups'].fillna('').astype(str).str.lower()
            approved_ids = set(catalog.loc[groups.str.contains(r'\bapproved\b', regex=True), 'drugbank_id'])

        # Utility/formulation clues in indication text (e.g., diluents/hydration vehicles).
        indication_by_id = {}
        if 'indication' in catalog.columns:
            for row in catalog[['drugbank_id', 'indication']].itertuples(index=False):
                indication_by_id[row.drugbank_id] = str(row.indication or '').lower()

        utility_indication_pattern = (
            r'dilut|dissolv|irrigat|delivery system|'
            r'source of electrolytes|source of water|for hydration'
        )
        utility_indication_re = re.compile(utility_indication_pattern, re.IGNORECASE)

        # Precompute direct DrugBank hit counts from SQLite to avoid excluding clinically relevant drugs.
        direct_hit_counts = {}
        conn = self.feature_extractor._db_conn
        rows = conn.execute(
            """
            WITH all_hits AS (
                SELECT drug_1_id AS id FROM known_interactions WHERE direct_drugbank_hit = 1
                UNION ALL
                SELECT drug_2_id AS id FROM known_interactions WHERE direct_drugbank_hit = 1
            )
            SELECT id, COUNT(*) AS hit_count
            FROM all_hits
            GROUP BY id
            """
        ).fetchall()
        for row in rows:
            direct_hit_counts[row['id']] = int(row['hit_count'] or 0)

        names = []
        for drugbank_id, name in self.feature_extractor.id_to_name.items():
            if drugbank_id not in self.smiles_dict:
                continue
            if approved_ids is not None and drugbank_id not in approved_ids:
                continue

            indication = indication_by_id.get(drugbank_id, '')
            is_utility_indication = bool(utility_indication_re.search(indication))
            hit_count = direct_hit_counts.get(drugbank_id, 0)
            # Exclude only when both clues agree: utility-like indication + weak DDI coverage.
            if is_utility_indication and hit_count <= 25:
                continue

            names.append(name)

        return sorted(set(names))


def main():
    parser = argparse.ArgumentParser(description='DrugInsight DDI prediction')
    parser.add_argument('drug_a', type=str, help='First drug name or DrugBank ID')
    parser.add_argument('drug_b', type=str, help='Second drug name or DrugBank ID')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    args = parser.parse_args()

    predictor = DDIPredictor()
    result = predictor.predict(args.drug_a, args.drug_b)
    if 'error' in result:
        print(f"\nError: {result['error']}\n")
        return

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"\n{'=' * 62}")
    print('  DRUGINSIGHT -- DDI PREDICTION REPORT')
    print(f"{'=' * 62}")
    print(f"  Drug A : {result['drug_a']:30s} ({result['drugbank_id_a']})")
    print(f"  Drug B : {result['drug_b']:30s} ({result['drugbank_id_b']})")
    print(f"{'-' * 62}")
    print(f"  Tier         : {result['evidence_tier']}")
    print(f"  Decision     : {result['decision_source']}")
    print(f"  Interaction  : {'YES' if result['interaction'] else 'NO'}")
    print(f"  Severity     : {result['severity']} ({result['severity_source']})")
    print(f"  Risk Index   : {result['risk_index']} / 100")
    print(f"  Confidence   : {result['confidence']}")
    print(f"{'-' * 62}")
    print(f"  Summary:\n    {result['summary']}")
    print(f"\n  Mechanism:\n    {result['mechanism']}")
    print(f"\n  Recommendation:\n    {result['recommendation']}")
    print(f"{'-' * 62}")
    print(f"  Shared enzymes    : {result['evidence']['drugbank']['shared_enzymes'] or 'none'}")
    print(f"  Shared targets    : {result['evidence']['drugbank']['shared_targets'] or 'none'}")
    print(f"  TWOSIDES top cond : {result['evidence']['twosides']['top_condition'] or 'none'}")
    print(f"  ML raw prob       : {result['component_scores']['ml_score']:.3f}")
    print(f"  Overall certainty : {result['uncertainty']['overall_confidence']}")
    print(f"{'=' * 62}\n")


if __name__ == '__main__':
    main()
