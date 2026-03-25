import argparse
import json
import os

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
    def __init__(self, model_path=MODEL_PATH, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.model_path = resolve_model_path(model_path)
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

    def _severity_from_risk(self, risk_index):
        if risk_index >= 70:
            return 'Major', 2
        if risk_index >= 40:
            return 'Moderate', 1
        return 'Minor', 0

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

    def _direct_hit_result(self, context):
        probability = 0.72
        probability += 0.05 * min(int(context.get('shared_major_cyp_count', 0)), 2)
        probability += 0.03 * min(int(context.get('shared_enzyme_count', 0)), 3)
        probability += 0.02 * min(int(context.get('shared_target_count', 0)), 3)
        probability += 0.01 * min(int(context.get('shared_pathway_count', 0)), 2)
        probability += 0.02 if int(context.get('twosides_found', 0)) else 0.0
        probability += 0.03 if float(context.get('twosides_max_prr', 0.0) or 0.0) > 10 else 0.0
        probability = min(probability, 0.95)
        risk_index = int(round(probability * 100))
        severity, severity_idx = self._severity_from_risk(risk_index)

        component_scores = {
            'rule_score': 1.0,
            'ml_score': 0.0,
            'twosides_score': self._twosides_score(context),
            'weights': {'rule': 1.0, 'ml': 0.0, 'twosides': 0.0},
        }
        uncertainty = {
            'drugbank_confidence': 'found',
            'ml_confidence': 'not_run',
            'twosides_confidence': self._twosides_confidence(context),
            'overall_confidence': 'high',
            'confounding_flag': float(context.get('twosides_max_prr', 0.0) or 0.0) > 100,
        }
        return self._build_result(
            context=context,
            probability=probability,
            interaction=True,
            risk_index=risk_index,
            severity=severity,
            severity_idx=severity_idx,
            decision_source='drugbank_direct',
            severity_source='derived_rule_based_direct',
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
        rule_score = round(min(rule_score, 0.9), 4)

        twosides_score = self._twosides_score(context)
        if has_structural and int(context.get('twosides_found', 0) or 0):
            weights = {'rule': 0.45, 'ml': 0.35, 'twosides': 0.20}
        elif has_structural:
            weights = {'rule': 0.55, 'ml': 0.45, 'twosides': 0.0}
        else:
            weights = {'rule': 0.0, 'ml': 0.70, 'twosides': 0.30}

        fused_prob = (
            (weights['rule'] * rule_score) +
            (weights['ml'] * float(ml_prob)) +
            (weights['twosides'] * twosides_score)
        )
        fused_prob = round(min(max(fused_prob, 0.0), 1.0), 4)
        risk_index = int(round(fused_prob * 100))
        severity, severity_idx = self._severity_from_risk(risk_index)

        component_scores = {
            'rule_score': rule_score,
            'ml_score': round(float(ml_prob), 4),
            'twosides_score': twosides_score,
            'weights': weights,
        }

        confidence_scores = {
            'found': 3,
            'partial': 2,
            'not_found': 1,
            'high': 3,
            'moderate': 2,
            'weak': 1,
            'no_signal': 1,
            'low': 1,
        }
        drugbank_confidence = 'partial' if has_structural else 'not_found'
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
            'confounding_flag': float(context.get('twosides_max_prr', 0.0) or 0.0) > 100,
        }

        return self._build_result(
            context=context,
            probability=fused_prob,
            interaction=fused_prob >= 0.5,
            risk_index=risk_index,
            severity=severity,
            severity_idx=severity_idx,
            decision_source='fused',
            severity_source='derived_from_fusion',
            component_scores=component_scores,
            uncertainty=uncertainty,
        )

    def _ml_only_result(self, context, ml_prob):
        risk_index = int(round(float(ml_prob) * 100))
        severity, severity_idx = self._severity_from_risk(risk_index)
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
            'confounding_flag': False,
        }
        return self._build_result(
            context=context,
            probability=float(ml_prob),
            interaction=float(ml_prob) >= 0.5,
            risk_index=risk_index,
            severity=severity,
            severity_idx=severity_idx,
            decision_source='ml_only',
            severity_source='derived_from_ml',
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

        if context['evidence_tier'] == 'tier_1_direct_drugbank':
            return self._direct_hit_result(context)

        if context['drug_a']['id'] not in self.smiles_dict:
            return {'error': f"No molecular structure available for {context['drug_a']['name']} ({context['drug_a']['id']})"}
        if context['drug_b']['id'] not in self.smiles_dict:
            return {'error': f"No molecular structure available for {context['drug_b']['name']} ({context['drug_b']['id']})"}

        try:
            ml_prob = self._run_model(context)
        except Exception as exc:
            return {'error': f'Model inference failed: {exc}'}

        if context['evidence_tier'] == 'tier_2_evidence_fusion':
            return self._compute_fusion(context, ml_prob)
        return self._ml_only_result(context, ml_prob)

    def drug_names_with_smiles(self):
        return sorted(
            name for drugbank_id, name in self.feature_extractor.id_to_name.items()
            if drugbank_id in self.smiles_dict
        )


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
