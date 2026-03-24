import torch
import pandas as pd
import json
import argparse
import os
from rdkit import Chem, RDLogger

from mol_graph         import smiles_to_graph
from gnn_encoder       import GNNEncoder
from ddi_classifier    import DDIClassifier
from feature_extractor import FeatureExtractor
from explainer         import Explainer

RDLogger.DisableLog('rdApp.*')

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_PATH = os.path.join(_ROOT, 'models', 'ddi_model.pt')
DATA_DIR   = os.path.join(_ROOT, 'data', 'processed')


class DDIPredictor:
    """
    End-to-end DDI prediction pipeline.
    Input:  two drug names or DrugBank IDs
    Output: interaction prediction + severity + risk index + explanation + uncertainty
    """

    def __init__(self, model_path=MODEL_PATH, data_dir=DATA_DIR):
        print("Initialising DDI prediction pipeline...")

        smiles_df        = pd.read_csv(os.path.join(data_dir, 'drugbank_smiles_filtered.csv'))
        self.smiles_dict = dict(zip(smiles_df['drugbank_id'], smiles_df['smiles']))

        self.gnn        = GNNEncoder().to(DEVICE)
        self.classifier = DDIClassifier(extra_features=6).to(DEVICE)

        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
        self.gnn.load_state_dict(checkpoint['gnn'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.gnn.eval()
        self.classifier.eval()
        print(f"Model loaded from {model_path}")

        self.feature_extractor = FeatureExtractor(data_dir)
        self.explainer         = Explainer()
        print("Pipeline ready.\n")

    # ── Fusion layer ──────────────────────────────────────────────────────────
    def _compute_fusion(self, context, ml_prob):
        """
        Combines three independent evidence sources into a unified risk index.

        rule_score    — based on DrugBank structural evidence (enzymes, targets, known interaction)
        twosides_score — based on pharmacovigilance signal (PRR)
        ml_score      — raw GNN + MLP interaction probability

        Returns fused probability, risk_index (0-100), severity label, and component scores.
        """

        # ── Rule score (DrugBank evidence) ─────────────────────────────────────
        ki = context.get('known_interaction')
        has_known = (
            ki is not None and
            ki.get('mechanism') is not None and
            str(ki.get('mechanism', '')).strip() not in ('', 'nan', 'None')
        )
        has_enzymes   = context['shared_enzyme_count'] > 0
        has_targets   = context['shared_target_count'] > 0
        has_pathways  = len(context['shared_pathways']) > 0

        if has_known:
            rule_score = 1.0
            drugbank_confidence = 'found'
        elif has_enzymes or has_targets:
            # Partial evidence — scale by number of shared features
            n_shared = context['shared_enzyme_count'] + context['shared_target_count']
            rule_score = min(0.5 + (n_shared * 0.05), 0.85)
            drugbank_confidence = 'partial'
        elif has_pathways:
            rule_score = 0.3
            drugbank_confidence = 'partial'
        else:
            rule_score = 0.0
            drugbank_confidence = 'not_found'

        # ── Twosides score (pharmacovigilance) ─────────────────────────────────
        max_prr = float(context.get('max_PRR', 0.0) or 0.0)
        if max_prr > 0:
            # Normalise PRR to 0-1; PRR > 10 = strong signal, clip at 50
            twosides_score = min(max_prr / 50.0, 1.0)
            # Confounding check: if PRR is very high it may reflect solo drug toxicity
            confounding_flag = max_prr > 100
            twosides_confidence = 'high' if max_prr > 10 else 'moderate' if max_prr > 3 else 'weak'
        else:
            twosides_score      = 0.0
            confounding_flag    = False
            twosides_confidence = 'no_signal'

        # ── ML score ───────────────────────────────────────────────────────────
        ml_score = float(ml_prob)
        # ML confidence based on distance from decision boundary (0.5)
        margin = abs(ml_score - 0.5)
        if margin > 0.35:
            ml_confidence = 'high'
        elif margin > 0.15:
            ml_confidence = 'moderate'
        else:
            ml_confidence = 'low'

        # ── Weighted fusion ────────────────────────────────────────────────────
        # Weights reflect reliability: DrugBank curated > ML > pharmacovigilance
        # If DrugBank has no evidence, ML carries more weight
        if drugbank_confidence == 'found':
            w_rule, w_ml, w_twosides = 0.60, 0.25, 0.15
        elif drugbank_confidence == 'partial':
            w_rule, w_ml, w_twosides = 0.40, 0.40, 0.20
        else:
            w_rule, w_ml, w_twosides = 0.10, 0.70, 0.20

        fused_prob = (w_rule * rule_score +
                      w_ml   * ml_score   +
                      w_twosides * twosides_score)
        fused_prob = round(min(max(fused_prob, 0.0), 1.0), 4)

        # ── Risk index (0-100) ─────────────────────────────────────────────────
        risk_index = int(round(fused_prob * 100))

        # ── Severity from risk index ───────────────────────────────────────────
        # Severity head is untrained (no severity labels yet), derive from fusion
        if risk_index >= 70:
            severity = 'Major'
            severity_idx = 2
        elif risk_index >= 40:
            severity = 'Moderate'
            severity_idx = 1
        else:
            severity = 'Minor'
            severity_idx = 0

        # ── Overall confidence ─────────────────────────────────────────────────
        confidence_scores = {
            'high': 3, 'moderate': 2, 'weak': 1,
            'found': 3, 'partial': 2, 'not_found': 1,
            'no_signal': 1, 'low': 1
        }
        avg_conf = (
            confidence_scores.get(drugbank_confidence, 1) +
            confidence_scores.get(ml_confidence, 1) +
            confidence_scores.get(twosides_confidence, 1)
        ) / 3.0

        if avg_conf >= 2.5:
            overall_confidence = 'high'
        elif avg_conf >= 1.5:
            overall_confidence = 'moderate'
        else:
            overall_confidence = 'low'

        return {
            'fused_prob':    fused_prob,
            'risk_index':    risk_index,
            'severity':      severity,
            'severity_idx':  severity_idx,
            'interaction':   fused_prob >= 0.5,
            'components': {
                'rule_score':      round(rule_score, 4),
                'ml_score':        round(ml_score, 4),
                'twosides_score':  round(twosides_score, 4),
                'weights':         {'rule': w_rule, 'ml': w_ml, 'twosides': w_twosides},
            },
            'uncertainty': {
                'drugbank_confidence':  drugbank_confidence,
                'ml_confidence':        ml_confidence,
                'twosides_confidence':  twosides_confidence,
                'overall_confidence':   overall_confidence,
                'confounding_flag':     confounding_flag,
            }
        }

    # ── Graph helper ──────────────────────────────────────────────────────────
    def _get_graph(self, drugbank_id):
        smiles = self.smiles_dict.get(drugbank_id)
        if not smiles:
            raise ValueError(f"No SMILES found for {drugbank_id}")
        graph = smiles_to_graph(str(smiles).strip())
        if graph is None:
            raise ValueError(f"Could not parse SMILES for {drugbank_id}")
        return graph

    # ── Main predict ──────────────────────────────────────────────────────────
    def predict(self, drug_a, drug_b):
        """
        Full DDI prediction for two drugs.
        Accepts drug names or DrugBank IDs.
        Returns structured output dict or {'error': message}.
        """

        # Edge case: same drug
        if str(drug_a).strip().lower() == str(drug_b).strip().lower():
            return {'error': f"Both inputs refer to the same drug: '{drug_a}'"}

        # ── 1. Resolve names → context ─────────────────────────────────────────
        try:
            context = self.feature_extractor.extract(drug_a, drug_b)
        except ValueError as e:
            return {'error': str(e)}

        id_a   = context['drug_a']['id']
        id_b   = context['drug_b']['id']
        name_a = context['drug_a']['name']
        name_b = context['drug_b']['name']

        # Edge case: SMILES missing
        if id_a not in self.smiles_dict:
            return {'error': f"No molecular structure available for {name_a} ({id_a})"}
        if id_b not in self.smiles_dict:
            return {'error': f"No molecular structure available for {name_b} ({id_b})"}

        # ── 2. Build graphs ────────────────────────────────────────────────────
        try:
            graph_a = self._get_graph(id_a)
            graph_b = self._get_graph(id_b)
        except ValueError as e:
            return {'error': str(e)}

        # ── 3. Extra features ──────────────────────────────────────────────────
        extra = torch.tensor([[
            min(float(context['shared_enzyme_count']), 21.0) / 21.0,
            min(float(context['shared_target_count']), 36.0) / 36.0,
            min(float(context.get('shared_transporter_count', 0)), 10.0) / 10.0,
            min(float(context.get('shared_carrier_count', 0)), 10.0) / 10.0,
            min(float(context.get('max_PRR', 0.0) or 0.0), 50.0) / 50.0,
            float(context.get('twosides_found', 0) or 0),
        ]], dtype=torch.float).to(DEVICE)

        # ── 4. GNN + classifier ────────────────────────────────────────────────
        from torch_geometric.data import Batch
        try:
            batch_a = Batch.from_data_list([graph_a]).to(DEVICE)
            batch_b = Batch.from_data_list([graph_b]).to(DEVICE)

            with torch.no_grad():
                embed_a = self.gnn(batch_a)
                embed_b = self.gnn(batch_b)
                prob_logit, _ = self.classifier(embed_a, embed_b, extra)
                ml_prob = torch.sigmoid(prob_logit).item()

                # Catch degenerate outputs
                if not (0.0 <= ml_prob <= 1.0):
                    ml_prob = 0.5
        except Exception as e:
            return {'error': f"Model inference failed: {str(e)}"}

        # ── 5. Fusion ──────────────────────────────────────────────────────────
        fusion = self._compute_fusion(context, ml_prob)

        # ── 6. Explanation ─────────────────────────────────────────────────────
        prediction_for_explainer = {
            'interaction':  fusion['interaction'],
            'probability':  fusion['fused_prob'],
            'severity_idx': fusion['severity_idx'],
        }
        explanation = self.explainer.explain(context, prediction_for_explainer)

        # ── 7. Final output ────────────────────────────────────────────────────
        return {
            'drug_a':        name_a,
            'drug_b':        name_b,
            'drugbank_id_a': id_a,
            'drugbank_id_b': id_b,
            'interaction':   fusion['interaction'],
            'probability':   fusion['fused_prob'],
            'risk_index':    fusion['risk_index'],
            'severity':      fusion['severity'],
            'confidence':    f"{fusion['fused_prob']*100:.1f}%",
            'summary':       explanation['summary'],
            'mechanism':     explanation['mechanism'],
            'recommendation': explanation['recommendation'],
            'evidence': {
                'drugbank': {
                    'shared_enzymes':  explanation['supporting_evidence']['shared_enzymes'],
                    'shared_targets':  explanation['supporting_evidence']['shared_targets'],
                    'shared_pathways': context['shared_pathways'],
                    'known_interaction': context['known_interaction'] is not None,
                },
                'twosides': {
                    'signal_found':     bool(context.get('twosides_found')),
                    'max_PRR':          context.get('max_PRR', 0.0),
                    'confounding_flag': fusion['uncertainty']['confounding_flag'],
                },
                'ml': {
                    'raw_probability': round(ml_prob, 4),
                    'confidence':      fusion['uncertainty']['ml_confidence'],
                },
            },
            'component_scores': fusion['components'],
            'uncertainty': fusion['uncertainty'],
            'full_explanation': explanation['full_text'],
        }
    def drug_names_with_smiles(self) -> list:
        """Return only drug names that have valid SMILES — safe for prediction."""
        return sorted(
            name for db_id, name in self.feature_extractor.id_to_name.items()
            if db_id in self.smiles_dict
        )

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='DrugInsight — DDI Prediction')
    parser.add_argument('drug_a', type=str, help='First drug name or DrugBank ID')
    parser.add_argument('drug_b', type=str, help='Second drug name or DrugBank ID')
    parser.add_argument('--json', action='store_true', help='Output raw JSON')
    args = parser.parse_args()

    predictor = DDIPredictor()
    result    = predictor.predict(args.drug_a, args.drug_b)

    if 'error' in result:
        print(f"\nError: {result['error']}\n")
        return

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Pretty print
    unc = result['uncertainty']

    print(f"\n{'='*62}")
    print(f"  DRUGINSIGHT -- DDI PREDICTION REPORT")
    print(f"{'='*62}")
    print(f"  Drug A : {result['drug_a']:30s} ({result['drugbank_id_a']})")
    print(f"  Drug B : {result['drug_b']:30s} ({result['drugbank_id_b']})")
    print(f"{'─'*62}")
    print(f"  Interaction : {'YES' if result['interaction'] else 'NO'}")
    print(f"  Severity    : {result['severity']}")
    print(f"  Risk Index  : {result['risk_index']} / 100")
    print(f"  Confidence  : {result['confidence']}")
    print(f"{'─'*62}")
    print(f"  Summary:")
    print(f"    {result['summary']}")
    print(f"\n  Mechanism:")
    print(f"    {result['mechanism']}")
    print(f"\n  Recommendation:")
    print(f"    {result['recommendation']}")
    print(f"{'─'*62}")
    ev = result['evidence']
    print(f"  Evidence Sources:")
    print(f"    DrugBank [{unc['drugbank_confidence']:10s}]  "
          f"enzymes={ev['drugbank']['shared_enzymes'] or 'none'}  "
          f"targets={ev['drugbank']['shared_targets'] or 'none'}")
    print(f"    TWOSIDES [{unc['twosides_confidence']:10s}]  "
          f"PRR={ev['twosides']['max_PRR']:.1f}"
          + ("  (confounding possible)" if ev['twosides']['confounding_flag'] else ""))
    print(f"    ML Model [{unc['ml_confidence']:10s}]  "
          f"raw_prob={result['component_scores']['ml_score']:.3f}")
    print(f"{'─'*62}")
    cs = result['component_scores']
    print(f"  Component Scores:")
    print(f"    Rule score     : {cs['rule_score']:.3f}  (w={cs['weights']['rule']})")
    print(f"    ML score       : {cs['ml_score']:.3f}  (w={cs['weights']['ml']})")
    print(f"    Twosides score : {cs['twosides_score']:.3f}  (w={cs['weights']['twosides']})")
    print(f"  Overall confidence : {unc['overall_confidence']}")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    main()
