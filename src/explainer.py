
SEVERITY_LABELS = {-1: 'Uncertain', 0: 'Minor', 1: 'Moderate', 2: 'Major'}
SEVERITY_COLORS = {-1: 'gray', 0: 'green', 1: 'orange', 2: 'red'}


class Explainer:
    """
    Generates a structured, mechanism-grounded explanation for a predicted DDI.
    Uses pharmacological context from FeatureExtractor — no LLM required.

    Explanation structure:
      1. Interaction summary (severity + confidence)
      2. Metabolic mechanism (shared enzymes / CYP interactions)
      3. Pharmacodynamic mechanism (shared targets)
      4. Pharmacovigilance signal (twosides PRR if available)
      5. Clinical recommendation
    """

    # Known CYP inhibitors/inducers/substrates — common clinical knowledge
    CYP_INHIBITORS = {
        'CYP2C9':  ['fluconazole', 'amiodarone', 'metronidazole', 'sulfonamides'],
        'CYP2C19': ['omeprazole', 'fluoxetine', 'fluvoxamine'],
        'CYP3A4':  ['ketoconazole', 'itraconazole', 'clarithromycin', 'ritonavir'],
        'CYP2D6':  ['fluoxetine', 'paroxetine', 'bupropion', 'quinidine'],
        'CYP1A2':  ['fluvoxamine', 'ciprofloxacin'],
    }

    CYP_INDUCERS = {
        'CYP3A4':  ['rifampicin', 'carbamazepine', 'phenytoin', 'st johns wort'],
        'CYP2C9':  ['rifampicin', 'carbamazepine'],
        'CYP1A2':  ['smoking', 'rifampicin'],
    }

    def _get_actions(self, enzyme_or_target):
        """Safely extract actions string."""
        actions = enzyme_or_target.get('actions', '')
        if isinstance(actions, list):
            return ', '.join(actions)
        return str(actions) if actions and str(actions) != 'nan' else 'unknown'

    def _enzyme_mechanism(self, context):
        """Generate metabolic mechanism sentence."""
        shared = context['shared_enzymes']
        if not shared:
            return None

        name_a = context['drug_a']['name']
        name_b = context['drug_b']['name']
        name_b_lower = name_b.lower()
        name_a_lower = name_a.lower()

        enzyme_names = [e.get('gene_name') or e.get('enzyme_name', 'unknown') for e in shared[:3]]
        enzyme_str   = ', '.join(enzyme_names)

        # Fix 1 — check known CYP inhibitors/inducers first (most specific)
        for e in shared:
            gene = (e.get('gene_name') or '').upper()

            if gene in self.CYP_INHIBITORS:
                if any(inh in name_b_lower for inh in self.CYP_INHIBITORS[gene]):
                    return (
                        f"{name_b} is a known inhibitor of {gene}, the primary enzyme "
                        f"responsible for {name_a} metabolism. This inhibition reduces "
                        f"{name_a} clearance, increasing its plasma concentration and "
                        f"risk of toxicity."
                    )
                if any(inh in name_a_lower for inh in self.CYP_INHIBITORS[gene]):
                    return (
                        f"{name_a} is a known inhibitor of {gene}, the primary enzyme "
                        f"responsible for {name_b} metabolism. This inhibition reduces "
                        f"{name_b} clearance, increasing its plasma concentration and "
                        f"risk of toxicity."
                    )

            if gene in self.CYP_INDUCERS:
                if any(ind in name_b_lower for ind in self.CYP_INDUCERS[gene]):
                    return (
                        f"{name_b} induces {gene}, accelerating {name_a} metabolism "
                        f"and reducing its plasma concentration, potentially leading to "
                        f"therapeutic failure."
                    )
                if any(ind in name_a_lower for ind in self.CYP_INDUCERS[gene]):
                    return (
                        f"{name_a} induces {gene}, accelerating {name_b} metabolism "
                        f"and reducing its plasma concentration, potentially leading to "
                        f"therapeutic failure."
                    )

        # Fall back to DrugBank actions on shared enzyme
        inhibition_notes = []
        for e in shared:
            gene = (e.get('gene_name') or '').upper()
            enzymes_b = context.get('enzymes_b', [])
            actions_b = next((self._get_actions(eb) for eb in enzymes_b
                              if eb.get('enzyme_id') == e.get('enzyme_id')), '')
            if 'inhibit' in actions_b.lower():
                inhibition_notes.append(
                    f"{name_b} inhibits {gene}, which may impair metabolism of "
                    f"{name_a}, potentially increasing its plasma concentration "
                    f"and the risk of dose-dependent adverse effects."
                )
            elif 'induc' in actions_b.lower():
                inhibition_notes.append(
                    f"{name_b} induces {gene}, which may accelerate metabolism of "
                    f"{name_a}, potentially reducing its therapeutic efficacy."
                )

        if inhibition_notes:
            return inhibition_notes[0]

        # Generic substrate competition
        return (
            f"Both {name_a} and {name_b} are metabolised by {enzyme_str}. "
            f"Co-administration may lead to competition for these enzymes, "
            f"altering the metabolism and plasma levels of one or both drugs "
            f"and increasing the risk of adverse effects or reduced efficacy."
        )

    def _target_mechanism(self, context):
        """Generate pharmacodynamic mechanism sentence."""
        shared = context['shared_targets']
        if not shared:
            return None

        name_a = context['drug_a']['name']
        name_b = context['drug_b']['name']

        target_names = [t.get('target_name', 'unknown') for t in shared[:3]]
        target_str   = ', '.join(target_names)

        return (
            f"Both drugs act on shared pharmacological targets: {target_str}. "
            f"Concurrent use may produce additive or synergistic effects at these targets, "
            f"increasing the risk of exaggerated pharmacodynamic responses."
        )

    def _pharmacovigilance_note(self, context):
        """Generate pharmacovigilance signal sentence if twosides data exists."""
        if not context.get('twosides_found'):
            return None

        prr = context.get('max_PRR', 0.0)
        name_a = context['drug_a']['name']
        name_b = context['drug_b']['name']

        if prr > 10:
            strength = "a strong"
        elif prr > 3:
            strength = "a moderate"
        else:
            strength = "a weak"

        return (
            f"Post-market surveillance data (TWOSIDES) shows {strength} adverse event signal "
            f"for this combination (PRR={prr:.1f}), suggesting real-world co-prescribing risks."
        )

    def _clinical_recommendation(self, severity_label, context):
        """Generate clinical recommendation based on severity."""
        name_a = context['drug_a']['name']
        name_b = context['drug_b']['name']
        has_enzymes = len(context['shared_enzymes']) > 0
        has_targets = len(context['shared_targets']) > 0

        if severity_label == 'Major':
            return (
                f"Avoid concurrent use of {name_a} and {name_b} if possible. "
                f"If co-administration is necessary, closely monitor for adverse effects"
                + (f" and consider dose adjustment" if has_enzymes else "")
                + f". Consult clinical guidelines or a pharmacist."
            )
        elif severity_label == 'Moderate':
            return (
                f"Use {name_a} and {name_b} together with caution. "
                + (f"Monitor drug levels and clinical response. " if has_enzymes else "")
                + (f"Watch for signs of enhanced pharmacodynamic effects. " if has_targets else "")
                + f"Consider dose reduction if adverse effects occur."
            )
        else:  # Minor
            return (
                f"The interaction between {name_a} and {name_b} is generally manageable. "
                f"Standard monitoring is recommended. No immediate dose adjustment required."
            )

    def explain(self, context, prediction):
        """
        Generate a full explanation.

        Args:
            context:    dict from FeatureExtractor.extract()
            prediction: dict with keys 'interaction' (bool), 'probability' (float),
                        'severity_idx' (int 0-2), 'confidence' (float)

        Returns:
            dict with 'summary', 'mechanism', 'recommendation', 'full_text'
        """
        name_a = context['drug_a']['name']
        name_b = context['drug_b']['name']
        prob   = prediction['probability']
        sev_idx = prediction.get('severity_idx', 1)
        sev_label = SEVERITY_LABELS[sev_idx]

        if not prediction['interaction']:
            return {
                'severity': 'None',
                'confidence': f"{prob*100:.1f}%",
                'summary': f"No significant interaction predicted between {name_a} and {name_b}.",
                'mechanism': "No shared metabolic enzymes or pharmacological targets were identified that would suggest a clinically significant interaction.",
                'recommendation': "Standard prescribing guidelines apply. No special precautions required based on available data.",
                'supporting_evidence': {
                    'shared_enzymes': [],
                    'shared_targets': [],
                    'shared_pathways': [],
                    'twosides_signal': False,
                    'max_PRR': 0.0,
                },
                'full_text': (
                    f"No significant interaction predicted between {name_a} and {name_b} "
                    f"(confidence: {prob*100:.1f}%). "
                    f"No shared metabolic enzymes or pharmacological targets were identified."
                )
            }

        # Fix 3 — use DrugBank mechanism text as primary source if available
        ki = context.get('known_interaction')
        db_mechanism = None
        if ki and ki.get('mechanism') and str(ki.get('mechanism', '')).strip() not in ('', 'nan', 'None'):
            db_mechanism = str(ki['mechanism']).strip()

        # Build mechanism paragraphs
        enzyme_text  = self._enzyme_mechanism(context)
        target_text  = self._target_mechanism(context)
        pvg_text     = self._pharmacovigilance_note(context)
        rec_text     = self._clinical_recommendation(sev_label, context)

        if db_mechanism:
            # Priority 1: DrugBank curated mechanism text
            mechanism_text = db_mechanism
        else:
            mechanism_parts = [p for p in [enzyme_text, target_text, pvg_text] if p]
            if mechanism_parts:
                mechanism_text  = ' '.join(mechanism_parts)
            else:
                # Priority 3: concise interaction-focused fallback for low-evidence (tier-3-like) cases
                mechanism_text = (
                    f"No specific interaction mechanism is confirmed for {name_a} + {name_b} "
                    "from DrugBank structure links or TWOSIDES pharmacovigilance signals. "
                    "This prediction is based on ML molecular-pattern inference only and should be "
                    "treated as a hypothesis pending external clinical verification."
                )
                    
        # Fix 2 — append severity-linked clinical consequence
        consequence = {
            'Major':    "This combination may cause serious or life-threatening adverse effects.",
            'Moderate': "This combination may cause clinically significant adverse effects requiring monitoring.",
            'Minor':    "This combination may cause minor adverse effects unlikely to require intervention.",
            'Uncertain': "Insufficient data — monitor patient closely (severity is uncertain due to lack of structural and clinical data).",
        }
        if sev_label in consequence:
            mechanism_text += f" {consequence[sev_label]}"

        if sev_label == 'Uncertain':
            summary = f"An interaction is known to exist between {name_a} and {name_b}, but its severity is {sev_label.lower()} due to missing data."
        else:
            summary = (
                f"A {sev_label.lower()} interaction is predicted between {name_a} and {name_b} "
                f"(confidence: {prob*100:.1f}%)."
            )

        full_text = f"{summary} {mechanism_text} {rec_text}"

        return {
            'severity':       sev_label,
            'severity_color': SEVERITY_COLORS[sev_idx],
            'confidence':     f"{prob*100:.1f}%",
            'summary':        summary,
            'mechanism':      mechanism_text,
            'recommendation': rec_text,
            'supporting_evidence': {
                'shared_enzymes': [e.get('gene_name') or e.get('enzyme_name', '') for e in context['shared_enzymes'] if e.get('gene_name') or e.get('enzyme_name')],
                'shared_targets': [t.get('target_name') for t in context['shared_targets'] if t.get('target_name')],
                'shared_pathways': context['shared_pathways'],
                'twosides_signal': bool(context.get('twosides_found')),
                'max_PRR':         context.get('max_PRR', 0.0),
            },
            'full_text': full_text,
        }


# ── Test ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    explainer = Explainer()

    # Mock context and prediction for testing
    mock_context = {
        'drug_a': {'id': 'DB00682', 'name': 'Warfarin'},
        'drug_b': {'id': 'DB00227', 'name': 'Lovastatin'},
        'shared_enzymes': [
            {'enzyme_id': 'P11712', 'enzyme_name': 'Cytochrome P450 2C9',
             'gene_name': 'CYP2C9', 'actions': 'inhibitor'}
        ],
        'shared_targets': [],
        'shared_pathways': [],
        'twosides_found': 1,
        'max_PRR': 8.5,
    }

    mock_prediction = {
        'interaction': True,
        'probability': 0.87,
        'severity_idx': 2,
        'confidence': 0.87,
    }

    result = explainer.explain(mock_context, mock_prediction)
    print(f"Severity:    {result['severity']}")
    print(f"Confidence:  {result['confidence']}")
    print(f"Summary:     {result['summary']}")
    print(f"Mechanism:   {result['mechanism']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Evidence:    {result['supporting_evidence']}")
