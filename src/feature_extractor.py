import pandas as pd
import os
import numpy as np


class FeatureExtractor:
    """
    Pulls structured pharmacological context for a drug pair from DrugBank CSVs.
    """

    def __init__(self, data_dir='data/processed'):
        print("Loading pharmacological databases...")

        drugs_df = pd.read_csv(os.path.join(data_dir, 'drugbank_drugs.csv'))
        name_col = 'name' if 'name' in drugs_df.columns else drugs_df.columns[1]
        id_col   = 'drugbank_id' if 'drugbank_id' in drugs_df.columns else drugs_df.columns[0]

        # 1. Exact name lookup
        self.name_to_id = {}
        for _, row in drugs_df.iterrows():
            self.name_to_id[str(row[name_col]).lower().strip()] = row[id_col]

        # 2. Synonyms
        for _, row in drugs_df.iterrows():
            synonyms = str(row.get('synonyms', ''))
            if synonyms and synonyms != 'nan':
                for syn in synonyms.split('|'):
                    syn = syn.strip().lower()
                    if syn and syn not in self.name_to_id:
                        self.name_to_id[syn] = row[id_col]

        # 3. Common aliases — must be after name_to_id is built
        COMMON_ALIASES = {
            'aspirin':    'acetylsalicylic acid',
            'tylenol':    'acetaminophen',
            'advil':      'ibuprofen',
            'motrin':     'ibuprofen',
            'glucophage': 'metformin',
            'zocor':      'simvastatin',
            'lipitor':    'atorvastatin',
            'coumadin':   'warfarin',
            'prozac':     'fluoxetine',
            'zoloft':     'sertraline',
            'prinivil':   'lisinopril',
            'norvasc':    'amlodipine',
        }
        for alias, canonical in COMMON_ALIASES.items():
            if alias not in self.name_to_id and canonical in self.name_to_id:
                self.name_to_id[alias] = self.name_to_id[canonical]

        self.id_to_name = dict(zip(drugs_df[id_col], drugs_df[name_col]))

        # Enzyme data
        enzymes_df        = pd.read_csv(os.path.join(data_dir, 'drugbank_enzymes.csv'))
        self.drug_enzymes = (
            enzymes_df.groupby('drugbank_id')
            .apply(lambda x: x[['enzyme_id', 'enzyme_name', 'gene_name', 'actions']]
                   .to_dict('records'), include_groups=False)
            .to_dict()
        )
        self.drug_enzyme_ids = (
            enzymes_df.groupby('drugbank_id')['enzyme_id']
            .apply(set).to_dict()
        )

        # Target data
        targets_df        = pd.read_csv(os.path.join(data_dir, 'drugbank_targets.csv'))
        self.drug_targets = (
            targets_df.groupby('drugbank_id')
            .apply(lambda x: x[['target_id', 'target_name', 'gene_name', 'actions', 'known_action']]
                   .to_dict('records'), include_groups=False)
            .to_dict()
        )
        self.drug_target_ids = (
            targets_df.groupby('drugbank_id')['target_id']
            .apply(set).to_dict()
        )

        # Transporter data
        transporters_df = pd.read_csv(os.path.join(data_dir, 'drugbank_transporters.csv'))
        self.drug_transporters = (
            transporters_df.groupby('drugbank_id')['transporter_id']
            .apply(set).to_dict()
        )

        # Carrier data
        carriers_df = pd.read_csv(os.path.join(data_dir, 'drugbank_carriers.csv'))
        self.drug_carriers = (
            carriers_df.groupby('drugbank_id')['carrier_id']
            .apply(set).to_dict()
        )

        # Pathway data
        pathways_df = pd.read_csv(os.path.join(data_dir, 'drugbank_pathways.csv'))
        if 'drugbank_id' in pathways_df.columns and 'pathway_name' in pathways_df.columns:
            self.drug_pathways = (
                pathways_df.groupby('drugbank_id')['pathway_name']
                .apply(list).to_dict()
            )
        else:
            self.drug_pathways = {}

        # Known interactions
        self.known_interactions = pd.read_csv(
            os.path.join(data_dir, 'drugbank_interactions_enriched.csv.gz'),
            compression='gzip'
        )
        self.known_pairs = set(zip(
            self.known_interactions['drug_1_id'].astype(str),
            self.known_interactions['drug_2_id'].astype(str),
        ))

        print("Feature extractor ready.")

    def pair_features(self, id_a, id_b):
        """
        Lightweight pair feature computation for sampling/training.
        Returns dict with the same numeric feature keys used by the model.
        """
        id_a = str(id_a)
        id_b = str(id_b)
        shared_enzyme_count = len(self.drug_enzyme_ids.get(id_a, set()) & self.drug_enzyme_ids.get(id_b, set()))
        shared_target_count = len(self.drug_target_ids.get(id_a, set()) & self.drug_target_ids.get(id_b, set()))
        shared_transporter_count = len(
            self.drug_transporters.get(id_a, set()) & self.drug_transporters.get(id_b, set())
        )
        shared_carrier_count = len(
            self.drug_carriers.get(id_a, set()) & self.drug_carriers.get(id_b, set())
        )
        shared_pathways = set(self.drug_pathways.get(id_a, [])) & set(self.drug_pathways.get(id_b, []))

        return {
            'shared_enzyme_count': shared_enzyme_count,
            'shared_target_count': shared_target_count,
            'shared_transporter_count': shared_transporter_count,
            'shared_carrier_count': shared_carrier_count,
            'shared_pathways_count': len(shared_pathways),
            # These only exist when the pair is a known interaction row in our DB.
            'max_PRR': 0.0,
            'twosides_found': 0,
        }

    def sample_hard_negatives(
        self,
        drug_pool,
        positive_pairs,
        n,
        seed=42,
        candidate_multiplier=50,
        hard_fraction=0.7,
    ):
        """
        Hard-negative sampling for DDI classification.

        Strategy:
        - Sample many candidate non-interacting pairs.
        - Score candidates by "plausibility" (shared enzymes/targets/transporters/carriers/pathways).
        - Keep a mix of hard negatives (high score) and easy negatives (random), to avoid training collapse.
        """
        rng = np.random.default_rng(seed)
        drug_pool = list(drug_pool)
        positive_pairs = set((str(a), str(b)) for a, b in positive_pairs)

        n_candidates = max(int(n * candidate_multiplier), n * 5)
        a_idx = rng.integers(0, len(drug_pool), size=n_candidates)
        b_idx = rng.integers(0, len(drug_pool), size=n_candidates)

        rows = []
        seen = set()
        for ai, bi in zip(a_idx, b_idx):
            a = str(drug_pool[ai])
            b = str(drug_pool[bi])
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)

            if (a, b) in positive_pairs or (b, a) in positive_pairs:
                continue
            if (a, b) in self.known_pairs or (b, a) in self.known_pairs:
                continue

            feats = self.pair_features(a, b)
            hardness = (
                2.0 * feats['shared_enzyme_count']
                + 2.0 * feats['shared_target_count']
                + 1.0 * feats['shared_transporter_count']
                + 1.0 * feats['shared_carrier_count']
                + 0.5 * feats['shared_pathways_count']
            )

            rows.append({
                'drug_1_id': a,
                'drug_2_id': b,
                'label': 0,
                **feats,
                '_hardness': hardness,
            })

            if len(rows) >= n_candidates:
                break

        if not rows:
            return pd.DataFrame(columns=[
                'drug_1_id', 'drug_2_id', 'label',
                'shared_enzyme_count', 'shared_target_count',
                'shared_transporter_count', 'shared_carrier_count',
                'max_PRR', 'twosides_found',
            ])

        df = pd.DataFrame(rows)
        df = df.sort_values('_hardness', ascending=False).reset_index(drop=True)

        n_hard = int(round(n * float(hard_fraction)))
        n_easy = n - n_hard

        hard = df.head(max(n_hard, 0))
        if n_easy > 0:
            easy_pool = df.tail(max(len(df) - len(hard), 0))
            if len(easy_pool) > 0:
                easy = easy_pool.sample(n=min(n_easy, len(easy_pool)), random_state=seed)
            else:
                easy = df.sample(n=min(n_easy, len(df)), random_state=seed)
            out = pd.concat([hard, easy], ignore_index=True)
        else:
            out = hard

        out = out.drop(columns=['_hardness', 'shared_pathways_count'], errors='ignore')
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return out.head(n)

    def resolve_drug(self, drug_input):
        drug_input = str(drug_input).strip()

        upper = drug_input.upper()
        if upper in self.id_to_name:
            return upper, self.id_to_name[upper]

        lower = drug_input.lower()
        if lower in self.name_to_id:
            db_id = self.name_to_id[lower]
            return db_id, self.id_to_name.get(db_id, drug_input)

        if len(lower) >= 4:
            matches = [(n, d) for n, d in self.name_to_id.items() if n.startswith(lower)]
            if matches:
                matches.sort(key=lambda x: len(x[0]))
                name, db_id = matches[0]
                return db_id, self.id_to_name.get(db_id, name)

            matches = [(n, d) for n, d in self.name_to_id.items() if lower in n]
            if matches:
                matches.sort(key=lambda x: len(x[0]))
                name, db_id = matches[0]
                return db_id, self.id_to_name.get(db_id, name)

        raise ValueError(f"Drug '{drug_input}' not found in DrugBank database.")

    def get_shared_enzymes(self, id_a, id_b):
        enzymes_a = {e['enzyme_id']: e for e in self.drug_enzymes.get(id_a, [])}
        enzymes_b = {e['enzyme_id']: e for e in self.drug_enzymes.get(id_b, [])}
        results   = [enzymes_a[eid] for eid in set(enzymes_a) & set(enzymes_b)]
        for e in results:
            if str(e.get('gene_name', '')) == 'nan':
                e['gene_name'] = e.get('enzyme_name', 'Unknown').split(' ')[-1]
            gene = e.get('gene_name', '')
            if gene and not gene.startswith('CYP'):
                e['gene_name'] = 'CYP' + gene
        return results

    def get_shared_targets(self, id_a, id_b):
        targets_a = {t['target_id']: t for t in self.drug_targets.get(id_a, [])}
        targets_b = {t['target_id']: t for t in self.drug_targets.get(id_b, [])}
        return [targets_a[tid] for tid in set(targets_a) & set(targets_b)]

    def get_shared_pathways(self, id_a, id_b):
        return list(set(self.drug_pathways.get(id_a, [])) &
                    set(self.drug_pathways.get(id_b, [])))

    def get_known_interaction(self, id_a, id_b):
        mask = (
            (self.known_interactions['drug_1_id'] == id_a) &
            (self.known_interactions['drug_2_id'] == id_b)
        ) | (
            (self.known_interactions['drug_1_id'] == id_b) &
            (self.known_interactions['drug_2_id'] == id_a)
        )
        matches = self.known_interactions[mask]
        if len(matches) == 0:
            return None
        row = matches.iloc[0]
        # Only return real DrugBank interactions — sampled negatives
        # exist in the CSV but have no mechanism text
        mechanism = row.get('mechanism', None)
        if pd.isna(mechanism) or str(mechanism).strip() in ('', 'nan', 'None'):
            return None
        return row.to_dict()

    def extract(self, drug_a, drug_b):
        id_a, name_a = self.resolve_drug(drug_a)
        id_b, name_b = self.resolve_drug(drug_b)

        shared_enzymes    = self.get_shared_enzymes(id_a, id_b)
        shared_targets    = self.get_shared_targets(id_a, id_b)
        shared_pathways   = self.get_shared_pathways(id_a, id_b)
        known_interaction = self.get_known_interaction(id_a, id_b)

        shared_transporter_count = len(
            self.drug_transporters.get(id_a, set()) &
            self.drug_transporters.get(id_b, set())
        )
        shared_carrier_count = len(
            self.drug_carriers.get(id_a, set()) &
            self.drug_carriers.get(id_b, set())
        )

        return {
            'drug_a': {'id': id_a, 'name': name_a},
            'drug_b': {'id': id_b, 'name': name_b},
            'shared_enzymes':           shared_enzymes,
            'shared_targets':           shared_targets,
            'shared_pathways':          shared_pathways,
            'enzymes_a':                self.drug_enzymes.get(id_a, []),
            'enzymes_b':                self.drug_enzymes.get(id_b, []),
            'targets_a':                self.drug_targets.get(id_a, []),
            'targets_b':                self.drug_targets.get(id_b, []),
            'known_interaction':        known_interaction,
            'shared_enzyme_count':      len(shared_enzymes),
            'shared_target_count':      len(shared_targets),
            'shared_transporter_count': shared_transporter_count,
            'shared_carrier_count':     shared_carrier_count,
            'max_PRR':         known_interaction.get('max_PRR', 0.0) if known_interaction else 0.0,
            'twosides_found':  1 if (known_interaction and known_interaction.get('twosides_found', 0)) else 0,
        }


# ── Test ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fe = FeatureExtractor()
    for pair in [('Warfarin', 'Aspirin'), ('Warfarin', 'Fluconazole')]:
        print(f"\n{'─'*50}")
        ctx = fe.extract(*pair)
        print(f"Drug A: {ctx['drug_a']}")
        print(f"Drug B: {ctx['drug_b']}")
        print(f"Shared enzymes ({ctx['shared_enzyme_count']}): "
              f"{[e.get('gene_name') for e in ctx['shared_enzymes']]}")
        print(f"Shared targets ({ctx['shared_target_count']})")
        print(f"Shared transporters: {ctx['shared_transporter_count']}")
        print(f"Shared carriers:     {ctx['shared_carrier_count']}")
        print(f"Known interaction:   {ctx['known_interaction'] is not None}")
