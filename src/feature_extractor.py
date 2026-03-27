import json
import os
import re
import sqlite3
from functools import lru_cache

import numpy as np
import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
DEFAULT_FEATURE_METADATA_PATH = os.path.join(DEFAULT_DATA_DIR, 'feature_metadata.json')
MAJOR_CYPS = ('CYP3A4', 'CYP2D6', 'CYP2C9')
FEATURE_COLUMNS = [
    'shared_enzyme_count',
    'shared_target_count',
    'shared_transporter_count',
    'shared_carrier_count',
    'shared_pathway_count',
    'shared_major_cyp_count',
    'cyp3a4_shared',
    'cyp2d6_shared',
    'cyp2c9_shared',
    'twosides_max_prr',
    'twosides_num_signals',
    'twosides_found',
]
COMMON_ALIASES = {
    'aspirin': 'acetylsalicylic acid',
    'tylenol': 'acetaminophen',
    'advil': 'ibuprofen',
    'motrin': 'ibuprofen',
    'glucophage': 'metformin',
    'zocor': 'simvastatin',
    'lipitor': 'atorvastatin',
    'coumadin': 'warfarin',
    'prozac': 'fluoxetine',
    'zoloft': 'sertraline',
    'prinivil': 'lisinopril',
    'norvasc': 'amlodipine',
}


def normalize_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    text = str(value).lower().strip()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def canonical_pair_ids(id_a, id_b):
    id_a = str(id_a).strip()
    id_b = str(id_b).strip()
    return (id_a, id_b) if id_a <= id_b else (id_b, id_a)


def canonical_pair_key(id_a, id_b):
    pair = canonical_pair_ids(id_a, id_b)
    return f'{pair[0]}||{pair[1]}'


def load_feature_metadata(feature_metadata_path=DEFAULT_FEATURE_METADATA_PATH):
    with open(feature_metadata_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def build_normalized_feature_vector(record, feature_metadata):
    caps = feature_metadata['feature_caps']
    vector = []
    for feature_name in feature_metadata['feature_order']:
        raw_value = float(record.get(feature_name, 0) or 0)
        cap = float(caps.get(feature_name, 1.0) or 1.0)
        if feature_name.endswith('_shared') or feature_name == 'twosides_found':
            vector.append(1.0 if raw_value > 0 else 0.0)
        else:
            vector.append(min(raw_value, cap) / cap if cap > 0 else 0.0)
    return vector


class FeatureExtractor:
    """
    Loads the canonical processed data and computes pair features on demand.
    """

    def __init__(self, data_dir=DEFAULT_DATA_DIR):
        self.data_dir = os.path.abspath(data_dir)
        self.feature_metadata = load_feature_metadata(
            os.path.join(self.data_dir, 'feature_metadata.json')
        )

        # ── Small objects: kept in memory (~45 MB combined) ──────────────────
        self.drug_catalog = pd.read_csv(os.path.join(self.data_dir, 'drug_catalog.csv'))
        self.id_to_name = dict(zip(self.drug_catalog['drugbank_id'], self.drug_catalog['name']))

        cols = ['indication', 'pharmacodynamics', 'mechanism_of_action', 'metabolism', 'absorption', 'half_life', 'toxicity', 'categories', 'description']
        self.drug_info = {}
        for row in self.drug_catalog.itertuples(index=False):
            info = {}
            for col in cols:
                if hasattr(row, col):
                    val = str(getattr(row, col)).strip()
                    if val.lower() not in ['', 'nan', 'none']:
                        info[col] = val
            self.drug_info[row.drugbank_id] = info

        self.name_to_ids = {}
        self.synonym_to_ids = {}
        for row in self.drug_catalog.itertuples(index=False):
            name_key = normalize_text(getattr(row, 'normalized_name', row.name))
            if name_key:
                self.name_to_ids.setdefault(name_key, set()).add(row.drugbank_id)

            normalized_synonyms = getattr(row, 'normalized_synonyms', '')
            if not pd.isna(normalized_synonyms):
                for synonym in str(normalized_synonyms).split('|'):
                    synonym_key = normalize_text(synonym)
                    if synonym_key:
                        self.synonym_to_ids.setdefault(synonym_key, set()).add(row.drugbank_id)

        for alias, canonical in COMMON_ALIASES.items():
            canonical_key = normalize_text(canonical)
            alias_key = normalize_text(alias)
            candidate_ids = self.name_to_ids.get(canonical_key, set())
            if len(candidate_ids) == 1:
                self.name_to_ids.setdefault(alias_key, set()).update(candidate_ids)

        smiles_df = pd.read_csv(os.path.join(self.data_dir, 'drugbank_smiles_filtered.csv'))
        self.smiles_dict = dict(zip(smiles_df['drugbank_id'], smiles_df['smiles']))

        self.drug_enzymes, self.drug_enzyme_ids, self.drug_major_cyps = self._load_enzymes()
        self.drug_targets, self.drug_target_ids = self._load_targets()
        self.drug_transporters = self._load_id_sets('drugbank_transporters.csv', 'transporter_id')
        self.drug_carriers = self._load_id_sets('drugbank_carriers.csv', 'carrier_id')
        self.drug_pathways = self._load_pathways()

        # ── Large objects: served from SQLite (saves ~1.5 GB peak RAM) ──────
        self.db_path = os.path.join(self.data_dir, 'druginsight.db')
        self._ensure_database()
        self._db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db_conn.row_factory = sqlite3.Row


    def _load_enzymes(self):
        enzymes_df = pd.read_csv(os.path.join(self.data_dir, 'drugbank_enzymes.csv'))
        enzymes_df['enzyme_id'] = enzymes_df['enzyme_id'].fillna('').astype(str).str.strip()
        enzymes_df['gene_name'] = enzymes_df['gene_name'].fillna('').astype(str).str.upper().str.strip()
        enzymes_df = enzymes_df[enzymes_df['enzyme_id'] != '']
        records = {}
        for drugbank_id, frame in enzymes_df.groupby('drugbank_id'):
            records[drugbank_id] = frame[
                ['enzyme_id', 'enzyme_name', 'gene_name', 'actions', 'uniprot_id']
            ].to_dict('records')
        ids = enzymes_df.groupby('drugbank_id')['enzyme_id'].apply(set).to_dict()
        major_cyps = (
            enzymes_df[enzymes_df['gene_name'].isin(MAJOR_CYPS)]
            .groupby('drugbank_id')['gene_name']
            .apply(lambda values: set(values.dropna()))
            .to_dict()
        )
        return records, ids, major_cyps

    def _load_targets(self):
        targets_df = pd.read_csv(os.path.join(self.data_dir, 'drugbank_targets.csv'))
        targets_df['target_id'] = targets_df['target_id'].fillna('').astype(str).str.strip()
        targets_df['gene_name'] = targets_df['gene_name'].fillna('').astype(str).str.upper().str.strip()
        targets_df = targets_df[targets_df['target_id'] != '']
        records = {}
        for drugbank_id, frame in targets_df.groupby('drugbank_id'):
            records[drugbank_id] = frame[
                ['target_id', 'target_name', 'gene_name', 'actions', 'known_action', 'uniprot_id']
            ].to_dict('records')
        ids = targets_df.groupby('drugbank_id')['target_id'].apply(set).to_dict()
        return records, ids

    def _load_id_sets(self, filename, id_column):
        df = pd.read_csv(os.path.join(self.data_dir, filename))
        if id_column not in df.columns:
            id_column = df.columns[2]
        df[id_column] = df[id_column].fillna('').astype(str).str.strip()
        df = df[df[id_column] != '']
        return df.groupby('drugbank_id')[id_column].apply(set).to_dict()

    def _load_pathways(self):
        pathways_df = pd.read_csv(os.path.join(self.data_dir, 'drugbank_pathways.csv'))
        pathways_df['pathway_name'] = pathways_df['pathway_name'].fillna('').astype(str).str.strip()
        return (
            pathways_df.groupby('drugbank_id')['pathway_name']
            .apply(lambda values: set(v for v in values if v))
            .to_dict()
        )

    def resolve_drug(self, drug_input):
        drug_input = str(drug_input).strip()
        upper = drug_input.upper()
        if upper in self.id_to_name:
            return upper, self.id_to_name[upper]

        normalized = normalize_text(drug_input)
        for lookup in (self.name_to_ids, self.synonym_to_ids):
            candidate_ids = lookup.get(normalized, set())
            if len(candidate_ids) == 1:
                drugbank_id = next(iter(candidate_ids))
                return drugbank_id, self.id_to_name.get(drugbank_id, drug_input)

        if len(normalized) >= 4:
            for key, candidate_ids in self.name_to_ids.items():
                if key.startswith(normalized) and len(candidate_ids) == 1:
                    drugbank_id = next(iter(candidate_ids))
                    return drugbank_id, self.id_to_name.get(drugbank_id, drug_input)
            for key, candidate_ids in self.name_to_ids.items():
                if normalized in key and len(candidate_ids) == 1:
                    drugbank_id = next(iter(candidate_ids))
                    return drugbank_id, self.id_to_name.get(drugbank_id, drug_input)

        raise ValueError(f"Drug '{drug_input}' not found in DrugBank database.")

    def _ensure_database(self):
        """Build the SQLite database from CSVs if it does not exist yet."""
        if os.path.exists(self.db_path):
            return
        from build_sqlite_db import build_database
        build_database()

    def _query_db(self, table, pair_key):
        """Run an indexed SELECT on pair_key. Returns a dict or None."""
        cursor = self._db_conn.execute(
            f'SELECT * FROM {table} WHERE pair_key = ?', (pair_key,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def get_known_interaction(self, id_a, id_b):
        pair_key = canonical_pair_key(id_a, id_b)
        row = self._query_db('known_interactions', pair_key)
        if row is None:
            return None
        if int(row.get('direct_drugbank_hit', 0) or 0) != 1:
            return None
        row['mechanism'] = row.get('mechanism_primary', '')
        return row

    def get_twosides_signal(self, id_a, id_b):
        pair_key = canonical_pair_key(id_a, id_b)
        return self._query_db('twosides_pairs', pair_key)

    def _is_excluded_pair(self, pair_key):
        """Check if a pair_key exists in either known_interactions or twosides_pairs."""
        cursor = self._db_conn.execute(
            'SELECT 1 FROM known_interactions WHERE pair_key = ? '
            'UNION SELECT 1 FROM twosides_pairs WHERE pair_key = ? LIMIT 1',
            (pair_key, pair_key)
        )
        return cursor.fetchone() is not None

    def get_shared_enzymes(self, id_a, id_b):
        enzymes_a = {row['enzyme_id']: dict(row) for row in self.drug_enzymes.get(id_a, [])}
        enzymes_b = {row['enzyme_id']: dict(row) for row in self.drug_enzymes.get(id_b, [])}
        shared_ids = sorted(set(enzymes_a) & set(enzymes_b))
        return [enzymes_a[enzyme_id] for enzyme_id in shared_ids]

    def get_shared_targets(self, id_a, id_b):
        targets_a = {row['target_id']: dict(row) for row in self.drug_targets.get(id_a, [])}
        targets_b = {row['target_id']: dict(row) for row in self.drug_targets.get(id_b, [])}
        shared_ids = sorted(set(targets_a) & set(targets_b))
        return [targets_a[target_id] for target_id in shared_ids]

    def get_shared_pathways(self, id_a, id_b):
        return sorted(self.drug_pathways.get(id_a, set()) & self.drug_pathways.get(id_b, set()))

    def pair_features(self, id_a, id_b):
        id_a = str(id_a)
        id_b = str(id_b)
        pair_key = canonical_pair_key(id_a, id_b)
        shared_major_cyps = self.drug_major_cyps.get(id_a, set()) & self.drug_major_cyps.get(id_b, set())
        twosides = self.get_twosides_signal(id_a, id_b) or {}
        known = self.get_known_interaction(id_a, id_b)

        features = {
            'pair_key': pair_key,
            'shared_enzyme_count': len(self.drug_enzyme_ids.get(id_a, set()) & self.drug_enzyme_ids.get(id_b, set())),
            'shared_target_count': len(self.drug_target_ids.get(id_a, set()) & self.drug_target_ids.get(id_b, set())),
            'shared_transporter_count': len(self.drug_transporters.get(id_a, set()) & self.drug_transporters.get(id_b, set())),
            'shared_carrier_count': len(self.drug_carriers.get(id_a, set()) & self.drug_carriers.get(id_b, set())),
            'shared_pathway_count': len(self.drug_pathways.get(id_a, set()) & self.drug_pathways.get(id_b, set())),
            'shared_major_cyp_count': len(shared_major_cyps),
            'cyp3a4_shared': int('CYP3A4' in shared_major_cyps),
            'cyp2d6_shared': int('CYP2D6' in shared_major_cyps),
            'cyp2c9_shared': int('CYP2C9' in shared_major_cyps),
            'twosides_found': int(twosides.get('twosides_found', 0) or 0),
            'twosides_max_prr': float(twosides.get('twosides_max_prr', 0.0) or 0.0),
            'twosides_mean_prr': float(twosides.get('twosides_mean_prr', 0.0) or 0.0),
            'twosides_num_signals': int(twosides.get('twosides_num_signals', 0) or 0),
            'twosides_total_coreports': float(twosides.get('twosides_total_coreports', 0.0) or 0.0),
            'twosides_mean_report_freq': float(twosides.get('twosides_mean_report_freq', 0.0) or 0.0),
            'twosides_top_condition': twosides.get('twosides_top_condition', ''),
            'twosides_mapping_source': twosides.get('twosides_mapping_source', ''),
            'twosides_mapping_status': twosides.get('twosides_mapping_status', 'not_mapped'),
            'direct_drugbank_hit': int(known is not None),
            'both_have_smiles': int(id_a in self.smiles_dict and id_b in self.smiles_dict),
        }
        features['max_PRR'] = features['twosides_max_prr']
        return features

    def determine_evidence_tier(self, features):
        if int(features.get('direct_drugbank_hit', 0) or 0) == 1:
            return 'tier_1_direct_drugbank'
        has_structured_evidence = any(
            float(features.get(name, 0) or 0) > 0
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
                'twosides_found',
            )
        )
        return 'tier_2_evidence_fusion' if has_structured_evidence else 'tier_3_structure_only'

    def feature_vector(self, features):
        return build_normalized_feature_vector(features, self.feature_metadata)

    def sample_hard_negatives(
        self,
        drug_pool,
        positive_pairs,
        n,
        seed=42,
        candidate_multiplier=50,
        hard_fraction=0.7,
    ):
        rng = np.random.default_rng(seed)
        drug_pool = list(drug_pool)
        positive_pair_keys = {canonical_pair_key(a, b) for a, b in positive_pairs}

        n_candidates = max(int(n * candidate_multiplier), n * 5)
        a_idx = rng.integers(0, len(drug_pool), size=n_candidates)
        b_idx = rng.integers(0, len(drug_pool), size=n_candidates)

        rows = []
        seen = set()
        for ai, bi in zip(a_idx, b_idx):
            drug_a = str(drug_pool[ai])
            drug_b = str(drug_pool[bi])
            if drug_a == drug_b:
                continue

            pair_key = canonical_pair_key(drug_a, drug_b)
            if pair_key in seen:
                continue
            seen.add(pair_key)

            if pair_key in positive_pair_keys or self._is_excluded_pair(pair_key):
                continue

            features = self.pair_features(drug_a, drug_b)
            hardness = (
                2.0 * features['shared_enzyme_count']
                + 2.0 * features['shared_target_count']
                + 1.0 * features['shared_transporter_count']
                + 1.0 * features['shared_carrier_count']
                + 0.5 * features['shared_pathway_count']
                + 2.5 * features['shared_major_cyp_count']
            )
            rows.append({
                'pair_key': pair_key,
                'drug_1_id': canonical_pair_ids(drug_a, drug_b)[0],
                'drug_2_id': canonical_pair_ids(drug_a, drug_b)[1],
                'drug_1_name': self.id_to_name.get(canonical_pair_ids(drug_a, drug_b)[0], canonical_pair_ids(drug_a, drug_b)[0]),
                'drug_2_name': self.id_to_name.get(canonical_pair_ids(drug_a, drug_b)[1], canonical_pair_ids(drug_a, drug_b)[1]),
                'label': 0,
                'direct_drugbank_hit': 0,
                'mechanism_primary': '',
                **features,
                '_hardness': hardness,
            })
            if len(rows) >= n_candidates:
                break

        if not rows:
            columns = ['pair_key', 'drug_1_id', 'drug_2_id', 'label', *FEATURE_COLUMNS]
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(rows).sort_values('_hardness', ascending=False).reset_index(drop=True)
        n_hard = int(round(n * float(hard_fraction)))
        n_easy = max(n - n_hard, 0)

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

        out = out.drop(columns=['_hardness'], errors='ignore')
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return out.head(n)

    def extract(self, drug_a, drug_b):
        id_a, name_a = self.resolve_drug(drug_a)
        id_b, name_b = self.resolve_drug(drug_b)

        features = self.pair_features(id_a, id_b)
        evidence_tier = self.determine_evidence_tier(features)
        known_interaction = self.get_known_interaction(id_a, id_b)

        context = {
            'drug_a': {'id': id_a, 'name': name_a},
            'drug_b': {'id': id_b, 'name': name_b},
            'drug_a_info': self.drug_info.get(id_a, {}),
            'drug_b_info': self.drug_info.get(id_b, {}),
            'shared_enzymes': self.get_shared_enzymes(id_a, id_b),
            'shared_targets': self.get_shared_targets(id_a, id_b),
            'shared_pathways': self.get_shared_pathways(id_a, id_b),
            'shared_major_cyps': sorted(self.drug_major_cyps.get(id_a, set()) & self.drug_major_cyps.get(id_b, set())),
            'enzymes_a': self.drug_enzymes.get(id_a, []),
            'enzymes_b': self.drug_enzymes.get(id_b, []),
            'targets_a': self.drug_targets.get(id_a, []),
            'targets_b': self.drug_targets.get(id_b, []),
            'known_interaction': known_interaction,
            'evidence_tier': evidence_tier,
            'model_feature_names': list(self.feature_metadata['feature_order']),
            'model_feature_values': {name: features.get(name, 0) for name in self.feature_metadata['feature_order']},
            'feature_vector': self.feature_vector(features),
            'feature_metadata': self.feature_metadata,
            **features,
        }
        return context


@lru_cache(maxsize=4)
def get_feature_extractor(data_dir=DEFAULT_DATA_DIR):
    return FeatureExtractor(data_dir=data_dir)


def extract(drug_a, drug_b, data_dir=DEFAULT_DATA_DIR):
    return get_feature_extractor(data_dir).extract(drug_a, drug_b)


if __name__ == '__main__':
    print(extract('Warfarin', 'Aspirin'))
