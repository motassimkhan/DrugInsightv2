import argparse
import json
import math
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog('rdApp.*')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
RAW_TWOSIDES_CANDIDATES = [
    os.path.join(ROOT_DIR, 'data', 'raw', 'TWOSIDES.csv'),
    os.path.join(ROOT_DIR, 'TWOSIDES.csv'),
]
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
MAJOR_CYPS = ('CYP3A4', 'CYP2D6', 'CYP2C9')
MANUAL_TWOSIDES_ALIASES = {
    'aspirin': 'acetylsalicylic acid',
    'beclomethasone': 'beclometasone',
    'cephalothin': 'cefalotin',
    'clavulanate': 'clavulanic acid',
    'cromolyn': 'cromoglicic acid',
    'dipyrone': 'metamizole',
    'dothiepin': 'dosulepin',
    'epoetin alfa': 'erythropoietin',
    'ethinyl estradiol': 'ethinylestradiol',
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


def resolve_twosides_path():
    for candidate in RAW_TWOSIDES_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError('Could not find raw TWOSIDES.csv in data/raw or repository root.')


def ensure_output_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def build_drug_catalog():
    drugs_df = pd.read_csv(os.path.join(DATA_DIR, 'drugbank_drugs.csv'))
    smiles_df = pd.read_csv(os.path.join(DATA_DIR, 'drugbank_smiles.csv'))
    external_ids_df = pd.read_csv(os.path.join(DATA_DIR, 'drugbank_external_ids.csv'))

    valid_smiles = {}
    invalid_smiles = set()
    for row in smiles_df.itertuples(index=False):
        smiles = '' if pd.isna(row.smiles) else str(row.smiles).strip()
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.add(row.drugbank_id)
            continue
        valid_smiles.setdefault(row.drugbank_id, Chem.MolToSmiles(mol))

    smiles_filtered = pd.DataFrame(
        [
            {
                'drugbank_id': drugbank_id,
                'drug_name': drugs_df.loc[drugs_df['drugbank_id'] == drugbank_id, 'name'].iloc[0]
                if drugbank_id in set(drugs_df['drugbank_id'])
                else drugbank_id,
                'smiles': smiles,
            }
            for drugbank_id, smiles in valid_smiles.items()
        ]
    ).sort_values('drugbank_id')

    rxcui_df = external_ids_df[external_ids_df['resource'].astype(str).str.strip() == 'RxCUI'].copy()
    rxcui_df['identifier'] = rxcui_df['identifier'].astype(str).str.strip()
    rxcui_by_drug = (
        rxcui_df.groupby('drugbank_id')['identifier']
        .apply(lambda values: '|'.join(sorted(set(v for v in values if v and v != 'nan'))))
        .to_dict()
    )
    rxcui_to_ids = (
        rxcui_df.groupby('identifier')['drugbank_id']
        .apply(lambda values: set(values.dropna().astype(str)))
        .to_dict()
    )

    catalog = drugs_df.copy()
    catalog['normalized_name'] = catalog['name'].apply(normalize_text)
    catalog['normalized_synonyms'] = catalog['synonyms'].fillna('').astype(str).apply(
        lambda raw: '|'.join(
            sorted({normalized for normalized in (normalize_text(item) for item in raw.split('|')) if normalized})
        )
    )
    catalog['canonical_smiles'] = catalog['drugbank_id'].map(valid_smiles).fillna('')
    catalog['has_smiles'] = catalog['drugbank_id'].isin(valid_smiles).astype(int)
    catalog['smiles_invalid_or_missing'] = catalog['drugbank_id'].apply(
        lambda drugbank_id: int(drugbank_id in invalid_smiles or drugbank_id not in valid_smiles)
    )
    catalog['drug_type'] = catalog['type'].fillna('').astype(str)
    catalog['rxcui_ids'] = catalog['drugbank_id'].map(rxcui_by_drug).fillna('')

    exact_name_map = {}
    synonym_map = {}
    for row in catalog.itertuples(index=False):
        if row.normalized_name:
            exact_name_map.setdefault(row.normalized_name, set()).add(row.drugbank_id)
        for synonym in str(row.normalized_synonyms).split('|'):
            synonym = normalize_text(synonym)
            if synonym:
                synonym_map.setdefault(synonym, set()).add(row.drugbank_id)

    drug_catalog_path = os.path.join(DATA_DIR, 'drug_catalog.csv')
    smiles_filtered_path = os.path.join(DATA_DIR, 'drugbank_smiles_filtered.csv')
    catalog.to_csv(drug_catalog_path, index=False)
    smiles_filtered.to_csv(smiles_filtered_path, index=False)

    return {
        'catalog': catalog,
        'smiles_filtered': smiles_filtered,
        'exact_name_map': exact_name_map,
        'synonym_map': synonym_map,
        'rxcui_to_ids': rxcui_to_ids,
    }


def load_entity_lookups():
    enzymes_df = pd.read_csv(os.path.join(DATA_DIR, 'drugbank_enzymes.csv'))
    enzymes_df['gene_name'] = enzymes_df['gene_name'].fillna('').astype(str).str.upper().str.strip()

    targets_df = pd.read_csv(os.path.join(DATA_DIR, 'drugbank_targets.csv'))
    transporters_df = pd.read_csv(os.path.join(DATA_DIR, 'drugbank_transporters.csv'))
    carriers_df = pd.read_csv(os.path.join(DATA_DIR, 'drugbank_carriers.csv'))
    pathways_df = pd.read_csv(os.path.join(DATA_DIR, 'drugbank_pathways.csv'))

    transporters_col = 'transporter_id' if 'transporter_id' in transporters_df.columns else transporters_df.columns[2]
    carriers_col = 'carrier_id' if 'carrier_id' in carriers_df.columns else carriers_df.columns[2]

    return {
        'enzyme_ids': enzymes_df.groupby('drugbank_id')['enzyme_id'].apply(set).to_dict(),
        'target_ids': targets_df.groupby('drugbank_id')['target_id'].apply(set).to_dict(),
        'transporter_ids': transporters_df.groupby('drugbank_id')[transporters_col].apply(set).to_dict(),
        'carrier_ids': carriers_df.groupby('drugbank_id')[carriers_col].apply(set).to_dict(),
        'pathway_names': pathways_df.groupby('drugbank_id')['pathway_name'].apply(
            lambda values: set(v for v in values.fillna('').astype(str) if v)
        ).to_dict(),
        'major_cyps': (
            enzymes_df[enzymes_df['gene_name'].isin(MAJOR_CYPS)]
            .groupby('drugbank_id')['gene_name']
            .apply(lambda values: set(values.dropna()))
            .to_dict()
        ),
    }


def map_twosides_concept(concept_name, rxnorm_id, exact_name_map, synonym_map, rxcui_to_ids):
    normalized_name = normalize_text(concept_name)

    exact_matches = exact_name_map.get(normalized_name, set())
    if exact_matches:
        if len(exact_matches) == 1:
            return next(iter(exact_matches)), 'exact_name', 'mapped'
        return None, None, 'ambiguous'

    synonym_matches = synonym_map.get(normalized_name, set())
    if synonym_matches:
        if len(synonym_matches) == 1:
            return next(iter(synonym_matches)), 'synonym', 'mapped'
        return None, None, 'ambiguous'

    rxnorm_key = '' if pd.isna(rxnorm_id) else str(rxnorm_id).strip()
    if rxnorm_key:
        rxcui_matches = rxcui_to_ids.get(rxnorm_key, set())
        if rxcui_matches:
            if len(rxcui_matches) == 1:
                return next(iter(rxcui_matches)), 'rxcui', 'mapped'
            return None, None, 'ambiguous'

    alias_target = MANUAL_TWOSIDES_ALIASES.get(normalized_name)
    if alias_target:
        alias_matches = exact_name_map.get(normalize_text(alias_target), set())
        if len(alias_matches) == 1:
            return next(iter(alias_matches)), 'manual_alias', 'mapped'
        if alias_matches:
            return None, None, 'ambiguous'

    return None, None, 'unresolved'


def rebuild_twosides(catalog_bundle):
    twosides_path = resolve_twosides_path()
    raw = pd.read_csv(
        twosides_path,
        usecols=[
            'drug_1_rxnorn_id',
            'drug_1_concept_name',
            'drug_2_rxnorm_id',
            'drug_2_concept_name',
            'condition_concept_name',
            'A',
            'PRR',
            'mean_reporting_frequency',
        ],
        low_memory=False,
    )
    raw = raw.rename(
        columns={
            'drug_1_rxnorn_id': 'drug_1_rxnorm_id',
            'drug_2_rxnorm_id': 'drug_2_rxnorm_id',
        }
    )
    raw['A'] = pd.to_numeric(raw['A'], errors='coerce').fillna(0.0)
    raw['PRR'] = pd.to_numeric(raw['PRR'], errors='coerce').fillna(0.0)
    raw['mean_reporting_frequency'] = pd.to_numeric(raw['mean_reporting_frequency'], errors='coerce').fillna(0.0)

    id_to_name = dict(zip(catalog_bundle['catalog']['drugbank_id'], catalog_bundle['catalog']['name']))
    cache = {}
    mapping_stats = {
        'exact_name_matches': 0,
        'synonym_matches': 0,
        'rxcui_matches': 0,
        'manual_alias_matches': 0,
        'unresolved': 0,
        'ambiguous': 0,
        'dropped': 0,
    }

    left_ids = []
    left_sources = []
    left_statuses = []
    right_ids = []
    right_sources = []
    right_statuses = []

    for row in raw.itertuples(index=False):
        left_cache_key = ('left', normalize_text(row.drug_1_concept_name), str(row.drug_1_rxnorm_id).strip())
        right_cache_key = ('right', normalize_text(row.drug_2_concept_name), str(row.drug_2_rxnorm_id).strip())

        if left_cache_key not in cache:
            cache[left_cache_key] = map_twosides_concept(
                row.drug_1_concept_name,
                row.drug_1_rxnorm_id,
                catalog_bundle['exact_name_map'],
                catalog_bundle['synonym_map'],
                catalog_bundle['rxcui_to_ids'],
            )
        if right_cache_key not in cache:
            cache[right_cache_key] = map_twosides_concept(
                row.drug_2_concept_name,
                row.drug_2_rxnorm_id,
                catalog_bundle['exact_name_map'],
                catalog_bundle['synonym_map'],
                catalog_bundle['rxcui_to_ids'],
            )

        left_id, left_source, left_status = cache[left_cache_key]
        right_id, right_source, right_status = cache[right_cache_key]

        left_ids.append(left_id)
        left_sources.append(left_source or '')
        left_statuses.append(left_status)
        right_ids.append(right_id)
        right_sources.append(right_source or '')
        right_statuses.append(right_status)

        for source, status in ((left_source, left_status), (right_source, right_status)):
            if status == 'mapped':
                mapping_stats[f'{source}_matches'] += 1
            else:
                mapping_stats[status] += 1

    raw['drug_1_id'] = left_ids
    raw['drug_1_mapping_source'] = left_sources
    raw['drug_1_mapping_status'] = left_statuses
    raw['drug_2_id'] = right_ids
    raw['drug_2_mapping_source'] = right_sources
    raw['drug_2_mapping_status'] = right_statuses

    mapped = raw[
        (raw['drug_1_mapping_status'] == 'mapped') &
        (raw['drug_2_mapping_status'] == 'mapped') &
        raw['drug_1_id'].notna() &
        raw['drug_2_id'].notna()
    ].copy()
    before_same_drug_filter = len(mapped)
    mapped = mapped[mapped['drug_1_id'] != mapped['drug_2_id']].copy()
    mapping_stats['dropped'] += int(len(raw) - before_same_drug_filter)
    mapping_stats['dropped'] += int(before_same_drug_filter - len(mapped))

    drug_1_ids = []
    drug_2_ids = []
    for left_id, right_id in zip(mapped['drug_1_id'], mapped['drug_2_id']):
        first_id, second_id = canonical_pair_ids(left_id, right_id)
        drug_1_ids.append(first_id)
        drug_2_ids.append(second_id)

    mapped['drug_1_id'] = drug_1_ids
    mapped['drug_2_id'] = drug_2_ids
    mapped['drug_1_name'] = mapped['drug_1_id'].map(id_to_name).fillna(mapped['drug_1_concept_name'])
    mapped['drug_2_name'] = mapped['drug_2_id'].map(id_to_name).fillna(mapped['drug_2_concept_name'])
    mapped['pair_key'] = mapped['drug_1_id'] + '||' + mapped['drug_2_id']
    mapped['row_mapping_source'] = mapped.apply(
        lambda row: '|'.join(sorted({
            source for source in [row['drug_1_mapping_source'], row['drug_2_mapping_source']] if source
        })),
        axis=1,
    )

    top_condition = (
        mapped.sort_values(['pair_key', 'PRR', 'A'], ascending=[True, False, False])
        .drop_duplicates(subset=['pair_key'])
        [['pair_key', 'condition_concept_name']]
        .rename(columns={'condition_concept_name': 'twosides_top_condition'})
    )
    mapping_sources = (
        mapped.groupby('pair_key')['row_mapping_source']
        .apply(lambda values: '|'.join(sorted(set(v for v in values if v))))
        .reset_index(name='twosides_mapping_source')
    )

    twosides_pairs = (
        mapped.groupby(['pair_key', 'drug_1_id', 'drug_1_name', 'drug_2_id', 'drug_2_name'], as_index=False)
        .agg(
            twosides_num_signals=('condition_concept_name', 'size'),
            twosides_max_prr=('PRR', 'max'),
            twosides_mean_prr=('PRR', 'mean'),
            twosides_total_coreports=('A', 'sum'),
            twosides_mean_report_freq=('mean_reporting_frequency', 'mean'),
        )
    )
    twosides_pairs['twosides_found'] = 1
    twosides_pairs = twosides_pairs.merge(top_condition, on='pair_key', how='left')
    twosides_pairs = twosides_pairs.merge(mapping_sources, on='pair_key', how='left')
    twosides_pairs['twosides_mapping_status'] = 'mapped'

    output_path = os.path.join(DATA_DIR, 'twosides_mapped.csv')
    twosides_pairs.to_csv(output_path, index=False)
    return twosides_pairs, mapping_stats


def compute_pair_features(df, lookups, valid_smiles_ids):
    empty = frozenset()
    drug_1_ids = df['drug_1_id'].astype(str).tolist()
    drug_2_ids = df['drug_2_id'].astype(str).tolist()

    shared_major_cyps = [
        lookups['major_cyps'].get(first_id, empty) & lookups['major_cyps'].get(second_id, empty)
        for first_id, second_id in zip(drug_1_ids, drug_2_ids)
    ]
    df['shared_enzyme_count'] = [
        len(lookups['enzyme_ids'].get(first_id, empty) & lookups['enzyme_ids'].get(second_id, empty))
        for first_id, second_id in zip(drug_1_ids, drug_2_ids)
    ]
    df['shared_target_count'] = [
        len(lookups['target_ids'].get(first_id, empty) & lookups['target_ids'].get(second_id, empty))
        for first_id, second_id in zip(drug_1_ids, drug_2_ids)
    ]
    df['shared_transporter_count'] = [
        len(lookups['transporter_ids'].get(first_id, empty) & lookups['transporter_ids'].get(second_id, empty))
        for first_id, second_id in zip(drug_1_ids, drug_2_ids)
    ]
    df['shared_carrier_count'] = [
        len(lookups['carrier_ids'].get(first_id, empty) & lookups['carrier_ids'].get(second_id, empty))
        for first_id, second_id in zip(drug_1_ids, drug_2_ids)
    ]
    df['shared_pathway_count'] = [
        len(lookups['pathway_names'].get(first_id, empty) & lookups['pathway_names'].get(second_id, empty))
        for first_id, second_id in zip(drug_1_ids, drug_2_ids)
    ]
    df['shared_major_cyp_count'] = [len(values) for values in shared_major_cyps]
    df['cyp3a4_shared'] = [int('CYP3A4' in values) for values in shared_major_cyps]
    df['cyp2d6_shared'] = [int('CYP2D6' in values) for values in shared_major_cyps]
    df['cyp2c9_shared'] = [int('CYP2C9' in values) for values in shared_major_cyps]
    df['both_have_smiles'] = [
        int(first_id in valid_smiles_ids and second_id in valid_smiles_ids)
        for first_id, second_id in zip(drug_1_ids, drug_2_ids)
    ]
    return df


def rebuild_drugbank_interactions(catalog_bundle, twosides_pairs, lookups, **kwargs):
    interactions_path = kwargs.get('interactions_path',
        os.path.join(DATA_DIR, 'drugbank_interactions.csv.gz')
    )
    interactions_df = pd.read_csv(
        interactions_path,
        usecols=['drug_1_id', 'drug_1_name', 'drug_2_id', 'drug_2_name', 'mechanism'],
        low_memory=False,
    )
    interactions_df['mechanism'] = interactions_df['mechanism'].fillna('').astype(str).str.strip()

    original_drug_1_ids = interactions_df['drug_1_id'].astype(str).to_numpy()
    original_drug_2_ids = interactions_df['drug_2_id'].astype(str).to_numpy()
    mask = original_drug_1_ids <= original_drug_2_ids

    interactions_df['drug_1_id'] = np.where(mask, original_drug_1_ids, original_drug_2_ids)
    interactions_df['drug_2_id'] = np.where(mask, original_drug_2_ids, original_drug_1_ids)

    id_to_name = dict(zip(catalog_bundle['catalog']['drugbank_id'], catalog_bundle['catalog']['name']))
    interactions_df['drug_1_name'] = interactions_df['drug_1_id'].map(id_to_name).fillna(interactions_df['drug_1_name'])
    interactions_df['drug_2_name'] = interactions_df['drug_2_id'].map(id_to_name).fillna(interactions_df['drug_2_name'])
    interactions_df['pair_key'] = interactions_df['drug_1_id'] + '||' + interactions_df['drug_2_id']
    interactions_df['mechanism_len'] = interactions_df['mechanism'].str.len()
    interactions_df = (
        interactions_df.sort_values(['pair_key', 'mechanism_len'], ascending=[True, False])
        .drop_duplicates(subset=['pair_key'])
        .rename(columns={'mechanism': 'mechanism_primary'})
        .drop(columns=['mechanism_len'])
        .reset_index(drop=True)
    )

    interactions_df = compute_pair_features(
        interactions_df,
        lookups=lookups,
        valid_smiles_ids=set(catalog_bundle['smiles_filtered']['drugbank_id']),
    )
    interactions_df = interactions_df.merge(
        twosides_pairs[
            [
                'pair_key',
                'twosides_found',
                'twosides_max_prr',
                'twosides_mean_prr',
                'twosides_num_signals',
                'twosides_total_coreports',
                'twosides_mean_report_freq',
                'twosides_top_condition',
                'twosides_mapping_source',
                'twosides_mapping_status',
            ]
        ],
        on='pair_key',
        how='left',
    )

    interactions_df['label'] = 1
    interactions_df['direct_drugbank_hit'] = 1
    for column, default in (
        ('twosides_found', 0),
        ('twosides_max_prr', 0.0),
        ('twosides_mean_prr', 0.0),
        ('twosides_num_signals', 0),
        ('twosides_total_coreports', 0.0),
        ('twosides_mean_report_freq', 0.0),
        ('twosides_top_condition', ''),
        ('twosides_mapping_source', ''),
        ('twosides_mapping_status', 'not_mapped'),
    ):
        interactions_df[column] = interactions_df[column].fillna(default)

    interactions_df['mechanism'] = interactions_df['mechanism_primary']
    interactions_df['max_PRR'] = interactions_df['twosides_max_prr']

    ordered_columns = [
        'pair_key',
        'drug_1_id',
        'drug_2_id',
        'drug_1_name',
        'drug_2_name',
        'label',
        'direct_drugbank_hit',
        'mechanism_primary',
        'mechanism',
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
        'twosides_max_prr',
        'twosides_mean_prr',
        'twosides_num_signals',
        'twosides_total_coreports',
        'twosides_mean_report_freq',
        'twosides_top_condition',
        'twosides_mapping_source',
        'twosides_mapping_status',
        'both_have_smiles',
        'max_PRR',
    ]
    interactions_df = interactions_df[ordered_columns]
    output_path = os.path.join(DATA_DIR, 'drugbank_interactions_enriched.csv.gz')
    interactions_df.to_csv(output_path, index=False)
    return interactions_df


def build_feature_metadata(enriched_pairs, twosides_pairs):
    feature_caps = {}
    for feature_name in FEATURE_COLUMNS:
        if feature_name in ('cyp3a4_shared', 'cyp2d6_shared', 'cyp2c9_shared', 'twosides_found'):
            feature_caps[feature_name] = 1
            continue

        if feature_name in twosides_pairs.columns and not twosides_pairs.empty:
            series = pd.to_numeric(twosides_pairs[feature_name], errors='coerce').fillna(0.0)
        else:
            series = pd.to_numeric(enriched_pairs[feature_name], errors='coerce').fillna(0.0)

        if series.empty:
            feature_caps[feature_name] = 1
            continue

        quantile_value = float(series.quantile(0.99))
        if feature_name == 'shared_major_cyp_count':
            feature_caps[feature_name] = 3
        elif feature_name == 'twosides_max_prr':
            feature_caps[feature_name] = round(max(1.0, quantile_value), 4)
        else:
            feature_caps[feature_name] = max(1, int(math.ceil(quantile_value)))

    metadata = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'feature_order': FEATURE_COLUMNS,
        'extra_dim': len(FEATURE_COLUMNS),
        'feature_caps': feature_caps,
        'major_cyps': list(MAJOR_CYPS),
    }
    output_path = os.path.join(DATA_DIR, 'feature_metadata.json')
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def build_manifest(catalog_bundle, twosides_pairs, enriched_pairs, feature_metadata, mapping_stats):
    manifest = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'row_counts': {
            'twosides_mapped.csv': int(len(twosides_pairs)),
        },
        'rows': {
            'drugbank_interactions_enriched.csv.gz': int(len(enriched_pairs)),
            'drug_catalog.csv': int(len(catalog_bundle['catalog'])),
            'drugbank_smiles_filtered.csv': int(len(catalog_bundle['smiles_filtered'])),
        },
        'feature_metadata': feature_metadata,
        'twosides_mapping_stats': mapping_stats,
        'source_paths': {
            'twosides': resolve_twosides_path(),
        },
        'artifacts': {
            'drugbank_interactions': os.path.join(DATA_DIR, 'drugbank_interactions.csv.gz'),
            'drug_catalog': os.path.join(DATA_DIR, 'drug_catalog.csv'),
            'drugbank_drugs': os.path.join(DATA_DIR, 'drugbank_drugs.csv'),
        },
    }
    output_path = os.path.join(DATA_DIR, 'preprocess_manifest.json')
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def rebuild_all():
    ensure_output_dir()
    catalog_bundle = build_drug_catalog()
    lookups = load_entity_lookups()
    twosides_pairs, mapping_stats = rebuild_twosides(catalog_bundle)
    enriched_pairs = rebuild_drugbank_interactions(catalog_bundle, twosides_pairs, lookups)
    feature_metadata = build_feature_metadata(enriched_pairs, twosides_pairs)
    manifest = build_manifest(catalog_bundle, twosides_pairs, enriched_pairs, feature_metadata, mapping_stats)
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description='DrugInsight preprocessing rebuild')
    parser.add_argument('--rebuild-all', action='store_true', help='Rebuild all processed outputs')
    args = parser.parse_args()

    if not args.rebuild_all:
        parser.error('Pass --rebuild-all to regenerate the processed data artifacts.')

    rebuild_all()


if __name__ == '__main__':
    main()
