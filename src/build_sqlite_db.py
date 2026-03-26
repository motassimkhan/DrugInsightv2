import os
import sqlite3
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
DB_PATH = os.path.join(DATA_DIR, 'druginsight.db')

def build_database():
    if os.path.exists(DB_PATH):
        return

    print("Building local SQLite database (low-memory chunked approach)...")
    
    # Use a database connection.
    conn = sqlite3.connect(DB_PATH)

    interactions_path = os.path.join(DATA_DIR, 'drugbank_interactions_enriched.csv.gz')
    usecols = [
        'pair_key',
        'drug_1_id',
        'drug_2_id',
        'drug_1_name',
        'drug_2_name',
        'direct_drugbank_hit',
        'mechanism_primary',
        'twosides_found',
        'twosides_max_prr',
        'twosides_mean_prr',
        'twosides_num_signals',
        'twosides_total_coreports',
        'twosides_mean_report_freq',
        'twosides_top_condition',
    ]
    
    # 2. dtype specification to skip cost of inference and string parsing latency
    dtypes = {'pair_key': str, 'drug_1_id': str, 'drug_2_id': str, 'drug_1_name': str, 'drug_2_name': str}
    
    # 1. Chunked ingestion must not accumulate into memory. Write directly to SQLite 
    # per chunk and then immediately drop chunk context.
    
    if os.path.exists(interactions_path):
        for chunk in pd.read_csv(interactions_path, usecols=usecols, dtype=dtypes, chunksize=50000, low_memory=False):
            chunk.to_sql('known_interactions', conn, if_exists='append', index=False)
            
    twosides_path = os.path.join(DATA_DIR, 'twosides_mapped.csv')
    if os.path.exists(twosides_path):
        for chunk in pd.read_csv(twosides_path, dtype=dtypes, chunksize=50000, low_memory=False):
            chunk.to_sql('twosides_pairs', conn, if_exists='append', index=False)

    # 3. Create all indexes after final chunk is committed
    print("Creating indexes on pair_key...")
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_enriched_pair_key ON known_interactions(pair_key);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_twosides_pair_key ON twosides_pairs(pair_key);")
    conn.commit()
    
    conn.close()
    print("Database build complete.")

if __name__ == '__main__':
    build_database()
