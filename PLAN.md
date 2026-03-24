# Master Plan: Multimodal Knowledge Graph for DrugInsightv2

**Agent Persona & Context:**
You are acting as an implementation agent for DrugInsightv2. Your strict directives are to implement a 4-phase Knowledge Graph over existing data using DrugBank, Reactome, STRING, and ChEMBL.

**Core Objective:**
Expand the current pipeline by extracting, mapping, and combining hierarchical and physical interactions into a unified Knowledge Graph, and training RotatE embeddings to perform downstream DDI predictions.

---

## 🚦 Optimization Directives for Claude Opus (Token Minimization & Reliability)

To ensure this large-scale project completes without hitting context limits, hallucinating, or looping during errors, follow these rules strictly:

1. **Token Economy (Strict Modularization):**
   - Write separate Python scripts for each discrete step (`prep_drugbank.py`, `prep_reactome.py`, etc.) and output intermediate files (e.g., `data/interim/graph_v1.parquet`).
   - Only output code diffs for fixes—do not re-print entire files if a 2-line fix is needed. Mention the exact location of the replacement snippet.
   - Do NOT run the parser on the full datasets initially. Implement an `--n-samples 5000` toggle in all scripts for dry-run validation.

2. **Automated Error Correction & State Saving (Corrective Measures):**
   - **OOM Errors**: Assume Reactome/STRING data will overwhelm Pandas. Use `chunksize=50000` when reading files, or use Parquet/Polars incrementally.
   - **ID Mapping Failures**: Between STRING (ENSP), Reactome (UniProt), ChEMBL (ChEMBL_ID), and DrugBank/SMILES, IDs will mismatch. Implement an explicit "orphan drop" logic in `utils.py` and print logging stats (e.g., `% of orphans dropped`) rather than crashing.
   - **NaN Exploits**: Fill/drop NaN values prior to building PyKEEN inputs. Check for exploding gradients (Loss = NaN) and automatically trigger gradient clipping in the trainer configuration upon detection.

---

## 🏗️ End-to-End Pipeline & Preprocessing Steps

The graph pipeline will be built in 4 modular phases. For all phases, keep triples in a standardized format: `[head_id, relation_type, tail_id, weight(optional)]`.

### Phase 1: DrugBank Baseline (Binary Foundations)
* **Goal:** Base graph of binary target/enzyme interactions.
* **Nodes:** `Drug`, `Protein` (Target/Enzyme/Transporter)
* **Relations:** `acts_on`, `inhibits`, `induces`, `binds`
* **Preprocessing:**
    1. Load DrugBank XML parsed tables.
    2. Build node dictionaries: `{db_id: 0, db_id_2: 1 ...}`
    3. Generate generic `Drug -> Protein` interaction triples.
    4. Save as `phase1_triples.tsv`.
    5. Run base PyKEEN RotatE training loop.

### Phase 2: Reactome Convergence (Pathway Hierarchies)
* **Goal:** Add hierarchical pathing so different targets converge on pathways.
* **Nodes:** `Protein`, `Reaction`, `Pathway`, `Complex`
* **Relations:** `participates_in`, `part_of`, `contains`
* **Preprocessing:**
    1. Download `UniProt2Reactome_All_Levels.txt`.
    2. Filter strictly by `Homo sapiens`.
    3. Map UniProt IDs to the `Protein` node IDs from Phase 1.
    4. Extract `Protein -> participates_in -> Reaction` and `Reaction -> part_of -> Pathway`.
    5. Combine with Phase 1 data -> `phase2_triples.tsv`.
    6. Retrain embeddings.

### Phase 3: STRING Integration (Physical Networks)
* **Goal:** Cross-pathway indirect interactions (high-confidence physical binding).
* **Nodes:** `Protein`
* **Relations:** `physically_interacts_with`
* **Preprocessing:**
    1. Read `9606.protein.links.v12.0.txt.gz`.
    2. **Crucial filter:** Keep only rows where `combined_score > 700` (high confidence) to prevent edge explosion.
    3. Use mapping tables to convert Ensembl Protein IDs (ENSP) to UniProt IDs matching Phase 1/2.
    4. Construct symmetric edges: `Protein A -> physically_interacts_with -> Protein B`.
    5. Append to graph -> `phase3_triples.tsv`. Retrain.

### Phase 4: ChEMBL & RotatE Adaptation (Quantitative Affinities)
* **Goal:** Replace binary DrugBank interactions with continuous binding affinities (IC50, Ki).
* **Nodes:** `Drug`, `Target`
* **Preprocessing:**
    1. Query local sqlite/ChEMBL CSV dumps for active compound-target pairs.
    2. Unify Drug dimensions by mapping ChEMBL IDs to SMILES, then to DrugBank IDs (using Unichem or rdkit canonicalization).
    3. Unify Target dimensions by mapping ChEMBL Target IDs to UniProt.
    4. **Transformation for RotatE:** Because native RotatE relies on discrete relations, bucket continuous IC50/Ki values into distinct categorical edges to simulate weights:
        - `IC50 < 100 nM`: `strongly_inhibits` / `strongly_binds`
        - `100 < IC50 < 1000 nM`: `moderately_binds`
        - `1000 < IC50 < 10000 nM`: `weakly_binds`
    5. Replace overlapping DrugBank binary edges with these specific gradient relation edges.
    6. Compile the final Knowledge Master Graph -> `phase4_master_triples.tsv`.

---

## 🧠 Downstream Classifier Integration (Final Stage)

Once `phase4_master_triples.tsv` embeddings are trained via PyKEEN RotatE:
1. Export the Entity Embedding matrices.
2. Given any drug combination $(D_1, D_2)$ during an unknown query:
   - Perform lookup to fetch embeddings $\vec{e_1}$ and $\vec{e_2}$.
   - Create fusion vector: $V = [\vec{e_1}, \vec{e_2}, \vec{e_1} \circ \vec{e_2}, |\vec{e_1} - \vec{e_2}|]$ (concatenation, element-wise product, and absolute difference).
   - Feed $V$ into the multi-class DDI predictor (XGBoost/MLP). 

**Final Output for the User:** Claude, as you work, keep track of completion in a `PROGRESS.md` document and ONLY notify the user when a complete Phase (1 through 4) successfully outputs a `.tsv` file and loss metric.
