# DrugInsight

DrugInsight is a drug-drug interaction (DDI) prediction project built around a graph neural network, curated DrugBank-derived evidence, and TWOSIDES pharmacovigilance signals. The repository includes a Streamlit interface for interactive predictions, a CLI workflow for inference, and training scripts for rebuilding the model.

## Quick Start

```bash
git lfs install
git clone https://github.com/AymanUzayr/DrugInsightv2.git
cd DrugInsightv2
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Platform-specific install options:

- Windows: install Git LFS, then run `git lfs install`
- macOS: `brew install git-lfs` then `git lfs install`
- Ubuntu/Debian: `sudo apt install git-lfs` then `git lfs install`

If you clone before installing Git LFS, large files may not be pulled correctly.

## What The Project Does

Given two drug names or DrugBank IDs, DrugInsight:

- resolves the drugs against processed DrugBank data
- builds molecular graphs from SMILES strings
- runs a GNN-based interaction model
- combines model output with curated interaction evidence and TWOSIDES signals
- returns an interaction decision, severity label, risk index, and explanation

## Main Entry Points

- `streamlit_app.py`: interactive web UI
- `src/predict.py`: CLI prediction workflow and `DDIPredictor` implementation
- `src/train.py`: model training pipeline

The `src/` directory is the authoritative code path for the current app. There are older duplicate scripts at the repository root; prefer the `src/` versions unless you are intentionally working with legacy code.

## Repository Layout

```text
.
|-- streamlit_app.py              # Streamlit application
|-- src/
|   |-- predict.py                # Inference pipeline
|   |-- train.py                  # Training pipeline
|   |-- feature_extractor.py      # DrugBank/TWOSIDES feature extraction
|   |-- explainer.py              # Explanation generation
|   |-- gnn_encoder.py            # Molecular graph encoder
|   |-- ddi_classifier.py         # Pairwise interaction classifier
|   |-- mol_graph.py              # SMILES -> graph conversion
|-- models/ddi_model.pt           # Trained checkpoint
|-- requirements.txt              # Python dependencies
|-- druginsight/                  # Checked-in virtual environment (not source code)
```

## Requirements

- Python 3.10+ recommended
- PyTorch and Torch Geometric compatible with your environment
- RDKit installed successfully
- Enough disk space for the tracked data assets

The project can run on CPU, but training is much more practical with CUDA available.

## Setup

1. Install Git LFS before cloning the repo.
2. Create and activate a virtual environment.
3. Install dependencies.

Example:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running The App

Start the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

The app loads:

- model weights from `models/ddi_model.pt`
- processed data from `data/processed`

Then open the local Streamlit URL shown in your terminal.

## CLI Prediction

Run a prediction from the command line:

```bash
python src/predict.py Warfarin Aspirin
```

JSON output:

```bash
python src/predict.py Warfarin Aspirin --json
```

Accepted inputs:

- common drug names
- DrugBank IDs such as `DB00682`

## Training

To retrain the model:

```bash
python src/train.py
```

Training currently:

- loads processed interaction and structure data from `data/processed`
- builds molecular graphs from filtered DrugBank SMILES
- performs negative sampling using extracted features
- trains the encoder and classifier
- saves the best checkpoint to `models/ddi_model.pt`

## Data Dependencies
Drugbank: https://go.drugbank.com/releases/help

TWOSIDES: https://tatonettilab.org/offsides/

- DrugBank interaction tables
- DrugBank structure and lookup tables
- TWOSIDES-derived feature tables
- SMILES mappings and RxNorm bridge data

Inference and training both depend on those files being present with the expected filenames.

## Notes For Contributors

- Prefer editing files under `src/` for model and pipeline work.
- Be careful with duplicate root-level scripts such as `predict.py` and `train.py`; they do not appear to be the primary runtime path.
- `druginsight/` is a checked-in virtual environment and usually should not be treated as project source.
- `requirements.txt` appears to be a broad environment export, so dependency cleanup may be useful later.

## Known Gaps

- No formal automated test suite is set up yet.
- No lint or type-check configuration is currently defined.
- The README now documents the current workflow, but some code paths still rely on hardcoded relative paths.


