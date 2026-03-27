# DrugInsightv2 Presentation: Slides 5 - 10

## 3. Model Training Process

### Slide 5: The Learning Paradigm
**Title: Supervised Learning for DDI Prediction**
*   **Objective:** Formulated as a supervised binary classification task (Interaction: True/False).
*   **Supervision Source:** Known positive interactions are drawn directly from curated DrugBank data.
*   **Generalization Goal:** The system is trained to generalize to unseen, novel drug pairs rather than merely memorizing known rows from the dataset.
*   **Loss Function:** Binary Cross-Entropy with Logits (BCEWithLogitsLoss).
*   **Optimization:** Adam optimizer with explicit gradient clipping and learning rate scheduling (ReduceLROnPlateau).

### Slide 6: Rigorous Splitting & Negative Sampling
**Title: Ensuring True Generalization**
*   **Drug-Level Cold-Start Split:** 
    *   Data is split 80/20 at the *drug* level, not the *pair* level.
    *   Validation evaluations are strictly performed on drugs completely unseen during training, simulating real-world novelty.
*   **Intelligent Hard Negative Sampling:**
    *   Avoids naive random sampling, which creates "easy" negatives that inflate metrics.
    *   **70/30 Split:** 70% hard negatives, 30% easy negatives.
    *   "Hardness" is scored by biological plausibility (e.g., non-interacting drugs that still share many targets or enzymes). Prevents the model from learning trivial shortcuts.

### Slide 7: Neural Architecture (GNN + MLP)
**Title: Molecular & Pharmacological Encoders**
*   **1. Molecular Graph Construction:**
    *   SMILES strings are parsed via RDKit into PyTorch Geometric (PyG) graphs.
    *   Nodes = Atoms (8 features), Edges = Bonds (6 features).
*   **2. Graph Neural Network (GNN) Encoder:**
    *   Uses **AttentiveFP** backbone (4 layers, 2 readout timesteps).
    *   Transforms the variable-size molecular graph into a fixed **256-dimensional dense embedding** per drug.
*   **3. Multi-Layer Perceptron (MLP) Classifier:**
    *   Inputs: Drug A embedding (256) + Drug B embedding (256) + Engineered Pair Features (6-dim) = **518-dimensional vector**.
    *   Trunk: 3-layer MLP (518 → 512 → 256 → 128) with BatchNorm, ReLU, and 50% Dropout.
    *   Output: Binary probability logit.

---

## 4. System Architecture & Evidence Fusion

### Slide 8: End-to-End System Flow
**Title: DrugInsight Architecture Overview**
*(Visual Placeholder: Insert the System Architecture Block Diagram here)*
*   **Inputs:** User provides two drugs (names or IDs).
*   **Resolution:** System resolves canonical IDs and extracts structural (SMILES) and tabular (TWOSIDES PRR, DrugBank shared metadata) features.
*   **Inference Pipeline:** SMILES → Graphs (PyG) → GNN Embeddings → Concatenation with Pair Features → MLP Classifier.
*   **Fusion & Explanation:** Raw Machine Learning scores are merged with rule-based heuristics and pharmacovigilance data to produce a final, interpretable risk index.

### Slide 9: Adaptive Evidence Fusion Layer
**Title: Combining Hybrid Evidence Streams**
*   **The Problem:** Neural networks alone are opaque "black boxes." Pure rules miss novel interactions.
*   **The Solution:** An adaptive, weighted fusion of three distinct signal channels:
    1.  **Rule Score (DrugBank):** Weights heavily when curated mechanisms (e.g., shared enzymes) exist. 
    2.  **ML Probability (GNN + MLP):** Acts as the primary signal when curated evidence is absent but structural similarity suggests interaction.
    3.  **PRR Signal (TWOSIDES):** Injects real-world post-market pharmacovigilance severity.
*   **Dynamic Weighting:** The fusion weights gracefully shift depending on evidence availability (e.g., Rule score drops from 60% weight to 10% weight if DrugBank data is missing).

### Slide 10: Rule-Based Explainer
**Title: Transparent, Mechanism-Grounded Logic**
*   **LLM-Free Explainability:** Generates structured, readable narratives without the hallucination risks or compute overhead of Large Language Models.
*   **Mechanistic Priority System:**
    1.  Uses explicit DrugBank curated text if available.
    2.  Infers metabolic mechanisms (e.g., known CYP inducer/inhibitor overlap).
    3.  Infers pharmacodynamic mechanisms (e.g., target or transporter competition).
*   **Actionable Clinical Outputs:**
    *   Translates fused probabilities into a **Risk Index (0-100)** and **Severity Tiers (Minor/Moderate/Major)**.
    *   Appends severity-conditioned clinical recommendations (e.g., "monitor dosing closely").
