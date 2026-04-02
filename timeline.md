# TIMELINE

## 10-Month Project Timeline: Explainable Drug-Drug Interaction Prediction Using Graph Neural Networks

| Month | Phase | Activities | Deliverables |
|-------|-------|------------|--------------|
| Month 1 | Project Initiation & Literature Survey | Problem definition, objective finalization, study of drug-drug interaction prediction, graph neural networks (GNNs), explainable AI in pharmacology, and tool selection | Literature survey report, project proposal |
| Month 2 | Requirement Analysis & System Design | Requirement analysis, system architecture design (GNN encoder, MLP classifier, evidence fusion layer, explainer), workflow modeling, database planning (SQLite) | System architecture & design document |
| Month 3 | Dataset Collection & Annotation | DrugBank structured data extraction (drugs, interactions, enzymes, targets, pathways, SMILES), TWOSIDES adverse event data collection, pair canonicalization, concept mapping | Annotated dataset with mapped drug pairs |
| Month 4 | Data Preprocessing & Feature Engineering | SMILES validation & canonicalization, molecular graph construction (atom/bond features via RDKit), 12-dimension feature extraction (shared enzymes, targets, CYP overlap, TWOSIDES PRR), hard negative sampling, feature normalization | Preprocessed training dataset & feature metadata |
| Month 5 | Model Training & Evaluation | AttentiveFP GNN encoder training, MLP pair classifier training, hyperparameter tuning (learning rates, dropout, weight decay), drug-level splitting, performance evaluation (AUC, AP, confusion matrix) | Trained & optimized GNN + MLP model |
| Month 6 | Backend Development | FastAPI development, model integration, evidence-tier routing logic, three-channel fusion (rule score, ML score, TWOSIDES score), severity & risk index computation | Functional backend system (API) |
| Month 7 | Frontend Development & Visualization | Streamlit UI design, drug search & selection module, prediction result visualization, risk & confidence panels, mechanistic explanation rendering, component score charts | User-friendly web interface |
| Month 8 | Explainability & Evidence Fusion | Rule-based explainer implementation, mechanism composition (DrugBank text, shared enzymes/CYP, TWOSIDES signals), severity-conditioned recommendations, uncertainty labeling per evidence channel | Explainable prediction output with evidence breakdown |
| Month 9 | Database Integration & System Testing | SQLite database construction, indexed pair-key retrieval, CLI/API/batch scoring integration testing, performance optimization, bug fixing | Tested and scalable system |
| Month 10 | Post-Deployment Feedback & Optimization | User acceptance testing, collection of user feedback, system performance optimization, refined technical documentation, and deployment packaging | Optimized version of the deployed system, user feedback summary report, updated technical documentation |

*Table 2: Timeline*
