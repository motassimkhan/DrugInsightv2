import streamlit as st
import pandas as pd
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import DDIPredictor, resolve_model_path

st.set_page_config(
    page_title="DrugInsight",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --navy:    #0a0e1a;
    --panel:   #0f1624;
    --border:  #1e2a3a;
    --border2: #253347;
    --amber:   #fe9ec7;
    --amber2:  #ffb8d8;
    --red:     #e05c5c;
    --orange:  #e0883a;
    --green:   #3aad6e;
    --text:    #e8eef5;
    --muted:   #5a7080;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'DM Sans', sans-serif;
    --display: 'Syne', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--navy) !important;
    color: var(--text);
}

.main .block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    max-width: 780px;
}

/* ── Header ── */
.di-header {
    text-align: center;
    margin-bottom: 2.8rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid var(--border);
}
.di-logo {
    font-family: var(--display);
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--text);
    margin-bottom: 0.3rem;
}
.di-logo span {
    color: var(--amber);
}
.di-tagline {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Input area ── */
.input-label {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

.stTextInput input {
    background: var(--panel) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 0.8rem !important;
    transition: border-color 0.15s ease !important;
}
.stTextInput input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px rgba(254, 158, 199, 0.12) !important;
}

/* ── Button ── */
button[kind="primary"] {
    background: var(--amber) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--display) !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.5rem !important;
    transition: background 0.15s ease, box-shadow 0.15s ease !important;
}
button[kind="primary"]:hover {
    background: var(--amber2) !important;
    box-shadow: 0 4px 16px rgba(254, 158, 199, 0.25) !important;
}

/* ── Result header ── */
.result-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.2rem 1.4rem;
    border-radius: 8px;
    margin-bottom: 1.2rem;
    border: 1px solid;
}
.result-major   { background: rgba(224,92,92,0.08);  border-color: rgba(224,92,92,0.3); }
.result-moderate{ background: rgba(224,136,58,0.08); border-color: rgba(224,136,58,0.3); }
.result-minor   { background: rgba(58,173,110,0.08); border-color: rgba(58,173,110,0.3); }
.result-none    { background: rgba(58,173,110,0.06); border-color: rgba(58,173,110,0.2); }

.sev-badge {
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    white-space: nowrap;
}
.badge-major    { background: rgba(224,92,92,0.2);  color: #e05c5c; }
.badge-moderate { background: rgba(224,136,58,0.2); color: #e0883a; }
.badge-minor    { background: rgba(58,173,110,0.2); color: #3aad6e; }
.badge-none     { background: rgba(58,173,110,0.15);color: #3aad6e; }

.result-title {
    font-family: var(--display);
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    flex: 1;
}

/* ── Metric row ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}
.metric-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.1rem;
    text-align: center;
}
.metric-val {
    font-family: var(--display);
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--amber);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-lbl {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── Risk bar ── */
.risk-bar-wrap {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
}
.risk-bar-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
}
.risk-track {
    background: var(--border);
    border-radius: 4px;
    height: 8px;
    position: relative;
    overflow: hidden;
}
.risk-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

/* ── Section blocks ── */
.section-block {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.section-head {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 0.7rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.section-text {
    font-family: var(--sans);
    font-size: 0.9rem;
    line-height: 1.65;
    color: #b8c8d8;
}

/* ── Evidence grid ── */
.ev-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.7rem;
    margin-bottom: 0.8rem;
}
.ev-cell {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1rem;
}
.ev-source {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.ev-conf {
    font-family: var(--display);
    font-size: 0.95rem;
    font-weight: 700;
}
.conf-high     { color: #3aad6e; }
.conf-moderate { color: var(--amber); }
.conf-low      { color: #e0883a; }
.conf-found    { color: #3aad6e; }
.conf-partial  { color: var(--amber); }
.conf-not_found{ color: var(--muted); }
.conf-no_signal{ color: var(--muted); }
.conf-weak     { color: #5a7080; }

/* ── Score bars ── */
.score-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.6rem;
}
.score-name {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    width: 110px;
    flex-shrink: 0;
}
.score-track {
    flex: 1;
    background: var(--border);
    border-radius: 3px;
    height: 6px;
    overflow: hidden;
}
.score-fill {
    height: 100%;
    border-radius: 3px;
    background: var(--amber);
}
.score-val {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text);
    width: 38px;
    text-align: right;
}

/* ── Tag chips ── */
.tag-list { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.4rem; }
.tag {
    font-family: var(--mono);
    font-size: 0.68rem;
    background: rgba(254,158,199,0.1);
    color: var(--amber);
    border: 1px solid rgba(254,158,199,0.2);
    border-radius: 4px;
    padding: 0.2rem 0.5rem;
}
.tag-muted {
    background: rgba(90,112,128,0.15);
    color: var(--muted);
    border-color: var(--border);
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--panel) !important;
    border-color: var(--border2) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}
.stSelectbox label { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.7rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Streamlit overrides ── */
.stSpinner > div { border-top-color: var(--amber) !important; }
[data-testid="stAlert"] { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_predictor():
    return DDIPredictor(model_path=resolve_model_path())



def risk_color(severity):
    return {'Major': '#e05c5c', 'Moderate': '#e0883a', 'Minor': '#3aad6e'}.get(severity, '#3aad6e')


def conf_class(val):
    return f"conf-{val.replace(' ', '_').replace('/', '_')}"


def render_risk_bar(risk_index, severity):
    color = risk_color(severity)
    st.markdown(f"""
    <div class="risk-bar-wrap">
        <div class="risk-bar-label">
            <span>Risk Index</span>
            <span style="color:{color}; font-weight:600">{risk_index} / 100</span>
        </div>
        <div class="risk-track">
            <div class="risk-fill" style="width:{risk_index}%; background:{color}"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_score_bars(component_scores):
    cs = component_scores
    scores = [
        ("Rule (DrugBank)", cs['rule_score'],     cs['weights']['rule']),
        ("ML Model",        cs['ml_score'],        cs['weights']['ml']),
        ("TWOSIDES",        cs['twosides_score'],  cs['weights']['twosides']),
    ]
    html = '<div class="section-block"><div class="section-head">Component Scores</div>'
    for name, score, weight in scores:
        pct = int(score * 100)
        html += f"""
        <div class="score-row">
            <div class="score-name">{name}</div>
            <div class="score-track"><div class="score-fill" style="width:{pct}%"></div></div>
            <div class="score-val">{score:.3f}</div>
        </div>"""
    html += f'<div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:0.7rem">Fusion weights: Rule {cs["weights"]["rule"]} · ML {cs["weights"]["ml"]} · TWOSIDES {cs["weights"]["twosides"]}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div class="di-header">
        <div class="di-logo">Drug<span>Insight</span></div>
        <div class="di-tagline">Neural DDI Prediction · Evidence Fusion · Explainable AI</div>
    </div>
    """, unsafe_allow_html=True)

    # Load predictor
    with st.spinner("Loading model..."):
        predictor = load_predictor()
        drug_list = predictor.drug_names_with_smiles()
    # Drug inputs — selectbox with search
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="input-label">First Drug</div>', unsafe_allow_html=True)
        drug_a = st.selectbox(" ", options=[""] + drug_list, index=0,
                               label_visibility="collapsed", key="drug_a_select")
    with col2:
        st.markdown('<div class="input-label">Second Drug</div>', unsafe_allow_html=True)
        drug_b = st.selectbox(" ", options=[""] + drug_list, index=0,
                               label_visibility="collapsed", key="drug_b_select")

    st.write("")
    run = st.button("Predict Interaction", type="primary", use_container_width=True)

    if run:
        if not drug_a or not drug_b:
            st.warning("Select both drugs to continue.")
            return
        if drug_a == drug_b:
            st.error("Both inputs refer to the same drug. Select two distinct drugs.")
            return

        with st.spinner("Running GNN inference and evidence fusion..."):
            result = predictor.predict(drug_a, drug_b)
            st.session_state['result']  = result
            st.session_state['last_a']  = drug_a
            st.session_state['last_b']  = drug_b

    # Clear if inputs changed
    if 'result' in st.session_state:
        if st.session_state.get('last_a') != drug_a or st.session_state.get('last_b') != drug_b:
            del st.session_state['result']
            st.rerun()

    if 'result' not in st.session_state:
        return

    result = st.session_state['result']

    if 'error' in result:
        st.error(result['error'])
        return

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Result header ──────────────────────────────────────────────────────────
    sev       = result['severity']
    sev_lower = sev.lower()
    found     = result['interaction']
    sev_class = sev_lower if found else 'none'

    badge_map = {
        'major': 'badge-major', 'moderate': 'badge-moderate',
        'minor': 'badge-minor', 'none': 'badge-none'
    }
    badge_text = sev if found else 'No Interaction'
    title_text = (
        f"{result['drug_a']} + {result['drug_b']} — {sev} Interaction Predicted"
        if found else
        f"No Significant Interaction: {result['drug_a']} + {result['drug_b']}"
    )

    st.markdown(f"""
    <div class="result-header result-{sev_class}">
        <span class="sev-badge {badge_map[sev_class]}">{badge_text}</span>
        <span class="result-title">{title_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-val" style="color:{risk_color(sev) if found else '#3aad6e'}">{result['risk_index']}</div>
            <div class="metric-lbl">Risk Index (Severity Scale)</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{result['confidence']}</div>
            <div class="metric-lbl">Interaction Probability (Fused)</div>
        </div>
        <div class="metric-box">
            <div class="metric-val" style="font-size:1.1rem;padding-top:0.3rem">{result['uncertainty']['overall_confidence'].upper()}</div>
            <div class="metric-lbl">Evidence Certainty</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk bar
    render_risk_bar(result['risk_index'], sev if found else 'Minor')

    # ── Mechanism ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="section-block">
        <div class="section-head">Pharmacological Mechanism</div>
        <div class="section-text">{result['mechanism']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Recommendation ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="section-block">
        <div class="section-head">Clinical Recommendation</div>
        <div class="section-text">{result['recommendation']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Evidence sources ───────────────────────────────────────────────────────
    ev  = result['evidence']
    unc = result['uncertainty']

    db_conf  = unc['drugbank_confidence']
    ml_conf  = unc['ml_confidence']
    ts_conf  = unc['twosides_confidence']
    prr      = ev['twosides']['max_PRR']
    prr_text = f"PRR {prr:.1f}" if prr > 0 else "No signal"

    st.markdown(f"""
    <div class="ev-grid">
        <div class="ev-cell">
            <div class="ev-source">DrugBank</div>
            <div class="ev-conf {conf_class(db_conf)}">{db_conf.replace('_', ' ').title()}</div>
            <div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:0.3rem">
                {'Known interaction' if ev['drugbank']['known_interaction'] else 'Not in database'}
            </div>
        </div>
        <div class="ev-cell">
            <div class="ev-source">ML Model</div>
            <div class="ev-conf {conf_class(ml_conf)}">{ml_conf.title()}</div>
            <div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:0.3rem">
                Raw: {result['component_scores']['ml_score']:.3f}
            </div>
        </div>
        <div class="ev-cell">
            <div class="ev-source">TWOSIDES</div>
            <div class="ev-conf {conf_class(ts_conf)}">{ts_conf.replace('_', ' ').title()}</div>
            <div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:0.3rem">
                {prr_text}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if ev['twosides']['confounding_flag']:
        st.warning("High PRR detected. Potential confounding — this signal may reflect individual drug toxicity rather than a true interaction.")

    # ── Shared biology ─────────────────────────────────────────────────────────
    enzymes  = ev['drugbank']['shared_enzymes']
    targets  = ev['drugbank']['shared_targets']
    pathways = ev['drugbank'].get('shared_pathways', [])

    if enzymes or targets or pathways:
        tags_html = '<div class="section-block"><div class="section-head">Shared Biology</div>'
        if enzymes:
            tags_html += '<div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-bottom:0.3rem">ENZYMES</div>'
            tags_html += '<div class="tag-list">' + ''.join(f'<span class="tag">{e}</span>' for e in enzymes) + '</div>'
        if targets:
            tags_html += '<div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:0.7rem;margin-bottom:0.3rem">TARGETS</div>'
            tags_html += '<div class="tag-list">' + ''.join(f'<span class="tag">{t}</span>' for t in targets) + '</div>'
        if pathways:
            tags_html += '<div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:0.7rem;margin-bottom:0.3rem">PATHWAYS</div>'
            tags_html += '<div class="tag-list">' + ''.join(f'<span class="tag tag-muted">{p}</span>' for p in pathways) + '</div>'
        tags_html += '</div>'
        st.markdown(tags_html, unsafe_allow_html=True)

    # ── Component scores ───────────────────────────────────────────────────────
    render_score_bars(result['component_scores'])

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top:2rem;padding-top:1rem;border-top:1px solid var(--border);
                font-family:var(--mono);font-size:0.62rem;color:var(--muted);text-align:center;
                line-height:1.8">
        {result['drug_a']} ({result['drugbank_id_a']}) · {result['drug_b']} ({result['drugbank_id_b']})<br>
        DrugInsight v0.1 · For research use only · Not a substitute for clinical judgment
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
