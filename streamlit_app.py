"""
CI/CD Failure Prediction — Conference Presentation Dashboard

Reads data directly from the GitHub repo's raw file URLs, so it always
shows the latest data the GitHub Actions pipeline has collected — no
manual refresh needed.

Run locally:   streamlit run streamlit_app.py
"""

import json
import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Early Prediction of CI/CD Pipeline Failures",
    page_icon="🛰️",
    layout="wide",
)

# ---- Config: point this at your repo ----
GITHUB_USER = "tharindra0622-sys"
GITHUB_REPO = "gha-demo"
BRANCH = "main"
BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}"
API_BASE = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}"

AUTHOR_NAME = "Tharindra"
INSTITUTION = "University of Peradeniya"
THESIS_TITLE = "Early Prediction of CI/CD Pipeline Failures in DevOps and AIOps"

# ---------------------------------------------------------------------
# Visual design
# ---------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
code, .stCode, [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; max-width: 1200px; }

/* ---- Hero banner ---- */
.hero-banner {
    background: linear-gradient(120deg, #4338CA 0%, #6366F1 50%, #8B5CF6 100%);
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.18);
    padding: 4px 12px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em;
    margin-bottom: 14px;
}
.hero-dot { width: 8px; height: 8px; border-radius: 50%; background: #4ADE80; animation: pulse 1.6s infinite; }
@keyframes pulse { 0%{opacity:1;} 50%{opacity:0.3;} 100%{opacity:1;} }
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem; font-weight: 700; line-height: 1.25;
    margin-bottom: 8px; max-width: 760px;
}
.hero-sub { font-size: 1rem; opacity: 0.92; margin-bottom: 4px; }
.hero-meta { font-size: 0.85rem; opacity: 0.75; margin-top: 10px; }

/* ---- KPI cards ---- */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 16px 18px 12px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="stMetricLabel"] { color: #6B7280 !important; font-size: 0.78rem !important; font-weight: 600; }
div[data-testid="stMetricValue"] { color: #4338CA !important; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #E5E7EB; }
.stTabs [data-baseweb="tab"] {
    background-color: #F5F5FF; border-radius: 10px 10px 0 0;
    padding: 10px 18px; color: #6366F1; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: white !important; background-color: #6366F1 !important;
}

/* ---- Status badges ---- */
.badge { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 0.76rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.badge-success { background: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
.badge-failure { background: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }
.badge-unknown { background: #F3F4F6; color: #6B7280; border: 1px solid #D1D5DB; }

[data-testid="stExpander"] { border: 1px solid #E5E7EB !important; border-radius: 12px !important; }

/* ---- Pipeline diagram ---- */
.pipeline-wrap { display: flex; flex-wrap: wrap; align-items: center; gap: 0; margin: 1.2rem 0; }
.pipe-box {
    background: #FFFFFF; border: 1.5px solid #C7D2FE; border-radius: 12px;
    padding: 14px 18px; min-width: 190px; text-align: center;
    box-shadow: 0 2px 6px rgba(99,102,241,0.08);
}
.pipe-box .emoji { font-size: 1.4rem; display: block; margin-bottom: 4px; }
.pipe-box .label { font-weight: 700; color: #312E81; font-size: 0.88rem; }
.pipe-box .sub { color: #6B7280; font-size: 0.74rem; margin-top: 2px; }
.pipe-arrow { color: #A5B4FC; font-size: 1.6rem; padding: 0 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_csv(path):
    return pd.read_csv(f"{BASE}/{path}")


@st.cache_data(ttl=300)
def load_json(path):
    r = requests.get(f"{BASE}/{path}")
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def list_dir(path):
    r = requests.get(f"{API_BASE}/contents/{path}?ref={BRANCH}")
    if r.status_code != 200:
        return []
    return [f["name"] for f in r.json() if f["name"].endswith(".json")]


def status_badge(value):
    v = str(value).lower()
    cls = "badge-success" if v == "success" else "badge-failure" if v == "failure" else "badge-unknown"
    return f'<span class="badge {cls}">{value}</span>'


def pipe_box(emoji, label, sub):
    return f'<div class="pipe-box"><span class="emoji">{emoji}</span><div class="label">{label}</div><div class="sub">{sub}</div></div>'


# ---------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-badge"><span class="hero-dot"></span> LIVE — updates automatically from GitHub Actions</div>
    <div class="hero-title">🛰️ {THESIS_TITLE}</div>
    <div class="hero-sub">Real GitHub Actions data → ML prediction → multi-tool LLM diagnosis, fully automated</div>
    <div class="hero-meta">{AUTHOR_NAME} · {INSTITUTION} · Source: {GITHUB_USER}/{GITHUB_REPO}</div>
</div>
""", unsafe_allow_html=True)

try:
    _preds = load_csv("data/predictions.csv")
    _summary = load_json("model/results_summary.json")
    _reports = list_dir("aiops/reports")

    _total_runs = len(_preds)
    _actual_failed = (_preds["metadata_conclusion"] == "failure").astype(int)
    _predicted_failed = _preds.get("predicted_label", pd.Series([0] * len(_preds))).astype(int)
    _live_accuracy = (_actual_failed == _predicted_failed).mean() if _total_runs else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Runs Collected", f"{_total_runs}")
    c2.metric("🏆 Best Model", _summary.get("best_model", "—"))
    c3.metric("🎯 Live Accuracy", f"{_live_accuracy:.0%}")
    c4.metric("🤖 Diagnoses Issued", f"{len(_reports)}")
except Exception:
    st.info("Live metrics will appear here once data has been collected.")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Model Performance", "📄  Live Test Data", "🎯  Prediction Accuracy",
    "🤖  AIOps Diagnoses", "🧭  System Architecture",
])

# ---------------- TAB 1: Model Performance ----------------
with tab1:
    st.subheader("Training Results")
    try:
        summary = load_json("model/results_summary.json")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best model", summary["best_model"])
        col2.metric("CV ROC-AUC", f"{summary['cv_results'][summary['best_model']]['roc_auc']['mean']:.3f}")
        col3.metric("Real-world ROC-AUC", f"{summary['real_world_results']['roc_auc']:.3f}")
        col4.metric("Real-world accuracy", f"{summary['real_world_results']['accuracy']:.1%}")

        st.markdown(f"**Selected features** (RFECV, {summary['n_features_final']} total)")
        st.code(", ".join(summary["rfecv_selected_features"]), language=None)

        st.markdown("**Cross-validation comparison across models**")
        cv_df = pd.DataFrame({
            model: {metric: vals["mean"] for metric, vals in metrics.items()}
            for model, metrics in summary["cv_results"].items()
        }).T
        st.dataframe(cv_df.style.highlight_max(axis=0, color="#E0E7FF"), use_container_width=True)

        st.markdown("**Top SHAP features**")
        st.code(", ".join(summary["shap_top_features"]), language=None)
    except Exception as e:
        st.error(f"Could not load results_summary.json: {e}")

# ---------------- TAB 2: Live Test Data ----------------
with tab2:
    st.subheader("Recent Real GitHub Actions Runs")
    st.caption("Collected automatically by the data pipeline")
    try:
        preds = load_csv("data/predictions.csv")
        st.caption(f"{len(preds)} runs collected so far")

        display = preds.sort_values("run_number", ascending=False).copy()
        display["outcome"] = display["metadata_conclusion"].apply(status_badge)
        display_cols = [
            "workflow_path", "run_number", "metadata_event", "outcome",
            "log_num_jobs", "log_total_steps", "log_error_steps",
            "predicted_failure_probability", "predicted_label",
        ]
        display_cols = [c for c in display_cols if c in display.columns]
        st.write(display[display_cols].to_html(escape=False, index=False), unsafe_allow_html=True)

        st.download_button(
            "⬇ Download full predictions CSV",
            preds.to_csv(index=False),
            file_name="predictions.csv",
        )
    except Exception as e:
        st.error(f"Could not load data/predictions.csv: {e}")

# ---------------- TAB 3: Prediction Accuracy ----------------
with tab3:
    st.subheader("Model Performance on Live, Unseen Data")
    try:
        preds = load_csv("data/predictions.csv")
        actual_failed = (preds["metadata_conclusion"] == "failure").astype(int)
        predicted_failed = preds["predicted_label"].astype(int)
        correct = (actual_failed == predicted_failed)

        col1, col2, col3 = st.columns(3)
        col1.metric("Runs evaluated", len(preds))
        col2.metric("Correct predictions", f"{correct.sum()} / {len(preds)}")
        col3.metric("Accuracy on live data", f"{correct.mean():.1%}")

        st.markdown("**Predicted failure probability per run**")
        chart_df = preds[["run_number", "predicted_failure_probability"]].sort_values("run_number")
        st.bar_chart(chart_df.set_index("run_number"), color="#6366F1")
    except Exception as e:
        st.error(f"Could not compute accuracy: {e}")

# ---------------- TAB 4: AIOps Diagnoses ----------------
with tab4:
    st.subheader("AI-Generated Diagnoses")
    st.caption("Produced by a 7-tool LangChain agent, for failed or high-risk runs")
    try:
        report_files = list_dir("aiops/reports")
        if not report_files:
            st.info("No diagnosis reports yet — they appear here once a run fails or is flagged.")
        for fname in sorted(report_files, reverse=True):
            report = load_json(f"aiops/reports/{fname}")
            diag = report.get("diagnosis", {})
            run = report.get("run", {})
            with st.expander(
                f"Run #{run.get('run_number')} — {run.get('workflow_path')} "
                f"· {diag.get('category', 'unknown')} · {diag.get('confidence', '?')} confidence"
            ):
                badge = status_badge(run.get("metadata_conclusion", "unknown"))
                st.markdown(f"**Actual outcome:** {badge}", unsafe_allow_html=True)
                st.write(f"**Predicted failure probability:** {run.get('predicted_failure_probability')}")
                st.write(f"**Root cause:** {diag.get('root_cause')}")
                evidence = diag.get("evidence_used") or diag.get("tool_calls_made")
                if evidence:
                    st.code(", ".join(evidence), language=None)
                st.write(f"**Suggested fix:** {diag.get('suggested_fix')}")
    except Exception as e:
        st.error(f"Could not load AIOps reports: {e}")

# ---------------- TAB 5: System Architecture (visual diagram) ----------------
with tab5:
    st.subheader("How This System Works")

    st.markdown(f"""
    <div class="pipeline-wrap">
        {pipe_box('⚙️', 'GitHub Actions', 'Real CI/CD pipelines run')}
        <span class="pipe-arrow">→</span>
        {pipe_box('📥', 'Collector', 'Extracts 22 features')}
        <span class="pipe-arrow">→</span>
        {pipe_box('🧠', 'LightGBM Model', 'Predicts failure risk')}
    </div>
    <div style="text-align:right; max-width:640px; color:#A5B4FC; font-size:1.6rem; line-height:1;">↓ failed or high-risk runs continue</div>
    <div class="pipeline-wrap">
        {pipe_box('🕵️', 'Control Agent', 'Flags failed / risky runs')}
        <span class="pipe-arrow">→</span>
        {pipe_box('🤖', 'LangChain Agent', '7 tools, reasons independently')}
        <span class="pipe-arrow">→</span>
        {pipe_box('📝', 'GitHub Issue', 'Human-reviewed recommendation')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### The 7 diagnostic tools")
    tool_cols = st.columns(4)
    tools_info = [
        ("📄", "Failure Log", "The raw error text"),
        ("🗂️", "Past Issues", "Has this happened before?"),
        ("📋", "Workflow File", "What was this step for?"),
        ("🟢", "GitHub Status", "Is the platform healthy?"),
        ("🌐", "Web Search", "Is this a known issue?"),
        ("📦", "Registry Status", "Is PyPI/npm down?"),
        ("🕓", "Recent Commits", "Could an earlier change be the cause?"),
    ]
    for i, (emoji, name, desc) in enumerate(tools_info):
        with tool_cols[i % 4]:
            st.markdown(f"**{emoji} {name}**")
            st.caption(desc)

    st.markdown("---")
    st.markdown(
        "**Design principle:** the agent only ever produces a recommendation — "
        "it never modifies code automatically. A human always reviews and decides."
    )
    st.caption(
        f"This dashboard reads live from "
        f"[{GITHUB_REPO}](https://github.com/{GITHUB_USER}/{GITHUB_REPO}) — "
        "every scheduled Action run updates the data shown here automatically."
    )
