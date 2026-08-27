"""
CI/CD Failure Prediction — Live Dashboard (professional edition)

Reads data directly from the gha-demo GitHub repo's raw file URLs, so it
always shows the latest data your GitHub Actions pipeline has collected —
no manual refresh or re-upload needed.

Run locally:   streamlit run streamlit_app.py
Deploy:        https://share.streamlit.io
"""

import json
import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CI/CD Failure Prediction — Live Dashboard",
    page_icon="🛰️",
    layout="wide",
)

# ---- Config: point this at your repo ----
GITHUB_USER = "tharindra0622-sys"
GITHUB_REPO = "gha-demo"
BRANCH = "main"
BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}"
API_BASE = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}"

# ---------------------------------------------------------------------
# Visual polish: fonts, metric cards, status badges, tab styling
# ---------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
code, .stCode, [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }

/* Hero title block */
.hero-title { font-size: 2.1rem; font-weight: 700; color: #E5E9F0; margin-bottom: 0.1rem; }
.hero-sub   { color: #8B95AA; font-size: 0.95rem; margin-bottom: 1.4rem; }

/* Metric strip cards */
div[data-testid="stMetric"] {
    background: #131B2E;
    border: 1px solid #1F2B45;
    border-radius: 10px;
    padding: 14px 16px 10px 16px;
}
div[data-testid="stMetricLabel"] { color: #8B95AA !important; font-size: 0.8rem !important; }
div[data-testid="stMetricValue"] { color: #14B8A6 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background-color: #131B2E; border-radius: 8px 8px 0 0;
    padding: 8px 16px; color: #8B95AA; font-weight: 500;
}
.stTabs [aria-selected="true"] { color: #14B8A6 !important; border-bottom: 2px solid #14B8A6; }

/* Status badges */
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.badge-success { background: rgba(34,197,94,0.15); color: #22C55E; border: 1px solid rgba(34,197,94,0.35); }
.badge-failure { background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.35); }
.badge-unknown { background: rgba(148,163,184,0.15); color: #94A3B8; border: 1px solid rgba(148,163,184,0.35); }

hr { border-color: #1F2B45 !important; }
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


# ---------------------------------------------------------------------
# Hero header + live metrics strip
# ---------------------------------------------------------------------
st.markdown('<div class="hero-title">🛰️ Early Prediction of CI/CD Pipeline Failures</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="hero-sub">Real GitHub Actions data → ML prediction → multi-tool LLM diagnosis, fully automated · '
    f'source: <a href="https://github.com/{GITHUB_USER}/{GITHUB_REPO}" style="color:#14B8A6;">'
    f'{GITHUB_USER}/{GITHUB_REPO}</a></div>',
    unsafe_allow_html=True,
)

try:
    _preds = load_csv("data/predictions.csv")
    _summary = load_json("model/results_summary.json")
    _reports = list_dir("aiops/reports")

    _total_runs = len(_preds)
    _actual_failed = (_preds["metadata_conclusion"] == "failure").astype(int)
    _predicted_failed = _preds.get("predicted_label", pd.Series([0] * len(_preds))).astype(int)
    _live_accuracy = (_actual_failed == _predicted_failed).mean() if _total_runs else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs Collected", f"{_total_runs}")
    c2.metric("Best Model", _summary.get("best_model", "—"))
    c3.metric("Live Accuracy", f"{_live_accuracy:.0%}")
    c4.metric("Diagnoses Issued", f"{len(_reports)}")
except Exception:
    st.info("Live metrics will appear here once data has been collected.")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Model Performance", "📄  Live Test Data", "🎯  Prediction Accuracy",
    "🤖  AIOps Diagnoses", "🧭  Pipeline Overview",
])

# ---------------- TAB 1: Model Performance ----------------
with tab1:
    st.subheader("Training results")
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
        st.dataframe(cv_df.style.highlight_max(axis=0, color="#134E4A"), use_container_width=True)

        st.markdown("**Top SHAP features**")
        st.code(", ".join(summary["shap_top_features"]), language=None)
    except Exception as e:
        st.error(f"Could not load results_summary.json: {e}")

# ---------------- TAB 2: Live Test Data ----------------
with tab2:
    st.subheader("Recent real GitHub Actions runs (collected automatically)")
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
    st.subheader("How well is the model doing on real, unseen data?")
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
        st.bar_chart(chart_df.set_index("run_number"), color="#14B8A6")
    except Exception as e:
        st.error(f"Could not compute accuracy: {e}")

# ---------------- TAB 4: AIOps Diagnoses ----------------
with tab4:
    st.subheader("AI-generated diagnoses for failed / flagged runs")
    try:
        report_files = list_dir("aiops/reports")
        if not report_files:
            st.info("No diagnosis reports yet — they appear here once a run fails or is flagged.")
        for fname in sorted(report_files, reverse=True):
            report = load_json(f"aiops/reports/{fname}")
            diag = report.get("diagnosis", {})
            run = report.get("run", {})
            badge = status_badge(run.get("metadata_conclusion", "unknown"))
            with st.expander(
                f"Run #{run.get('run_number')} — {run.get('workflow_path')} "
                f"· {diag.get('category', 'unknown')} · {diag.get('confidence', '?')} confidence"
            ):
                st.markdown(f"**Actual outcome:** {badge}", unsafe_allow_html=True)
                st.write(f"**Predicted failure probability:** {run.get('predicted_failure_probability')}")
                st.write(f"**Root cause:** {diag.get('root_cause')}")
                evidence = diag.get("evidence_used") or diag.get("tool_calls_made")
                if evidence:
                    st.code(", ".join(evidence), language=None)
                st.write(f"**Suggested fix:** {diag.get('suggested_fix')}")
    except Exception as e:
        st.error(f"Could not load AIOps reports: {e}")

# ---------------- TAB 5: Pipeline Overview ----------------
with tab5:
    st.subheader("How this pipeline works")
    st.markdown("""
    ```
    GitHub Actions runs (real CI/CD pipelines)
            ↓
    collect_test_data.py  → data/test_data.csv
            ↓
    predict_on_new_data.py (trained LightGBM model)  → data/predictions.csv
            ↓
    control_agent.py  → finds failed / predicted-failure runs
            ↓
    langchain_multi_tool_agent.py  → Claude (via LangChain) checks
        multiple tools — failure log, past issues, workflow file,
        GitHub status — before diagnosing root cause + suggesting a fix
            ↓
    GitHub Issue created (recommendation only — no code auto-changed)
    ```
    """)
    st.caption(
        f"This dashboard reads live from the "
        f"[{GITHUB_REPO}](https://github.com/{GITHUB_USER}/{GITHUB_REPO}) repo — "
        "every scheduled Action run updates the data shown here automatically."
    )
