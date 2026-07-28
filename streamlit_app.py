"""
CI/CD Failure Prediction — Live Dashboard

Reads data directly from the gha-demo GitHub repo's raw file URLs, so it
always shows the latest data your GitHub Actions pipeline has collected —
no manual refresh or re-upload needed.

Run locally:   streamlit run streamlit_app.py
Deploy:        https://share.streamlit.io  (see deployment steps)
"""

import json
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CI/CD Failure Prediction Dashboard", layout="wide")

# ---- Config: point this at your repo ----
GITHUB_USER = "tharindra0622-sys"
GITHUB_REPO = "gha-demo"
BRANCH = "main"
BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}"
API_BASE = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}"


@st.cache_data(ttl=300)  # refresh every 5 minutes
def load_csv(path):
    return pd.read_csv(f"{BASE}/{path}")


@st.cache_data(ttl=300)
def load_json(path):
    r = requests.get(f"{BASE}/{path}")
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def list_dir(path):
    """List files in a repo folder via the GitHub API (raw.githubusercontent
    can't list directories)."""
    r = requests.get(f"{API_BASE}/contents/{path}?ref={BRANCH}")
    if r.status_code != 200:
        return []
    return [f["name"] for f in r.json() if f["name"].endswith(".json")]


st.title("🔍 Early Prediction of CI/CD Pipeline Failures — Live Dashboard")
st.caption(
    "Real GitHub Actions data → ML prediction → AI diagnosis, fully automated. "
    f"Source repo: [{GITHUB_USER}/{GITHUB_REPO}](https://github.com/{GITHUB_USER}/{GITHUB_REPO})"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Performance", "📄 Live Test Data", "🎯 Prediction Accuracy",
    "🤖 AIOps Diagnoses", "🧭 Pipeline Overview",
])

# ---------------- TAB 1: Model Performance ----------------
with tab1:
    st.subheader("Training results (results_summary.json)")
    try:
        summary = load_json("model/results_summary.json")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best model", summary["best_model"])
        col2.metric("CV ROC-AUC", f"{summary['cv_results'][summary['best_model']]['roc_auc']['mean']:.3f}")
        col3.metric("Real-world ROC-AUC", f"{summary['real_world_results']['roc_auc']:.3f}")
        col4.metric("Real-world accuracy", f"{summary['real_world_results']['accuracy']:.1%}")

        st.markdown("**Selected features (RFECV, {} total)**".format(summary["n_features_final"]))
        st.write(", ".join(summary["rfecv_selected_features"]))

        st.markdown("**Cross-validation comparison across models**")
        cv_df = pd.DataFrame({
            model: {metric: vals["mean"] for metric, vals in metrics.items()}
            for model, metrics in summary["cv_results"].items()
        }).T
        st.dataframe(cv_df.style.highlight_max(axis=0, color="#1a3d1a"))

        st.markdown("**Top SHAP features**")
        st.write(", ".join(summary["shap_top_features"]))
    except Exception as e:
        st.error(f"Could not load results_summary.json: {e}")

# ---------------- TAB 2: Live Test Data ----------------
with tab2:
    st.subheader("Recent real GitHub Actions runs (collected automatically)")
    try:
        preds = load_csv("data/predictions.csv")
        st.write(f"{len(preds)} runs collected so far.")
        display_cols = [
            "workflow_path", "run_number", "metadata_event", "metadata_conclusion",
            "log_num_jobs", "log_total_steps", "log_error_steps",
            "predicted_failure_probability", "predicted_label",
        ]
        display_cols = [c for c in display_cols if c in preds.columns]
        st.dataframe(preds[display_cols].sort_values("run_number", ascending=False), use_container_width=True)

        st.download_button(
            "Download full predictions CSV",
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

        st.markdown("**Confusion breakdown**")
        conf = pd.DataFrame({
            "Actual failure": actual_failed, "Predicted failure": predicted_failed,
        }).value_counts().rename("count").reset_index()
        st.dataframe(conf)

        st.markdown("**Predicted failure probability per run**")
        chart_df = preds[["run_number", "predicted_failure_probability"]].sort_values("run_number")
        st.bar_chart(chart_df.set_index("run_number"))
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
            with st.expander(
                f"Run #{run.get('run_number')} — {run.get('workflow_path')} "
                f"({diag.get('category', 'unknown')}, {diag.get('confidence', '?')} confidence)"
            ):
                st.write(f"**Actual outcome:** {run.get('metadata_conclusion')}")
                st.write(f"**Predicted failure probability:** {run.get('predicted_failure_probability')}")
                st.write(f"**Root cause:** {diag.get('root_cause')}")
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
    llm_agent.py  → Gemini diagnoses root cause + suggests a fix
            ↓
    GitHub Issue created (recommendation only — no code auto-changed)
    ```
    """)
    st.markdown(
        "This dashboard reads live from the "
        f"[{GITHUB_REPO}](https://github.com/{GITHUB_USER}/{GITHUB_REPO}) repo — "
        "every scheduled Action run updates the data shown here automatically."
    )
