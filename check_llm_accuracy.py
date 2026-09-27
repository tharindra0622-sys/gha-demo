#!/usr/bin/env python3
"""
predict_on_new_data.py

Scores freshly collected GitHub Actions run data (from collect_test_data.py)
using the trained LightGBM model (best_model.pkl), reproducing the same
imputation/scaling pipeline used at training time.

WHAT THIS DOES
--------------
The model was trained on 22 RFECV-selected features out of a 192-column
feature matrix. imputer.pkl and scaler.pkl were fit on that full 192-column
matrix, so this script:
  1. Loads imputer.pkl to get the exact 192 column names + per-column median
     (imputer.statistics_) used at training time (SimpleImputer stores
     feature_names_in_ because it was fit on a pandas DataFrame).
  2. Loads scaler.pkl's per-column mean_/scale_ (same column order as the
     imputer, since StandardScaler was fit right after SimpleImputer on the
     same array).
  3. For each of the model's 22 needed features, looks up that column's
     saved median/mean/std by NAME, so imputation+scaling exactly matches
     training — without needing to reconstruct all 192 columns.
  4. Computes the 22 raw feature values from the collected CSV:
       - 14 come straight from the `log_*` columns collect_test_data.py
         already produces
       - is_main_branch: 1 if head branch == 'main'/'master', else 0
       - early_shell_ratio: shell / (shell + action) among first 3 steps
       - early_dur_zscore: (log_early3_avg_dur - 27.339791026265875)
         / 244.06872785106654
         -> this exact formula was reverse-engineered from your real
            processed_ml_data.csv (matched to 9 decimal places), so it's
            EXACT, not an approximation.
       - msg_word_count: word count of the commit message
       - metadata_event_enc / metadata_actor_type_enc /
         metadata_repository_owner_type_enc / metadata_triggering_actor_type_enc:
         *** APPROXIMATION *** — the original training-time category->number
         key was not recoverable from the files available. This script
         assigns numbers to categories alphabetically as they appear across
         your batch (the standard sklearn LabelEncoder convention), which is
         internally consistent but not guaranteed to match the exact codes
         used during training. Document this as a known limitation if you
         use this for your thesis evaluation.

USAGE
-----
    python predict_on_new_data.py --input data/test_data.csv --output predictions.csv
"""

import argparse
import json
import re
import joblib
import numpy as np
import pandas as pd

EARLY_DUR_MEAN = 27.339791026265875  # from processed_ml_data.csv, exact
EARLY_DUR_STD = 244.06872785106654   # from processed_ml_data.csv, exact


def build_lookup(imputer, scaler):
    """Map feature name -> (median, mean, std) using the imputer's saved
    column order (it has feature_names_in_ because it was fit on a
    DataFrame); scaler.mean_/scale_ share that same column order."""
    names = list(imputer.feature_names_in_)
    medians = imputer.statistics_
    means = scaler.mean_
    stds = scaler.scale_
    return {
        name: (medians[i], means[i], stds[i])
        for i, name in enumerate(names)
    }


def encode_categories(series):
    """APPROXIMATION: alphabetical LabelEncoder-style mapping fit fresh on
    whatever categories appear in this batch. See module docstring."""
    cats = sorted(series.dropna().unique().tolist())
    mapping = {cat: i for i, cat in enumerate(cats)}
    return series.map(mapping), mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="collected CSV (runs_200_2 schema)")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--imputer", default="imputer.pkl")
    ap.add_argument("--scaler", default="scaler.pkl")
    ap.add_argument("--model", default="best_model.pkl")
    ap.add_argument("--results-summary", default="results_summary.json")
    args = ap.parse_args()

    imputer = joblib.load(args.imputer)
    scaler = joblib.load(args.scaler)
    model = joblib.load(args.model)
    with open(args.results_summary) as f:
        summary = json.load(f)
    selected_features = summary["rfecv_selected_features"]

    lookup = build_lookup(imputer, scaler)

    df = pd.read_csv(args.input)
    n = len(df)
    out = pd.DataFrame(index=df.index)

    # --- direct log_* passthroughs (exact) ---
    direct_cols = [
        "log_num_jobs", "log_total_steps", "log_shell_steps", "log_action_steps",
        "log_has_linux", "log_has_macos", "log_has_windows", "log_num_os_types",
        "log_early3_total_dur", "log_early3_max_dur", "log_early3_min_dur",
        "log_early3_shell_count", "log_early3_action_count", "log_early3_avg_dur",
    ]
    for col in direct_cols:
        out[col] = pd.to_numeric(df.get(col), errors="coerce")

    # bools -> 0/1
    for col in ["log_has_linux", "log_has_macos", "log_has_windows"]:
        out[col] = out[col].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(out[col])
        out[col] = out[col].astype(float)

    # --- is_main_branch (exact) ---
    out["is_main_branch"] = df.get("metadata_head_branch").astype(str).str.lower().isin(
        ["main", "master"]
    ).astype(int)

    # --- early_shell_ratio (exact) ---
    shell = out["log_early3_shell_count"].fillna(0)
    action = out["log_early3_action_count"].fillna(0)
    denom = (shell + action).replace(0, np.nan)
    out["early_shell_ratio"] = (shell / denom).fillna(0)

    # --- early_dur_zscore (exact formula, reverse-engineered) ---
    out["early_dur_zscore"] = (out["log_early3_avg_dur"] - EARLY_DUR_MEAN) / EARLY_DUR_STD

    # --- msg_word_count (exact) ---
    msgs = df.get("metadata_head_commit_message").fillna("").astype(str)
    out["msg_word_count"] = msgs.apply(lambda s: len(re.findall(r"\S+", s)))

    # --- categorical encodings (APPROXIMATION — see docstring) ---
    cat_source_cols = {
        "metadata_event_enc": "metadata_event",
        "metadata_actor_type_enc": "metadata_actor_type",
        "metadata_repository_owner_type_enc": "metadata_repository_owner_type",
        "metadata_triggering_actor_type_enc": "metadata_triggering_actor_type",
    }
    encoding_maps = {}
    for enc_col, src_col in cat_source_cols.items():
        encoded, mapping = encode_categories(df.get(src_col, pd.Series([None] * n)))
        out[enc_col] = encoded
        encoding_maps[enc_col] = mapping

    print("Category encodings used (APPROXIMATION, not guaranteed to match training):")
    for enc_col, mapping in encoding_maps.items():
        print(f"  {enc_col}: {mapping}")

    # --- impute + scale each of the 22 selected features using training stats ---
    X = np.zeros((n, len(selected_features)))
    for j, feat in enumerate(selected_features):
        if feat not in lookup:
            raise ValueError(f"Feature '{feat}' not found in imputer's saved columns")
        median, mean, std = lookup[feat]
        col = out[feat].astype(float)
        col = col.fillna(median)
        X[:, j] = (col - mean) / (std if std != 0 else 1.0)

    # --- predict ---
    proba_failure = model.predict_proba(X)[:, 1]
    pred_label = model.predict(X)

    result = df.copy()
    result["predicted_failure_probability"] = proba_failure
    result["predicted_label"] = pred_label
    result.to_csv(args.output, index=False)
    print(f"\nWrote {n} predictions to {args.output}")
    print(result[["predicted_label", "predicted_failure_probability"]].describe())


if __name__ == "__main__":
    main()
