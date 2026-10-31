#!/usr/bin/env python3
"""
Step 1 - build the training data from YOUR OWN runs (run on EC2, repo root).

For every FAILED run of a known test pipeline it downloads the failing job's log from GitHub and
labels it with the injected fault (ground truth, not an LLM guess).

    export GITHUB_TOKEN=...  GITHUB_REPOSITORY=owner/repo
    python3 post_training/build_dataset.py
Output: post_training/data/{train,test_seen,test_heldout}.jsonl
"""
import os, sys, json, random, argparse
from pathlib import Path
import pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "post_training")
import llm_agent
from common import EXPECTED, workflow_key, train_label, user_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--predictions", default="data/predictions.csv")
ap.add_argument("--holdout", default="matrix_dependency_test",
                help="pipelines kept completely out of training (tests generalisation)")
ap.add_argument("--max-per-pipeline", type=int, default=40)
args = ap.parse_args()
holdout = set(filter(None, args.holdout.split(",")))

token = os.environ["GITHUB_TOKEN"]; owner, repo = os.environ["GITHUB_REPOSITORY"].split("/")
out_dir = Path("post_training/data"); out_dir.mkdir(parents=True, exist_ok=True)
cache_path = out_dir / "log_cache.json"
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

df = pd.read_csv(args.predictions).drop_duplicates(subset="metadata_id", keep="last")
df = df[df["metadata_conclusion"] == "failure"].copy()
df["key"] = df["workflow_path"].map(workflow_key)
df = df[df["key"].isin(EXPECTED)].sort_values("run_number")

rows = []
for key, g in df.groupby("key"):
    for _, r in g.tail(args.max_per_pipeline).iterrows():     # most recent runs of each pipeline
        run_id = str(r["metadata_id"])
        if run_id not in cache:
            try:
                job, _ = llm_agent.find_failing_job_and_step(llm_agent.get_jobs(owner, repo, run_id, token))
                cache[run_id] = llm_agent.download_log_text_for_job(owner, repo, run_id, job, token) if job else ""
            except Exception as e:
                print("skip", run_id, e); continue
            cache_path.write_text(json.dumps(cache))
        log = cache[run_id]
        if len(log.strip()) < 50:
            continue
        rows.append({"run_id": run_id, "run_number": int(r["run_number"]), "pipeline": key,
                     "prompt": user_prompt(r["workflow_path"], r.get("metadata_head_commit_message", ""), log),
                     "label": train_label(key), "accepted": sorted(EXPECTED[key])})

held = [r for r in rows if r["pipeline"] in holdout]
seen = [r for r in rows if r["pipeline"] not in holdout]
random.seed(42)
# later 20% of each seen pipeline's runs are the test_seen set (time-based, never trained on)
train, test_seen = [], []
for key in {r["pipeline"] for r in seen}:
    g = sorted([r for r in seen if r["pipeline"] == key], key=lambda r: r["run_number"])
    cut = max(1, int(len(g) * 0.8)); train += g[:cut]; test_seen += g[cut:]
random.shuffle(train)
for name, data in [("train", train), ("test_seen", test_seen), ("test_heldout", held)]:
    (out_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in data))
    print(f"{name}: {len(data)} examples", {k: sum(1 for x in data if x['pipeline'] == k) for k in sorted({x['pipeline'] for x in data})})
