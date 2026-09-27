name: Collect CI/CD Test Data

on:
  schedule:
    - cron: "0 */6 * * *"   # every 6 hours; tune to how often your other workflows run
  workflow_dispatch: {}       # lets you trigger it manually from the Actions tab

permissions:
  actions: read      # to read run/job data and download logs
  contents: write    # to commit the CSV + checkpoint file back to the repo

jobs:
  collect:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install requests pyyaml joblib scikit-learn lightgbm pandas numpy

      - name: Extract log features from recent completed runs
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          OUT_CSV: data/test_data.csv
          STATE_FILE: data/state/last_checkpoint.txt
          MAX_RUNS: "20"
        run: python scripts/collect_test_data.py

      - name: Score new rows with the trained model
        run: |
          python scripts/predict_on_new_data.py \
            --input data/test_data.csv \
            --output data/predictions.csv \
            --imputer model/imputer.pkl \
            --scaler model/scaler.pkl \
            --model model/best_model.pkl \
            --results-summary model/results_summary.json

      - name: Commit new test data and predictions
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/test_data.csv data/state/last_checkpoint.txt data/predictions.csv
          if ! git diff --cached --quiet; then
            git commit -m "chore: append new CI/CD test-data rows + predictions [skip ci]"
            git push
          else
            echo "No new rows to commit."
          fi
