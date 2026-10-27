# ChurnLab

A churn-prediction workbench (TelcoGuard prototype) with an Orange-style workflow, plus explainability and campaign-profit optimisation.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/streamlit run app.py
```

Open http://localhost:8501 and click **Run autopilot** on the Overview page, or work through steps 1–8 in the sidebar.

XGBoost needs the OpenMP runtime. It is present on Streamlit Community Cloud and most Linux machines; on macOS run `brew install libomp`. Without it, the other 9 models still work.

## Deploy (Streamlit Community Cloud)

Push this repo to GitHub, then at share.streamlit.io choose **Create app**, pick the repo and branch `main`, set the main file to `app.py`, and under **Advanced settings** choose Python 3.12.

## Workflow

| Step | What it does | Orange equivalent |
|---|---|---|
| 1 Data | Load the bundled CSV or upload CSV/Excel; auto-cleaning, type detection, target and ID selection, row filter | File, Data Table, Select Columns |
| 2 Explore | Auto insights, feature statistics, distributions, segment churn rates, scatter, box plot, correlation | Feature Statistics, Distributions, Scatter Plot, Box Plot |
| 3 Prepare & rank | Mutual information, univariate AUC, chi²/ANOVA ranking; scaling, imputation, engineered telecom features | Rank, Preprocess |
| 4 Train | 10 models, stratified k-fold CV and an untouched holdout, class re-weighting, randomised hyperparameter search | Test & Score, learner widgets |
| 5 Evaluate | ROC, precision-recall, lift/gains, calibration, confusion matrix, threshold sweep | ROC Analysis, Confusion Matrix, Lift Curve, Calibration Plot |
| 6 Explain | SHAP drivers (global and per customer), permutation importance, dependence, odds ratios, tree rules | Explain Model, Nomogram, Tree Viewer |
| 7 Decide | Campaign economics, profit curve, profit-optimal threshold | — |
| 8 Act | Customer simulator with what-if levers, out-of-fold retention list with reasons, batch scoring, model export | Predictions (partial) |

## Layout

- `churnlab/`: ML engine with no UI code (data, explore, features, models, evaluate, explain)
- `views/`: Streamlit pages
- `tests/`: engine tests for every model plus an end-to-end app test (`.venv/bin/python -m pytest tests`)
