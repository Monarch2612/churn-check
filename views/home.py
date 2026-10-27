import streamlit as st

from churnlab import models
from churnlab.features import is_telco
from views.common import optimal_threshold, pct, run_training, step_done

ss = st.session_state

st.title("ChurnLab")
st.markdown("Predict which customers are about to leave, understand **why**, and decide **who to contact** "
            "for the best return on your retention budget.")

schema = ss.schema
y_rate = (ss.df[schema.target].astype(str) == schema.positive).mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(ss.df):,}")
c2.metric("Features", len(schema.features))
c3.metric("Target", f"{schema.target} = {schema.positive}")
c4.metric(f"{schema.target} rate", pct(y_rate))
st.caption(f"Dataset: **{ss.source_name}**")

st.subheader("Workflow")
steps = [
    ("views/data_page.py", "1 · Data", "Load a CSV or Excel file, check column types, pick the target.", "data"),
    ("views/explore.py", "2 · Explore", "Statistics, distributions, segments and auto-generated insights.", "data"),
    ("views/prepare.py", "3 · Prepare & rank", "Rank features by predictive power and configure preprocessing.", "prep"),
    ("views/train.py", "4 · Train", "Train and cross-validate up to 10 models, optionally auto-tuned.", "exp"),
    ("views/evaluate.py", "5 · Evaluate", "ROC, precision-recall, confusion matrix, calibration and lift.", "exp"),
    ("views/explain_page.py", "6 · Explain", "SHAP drivers, permutation importance, odds ratios, tree rules.", "exp"),
    ("views/decide.py", "7 · Decide", "Turn scores into money: find the profit-maximising threshold.", "decided"),
    ("views/act.py", "8 · Act", "Score customers, run what-ifs, export the retention list and model.", "decided"),
]
cols = st.columns(4)
for i, (page, title, text, step) in enumerate(steps):
    with cols[i % 4].container(border=True, height=160):
        done = step_done(step)
        st.page_link(page, label=title, icon=":material/check_circle:" if done else ":material/arrow_forward:")
        st.caption(text)

st.subheader("Autopilot")
st.markdown("Don't want to click through? Autopilot accepts default preprocessing"
            + (" (with telecom-specific engineered features)" if is_telco(schema) else "")
            + ", trains the core model set with 5-fold cross-validation, selects the best model by AUC and "
              "sets the threshold that maximises expected campaign profit.")
tune = st.toggle("Also auto-tune the two strongest model families (slower, about 1 to 2 minutes)", value=False)
if st.button("Run autopilot", type="primary", icon=":material/rocket_launch:"):
    ss.prep_confirmed = True
    cfg = models.TrainConfig(tune=[k for k in ("hgb", "rf") if k in models.REGISTRY] if tune else [])
    exp = run_training(cfg)
    ss.threshold, _ = optimal_threshold(exp, ss.active, ss.econ)
    ss.decided = True
    best = exp.leaderboard.iloc[0]
    ss.autopilot_msg = (f"Best model: **{best['model']}** with cross-validated AUC {best['CV AUC']:.3f} "
                        f"and test AUC {best['Test AUC']:.3f}. Profit-optimal threshold: {ss.threshold:.2f}.")
    st.rerun()
if ss.get("autopilot_msg") and ss.exp is not None:
    st.success(ss.autopilot_msg)
    st.page_link("views/act.py", label="See the retention list", icon=":material/campaign:")

if not models.HAS_XGB:
    st.caption("XGBoost is unavailable because the OpenMP runtime is missing. Install it with "
               "`brew install libomp` and restart to enable it. All other models work without it.")
