import plotly.express as px
import streamlit as st

from churnlab import models
from views.common import ACCENT, run_training

ss = st.session_state
st.title("4 · Train")

if not ss.prep_confirmed:
    st.info("Using default preprocessing. Visit **Prepare & rank** to customise it.", icon=":material/tune:")

available = list(models.REGISTRY)
with st.form("train"):
    chosen = st.multiselect("Models", available, default=models.DEFAULT_MODELS,
                            format_func=lambda k: models.REGISTRY[k].label)
    with st.expander("What each model does"):
        for k in available:
            spec = models.REGISTRY[k]
            st.markdown(f"**{spec.label}**: {spec.blurb}")
    c1, c2, c3 = st.columns(3)
    folds = c1.slider("Cross-validation folds", 3, 10, 5)
    test_size = c2.slider("Holdout test size", 0.1, 0.4, 0.2, 0.05)
    balanced = c3.toggle("Re-weight classes", value=False,
                         help="Gives the minority class more weight. Raises recall and lowers precision at a 0.5 "
                              "threshold. Usually unnecessary because you pick the threshold later.")
    tunable = [k for k in available if models.REGISTRY[k].search]
    tune = st.multiselect("Auto-tune hyperparameters (randomised search, maximising AUC)", tunable,
                          format_func=lambda k: models.REGISTRY[k].label)
    n_iter = st.slider("Tuning budget (candidates per model)", 5, 60, 20, disabled=False)
    submitted = st.form_submit_button("Train", type="primary", icon=":material/play_arrow:")

if submitted:
    if not chosen:
        st.error("Select at least one model.")
        st.stop()
    ss.prep_confirmed = True
    cfg = models.TrainConfig(models=chosen, folds=folds, test_size=test_size, balanced=balanced,
                             tune=[t for t in tune if t in chosen], tune_iter=n_iter)
    run_training(cfg)
    st.rerun()

exp = ss.exp
if exp is None:
    st.stop()

st.subheader("Test & score")
st.caption(f"Trained on {len(exp.y_train):,} rows with {exp.train_cfg.folds}-fold stratified CV; "
           f"holdout of {len(exp.y_test):,} rows never seen during training or tuning. "
           "Metrics that need a yes/no decision use a 0.5 threshold. Choose a better one on **Decide**.")
board = exp.leaderboard
view_cols = ["model", "CV AUC", "Test AUC", "Test PR-AUC", "Test accuracy", "Test F1", "Test precision",
             "Test recall", "Test MCC", "Test log loss", "Test Brier", "fit time (s)"]
st.dataframe(board[view_cols], hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.3f") for c in view_cols[1:]})
with st.expander("Cross-validation metrics (all)"):
    st.dataframe(board[[c for c in board.columns if c.startswith("CV ") or c == "model"]], hide_index=True,
                 width="stretch")

fig = px.bar(board.iloc[::-1], x="Test AUC", y="model", orientation="h", color_discrete_sequence=[ACCENT],
             text=board.iloc[::-1]["Test AUC"].map("{:.3f}".format))
fig.update_layout(height=60 + 40 * len(board), xaxis_range=[0.45, 1], yaxis_title=None)
st.plotly_chart(fig, width="stretch")

best = board.iloc[0]
st.success(f"Best by cross-validated AUC: **{best['model']}** (CV {best['CV AUC']:.3f}, test {best['Test AUC']:.3f}). "
           "It is set as the active model. Change it on any later page.")
st.page_link("views/evaluate.py", label="Next: evaluate", icon=":material/arrow_forward:")

with st.expander("Hyperparameters used"):
    for k, m in exp.models.items():
        st.markdown(f"**{m.label}**")
        st.json({k2: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                 for k2, v in m.params.items()}, expanded=False)
