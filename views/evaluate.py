import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from churnlab import evaluate
from views.common import PALETTE, model_picker, require_experiment

ss = st.session_state
exp = require_experiment()
st.title("5 · Evaluate")
st.caption(f"All curves use the holdout test set ({len(exp.y_test):,} customers).")

keys = list(exp.leaderboard.index)
compare = st.multiselect("Compare models", keys, default=[k for k in keys if k != "baseline"][:4] or keys,
                         format_func=lambda k: exp.models[k].label)
y = exp.y_test

tabs = st.tabs(["ROC", "Precision-recall", "Lift & gains", "Calibration", "Confusion matrix", "Threshold sweep"])

with tabs[0]:
    fig = go.Figure()
    for i, k in enumerate(compare):
        r = evaluate.roc(y, exp.models[k].test_proba)
        auc = exp.leaderboard.loc[k, "Test AUC"]
        fig.add_scatter(x=r["fpr"], y=r["tpr"], name=f"{exp.models[k].label} ({auc:.3f})",
                        line=dict(color=PALETTE[i % len(PALETTE)]))
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="gray"))
    fig.update_layout(height=480, xaxis_title="False positive rate", yaxis_title="True positive rate (recall)")
    st.plotly_chart(fig, width="stretch")

with tabs[1]:
    fig = go.Figure()
    for i, k in enumerate(compare):
        r = evaluate.pr(y, exp.models[k].test_proba)
        fig.add_scatter(x=r["recall"], y=r["precision"], name=exp.models[k].label,
                        line=dict(color=PALETTE[i % len(PALETTE)]))
    fig.add_hline(y=y.mean(), line_dash="dash", annotation_text="base rate")
    fig.update_layout(height=480, xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig, width="stretch")

with tabs[2]:
    st.markdown("If you contact the top *x%* riskiest customers, what share of all churners do you reach?")
    frames = []
    for k in compare:
        g = evaluate.gains(y, exp.models[k].test_proba)
        g["model"] = exp.models[k].label
        frames.append(g)
    g = pd.concat(frames)
    a, b = st.columns(2)
    fig = px.line(g, x="% contacted", y="% of churners caught", color="model", markers=True,
                  color_discrete_sequence=PALETTE)
    fig.add_scatter(x=[0, 100], y=[0, 100], name="Random", line=dict(dash="dash", color="gray"))
    fig.update_layout(height=420, title="Cumulative gains")
    a.plotly_chart(fig, width="stretch")
    fig = px.line(g, x="% contacted", y="lift", color="model", markers=True, color_discrete_sequence=PALETTE)
    fig.add_hline(y=1, line_dash="dash")
    fig.update_layout(height=420, title="Lift")
    b.plotly_chart(fig, width="stretch")

with tabs[3]:
    st.markdown("A well-calibrated model's predicted probabilities match observed frequencies "
                "(points on the diagonal). This matters when you use the scores as real probabilities, "
                "e.g. for revenue-at-risk.")
    fig = go.Figure()
    for i, k in enumerate(compare):
        c = evaluate.calibration(y, exp.models[k].test_proba)
        fig.add_scatter(x=c["predicted"], y=c["observed"], mode="lines+markers", name=exp.models[k].label,
                        line=dict(color=PALETTE[i % len(PALETTE)]))
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Perfect", line=dict(dash="dash", color="gray"))
    fig.update_layout(height=480, xaxis_title="Mean predicted probability", yaxis_title="Observed frequency")
    st.plotly_chart(fig, width="stretch")

with tabs[4]:
    a, b = st.columns([1, 2])
    with a:
        key = model_picker(exp, key="cm_model")
        t = st.slider("Threshold", 0.05, 0.95, float(ss.threshold), 0.01, key="cm_t")
        proba = exp.models[key].test_proba
        m = evaluate.metric_row(y, proba, threshold=t)
        for name in ["precision", "recall", "F1", "accuracy"]:
            st.metric(name.capitalize(), f"{m[name]:.3f}")
    with b:
        cm = evaluate.confusion(y, proba, t)
        labels = [f"Not {exp.schema.positive}", exp.schema.positive]
        text = [[f"{v:,}<br>{v / cm.sum():.1%}" for v in row] for row in cm]
        fig = go.Figure(go.Heatmap(z=cm, x=[f"Predicted {l}" for l in labels], y=[f"Actual {l}" for l in labels],
                                   text=text, texttemplate="%{text}", colorscale="Purples", showscale=False))
        fig.update_layout(height=420, yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"At {t:.2f}: **{tp}** churners caught, **{fn}** missed, **{fp}** loyal customers "
                   f"contacted unnecessarily.")

with tabs[5]:
    key = model_picker(exp, key="sweep_model")
    table = evaluate.threshold_table(y, exp.models[key].test_proba)
    long = table.melt(id_vars="threshold", value_vars=["precision", "recall", "F1"])
    fig = px.line(long, x="threshold", y="value", color="variable", color_discrete_sequence=PALETTE)
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")
    best = table.loc[table["F1"].idxmax()]
    st.caption(f"F1 peaks at threshold {best['threshold']:.2f} (F1 {best['F1']:.3f}, flags {best['flagged %']:.0f}% "
               "of customers). For a money-based choice, use **Decide**.")
