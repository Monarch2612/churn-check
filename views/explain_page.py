import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from churnlab import explain
from views.common import ACCENT, NEG_COLOR, POS_COLOR, get_explainer, model_picker, require_experiment

ss = st.session_state
exp = require_experiment()
st.title("6 · Explain")

key = model_picker(exp)
if key == "baseline":
    st.info("The majority baseline uses no features, so there is nothing to explain. Pick another model.")
    st.stop()
pipe = exp.models[key].pipeline
ex = get_explainer(exp, key)
unit = "change in log-odds" if ex.units == "log-odds" else "change in probability"

tabs = st.tabs(["Global drivers", "Dependence", "Model internals", "Single customer"])

cache_key = ("global", key)
if cache_key not in ss.explainers:
    with st.spinner("Computing SHAP contributions on test customers…"):
        ss.explainers[cache_key] = ex.global_summary(exp.X_test)
importance, contrib, shown = ss.explainers[cache_key]

with tabs[0]:
    a, b = st.columns(2)
    with a:
        st.markdown("**SHAP importance**: average size of each feature's push on individual predictions.")
        fig = px.bar(importance.iloc[::-1], x="mean |contribution|", y="feature", orientation="h",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(height=max(420, 24 * len(importance)), yaxis_title=None, xaxis_title=f"mean |{unit}|")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Permutation importance**: how much test AUC drops when a feature's values are shuffled.")
        pkey = ("perm", key)
        if pkey not in ss.explainers:
            with st.spinner("Shuffling features…"):
                ss.explainers[pkey] = explain.permutation_table(pipe, exp.X_test, exp.y_test)
        perm = ss.explainers[pkey]
        fig = px.bar(perm.iloc[::-1], x="AUC drop", y="feature", orientation="h", error_x="std",
                     color_discrete_sequence=[NEG_COLOR])
        fig.update_layout(height=max(420, 24 * len(perm)), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Direction of effect**: each dot is a customer. Right of zero pushes toward "
                f"*{exp.schema.positive}*.")
    top = list(importance["feature"][:12])
    rows = []
    for f in top:
        vals = shown[f]
        if pd.api.types.is_numeric_dtype(vals):
            rank = vals.rank(pct=True)
            label = vals.map("{:,.1f}".format)
        else:
            rank = pd.Series(0.5, index=vals.index)
            label = vals.astype(str)
        rows.append(pd.DataFrame({"feature": f, "contribution": contrib[f], "value": label, "relative value": rank}))
    swarm = pd.concat(rows)
    fig = px.scatter(swarm, x="contribution", y="feature", color="relative value", hover_data=["value"],
                     category_orders={"feature": top}, color_continuous_scale=[NEG_COLOR, "#D3D1C7", POS_COLOR],
                     opacity=0.6)
    fig.update_traces(marker=dict(size=6))
    fig.add_vline(x=0, line_color="gray")
    fig.update_layout(height=60 + 34 * len(top), yaxis_title=None, xaxis_title=unit,
                      coloraxis_colorbar=dict(title="value<br>(low→high)"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Colour shows the feature value relative to other customers (numeric features). Categorical features are grey.")

with tabs[1]:
    f = st.selectbox("Feature", list(importance["feature"]))
    dep = pd.DataFrame({"value": shown[f], "contribution": contrib[f]})
    if pd.api.types.is_numeric_dtype(dep["value"]):
        fig = px.scatter(dep, x="value", y="contribution", opacity=0.6, color_discrete_sequence=[ACCENT])
    else:
        fig = px.box(dep, x="value", y="contribution", color_discrete_sequence=[ACCENT], points="all")
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(height=450, xaxis_title=f, yaxis_title=unit)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"How the model's output moves as **{f}** changes, holding everything else as it was.")

with tabs[2]:
    odds = explain.odds_ratios(pipe)
    rules = explain.tree_rules(pipe)
    if odds is not None:
        st.markdown("**Odds ratios.** Above 1 raises the odds of churn, below 1 lowers them. Numeric terms are "
                    "per one standard deviation when standard scaling is on.")
        st.dataframe(odds, hide_index=True, width="stretch",
                     column_config={"coefficient": st.column_config.NumberColumn(format="%.3f"),
                                    "odds ratio": st.column_config.NumberColumn(format="%.2f")})
    elif rules is not None:
        depth = st.slider("Rule depth", 1, 8, 4)
        st.markdown("**Decision rules** (weights = [not churn, churn] training counts per leaf):")
        st.code(explain.tree_rules(pipe, depth), language="text")
    else:
        st.info(f"{exp.models[key].label} has no compact internal form. Use SHAP and permutation importance, "
                "or train a logistic regression or decision tree for a white-box view.")
    st.markdown("**Hyperparameters**")
    st.json({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
             for k, v in exp.models[key].params.items()}, expanded=False)

with tabs[3]:
    proba = exp.models[key].test_proba
    order = pd.Series(proba, index=exp.X_test.index).sort_values(ascending=False)
    pick = st.radio("Pick a test customer", ["Highest risk", "Lowest risk", "Median risk", "By row"], horizontal=True)
    if pick == "Highest risk":
        idx = order.index[0]
    elif pick == "Lowest risk":
        idx = order.index[-1]
    elif pick == "Median risk":
        idx = order.index[len(order) // 2]
    else:
        idx = st.selectbox("Row index", list(exp.X_test.index))
    row = exp.X_test.loc[[idx]]
    p = float(pipe.predict_proba(row)[:, 1][0])
    actual = exp.y_test[list(exp.X_test.index).index(idx)]
    c1, c2 = st.columns(2)
    c1.metric("Predicted risk", f"{p:.1%}")
    c2.metric("Actually churned?", exp.schema.positive if actual else f"not {exp.schema.positive}")
    local = ex.local(row).head(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=local["contribution"], y=local["feature"] + " = " + local["value"], orientation="h",
                           marker_color=[POS_COLOR if v > 0 else NEG_COLOR for v in local["contribution"]]))
    fig.add_vline(x=0, line_color="gray")
    fig.update_layout(height=460, xaxis_title=unit, title="Why this prediction")
    st.plotly_chart(fig, width="stretch")
    st.caption("Red bars push toward churn; green bars push away. Bars are measured from the average customer.")
