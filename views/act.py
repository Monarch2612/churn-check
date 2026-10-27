import io
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churnlab import data, evaluate, explain, models
from views.common import NEG_COLOR, POS_COLOR, get_explainer, model_picker, money, require_experiment

ss = st.session_state
exp = require_experiment()
schema = exp.schema
st.title("8 · Act")

key = model_picker(exp)
if key == "baseline":
    st.info("Pick a real model. The baseline gives every customer the same score.")
    st.stop()
pipe = exp.models[key].pipeline
t = ss.threshold
ex = get_explainer(exp, key)
st.caption(f"Flagging customers with risk ≥ **{t:.2f}**"
           + ("" if ss.decided else ". Pick a profit-optimal threshold on **Decide**."))

ACTIONABLE = ["Contract", "PaymentMethod", "PaperlessBilling", "OnlineSecurity", "TechSupport", "OnlineBackup",
              "DeviceProtection", "StreamingTV", "StreamingMovies"]


def lever_options(col: str, values) -> list[str]:
    # "No internet service" style values describe a missing base product, not an add-on the business can offer.
    return sorted(v for v in values if not (v.startswith("No ") and v.endswith(" service")))


def tier(p: float) -> str:
    return "High" if p >= t else ("Medium" if p >= t / 2 else "Low")


def top_reasons(contrib: pd.DataFrame, shown: pd.DataFrame, n: int = 3) -> list[str]:
    out = []
    for i in range(len(contrib)):
        row = contrib.iloc[i]
        pos = row[row > 0].sort_values(ascending=False).head(n)
        out.append("; ".join(f"{f} = {shown.iloc[i][f]:.0f}" if isinstance(shown.iloc[i][f], (int, float, np.floating))
                             else f"{f} = {shown.iloc[i][f]}" for f in pos.index))
    return out


tabs = st.tabs(["Customer simulator", "Retention priority list", "Score a new file", "Export model"])

with tabs[0]:
    X_all = pd.concat([exp.X_train, exp.X_test])
    st.markdown("Describe a customer, or adjust an existing one, to get an instant risk score, the reasons "
                "behind it, and the changes most likely to keep them.")
    seed = st.selectbox("Start from", ["Typical customer", "Random high-risk test customer"])
    if seed == "Typical customer":
        base = {c: float(X_all[c].median()) for c in schema.numeric}
        base |= {c: X_all[c].mode().iloc[0] for c in schema.categorical}
    else:
        risky = exp.X_test[exp.models[key].test_proba >= max(t, 0.5)]
        base = (risky if len(risky) else exp.X_test).sample(1, random_state=7).iloc[0].to_dict()
    with st.form("customer"):
        cols = st.columns(4)
        values = {}
        for i, c in enumerate(schema.features):
            with cols[i % 4]:
                if c in schema.numeric:
                    lo, hi = float(X_all[c].min()), float(X_all[c].max())
                    v = base.get(c)
                    v = float(np.clip(v if pd.notna(v) else X_all[c].median(), lo, hi))
                    values[c] = st.number_input(c, lo, hi, v)
                else:
                    opts = sorted(X_all[c].dropna().astype(str).unique())
                    cur = str(base.get(c))
                    values[c] = st.selectbox(c, opts, index=opts.index(cur) if cur in opts else 0)
        st.form_submit_button("Score customer", type="primary")
    row = pd.DataFrame([values])[schema.features]
    p = float(pipe.predict_proba(row)[:, 1][0])

    a, b = st.columns([1, 2])
    with a:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=100 * p, number={"suffix": "%"},
            title={"text": f"{tier(p)} risk"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": POS_COLOR if p >= t else NEG_COLOR},
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 100 * t}}))
        gauge.update_layout(height=300, margin=dict(t=60, b=10))
        st.plotly_chart(gauge, width="stretch")
    with b:
        local = ex.local(row).head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=local["contribution"], y=local["feature"] + " = " + local["value"], orientation="h",
                               marker_color=[POS_COLOR if v > 0 else NEG_COLOR for v in local["contribution"]]))
        fig.add_vline(x=0, line_color="gray")
        fig.update_layout(height=320, title="Why", margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**What-if levers**: single changes that move this customer's risk the most.")
    default_levers = [c for c in ACTIONABLE if c in schema.categorical] or schema.categorical[:6]
    lever_cols = st.multiselect("Features the business can change", schema.categorical, default=default_levers)
    levers = {c: lever_options(c, X_all[c].dropna().astype(str).unique()) for c in lever_cols}
    wi = explain.what_if(pipe, row, levers)
    if len(wi):
        wi["new risk"] = wi["new risk"].map("{:.1%}".format)
        wi["risk change"] = wi["risk change"].map("{:+.1%}".format)
        st.dataframe(wi.head(8), hide_index=True, width="stretch")
        st.caption("These are model associations, not proven causal effects. Test interventions with a control group.")

with tabs[1]:
    st.markdown("Every customer is scored **out-of-fold**: by a model that never saw them during training. "
                "This gives honest risk estimates for the whole base.")
    oof_key = ("oof", key)
    X_full, y_full = data.split_xy(ss.df, schema)
    if oof_key not in ss.explainers:
        with st.spinner("Scoring all customers with cross-validation…"):
            ss.explainers[oof_key] = models.out_of_fold_scores(exp, key, X_full, y_full)
    scores = ss.explainers[oof_key]
    active_only = st.toggle(f"Only customers who have not already churned ({schema.target} ≠ {schema.positive})",
                            value=True)
    base = ss.df.loc[X_full.index]
    value = evaluate.customer_value(X_full, schema.value_col, ss.econ)
    table = pd.DataFrame({"risk": scores, "tier": [tier(s) for s in scores],
                          "value at risk": scores * value, schema.target: base[schema.target].astype(str).to_numpy()},
                         index=X_full.index)
    for c in schema.id_cols[:1]:
        table.insert(0, c, base[c].to_numpy())
    if active_only:
        table = table[table[schema.target] != schema.positive]
    table = table.sort_values("value at risk", ascending=False)
    flagged = table[table["risk"] >= t]

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers flagged", f"{len(flagged):,}", f"{len(flagged) / max(len(table), 1):.0%} of list",
              delta_color="off")
    c2.metric("Revenue at risk (flagged)", money(flagged["value at risk"].sum()))
    c3.metric("Average risk (flagged)", f"{flagged['risk'].mean():.1%}" if len(flagged) else "–")

    n_explain = st.slider("Add top reasons for the first N customers", 0, 300, 100, 25,
                          help="Model-agnostic explainers (kNN, SVM, neural net, naive Bayes) are slower.")
    show = table.head(max(n_explain, 500)).copy()
    if n_explain:
        rkey = ("reasons", key, n_explain, active_only)
        if rkey not in ss.explainers:
            with st.spinner("Explaining top customers…"):
                sub = X_full.loc[show.index[:n_explain]]
                contrib = ex.contributions(sub)
                ss.explainers[rkey] = top_reasons(contrib, explain.display_frame(pipe, sub))
        reasons = ss.explainers[rkey]
        show["top reasons"] = reasons + [""] * (len(show) - len(reasons))
    st.dataframe(show, width="stretch", height=480,
                 column_config={"risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                "value at risk": st.column_config.NumberColumn(format="$%.0f")})
    st.download_button("Download full list (CSV)", table.to_csv(index=False).encode(), "retention_priority_list.csv",
                       "text/csv", icon=":material/download:")

with tabs[2]:
    st.markdown("Upload new customers with the same columns as the training data. Scores come from the selected "
                "model **refitted on all labelled data**.")
    upload = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"], key="score_upload")
    if upload is not None:
        new = data.load(upload)
        missing = [c for c in schema.features if c not in new.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            full_key = ("full", key)
            if full_key not in ss.explainers:
                with st.spinner("Refitting on all data…"):
                    ss.explainers[full_key] = models.refit_full(exp, key)
            X_new = new[schema.features].copy()
            for c in schema.categorical:
                X_new[c] = X_new[c].astype(str).where(X_new[c].notna(), np.nan)
            risk = ss.explainers[full_key].predict_proba(X_new)[:, 1]
            out = new.assign(churn_risk=risk, risk_tier=[tier(r) for r in risk]).sort_values("churn_risk", ascending=False)
            st.dataframe(out, width="stretch", height=420,
                         column_config={"churn_risk": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                                                      max_value=1)})
            st.download_button("Download scored file", out.to_csv(index=False).encode(), "scored_customers.csv",
                               "text/csv", icon=":material/download:")

with tabs[3]:
    st.markdown("Download the selected model, refitted on all labelled data with its full preprocessing "
                "pipeline. It is a single scikit-learn object.")
    full_key = ("full", key)
    if st.button("Prepare download", icon=":material/package_2:"):
        if full_key not in ss.explainers:
            with st.spinner("Refitting on all data…"):
                ss.explainers[full_key] = models.refit_full(exp, key)
    if full_key in ss.explainers:
        meta = {"model": exp.models[key].label, "target": schema.target, "positive_class": schema.positive,
                "features": schema.features, "numeric": schema.numeric, "categorical": schema.categorical,
                "threshold": t, "test_auc": float(exp.leaderboard.loc[key, "Test AUC"]),
                "cv_auc": float(exp.leaderboard.loc[key, "CV AUC"])}
        buf = io.BytesIO()
        joblib.dump({"pipeline": ss.explainers[full_key], "meta": meta}, buf)
        st.download_button("Download model (.joblib)", buf.getvalue(), f"churnlab_{key}.joblib",
                           icon=":material/download:")
        st.json(meta, expanded=False)
        st.code(f"""import joblib, pandas as pd
bundle = joblib.load("churnlab_{key}.joblib")
pipe, meta = bundle["pipeline"], bundle["meta"]
df = pd.read_csv("new_customers.csv")
df["churn_risk"] = pipe.predict_proba(df[meta["features"]])[:, 1]
df["flag"] = df["churn_risk"] >= meta["threshold"]""", language="python")
        st.caption("To load the model elsewhere, the `churnlab` package must be importable "
                   "(it defines the engineered-feature step).")
