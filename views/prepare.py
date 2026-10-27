import plotly.express as px
import streamlit as st

from churnlab import explore
from churnlab.features import ENGINEERED, PrepConfig, is_telco
from views.common import ACCENT, reset_downstream

ss = st.session_state
df, schema = ss.df, ss.schema
st.title("3 · Prepare & rank")

key = (ss.source_name, schema.target, schema.positive, tuple(schema.features))
if ss.get("ranking_key") != key:
    with st.spinner("Scoring every feature…"):
        ss.ranking = explore.rank_features(df, schema)
    ss.ranking_key = key
ranking = ss.ranking

st.subheader("Feature ranking")
st.markdown("Each feature is scored **on its own** against the target. **Info gain** is mutual information "
            "(how much uncertainty the feature removes), **univariate AUC** is how well that feature alone separates "
            "classes (0.5 = useless), and the **p-value** comes from a chi-square test (categorical) or ANOVA (numeric).")
left, right = st.columns([3, 2])
with left:
    fig = px.bar(ranking.iloc[::-1], x="info gain (MI)", y="feature", orientation="h",
                 color_discrete_sequence=[ACCENT], hover_data=["univariate AUC", "p-value"])
    fig.update_layout(height=max(380, 24 * len(ranking)), yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
with right:
    st.dataframe(ranking[["feature", "info gain (MI)", "univariate AUC", "p-value"]], width="stretch",
                 height=max(380, 24 * len(ranking)),
                 column_config={"info gain (MI)": st.column_config.NumberColumn(format="%.4f"),
                                "univariate AUC": st.column_config.NumberColumn(format="%.3f"),
                                "p-value": st.column_config.NumberColumn(format="%.1e")})

weak = list(ranking.loc[ranking["p-value"] > 0.05, "feature"])

st.subheader("Preprocessing")
prep: PrepConfig = ss.prep
with st.form("prep"):
    c1, c2 = st.columns(2)
    scaler = c1.selectbox("Numeric scaling", ["standard", "robust", "minmax", "none"],
                          index=["standard", "robust", "minmax", "none"].index(prep.scaler),
                          help="Standard = z-score. Robust uses median/IQR (outlier-resistant). Trees ignore scaling.")
    impute = c2.selectbox("Numeric missing values", ["median", "mean"], index=["median", "mean"].index(prep.num_impute))
    telco = is_telco(schema)
    engineer = st.toggle("Add engineered telecom features", value=prep.engineer and telco, disabled=not telco,
                         help="Adds " + ", ".join(f"`{f}`" for f in ENGINEERED) + ". Computed per row, so no leakage.")
    drop = st.multiselect("Drop features", schema.features, default=[d for d in prep.drop if d in schema.features],
                          help="Statistically weak features (p > 0.05): " + (", ".join(weak) or "none"))
    st.caption("Categorical features are always one-hot encoded with unseen categories ignored, and missing "
               "categories get their own level. Every step is fitted inside cross-validation folds, so nothing leaks.")
    if st.form_submit_button("Apply preprocessing", type="primary"):
        ss.prep = PrepConfig(scaler=scaler, num_impute=impute, engineer=engineer, drop=tuple(drop))
        reset_downstream("prep")
        ss.prep_confirmed = True
        st.rerun()

if ss.prep_confirmed:
    st.success("Preprocessing applied.", icon=":material/check:")
    st.page_link("views/train.py", label="Next: train models", icon=":material/arrow_forward:")
if weak:
    st.caption(f"Tip: {', '.join(weak)} show no significant relationship with {schema.target} on their own. "
               "You can try dropping them, though they may still help in combination.")
