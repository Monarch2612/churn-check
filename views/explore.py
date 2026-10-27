import pandas as pd
import plotly.express as px
import streamlit as st

from churnlab import explore
from views.common import NEG_COLOR, POS_COLOR

ss = st.session_state
df, schema = ss.df, ss.schema
st.title("2 · Explore")

target_label = f"{schema.target}"
colors = {schema.positive: POS_COLOR}
for v in df[schema.target].dropna().astype(str).unique():
    colors.setdefault(v, NEG_COLOR)
plot_df = df.assign(**{schema.target: df[schema.target].astype(str)})

tabs = st.tabs(["Insights", "Feature statistics", "Distributions", "Segments", "Scatter", "Box plot", "Correlation"])

with tabs[0]:
    st.markdown("Automatically generated findings, ranked by how strongly each segment differs from the average.")
    for note in explore.insights(df, schema):
        st.markdown(f"- {note}")

with tabs[1]:
    stats = explore.feature_statistics(df, schema)
    st.dataframe(stats, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f")
                                for c in ["mean", "median", "std", "min", "max", "missing %"]})

with tabs[2]:
    col = st.selectbox("Feature", schema.features, key="dist_col")
    if col in schema.numeric:
        norm = st.toggle("Compare shapes (normalise each class)", value=True)
        fig = px.histogram(plot_df, x=col, color=schema.target, barmode="overlay", nbins=40, opacity=0.65,
                           histnorm="probability density" if norm else None, color_discrete_map=colors,
                           marginal="box")
    else:
        rates = explore.rate_by_category(df, schema, col)
        fig = px.histogram(plot_df, x=col, color=schema.target, barmode="group", color_discrete_map=colors,
                           category_orders={col: list(rates[col])})
    fig.update_layout(height=430, legend_title_text=target_label)
    st.plotly_chart(fig, width="stretch")
    if col in schema.categorical:
        st.dataframe(rates, hide_index=True, width="stretch",
                     column_config={"rate": st.column_config.ProgressColumn(f"{schema.positive} rate", format="percent",
                                                                            min_value=0, max_value=1),
                                    "lift": st.column_config.NumberColumn(format="%.2f×")})

with tabs[3]:
    st.markdown(f"**{schema.positive}** rate for every category of every categorical feature. "
                "The dashed line is the overall average.")
    base = (df[schema.target].astype(str) == schema.positive).mean()
    frames = []
    for c in schema.categorical:
        r = explore.rate_by_category(df, schema, c).rename(columns={c: "value"})
        r["feature"] = c
        frames.append(r)
    seg = pd.concat(frames)
    seg["segment"] = seg["feature"] + " = " + seg["value"].astype(str)
    seg = seg[seg["customers"] >= 30].sort_values("rate")
    fig = px.bar(seg, x="rate", y="segment", orientation="h", color="rate", color_continuous_scale="RdYlGn_r",
                 hover_data=["customers", "lift"])
    fig.add_vline(x=base, line_dash="dash")
    fig.update_layout(height=max(400, 18 * len(seg)), coloraxis_showscale=False, xaxis_tickformat=".0%",
                      yaxis_title=None)
    st.plotly_chart(fig, width="stretch")

with tabs[4]:
    if len(schema.numeric) < 2:
        st.info("Needs at least two numeric features.")
    else:
        a, b = st.columns(2)
        x = a.selectbox("X", schema.numeric, index=0)
        y = b.selectbox("Y", schema.numeric, index=min(1, len(schema.numeric) - 1))
        fig = px.scatter(plot_df.sample(min(len(plot_df), 3000), random_state=0), x=x, y=y, color=schema.target,
                         opacity=0.55, color_discrete_map=colors)
        fig.update_layout(height=480)
        st.plotly_chart(fig, width="stretch")

with tabs[5]:
    if not schema.numeric:
        st.info("No numeric features.")
    else:
        a, b = st.columns(2)
        num = a.selectbox("Numeric feature", schema.numeric, key="box_num")
        group = b.selectbox("Split by", ["(target only)"] + schema.categorical, key="box_group")
        fig = px.box(plot_df, x=None if group == "(target only)" else group, y=num, color=schema.target,
                     color_discrete_map=colors, points=False)
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch")

with tabs[6]:
    corr = explore.correlation(df, schema)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=460)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Pearson correlation between numeric features and the target (1 = {schema.positive}).")
