import plotly.graph_objects as go
import streamlit as st

from churnlab import evaluate
from churnlab.evaluate import Economics
from views.common import NEG_COLOR, POS_COLOR, ACCENT, model_picker, money, optimal_threshold, require_experiment

ss = st.session_state
exp = require_experiment()
st.title("7 · Decide")
st.markdown("A model is only useful if acting on it makes money. Enter your campaign economics. ChurnLab "
            "simulates the retention campaign on the holdout customers at every threshold and finds the most "
            "profitable one.")

key = model_picker(exp)
econ: Economics = ss.econ
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    cost = c1.number_input("Cost per contacted customer ($)", 0.0, 1000.0, float(econ.offer_cost), 5.0,
                           help="Discount, call-centre time and so on.")
    success = c2.slider("Offer success rate", 0.05, 1.0, float(econ.success_rate), 0.05,
                        help="Share of true churners who stay because of the offer.")
    horizon = c3.slider("Value horizon (months)", 1, 36, int(econ.horizon_months),
                        help="Months of revenue kept per saved customer.")
    default_value = c4.number_input("Monthly value if no column ($)", 0.0, 10000.0, float(econ.default_value), 5.0,
                                    disabled=exp.schema.value_col is not None)
econ = ss.econ = Economics(cost, success, horizon, default_value)
if exp.schema.value_col:
    st.caption(f"Each customer's value = their own `{exp.schema.value_col}` × {horizon} months.")

best_t, curve = optimal_threshold(exp, key, econ)
value = evaluate.customer_value(exp.X_test, exp.schema.value_col, econ)
bench = evaluate.benchmark_profits(exp.y_test, value, econ)
best = curve.loc[curve["profit"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Optimal threshold", f"{best_t:.2f}")
c2.metric("Expected profit (test set)", money(best["profit"]),
          delta=f"{money(best['profit'] - bench['Contact everyone'])} vs contacting everyone")
c3.metric("Customers to contact", f"{int(best['contacted']):,} of {len(exp.y_test):,}")
c4.metric("Share of perfect-foresight profit", f"{best['profit'] / bench['Perfect foresight']:.0%}"
          if bench["Perfect foresight"] > 0 else "n/a")

fig = go.Figure()
fig.add_scatter(x=curve["threshold"], y=curve["profit"], name="Profit", line=dict(color=ACCENT, width=3))
fig.add_scatter(x=curve["threshold"], y=curve["revenue saved"], name="Revenue saved", line=dict(color=NEG_COLOR, dash="dot"))
fig.add_scatter(x=curve["threshold"], y=-curve["campaign cost"], name="Campaign cost", line=dict(color=POS_COLOR, dash="dot"))
fig.add_vline(x=best_t, line_dash="dash", annotation_text=f"best {best_t:.2f}")
fig.add_hline(y=bench["Contact everyone"], line_color="gray", line_dash="dash", annotation_text="contact everyone")
fig.update_layout(height=460, xaxis_title="Threshold (contact if risk ≥ threshold)", yaxis_title="$ on test set",
                  yaxis_tickprefix="$")
st.plotly_chart(fig, width="stretch")

scale = len(ss.df) / len(exp.y_test)
st.markdown(f"Scaled to the full customer base of {len(ss.df):,}, that is roughly **{money(best['profit'] * scale)}** "
            f"per campaign, versus {money(bench['Contact everyone'] * scale)} for a blanket offer."
            .replace("$", r"\$"))

c1, c2 = st.columns(2)
if c1.button(f"Use threshold {best_t:.2f}", type="primary", icon=":material/check:"):
    ss.threshold, ss.decided = best_t, True
    st.rerun()
manual = c2.slider("…or set manually", 0.05, 0.95, float(ss.threshold), 0.01)
if manual != ss.threshold:
    ss.threshold, ss.decided = manual, True
    st.rerun()
if ss.decided:
    st.success(f"Threshold {ss.threshold:.2f} will be used to flag customers on **Act**.")
    st.page_link("views/act.py", label="Next: act", icon=":material/arrow_forward:")
