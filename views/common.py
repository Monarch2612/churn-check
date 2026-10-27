import pandas as pd
import streamlit as st

from churnlab import data, evaluate, explain, models
from churnlab.evaluate import Economics
from churnlab.features import PrepConfig

POS_COLOR = "#D85A30"
NEG_COLOR = "#1D9E75"
ACCENT = "#534AB7"
PALETTE = ["#534AB7", "#1D9E75", "#D85A30", "#185FA5", "#BA7517", "#993556", "#3B6D11", "#888780", "#0F6E56", "#A32D2D"]

STEPS = [
    ("data", "Data loaded"),
    ("prep", "Features prepared"),
    ("exp", "Models trained"),
    ("decided", "Threshold chosen"),
]


@st.cache_data(show_spinner="Loading data…")
def load_default() -> pd.DataFrame:
    return data.load()


def init_state():
    ss = st.session_state
    if "df" not in ss:
        ss.df = load_default()
        ss.source_name = data.DEFAULT_DATA.name
        ss.schema = data.infer_schema(ss.df)
    ss.setdefault("prep", PrepConfig())
    ss.setdefault("prep_confirmed", False)
    ss.setdefault("exp", None)
    ss.setdefault("active", None)
    ss.setdefault("threshold", 0.5)
    ss.setdefault("decided", False)
    ss.setdefault("econ", Economics())
    ss.setdefault("explainers", {})


def reset_downstream(from_step: str):
    ss = st.session_state
    order = ["data", "prep", "exp", "decided"]
    for step in order[order.index(from_step) + 1:]:
        if step == "prep":
            ss.prep_confirmed = False
        elif step == "exp":
            ss.exp, ss.active, ss.explainers = None, None, {}
        elif step == "decided":
            ss.decided, ss.threshold = False, 0.5


def step_done(step: str) -> bool:
    ss = st.session_state
    return {"data": "df" in ss, "prep": ss.get("prep_confirmed", False),
            "exp": ss.get("exp") is not None, "decided": ss.get("decided", False)}[step]


def sidebar_status():
    with st.sidebar:
        st.caption("Pipeline progress")
        for step, label in STEPS:
            icon = ":material/check_circle:" if step_done(step) else ":material/radio_button_unchecked:"
            st.markdown(f"{icon} {label}")
        ss = st.session_state
        if ss.get("exp") is not None and ss.active:
            st.divider()
            st.caption("Active model")
            st.markdown(f"**{ss.exp.models[ss.active].label}**  \nthreshold {ss.threshold:.2f}")


def require_experiment():
    if st.session_state.get("exp") is None:
        st.info("Train at least one model first.", icon=":material/model_training:")
        st.page_link("views/train.py", label="Go to Train", icon=":material/arrow_forward:")
        st.stop()
    return st.session_state.exp


def model_picker(exp, key: str = "active_picker", label: str = "Model") -> str:
    ss = st.session_state
    options = list(exp.leaderboard.index)
    current = ss.active if ss.active in options else options[0]
    choice = st.selectbox(label, options, index=options.index(current),
                          format_func=lambda k: exp.models[k].label, key=key)
    if choice != ss.active:
        ss.active = choice
    return choice


def get_explainer(exp, key: str) -> explain.Explainer:
    cache = st.session_state.explainers
    if key not in cache:
        with st.spinner("Building explainer…"):
            cache[key] = explain.Explainer(exp.models[key].pipeline, exp.X_train)
    return cache[key]


def run_training(cfg: models.TrainConfig):
    ss = st.session_state
    X, y = data.split_xy(ss.df, ss.schema)
    bar = st.progress(0.0, text="Starting…")
    exp = models.run_experiment(X, y, ss.schema, ss.prep, cfg, progress=lambda f, t: bar.progress(f, text=t))
    bar.empty()
    reset_downstream("prep")
    ss.exp = exp
    ss.active = exp.best_key()
    return exp


def optimal_threshold(exp, key: str, econ: Economics) -> tuple[float, pd.DataFrame]:
    value = evaluate.customer_value(exp.X_test, exp.schema.value_col, econ)
    curve = evaluate.profit_curve(exp.y_test, exp.models[key].test_proba, value, econ)
    return float(curve.loc[curve["profit"].idxmax(), "threshold"]), curve


def pct(x: float) -> str:
    return f"{x:.1%}"


def money(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"
