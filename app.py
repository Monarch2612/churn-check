import streamlit as st

from views.common import init_state, sidebar_status

st.set_page_config(page_title="ChurnLab", page_icon=":material/insights:", layout="wide")
init_state()

pages = {
    "Start": [st.Page("views/home.py", title="Overview", icon=":material/home:", default=True)],
    "Understand": [
        st.Page("views/data_page.py", title="1 · Data", icon=":material/table:"),
        st.Page("views/explore.py", title="2 · Explore", icon=":material/query_stats:"),
        st.Page("views/prepare.py", title="3 · Prepare & rank", icon=":material/tune:"),
    ],
    "Model": [
        st.Page("views/train.py", title="4 · Train", icon=":material/model_training:"),
        st.Page("views/evaluate.py", title="5 · Evaluate", icon=":material/monitoring:"),
        st.Page("views/explain_page.py", title="6 · Explain", icon=":material/psychology:"),
    ],
    "Act": [
        st.Page("views/decide.py", title="7 · Decide", icon=":material/payments:"),
        st.Page("views/act.py", title="8 · Act", icon=":material/campaign:"),
    ],
}

nav = st.navigation(pages)
sidebar_status()
nav.run()
