import streamlit as st

from churnlab import data
from views.common import load_default, reset_downstream

ss = st.session_state
st.title("1 · Data")

with st.container(border=True):
    st.markdown("**Source**")
    left, right = st.columns([1, 2])
    if left.button("Use bundled Telco dataset", icon=":material/dataset:"):
        ss.df, ss.source_name = load_default(), data.DEFAULT_DATA.name
        ss.schema = data.infer_schema(ss.df)
        reset_downstream("data")
        st.rerun()
    upload = right.file_uploader("…or upload your own CSV / Excel", type=["csv", "xlsx", "xls"])
    if upload is not None and upload.file_id != ss.get("last_upload"):
        ss.last_upload = upload.file_id
        try:
            df = data.load(upload)
        except Exception as e:
            st.error(f"Could not read that file: {e}")
            st.stop()
        ss.df, ss.source_name = df, upload.name
        ss.schema = data.infer_schema(df)
        reset_downstream("data")
        st.rerun()

df, schema = ss.df, ss.schema
st.caption(f"**{ss.source_name}**: {len(df):,} rows × {df.shape[1]} columns after cleaning "
           "(whitespace trimmed, numeric text converted, blanks treated as missing, duplicates removed).")

st.subheader("Roles")
c1, c2, c3 = st.columns(3)
binary_like = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 20]
target = c1.selectbox("Target (what to predict)", binary_like,
                      index=binary_like.index(schema.target) if schema.target in binary_like else 0)
values = sorted(df[target].dropna().astype(str).unique())
default_pos = schema.positive if target == schema.target else data.guess_positive(df[target])
positive = c2.selectbox("Positive class (the event)", values,
                        index=values.index(default_pos) if default_pos in values else 0)
candidates = [c for c in df.columns if c != target]
ignored = c3.multiselect("Ignore columns (IDs, leakage)", candidates,
                         default=[c for c in schema.id_cols if c in candidates])

if (target, positive, sorted(ignored)) != (schema.target, schema.positive, sorted(schema.id_cols)):
    new = data.infer_schema(df.drop(columns=ignored), target, positive)
    new.id_cols = ignored
    ss.schema = new
    reset_downstream("data")
    st.rerun()

num_choices = [c for c in schema.numeric]
value_col = st.selectbox("Monthly customer value column (used for revenue-at-risk)", ["(none)"] + num_choices,
                         index=(num_choices.index(schema.value_col) + 1) if schema.value_col else 0)
schema.value_col = None if value_col == "(none)" else value_col

st.subheader("Column profile")
profile = data.column_profile(df, schema)
st.dataframe(profile, hide_index=True, width="stretch",
             column_config={"missing %": st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=100)})

if len(df[target].dropna().unique()) > 2:
    st.info(f"`{target}` has more than two values. ChurnLab models **{positive}** versus everything else.")

st.subheader("Data table")
query = st.text_input("Filter rows (pandas query, e.g. `tenure < 6 and Contract == 'Month-to-month'`)")
view = df
if query:
    try:
        view = df.query(query)
    except Exception as e:
        st.warning(f"Invalid filter: {e}")
st.dataframe(view, width="stretch", height=380)
st.caption(f"Showing {len(view):,} rows")
