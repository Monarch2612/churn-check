from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "Telco-Customer-Churn.csv"

TARGET_HINTS = ("churn", "exited", "attrition", "left", "target", "label", "class")
VALUE_HINTS = ("monthlycharges", "monthly_charges", "monthly", "arpu", "revenue", "charges")


@dataclass
class Schema:
    target: str
    positive: str
    id_cols: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    value_col: str | None = None

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical


def load(source=None) -> pd.DataFrame:
    if source is None:
        source = DEFAULT_DATA
    name = getattr(source, "name", str(source)).lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(source)
    else:
        df = pd.read_csv(source)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype("string").str.strip()
            s = s.mask(s == "", pd.NA)
            converted = pd.to_numeric(s, errors="coerce")
            non_null = s.notna().sum()
            if non_null and converted.notna().sum() / non_null >= 0.95:
                df[col] = converted.astype(float)
            else:
                df[col] = s.astype(object).where(s.notna(), np.nan)
    return df.drop_duplicates()


def guess_target(df: pd.DataFrame) -> str:
    for hint in TARGET_HINTS:
        for col in df.columns:
            if col.lower() == hint and df[col].nunique(dropna=True) >= 2:
                return col
    for col in reversed(df.columns):
        if df[col].nunique(dropna=True) == 2:
            return col
    return df.columns[-1]


def guess_positive(series: pd.Series) -> str:
    values = [str(v) for v in series.dropna().unique()]
    for v in values:
        if v.lower() in ("yes", "1", "true", "churn", "churned", "exited", "y"):
            return v
    counts = series.astype(str).value_counts()
    return counts.index[-1]


def infer_schema(df: pd.DataFrame, target: str | None = None, positive: str | None = None) -> Schema:
    target = target or guess_target(df)
    positive = positive or guess_positive(df[target])
    id_cols, numeric, categorical = [], [], []
    n = len(df)
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        nunique = s.nunique(dropna=True)
        if nunique <= 1:
            continue
        is_num = pd.api.types.is_numeric_dtype(s)
        if not is_num and nunique > 0.9 * n:
            id_cols.append(col)
        elif is_num and nunique <= 2:
            categorical.append(col)
        elif is_num:
            numeric.append(col)
        elif nunique <= 50:
            categorical.append(col)
        else:
            id_cols.append(col)
    value_col = next((c for c in numeric if c.lower().replace(" ", "") in VALUE_HINTS), None)
    return Schema(target, str(positive), id_cols, numeric, categorical, value_col)


def split_xy(df: pd.DataFrame, schema: Schema) -> tuple[pd.DataFrame, np.ndarray]:
    data = df.dropna(subset=[schema.target])
    X = data[schema.features].copy()
    for col in schema.categorical:
        X[col] = X[col].astype(str).where(X[col].notna(), np.nan)
    y = (data[schema.target].astype(str) == schema.positive).astype(int).to_numpy()
    return X, y


def column_profile(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    rows = []
    roles = {schema.target: "target", **{c: "id" for c in schema.id_cols},
             **{c: "numeric" for c in schema.numeric}, **{c: "categorical" for c in schema.categorical}}
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "role": roles.get(col, "ignored"),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing %": round(100 * s.isna().mean(), 2),
            "unique": int(s.nunique(dropna=True)),
            "example": str(s.dropna().iloc[0]) if s.notna().any() else "",
        })
    return pd.DataFrame(rows)
