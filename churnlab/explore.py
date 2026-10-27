import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from .data import Schema, split_xy


def feature_statistics(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    rows = []
    for col in schema.numeric:
        s = df[col]
        rows.append({"feature": col, "type": "numeric", "mean": s.mean(), "median": s.median(),
                     "std": s.std(), "min": s.min(), "max": s.max(),
                     "missing %": 100 * s.isna().mean(), "mode": None, "distinct": s.nunique()})
    for col in schema.categorical:
        s = df[col].astype(str)
        rows.append({"feature": col, "type": "categorical", "mean": None, "median": None, "std": None,
                     "min": None, "max": None, "missing %": 100 * df[col].isna().mean(),
                     "mode": s.mode().iloc[0], "distinct": s.nunique()})
    return pd.DataFrame(rows)


def rate_by_category(df: pd.DataFrame, schema: Schema, col: str) -> pd.DataFrame:
    y = (df[schema.target].astype(str) == schema.positive).astype(int)
    g = pd.DataFrame({col: df[col].astype(str), "y": y}).groupby(col)["y"].agg(["mean", "count"])
    g = g.rename(columns={"mean": "rate", "count": "customers"}).reset_index()
    g["lift"] = g["rate"] / y.mean()
    return g.sort_values("rate", ascending=False)


def rank_features(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    X, y = split_xy(df, schema)
    rows = []
    for col in schema.features:
        s = X[col]
        if col in schema.numeric:
            filled = s.fillna(s.median()).to_numpy()
            mi = mutual_info_classif(filled.reshape(-1, 1), y, random_state=0)[0]
            auc = roc_auc_score(y, filled)
            auc = max(auc, 1 - auc)
            groups = [filled[y == k] for k in (0, 1)]
            stat, p = stats.f_oneway(*groups)
            test = "ANOVA F"
        else:
            codes = s.fillna("missing").astype("category").cat.codes.to_numpy()
            mi = mutual_info_classif(codes.reshape(-1, 1), y, discrete_features=True, random_state=0)[0]
            rates = pd.Series(y).groupby(codes).mean()
            auc = roc_auc_score(y, pd.Series(codes).map(rates))
            table = pd.crosstab(codes, y)
            stat, p, _, _ = stats.chi2_contingency(table)
            test = "Chi²"
        rows.append({"feature": col, "type": "numeric" if col in schema.numeric else "categorical",
                     "info gain (MI)": mi, "univariate AUC": auc, "test": test,
                     "statistic": stat, "p-value": p})
    out = pd.DataFrame(rows).sort_values("info gain (MI)", ascending=False).reset_index(drop=True)
    out.index += 1
    return out


def insights(df: pd.DataFrame, schema: Schema, max_items: int = 8) -> list[str]:
    y = (df[schema.target].astype(str) == schema.positive).astype(int)
    base = y.mean()
    notes = [f"Overall {schema.target} = {schema.positive} rate is **{base:.1%}** "
             f"({y.sum():,} of {len(y):,} rows)."]
    segments = []
    for col in schema.categorical:
        g = rate_by_category(df, schema, col)
        g = g[g["customers"] >= max(30, 0.02 * len(df))]
        if len(g) < 2:
            continue
        top, low = g.iloc[0], g.iloc[-1]
        segments.append((top["rate"] - low["rate"], col, top, low))
    segments.sort(key=lambda t: t[0], reverse=True)
    for gap, col, top, low in segments[: max_items - 3]:
        notes.append(f"**{col} = {top[col]}** has a {top['rate']:.1%} rate ({top['lift']:.1f}× average) "
                     f"vs {low['rate']:.1%} for {col} = {low[col]}.")
    for col in schema.numeric:
        a, b = df.loc[y == 1, col].median(), df.loc[y == 0, col].median()
        if pd.notna(a) and pd.notna(b) and b != 0 and abs(a - b) / abs(b) > 0.15:
            direction = "higher" if a > b else "lower"
            notes.append(f"Positive cases have {direction} **{col}** (median {a:,.1f} vs {b:,.1f}).")
    return notes[:max_items + 2]


def correlation(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    num = df[schema.numeric].copy()
    num[schema.target] = (df[schema.target].astype(str) == schema.positive).astype(int)
    return num.corr(numeric_only=True)
