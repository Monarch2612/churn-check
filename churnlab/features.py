from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from .data import Schema

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
ENGINEERED = ["n_addons", "avg_monthly_spend", "charge_gap", "tenure_band"]


@dataclass
class PrepConfig:
    scaler: str = "standard"
    num_impute: str = "median"
    engineer: bool = True
    drop: tuple[str, ...] = ()


def is_telco(schema: Schema) -> bool:
    cols = set(schema.features)
    return {"tenure", "MonthlyCharges", "TotalCharges"} <= cols and len(cols & set(ADDON_COLS)) >= 3


class TelcoFeatures(BaseEstimator, TransformerMixin):
    """Stateless domain features; safe inside CV because nothing is learned from y."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        addons = [c for c in ADDON_COLS if c in X.columns]
        X["n_addons"] = (X[addons] == "Yes").sum(axis=1).astype(float)
        tenure = X["tenure"].astype(float)
        X["avg_monthly_spend"] = np.where(tenure > 0, X["TotalCharges"] / tenure.replace(0, np.nan), X["MonthlyCharges"])
        X["avg_monthly_spend"] = X["avg_monthly_spend"].fillna(X["MonthlyCharges"])
        X["charge_gap"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
        X["tenure_band"] = pd.cut(tenure, [-1, 6, 12, 24, 48, 1e9],
                                  labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]).astype(str)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features) + ENGINEERED)


def feature_lists(schema: Schema, cfg: PrepConfig) -> tuple[list[str], list[str]]:
    numeric = [c for c in schema.numeric if c not in cfg.drop]
    categorical = [c for c in schema.categorical if c not in cfg.drop]
    if cfg.engineer and is_telco(schema):
        numeric += ["n_addons", "avg_monthly_spend", "charge_gap"]
        categorical += ["tenure_band"]
    return numeric, categorical


def build_preprocessor(schema: Schema, cfg: PrepConfig) -> Pipeline:
    numeric, categorical = feature_lists(schema, cfg)
    scalers = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler(), "none": "passthrough"}
    num_pipe = Pipeline([("impute", SimpleImputer(strategy=cfg.num_impute)), ("scale", scalers[cfg.scaler])])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
                                remainder="drop", verbose_feature_names_out=False)
    steps = []
    if cfg.engineer and is_telco(schema):
        steps.append(("domain", TelcoFeatures()))
    steps.append(("columns", columns))
    return Pipeline(steps)


def output_groups(preprocessor: Pipeline) -> tuple[list[str], list[str]]:
    """Map every transformed column back to the business feature it came from."""
    ct: ColumnTransformer = preprocessor.named_steps["columns"]
    names, groups = [], []
    for name, trans, cols in ct.transformers_:
        if name == "num":
            names += list(cols)
            groups += list(cols)
        elif name == "cat":
            enc = trans.named_steps["onehot"]
            for col, cats in zip(cols, enc.categories_):
                for cat in cats:
                    names.append(f"{col} = {cat}")
                    groups.append(col)
    return names, groups
