import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from .features import output_groups
from .models import HAS_XGB

if HAS_XGB:
    import xgboost as xgb
    XGB_TYPES = (xgb.XGBClassifier,)
else:
    XGB_TYPES = ()

TREE_MODELS = (DecisionTreeClassifier, RandomForestClassifier, HistGradientBoostingClassifier)


def permutation_table(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, repeats: int = 5) -> pd.DataFrame:
    r = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=repeats, random_state=0, n_jobs=-1)
    out = pd.DataFrame({"feature": X.columns, "AUC drop": r.importances_mean, "std": r.importances_std})
    return out.sort_values("AUC drop", ascending=False).reset_index(drop=True)


def display_frame(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    prep = pipe.named_steps["prep"]
    return prep.named_steps["domain"].transform(X) if "domain" in prep.named_steps else X


class Explainer:
    """Additive per-feature contributions, aggregated back to business features.

    Linear and tree models get exact SHAP values in log-odds; any other model falls back to a
    model-agnostic permutation SHAP estimate in probability units.
    """

    def __init__(self, pipe: Pipeline, background: pd.DataFrame):
        self.pipe = pipe
        self.prep = pipe.named_steps["prep"]
        self.model = pipe.named_steps["model"]
        self.names, self.groups = output_groups(self.prep)
        bg = background.sample(min(len(background), 200), random_state=0)
        self.bg = self.prep.transform(bg)
        self.units = "log-odds"
        self._tree = None
        self._generic = None
        if isinstance(self.model, TREE_MODELS):
            self._tree = shap.TreeExplainer(self.model)
            if not isinstance(self.model, HistGradientBoostingClassifier):
                self.units = "probability"
        elif not isinstance(self.model, (LogisticRegression, *XGB_TYPES)):
            masker = shap.maskers.Independent(self.bg, max_samples=100)
            self._generic = shap.PermutationExplainer(lambda Z: self.model.predict_proba(Z)[:, 1], masker)
            self.units = "probability"

    def raw_contributions(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.prep.transform(X)
        if isinstance(self.model, LogisticRegression):
            return (Z - self.bg.mean(axis=0)) * self.model.coef_[0]
        if isinstance(self.model, XGB_TYPES):
            return self.model.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]
        if self._tree is not None:
            values = self._tree.shap_values(Z, check_additivity=False)
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[..., 1] if values.shape[-1] == 2 else values[1]
            return values
        return self._generic(Z, silent=True).values

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        raw = self.raw_contributions(X)
        frame = pd.DataFrame(raw, columns=self.names)
        return frame.T.groupby(self.groups, sort=False).sum().T

    def local(self, row: pd.DataFrame) -> pd.DataFrame:
        contrib = self.contributions(row).iloc[0]
        shown = display_frame(self.pipe, row).iloc[0]
        out = pd.DataFrame({"feature": contrib.index, "value": [shown.get(f, "") for f in contrib.index],
                            "contribution": contrib.to_numpy()})
        out["value"] = out["value"].map(lambda v: f"{v:,.2f}" if isinstance(v, (float, np.floating)) else str(v))
        return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    def global_summary(self, X: pd.DataFrame, max_rows: int = 400) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = max_rows if self._generic is None else min(max_rows, 120)
        sample = X.sample(min(len(X), n), random_state=1)
        contrib = self.contributions(sample)
        importance = contrib.abs().mean().sort_values(ascending=False).rename("mean |contribution|").reset_index()
        importance.columns = ["feature", "mean |contribution|"]
        return importance, contrib.reset_index(drop=True), display_frame(self.pipe, sample).reset_index(drop=True)


def odds_ratios(pipe: Pipeline) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    if not isinstance(model, LogisticRegression):
        return None
    names, groups = output_groups(pipe.named_steps["prep"])
    coef = model.coef_[0]
    out = pd.DataFrame({"term": names, "feature": groups, "coefficient": coef, "odds ratio": np.exp(coef)})
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def tree_rules(pipe: Pipeline, max_depth: int = 4) -> str | None:
    model = pipe.named_steps["model"]
    if not isinstance(model, DecisionTreeClassifier):
        return None
    names, _ = output_groups(pipe.named_steps["prep"])
    return export_text(model, feature_names=names, max_depth=max_depth, show_weights=True)


def what_if(pipe: Pipeline, row: pd.DataFrame, levers: dict[str, list[str]]) -> pd.DataFrame:
    base = float(pipe.predict_proba(row)[:, 1][0])
    trials = []
    for col, options in levers.items():
        current = str(row.iloc[0][col])
        for option in options:
            if option == current:
                continue
            alt = row.copy()
            alt[col] = option
            trials.append((col, current, option, alt))
    if not trials:
        return pd.DataFrame(columns=["change", "from", "to", "new risk", "risk change"])
    probs = pipe.predict_proba(pd.concat([t[3] for t in trials]))[:, 1]
    out = pd.DataFrame({"change": [t[0] for t in trials], "from": [t[1] for t in trials],
                        "to": [t[2] for t in trials], "new risk": probs})
    out["risk change"] = out["new risk"] - base
    return out.sort_values("risk change").reset_index(drop=True)
