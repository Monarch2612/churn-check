import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data import Schema
from .evaluate import metric_row
from .features import PrepConfig, build_preprocessor

SEED = 42

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # missing libomp on macOS raises XGBoostError, not ImportError
    HAS_XGB = False


@dataclass
class ModelSpec:
    key: str
    label: str
    make: callable
    blurb: str
    search: dict = field(default_factory=dict)
    supports_weight: bool = True


def _xgb(balanced, pos_weight):
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9,
                         colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=-1,
                         scale_pos_weight=pos_weight if balanced else 1.0)


REGISTRY: dict[str, ModelSpec] = {s.key: s for s in [
    ModelSpec("baseline", "Majority baseline", lambda b, w: DummyClassifier(strategy="prior"),
              "Always predicts the base rate. Any useful model must beat this.", supports_weight=False),
    ModelSpec("logreg", "Logistic regression",
              lambda b, w: LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced" if b else None),
              "Linear, fast, fully interpretable via odds ratios.",
              {"model__C": loguniform(1e-3, 1e2)}),
    ModelSpec("tree", "Decision tree",
              lambda b, w: DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=SEED,
                                                  class_weight="balanced" if b else None),
              "Human-readable if/then rules.",
              {"model__max_depth": randint(2, 12), "model__min_samples_leaf": randint(5, 100)}),
    ModelSpec("rf", "Random forest",
              lambda b, w: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED,
                                                  class_weight="balanced_subsample" if b else None),
              "Many trees voting; robust default.",
              {"model__n_estimators": randint(150, 500), "model__max_depth": randint(3, 20),
               "model__min_samples_leaf": randint(1, 30), "model__max_features": uniform(0.2, 0.7)}),
    ModelSpec("hgb", "Gradient boosting",
              lambda b, w: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                                          l2_regularization=1.0, random_state=SEED,
                                                          class_weight="balanced" if b else None),
              "Sequential trees correcting each other's mistakes.",
              {"model__learning_rate": loguniform(0.01, 0.3), "model__max_leaf_nodes": randint(4, 40),
               "model__max_iter": randint(100, 600), "model__l2_regularization": loguniform(1e-3, 10)}),
    *([ModelSpec("xgb", "XGBoost", _xgb, "Industry-standard boosted trees.",
                 {"model__n_estimators": randint(100, 600), "model__max_depth": randint(2, 8),
                  "model__learning_rate": loguniform(0.01, 0.3), "model__subsample": uniform(0.6, 0.4),
                  "model__colsample_bytree": uniform(0.5, 0.5), "model__min_child_weight": randint(1, 10)})]
      if HAS_XGB else []),
    ModelSpec("knn", "k-nearest neighbours",
              lambda b, w: KNeighborsClassifier(n_neighbors=25, weights="distance"),
              "Predicts from the most similar customers.",
              {"model__n_neighbors": randint(5, 80)}, supports_weight=False),
    ModelSpec("nb", "Naive Bayes", lambda b, w: GaussianNB(),
              "Probabilistic baseline assuming independent features.", supports_weight=False),
    ModelSpec("svm", "Support vector machine",
              lambda b, w: CalibratedClassifierCV(SVC(C=1.0, random_state=SEED,
                                                      class_weight="balanced" if b else None), ensemble=False),
              "Maximum-margin classifier with RBF kernel (slowest).",
              {"model__estimator__C": loguniform(1e-2, 1e2), "model__estimator__gamma": loguniform(1e-4, 1e-1)}),
    ModelSpec("mlp", "Neural network",
              lambda b, w: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=600,
                                         early_stopping=True, random_state=SEED),
              "Two-layer perceptron.",
              {"model__alpha": loguniform(1e-5, 1e-1), "model__learning_rate_init": loguniform(1e-4, 1e-2)},
              supports_weight=False),
]}

DEFAULT_MODELS = [k for k in ["baseline", "logreg", "tree", "rf", "hgb", "xgb"] if k in REGISTRY]


@dataclass
class TrainConfig:
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    folds: int = 5
    test_size: float = 0.2
    balanced: bool = False
    tune: list[str] = field(default_factory=list)
    tune_iter: int = 20


@dataclass
class TrainedModel:
    key: str
    label: str
    pipeline: Pipeline
    cv_proba: np.ndarray
    test_proba: np.ndarray
    fit_seconds: float
    params: dict


@dataclass
class Experiment:
    schema: Schema
    prep: PrepConfig
    train_cfg: TrainConfig
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    models: dict[str, TrainedModel]
    leaderboard: pd.DataFrame

    def best_key(self) -> str:
        return self.leaderboard.index[0]


def make_pipeline(key: str, schema: Schema, prep: PrepConfig, balanced: bool, pos_weight: float) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(schema, prep)),
                     ("model", REGISTRY[key].make(balanced, pos_weight))])


def run_experiment(X: pd.DataFrame, y: np.ndarray, schema: Schema, prep: PrepConfig, cfg: TrainConfig,
                   progress=None) -> Experiment:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=SEED)
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    cv = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=SEED)
    trained, rows = {}, []
    for i, key in enumerate(cfg.models):
        spec = REGISTRY[key]
        if progress:
            progress(i / len(cfg.models), f"Training {spec.label}…")
        pipe = make_pipeline(key, schema, prep, cfg.balanced, pos_weight)
        if key in cfg.tune and spec.search:
            search = RandomizedSearchCV(pipe, spec.search, n_iter=cfg.tune_iter, scoring="roc_auc", cv=cv,
                                        random_state=SEED, n_jobs=-1)
            search.fit(X_train, y_train)
            pipe = clone(pipe).set_params(**search.best_params_)
        cv_proba = cross_val_predict(clone(pipe), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        seconds = time.perf_counter() - start
        test_proba = pipe.predict_proba(X_test)[:, 1]
        params = {k.replace("model__", ""): v for k, v in pipe.get_params().items()
                  if k.startswith("model__") and k.count("__") == 1}
        label = spec.label + (" (tuned)" if key in cfg.tune and spec.search else "")
        trained[key] = TrainedModel(key, label, pipe, cv_proba, test_proba, seconds, params)
        row = {"model": label, **metric_row(y_train, cv_proba, prefix="CV "),
               **metric_row(y_test, test_proba, prefix="Test "), "fit time (s)": seconds}
        rows.append(pd.Series(row, name=key))
    if progress:
        progress(1.0, "Done")
    board = pd.DataFrame(rows).sort_values("CV AUC", ascending=False)
    return Experiment(schema, prep, cfg, X_train, X_test, y_train, y_test, trained, board)


def refit_full(exp: Experiment, key: str) -> Pipeline:
    pipe = clone(exp.models[key].pipeline)
    X = pd.concat([exp.X_train, exp.X_test])
    y = np.concatenate([exp.y_train, exp.y_test])
    return pipe.fit(X, y)


def out_of_fold_scores(exp: Experiment, key: str, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Honest scores for every row: each customer is scored by a model that never saw them."""
    cv = StratifiedKFold(n_splits=exp.train_cfg.folds, shuffle=True, random_state=SEED)
    return cross_val_predict(clone(exp.models[key].pipeline), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
