import numpy as np
import pytest

from churnlab import data, evaluate, explain, explore, models
from churnlab.features import PrepConfig


@pytest.fixture(scope="module")
def setup():
    df = data.load()
    schema = data.infer_schema(df)
    X, y = data.split_xy(df, schema)
    return df, schema, X, y


def test_schema(setup):
    df, schema, X, y = setup
    assert schema.target == "Churn" and schema.positive == "Yes"
    assert schema.id_cols == ["customerID"]
    assert {"tenure", "MonthlyCharges", "TotalCharges"} <= set(schema.numeric)
    assert "SeniorCitizen" in schema.categorical
    assert schema.value_col == "MonthlyCharges"
    assert df["TotalCharges"].dtype == float
    assert abs(y.mean() - 0.265) < 0.01


def test_explore(setup):
    df, schema, _, _ = setup
    ranking = explore.rank_features(df, schema)
    assert ranking.iloc[0]["feature"] in {"Contract", "tenure", "OnlineSecurity", "TechSupport"}
    assert len(explore.insights(df, schema)) >= 4


@pytest.mark.parametrize("key", list(models.REGISTRY))
def test_every_model_and_explainer(setup, key):
    _, schema, X, y = setup
    cfg = models.TrainConfig(models=[key], folds=3)
    exp = models.run_experiment(X, y, schema, PrepConfig(), cfg)
    row = exp.leaderboard.iloc[0]
    if key != "baseline":
        assert row["Test AUC"] > 0.75
    pipe = exp.models[key].pipeline
    ex = explain.Explainer(pipe, exp.X_train)
    local = ex.local(exp.X_test.iloc[[0]])
    assert len(local) == len(schema.features) + 4
    assert np.isfinite(local["contribution"]).all()


def test_business_and_whatif(setup):
    _, schema, X, y = setup
    exp = models.run_experiment(X, y, schema, PrepConfig(), models.TrainConfig(models=["logreg"], folds=3))
    econ = evaluate.Economics()
    value = evaluate.customer_value(exp.X_test, schema.value_col, econ)
    curve = evaluate.profit_curve(exp.y_test, exp.models["logreg"].test_proba, value, econ)
    bench = evaluate.benchmark_profits(exp.y_test, value, econ)
    assert curve["profit"].max() > bench["Contact everyone"]
    row = exp.X_test.iloc[[0]]
    levers = {"Contract": ["Month-to-month", "One year", "Two year"]}
    wi = explain.what_if(exp.models["logreg"].pipeline, row, levers)
    assert len(wi) == 2
