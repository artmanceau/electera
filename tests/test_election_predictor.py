import numpy as np
import pandas as pd
import pytest

from electera.components.modelling.election_predictor import ElectionPredictor


class DummyModel:
    def __init__(self, value=0.5):
        self.value = value

    def infer(self, X):
        return np.full(len(X), self.value)

    def infer_multiple(self, X):
        return np.full(len(X), self.value * 2)

    def get_features(self):
        return ["inscrits"]


@pytest.fixture
def predictor():
    ep = ElectionPredictor(trends=["a", "b", "par"])
    ep.add_model("a", DummyModel(0.2))
    ep.add_model("b", DummyModel(0.3))
    ep.add_model("par", DummyModel(0.5))
    return ep


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "codecommune": ["001", "002"],
            "inscrits": [100, 200],
            "previouspvotea": [0.1, 0.2],
            "previouspvoteb": [0.1, 0.2],
            "inscrits": [100, 200],
        }
    )


def test_predict_votes_basic(predictor, sample_df):
    X = sample_df.copy()
    X["previouspvotea"] = 0.1
    X["previouspvoteb"] = 0.2

    predictor.features["a"] = ["inscrits"]
    predictor.features["b"] = ["inscrits"]
    predictor.features["par"] = ["inscrits"]

    result = predictor.predict_votes(X)

    assert "pvotea" in result.columns
    assert "pvoteb" in result.columns
    assert "pvotepar" in result.columns
    assert np.isclose(result[["pvotea", "pvoteb"]].sum(axis=1).mean(), 1.0, atol=1e-3)


def test_predict_votes_agg(predictor, sample_df):
    X = sample_df.copy()
    X["previouspvotea"] = 0.1
    X["previouspvoteb"] = 0.2

    predictor.features["a"] = ["inscrits"]
    predictor.features["b"] = ["inscrits"]
    predictor.features["par"] = ["inscrits"]

    agg = predictor.predict_votes(X, agg=True)

    assert isinstance(agg, dict)
    assert "tot_par" in agg
    assert "tot_votea" in agg


def test_add_model_sets_features():
    ep = ElectionPredictor(trends=["a"])
    model = DummyModel()

    ep.add_model("a", model)

    assert ep.models["a"] is model
    assert ep.features["a"] == ["inscrits"]


def test_evaluate_predictions():
    ep = ElectionPredictor(trends=["a"])

    X_pred = pd.DataFrame(
        {
            "codecommune": ["001"],
            "pvotea": [0.6],
            "inscrits": [100],
            "exprimes": [90],
        }
    )

    X_true = pd.DataFrame(
        {
            "codecommune": ["001"],
            "pvotea": [0.5],
            "inscrits": [100],
            "exprimes": [90],
        }
    )

    merged = ep.evaluate_predictions(X_pred, X_true)

    assert "pvotea_diff" in merged.columns
    assert np.isclose(merged["pvotea_diff"].iloc[0], 0.1)
