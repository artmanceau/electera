import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from electera.components.data_processing.data_loader import DataLoader
from electera.components.explanability.core_explanability import ExplainCore
from electera.components.modelling.election_predictor import ElectionPredictor
from electera.pipeline.election_backtester import BackTester
from electera.pipeline.train_models import ElectionModelTrainer


class DummyEstimator:
    def __init__(self, value=42):
        self.value = value
        self.best_models = [self]

    def predict(self, X):
        return np.full(len(X), self.value)

    def infer(self, X):
        return np.full(len(X), self.value)

    def get_features(self):
        return ["inscrits"]


# ---------------------------------------------------------------------------
# Tests for DataLoader dump_joblib & load_joblib
# ---------------------------------------------------------------------------


def test_data_loader_dump_load_joblib_lzma(tmp_path):
    obj = {"test_key": "test_value", "numbers": [1, 2, 3]}
    file_path = str(tmp_path / "model.joblib")

    DataLoader.dump_joblib(obj, file_path, compress="lzma")
    assert os.path.exists(file_path)

    loaded = DataLoader.load_joblib(file_path)
    assert loaded == obj


def test_data_loader_load_pickle_backward_compat(tmp_path):
    obj = {"legacy": True, "data": [4, 5, 6]}
    file_path = str(tmp_path / "legacy_model.pkl")

    # Save with DataLoader.dump_pickle
    DataLoader.dump_pickle(obj, file_path)
    assert os.path.exists(file_path)

    # Should be loadable via load_joblib and load_pickle
    loaded_via_joblib = DataLoader.load_joblib(file_path)
    assert loaded_via_joblib == obj

    loaded_via_pickle = DataLoader.load_pickle(file_path)
    assert loaded_via_pickle == obj


# ---------------------------------------------------------------------------
# Tests for ExplainCore._load_model
# ---------------------------------------------------------------------------


def test_explain_core_load_joblib_model(tmp_path):
    predictor = ElectionPredictor(trends=["a"])
    dummy = DummyEstimator()
    predictor.add_model("a", dummy)

    # Save as .joblib in output/models/
    models_dir = tmp_path / "output" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(models_dir / "model_2022_pr_a_v1.joblib")
    DataLoader.dump_joblib(predictor, model_path, compress="lzma")

    loaded_model, _n_models = ExplainCore._load_model(
        data_path=f"{tmp_path}/",
        var="a",
        year="2022",
        type_="pr",
        vars_=["a"],
        model_version="v1",
    )
    assert loaded_model is not None
    assert "a" in loaded_model.models


def test_explain_core_load_fallback_pkl(tmp_path):
    predictor = ElectionPredictor(trends=["a"])
    dummy = DummyEstimator()
    predictor.add_model("a", dummy)

    models_dir = tmp_path / "output" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(models_dir / "model_2022_pr_a_v1.pkl")
    DataLoader.dump_pickle(predictor, model_path)

    loaded_model, _n_models = ExplainCore._load_model(
        data_path=f"{tmp_path}/",
        var="a",
        year="2022",
        type_="pr",
        vars_=["a"],
        model_version="v1",
    )
    assert loaded_model is not None
    assert "a" in loaded_model.models


# ---------------------------------------------------------------------------
# Tests for election_backtester caching & skipping
# ---------------------------------------------------------------------------


def test_backtester_find_model(tmp_path):
    config = MagicMock()
    config.data_path = f"{tmp_path}/"
    config.use_mlflow = False
    backtester = BackTester(config)

    model_dir = tmp_path / "output" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_dir_str = str(model_dir) + "/"

    # Neither exists yet
    assert backtester._find_model(model_dir_str, "test_model") is None

    # Save .joblib
    joblib_file = str(model_dir / "test_model.joblib")
    DataLoader.dump_joblib({"model": 1}, joblib_file, compress="lzma")

    found = backtester._find_model(model_dir_str, "test_model")
    assert found == joblib_file

    # If only .pkl exists
    os.remove(joblib_file)
    pkl_file = str(model_dir / "test_model.pkl")
    DataLoader.dump_pickle({"model": 2}, pkl_file)

    found_pkl = backtester._find_model(model_dir_str, "test_model")
    assert found_pkl == pkl_file


def test_backtester_skips_when_full_model_and_results_exist(tmp_path):
    config = MagicMock()
    config.data_path = f"{tmp_path}/"
    config.version = "v1"
    config.organize_vote = True
    config.use_mlflow = False

    backtester = BackTester(config)

    model_dir = tmp_path / "output" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    res_dir = tmp_path / "output" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    full_model_name = "model_2022_pr_taua_v1"
    predictor = ElectionPredictor(trends=["a"])
    DataLoader.dump_joblib(
        predictor, str(model_dir / f"{full_model_name}.joblib"), compress="lzma"
    )

    # Create dummy parquet result files in data_path output/results
    pd.DataFrame({"pred": [1]}).to_parquet(
        str(res_dir / "results_synth_2022_pr_taua_v1.parquet")
    )
    pd.DataFrame({"pred": [1]}).to_parquet(
        str(res_dir / "results_full_2022_pr_taua_v1.parquet")
    )

    backtester.process_and_split_dataset = MagicMock()
    backtester.organize_vote = MagicMock()

    backtester.run_backtest(
        data=None,
        k_year=2022,
        k_type="pr",
        k_political_trends=["taua"],
        model=DummyEstimator,
        model_args={},
        model_name="boosting",
    )

    # Everything should be skipped
    backtester.process_and_split_dataset.assert_not_called()
    backtester.organize_vote.assert_not_called()


def test_backtester_loads_full_model_when_results_absent(tmp_path):
    config = MagicMock()
    config.data_path = f"{tmp_path}/"
    config.version = "v1"
    config.organize_vote = False
    config.use_mlflow = False

    backtester = BackTester(config)

    model_dir = tmp_path / "output" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    full_model_name = "model_2022_pr_taua_v1"
    predictor = ElectionPredictor(trends=["a"])
    predictor.add_model("a", DummyEstimator(value=77))
    DataLoader.dump_joblib(
        predictor, str(model_dir / f"{full_model_name}.joblib"), compress="lzma"
    )

    backtester.process_and_split_dataset = MagicMock()

    backtester.run_backtest(
        data=None,
        k_year=2022,
        k_type="pr",
        k_political_trends=["taua"],
        model=MagicMock(),
        model_args={},
        model_name="boosting",
    )

    # Dataset split & model training were skipped
    backtester.process_and_split_dataset.assert_not_called()
    assert backtester.election_predictor.models["a"].value == 77


def test_backtester_loads_cached_trend_model(tmp_path):
    config = MagicMock()
    config.data_path = f"{tmp_path}/"
    config.dataset_path = "data/"
    config.version = "v1"
    config.organize_vote = False
    config.use_mlflow = False

    backtester = BackTester(config)

    model_dir = tmp_path / "output" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save a cached trend model
    cached_model = DummyEstimator(value=99)
    trend_model_name = "best_model_boosting_2022_pr_taua_v1"
    DataLoader.dump_joblib(
        cached_model, str(model_dir / f"{trend_model_name}.joblib"), compress="lzma"
    )

    # Setup dummy data containers on backtester with >= 5 rows for sample(5)
    n_rows = 10
    df_features = pd.DataFrame({"inscrits": np.arange(n_rows)})
    backtester.X_train = {"taua": df_features}
    backtester.y_train = {"taua": pd.Series(np.linspace(0.1, 0.5, n_rows))}
    backtester.X_val = {"taua": df_features}
    backtester.y_val = {"taua": pd.Series(np.linspace(0.1, 0.5, n_rows))}
    backtester.X_test = {"taua": df_features}
    backtester.y_test = {"taua": pd.Series(np.linspace(0.1, 0.5, n_rows))}
    backtester.y_prev = {"taua": pd.Series(np.linspace(0.1, 0.5, n_rows))}
    backtester.meta_train = {"taua": None}
    backtester.meta_val = {"taua": None}
    backtester.meta_test = {"taua": None}
    backtester.feature_names = {"taua": ["inscrits"]}

    # Mock process_and_split_dataset to keep our populated attributes
    backtester.process_and_split_dataset = MagicMock()

    # Model constructor mock that should NOT be instantiated because cached model is used
    model_constructor = MagicMock()

    backtester.run_backtest(
        data=None,
        k_year=2022,
        k_type="pr",
        k_political_trends=["taua"],
        model=model_constructor,
        model_args={},
        model_name="boosting",
    )

    # Model constructor was never invoked because cached model was loaded
    model_constructor.assert_not_called()
    assert "a" in backtester.election_predictor.models
    assert backtester.election_predictor.models["a"].value == 99


# ---------------------------------------------------------------------------
# Tests for train_models caching and best_model saving
# ---------------------------------------------------------------------------


def test_trainer_find_saved_model(tmp_path):
    trainer = ElectionModelTrainer.__new__(ElectionModelTrainer)
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    assert trainer._find_saved_model("linear_reg", model_dir) is None

    joblib_path = os.path.join(model_dir, "linear_reg.joblib")
    DataLoader.dump_joblib(DummyEstimator(1), joblib_path, compress="lzma")
    assert trainer._find_saved_model("linear_reg", model_dir) == joblib_path

    os.remove(joblib_path)
    pkl_path = os.path.join(model_dir, "linear_reg.pkl")
    DataLoader.dump_pickle(DummyEstimator(2), pkl_path)
    assert trainer._find_saved_model("linear_reg", model_dir) == pkl_path


def test_trainer_saves_best_model(tmp_path):
    trainer = ElectionModelTrainer.__new__(ElectionModelTrainer)
    model_dir_path = str(tmp_path / "models")
    os.makedirs(model_dir_path, exist_ok=True)

    m1 = DummyEstimator(1)
    m2 = DummyEstimator(2)
    trainer.models = {"model_1": m1, "model_2": m2}
    trainer.results = {
        "model_1": {"mse": 0.5, "mae": 0.7, "r2": 0.1},
        "model_2": {"mse": 0.2, "mae": 0.4, "r2": 0.3},
    }

    comparison_df = trainer.compare_models()
    assert not comparison_df.empty
    best_row = comparison_df.sort_values("MSE").iloc[0]
    assert best_row["Model"] == "model_2"

    best_model_path = os.path.join(model_dir_path, "best_model_pvotea_feat1.joblib")
    DataLoader.dump_joblib(
        trainer.models[best_row["Model"]], best_model_path, compress="lzma"
    )

    loaded_best = DataLoader.load_joblib(best_model_path)
    assert loaded_best.value == 2


def test_backtester_gpu_configuration_propagation():
    import copy
    from electera.pipeline.election_backtester import MODEL_ARGS, MODELS

    config = MagicMock()
    config.use_gpu = True
    config.models = ["meta_boosting", "meta_boosting_multiple", "boosting"]
    config.use_mlflow = False

    backtester = BackTester(config)

    for model_name in config.models:
        model_args = copy.deepcopy(MODEL_ARGS[model_name])
        if "use_gpu" in model_args or model_name in (
            "boosting",
            "meta_boosting",
            "meta_boosting_multiple",
        ):
            model_args["use_gpu"] = getattr(backtester.config, "use_gpu", False)

        assert model_args["use_gpu"] is True

        if model_name in ("meta_boosting", "meta_boosting_multiple"):
            model_cls = MODELS[model_name]
            instance = model_cls(**model_args)
            assert instance.use_gpu is True


def test_backtester_s3_does_not_save_locally(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = MagicMock()
    config.data_path = "s3://my-bucket/election_data/"
    config.use_mlflow = False
    backtester = BackTester(config)

    res_dir, model_dir = backtester._get_output_paths()
    assert res_dir == "s3://my-bucket/election_data/output/results/"
    assert model_dir == "s3://my-bucket/election_data/output/models/"
    # Verify no local output directory was created
    assert not os.path.exists("output")


def test_trainer_s3_does_not_save_locally(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = ElectionModelTrainer.__new__(ElectionModelTrainer)
    config = MagicMock()
    config.dataset_path = "s3://my-bucket/data/derived/processed/data.parquet"
    trainer.config = config
    trainer.models = {"m1": DummyEstimator(1)}
    trainer.results = {"m1": {"mse": 0.1, "mae": 0.2, "r2": 0.9}}

    comparison_df = trainer.compare_models()
    assert not comparison_df.empty
    # Verify no local data/ directory was created
    assert not os.path.exists("data")
