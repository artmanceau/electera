"""
Pipeline 2: Model Training and Evaluation
=========================================
This module handles model training, evaluation, and comparison for the election modeling project.
"""

import argparse
import os
import re
import shutil
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import xgboost as xgb

from electera.components.data_processing.data_loader import DataLoader, DataUtils
from electera.components.modelling.benchmark_models import BenchmarkModels
from electera.components.modelling.boosting.boosting import BoostingModel
from electera.components.modelling.data_split_pl import get_Xy_pl
from electera.components.modelling.evaluation import ModelEvaluator
from electera.components.modelling.meta_booster import (
    MetaBooster,
    MetaBoosterMultipleElections,
)
from electera.components.utils.config import TrainModelsConfig
from electera.components.utils.read_config import ConfigReader

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_inner_fold(
    X_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    n_inner_splits: int = 3,
    n_trials: int = 30,
    use_gpu: bool = False,
    early_stopping_rounds: int = 30,
    max_n_estimators: int = 1500,
) -> Dict[str, Any]:
    """Inner CV loop: Uses Optuna to find hyperparameters minimizing MSE across inner folds."""
    inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=42)
    device = "cuda" if use_gpu else "cpu"

    def objective(trial):
        params = {
            "n_estimators": max_n_estimators,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "tree_method": "hist",
            "device": device,
            "early_stopping_rounds": early_stopping_rounds,
            "eval_metric": "rmse",
            "random_state": 42,
        }

        if not use_gpu:
            params["n_jobs"] = -1

        fold_losses = []
        best_iterations = []

        for in_train_idx, in_val_idx in inner_cv.split(X_train_outer, y_train_outer):
            X_in_tr, X_in_val = X_train_outer[in_train_idx], X_train_outer[in_val_idx]
            y_in_tr, y_in_val = y_train_outer[in_train_idx], y_train_outer[in_val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_in_tr,
                y_in_tr,
                eval_set=[(X_in_val, y_in_val)],
                verbose=False,
            )

            preds = model.predict(X_in_val)
            fold_losses.append(mean_squared_error(y_in_val, preds))
            best_iterations.append(model.best_iteration)

        trial.set_user_attr("mean_best_n_estimators", int(np.mean(best_iterations)) + 1)
        return float(np.mean(fold_losses))

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["n_estimators"] = study.best_trial.user_attrs["mean_best_n_estimators"]
    return best_params


def nested_cross_validation_xgb(
    X: np.ndarray,
    y: np.ndarray,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    n_trials: int = 30,
    use_gpu: bool = False,
) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    """Executes the full nested cross-validation pipeline with optional GPU support."""
    outer_cv = KFold(n_splits=n_outer_splits, shuffle=True, random_state=123)
    outer_scores = []
    device = "cuda" if use_gpu else "cpu"

    logger.info(f"Nested CV Device: {device.upper()}")
    logger.info(f"Starting Nested CV: {n_outer_splits} Outer Folds x {n_inner_splits} Inner Folds")

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_tr_out, X_te_out = X[train_idx], X[test_idx]
        y_tr_out, y_te_out = y[train_idx], y[test_idx]

        best_params = optimize_inner_fold(
            X_tr_out,
            y_tr_out,
            n_inner_splits=n_inner_splits,
            n_trials=n_trials,
            use_gpu=use_gpu,
        )

        final_model_params = {
            **best_params,
            "tree_method": "hist",
            "device": device,
            "eval_metric": "rmse",
            "random_state": 42,
        }
        if not use_gpu:
            final_model_params["n_jobs"] = -1

        final_model = xgb.XGBRegressor(**final_model_params)
        final_model.fit(X_tr_out, y_tr_out, verbose=False)

        outer_preds = final_model.predict(X_te_out)
        mse = mean_squared_error(y_te_out, outer_preds)
        mae = mean_absolute_error(y_te_out, outer_preds)
        outer_scores.append({"mse": mse, "mae": mae})

        logger.info(
            f"Outer Fold {outer_fold + 1}/{n_outer_splits} | MSE: {mse:.4f} | MAE: {mae:.4f} | Trees: {best_params['n_estimators']} | Depth: {best_params['max_depth']}"
        )

    mean_mse = float(np.mean([s["mse"] for s in outer_scores]))
    std_mse = float(np.std([s["mse"] for s in outer_scores]))
    logger.info(f"Outer CV Generalization MSE: {mean_mse:.4f} (+/- {std_mse:.4f})")

    # Fit best model on the complete outer dataset
    best_params_full = optimize_inner_fold(
        X,
        y,
        n_inner_splits=n_inner_splits,
        n_trials=n_trials,
        use_gpu=use_gpu,
    )
    final_deployment_model = xgb.XGBRegressor(
        **best_params_full,
        tree_method="hist",
        device=device,
        eval_metric="rmse",
        random_state=42,
    )
    if not use_gpu:
        final_deployment_model.set_params(n_jobs=-1)

    final_deployment_model.fit(X, y, verbose=False)
    return final_deployment_model, {"nested_mse": mean_mse, "nested_mse_std": std_mse}


class ElectionModelTrainer:
    """Class to handle model training and evaluation pipeline"""

    def __init__(self):
        self.config = ConfigReader._read_config(
            "../config/train_models.json", TrainModelsConfig
        )
        self.models = {}
        self.results = {}
        self.predictions = {}
        self.model_data = {}
        self.input_examples = {}

    def _find_saved_model(
        self, model_name: str, model_dir_path: str = "output/models/"
    ) -> Optional[str]:
        fs = DataUtils._create_fs() if DataUtils._detect_s3(model_dir_path) else None
        for ext in [".joblib", ".pkl"]:
            cand = (
                f"{model_dir_path.rstrip('/')}/{model_name}{ext}"
                if DataUtils._detect_s3(model_dir_path)
                else os.path.join(model_dir_path, f"{model_name}{ext}")
            )
            if DataUtils._exists(cand, fs=fs):
                return cand
        return None

    def data_processing(
        self, data, var, feature_groups=["rank", "inscrits", "type", "geo"]
    ):
        logger.info("Preparing data splits...")
        container_names = (
            "X_train",
            "X_val",
            "X_test",
            "y_train",
            "y_val",
            "y_test",
            "y_prev",
            "meta_train",
            "meta_val",
            "meta_test",
        )
        self.feature_names = {}
        values = get_Xy_pl(
            data,
            vote_variable=f"pvote{var}",
            year=2022,
            election_type="presidentiel",
            predict_delta=self.config.predict_delta,
            predict_perc=self.config.predict_perc,
            selected_groups=feature_groups,
            selected_features=None,
            split_method_way="random",
        )
        for name, value in zip(container_names, values):
            setattr(self, name, value)

        self.feature_names = self.X_train.columns.tolist()
        logger.info(f"Data prepared: Train {self.X_train.shape}, Test {self.X_test.shape}")

    def compare_models(self):
        logger.info("Comparing models...")
        model_names = []
        mse_scores = []
        mae_scores = []
        r2_scores = []

        for model_name, results in self.results.items():
            model_names.append(model_name)
            mse_scores.append(results["mse"])
            mae_scores.append(results["mae"])
            r2_scores.append(results["r2"])

        comparison_df = pd.DataFrame(
            {"Model": model_names, "MSE": mse_scores, "MAE": mae_scores, "R²": r2_scores}
        )
        config = getattr(self, "config", None)
        dataset_path = getattr(config, "dataset_path", "") if config else ""
        if not DataUtils._detect_s3(dataset_path):
            os.makedirs("data", exist_ok=True)
            comparison_df.to_csv("data/comp_table.csv")
        return comparison_df

    def save_results(self, experiment_name=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name is None:
            experiment_name = f"pipeline_train_models_{timestamp}"

        mlflow.set_experiment(experiment_name)
        logger.info(f"Starting MLflow experiment: {experiment_name}")

        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=f"{model_name}_{timestamp}"):
                if hasattr(self, "config") and self.config is not None:
                    for key, value in self.config.model_dump().items():
                        mlflow.log_param(f"config_{key}", str(value)[:500])

                self._log_model_to_mlflow(model_name, model)
                mlflow.log_param("timestamp", timestamp)
                mlflow.log_param("model_name", model_name)
                mlflow.set_tag("experiment_type", "model_training")
                mlflow.set_tag("framework", "xgboost_nested_cv")

    def _log_model_to_mlflow(self, model_name: str, model) -> None:
        model_name_safe = re.sub(r"[:\s]", "_", model_name)
        model_results = self.results.get(model_name, {})
        model_preds = self.predictions.get(model_name, {})

        mlflow.sklearn.log_model(
            model,
            name=model_name_safe,
            registered_model_name=model_name_safe,
        )

        for metric_name, metric_value in model_results.items():
            if metric_name == "predictions":
                continue
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"{metric_name}_{model_name}", float(metric_value))

        artifacts_dir = tempfile.mkdtemp()
        pred_path = os.path.join(artifacts_dir, "predictions.csv")
        model_preds.to_csv(pred_path)
        mlflow.log_artifact(pred_path, artifact_path=f"{model_name_safe}/artifacts")
        shutil.rmtree(artifacts_dir, ignore_errors=True)


def run():
    trainer = ElectionModelTrainer()
    data = DataLoader.load_dataset(trainer.config.dataset_path, engine="polars")

    is_s3 = DataUtils._detect_s3(trainer.config.dataset_path)
    if is_s3:
        s3_base = (
            trainer.config.dataset_path.split("derived/")[0]
            if "derived/" in trainer.config.dataset_path
            else trainer.config.dataset_path.rsplit("/", 1)[0] + "/"
        )
        model_dir_path = f"{s3_base}output/models/"
    else:
        model_dir_path = "output/models/"
        os.makedirs(model_dir_path, exist_ok=True)

    def _save_model_file(m, m_name):
        fpath = (
            f"{model_dir_path}{m_name}.joblib"
            if is_s3
            else os.path.join(model_dir_path, f"{m_name}.joblib")
        )
        DataLoader.dump_joblib(m, fpath, compress="lzma")

    use_gpu = getattr(trainer.config, "use_gpu", False)

    for var in trainer.config.vote_variable:
        for feature_groups in [
            ["raw", "inscrits", "type", "geo", "annee", "previous_vote"],
            ["raw", "inscrits", "type", "geo", "annee"],
            ["raw", "inscrits", "type", "geo"],
            ["rank", "inscrits", "type", "geo"],
            ["rank", "inscrits", "type", "geo", "pct_change"],
        ]:
            feature_groups_str = "_".join(feature_groups)
            logger.info(f"Running for variable: {var} | features: {feature_groups_str}")

            trainer.data_processing(data, var, feature_groups)

            # 1. Trivial Model 1
            if "trivial_1" in trainer.config.models:
                model_name = f"trivial_1_{var}_{feature_groups_str}"
                saved_path = trainer._find_saved_model(model_name, model_dir_path)
                if saved_path:
                    model = DataLoader.load_joblib(saved_path)
                    trainer.models[model_name] = model
                    y_1 = trainer.y_prev.fillna(trainer.y_prev.mean())
                else:
                    bm = BenchmarkModels()
                    y_1 = bm.train_trivial_1(trainer.y_prev, trainer.y_test)
                    model = bm.get_model()
                    trainer.models[model_name] = model
                    _save_model_file(model, model_name)

                trainer.results[model_name] = ModelEvaluator.evaluate(trainer.y_test, y_1, model_name, extended=True)
                trainer.predictions[model_name] = pd.concat([trainer.y_test, y_1], axis=1)

            # 2. Linear Regression
            if "linear_reg" in trainer.config.models:
                model_name = f"linear_regression_{var}_{feature_groups_str}"
                saved_path = trainer._find_saved_model(model_name, model_dir_path)
                if saved_path:
                    model = DataLoader.load_joblib(saved_path)
                    trainer.models[model_name] = model
                    y_3 = model.predict(trainer.X_test)
                else:
                    bm = BenchmarkModels()
                    y_3 = bm.train_linear_model(trainer.X_train, trainer.y_train, trainer.X_test, linear_model=LinearRegression)
                    model = bm.get_model()
                    trainer.models[model_name] = model
                    _save_model_file(model, model_name)

                trainer.results[model_name] = ModelEvaluator.evaluate(trainer.y_test, y_3, model_name, extended=True)
                trainer.predictions[model_name] = pd.concat([trainer.y_test, pd.Series(y_3)], axis=1)

            # 3. Boosting via Nested Cross-Validation & Optuna
            if "boosting" in trainer.config.models:
                std_model_name = f"xgboost_nested_cv_{var}_{feature_groups_str}"
                saved_path = trainer._find_saved_model(std_model_name, model_dir_path)

                if saved_path:
                    logger.info(f"Model '{std_model_name}' already exists. Skipping computation.")
                    model = DataLoader.load_joblib(saved_path)
                    trainer.models[std_model_name] = model
                    preds = model.predict(trainer.X_test.to_numpy())
                else:
                    logger.info(f"Running Nested CV and Optuna optimization for {std_model_name}...")
                    model, cv_metrics = nested_cross_validation_xgb(
                        X=trainer.X_train.to_numpy(),
                        y=trainer.y_train.to_numpy(),
                        n_outer_splits=getattr(trainer.config, "n_splits_outer", 5),
                        n_inner_splits=getattr(trainer.config, "n_splits_inner", 3),
                        n_trials=getattr(trainer.config, "n_trials", 25),
                        use_gpu=use_gpu,
                    )
                    trainer.models[std_model_name] = model
                    _save_model_file(model, std_model_name)
                    preds = model.predict(trainer.X_test.to_numpy())

                trainer.results[std_model_name] = ModelEvaluator.evaluate(trainer.y_test, preds, std_model_name, extended=True)
                trainer.predictions[std_model_name] = pd.concat([trainer.y_test, pd.Series(preds, index=trainer.y_test.index)], axis=1)

            # 4. Meta Boosting
            if "meta_boosting" in trainer.config.models:
                model_name = f"meta_booster_{var}_{feature_groups_str}"
                saved_path = trainer._find_saved_model(model_name, model_dir_path)
                if saved_path:
                    meta_booster = DataLoader.load_joblib(saved_path)
                    trainer.models[model_name] = meta_booster
                    y_pred = meta_booster.infer(trainer.X_test)
                else:
                    meta_booster = MetaBooster(
                        method="xgboost",
                        objective_metric=mean_squared_error,
                        weighting="log",
                        n_splits_outer=3,
                        n_splits_inner=3,
                        n_trials=10,
                        use_gpu=use_gpu,
                    )
                    meta_booster.train(trainer.X_train, trainer.y_train, use_feature_selection=False)
                    trainer.models[model_name] = meta_booster
                    _save_model_file(meta_booster, model_name)
                    y_pred = meta_booster.infer(trainer.X_test)

                trainer.results[model_name] = ModelEvaluator.evaluate(trainer.y_test, y_pred, model_name, extended=True)
                trainer.predictions[model_name] = pd.concat([trainer.y_test, pd.Series(y_pred)], axis=1)

        comparison_df = trainer.compare_models()
        logger.success("\nModel Comparison:")
        logger.info(comparison_df.to_string(index=False))

        if trainer.config.use_MLFlow:
            trainer.save_results()

    return trainer


if __name__ == "__main__":
    trainer = run()
