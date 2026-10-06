"""
Election Backtester
=========================================

# Implement a backtesting logic.
# We train models over presidential election and legislative election (1er tour).
# The model is trained on all elections. One election is excluded from testing and taken for test.
# Hyperparameters are optimized via Nested Cross-Validation with Optuna.
"""

import copy
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import polars as pl
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error  # , r2_score
from sklearn.model_selection import KFold
import xgboost as xgb

import electera.components.mlflow.mlflow_utils as mlf_utils
from assets.delta_pred_features import CHAMPION_FEATURES_EXTENDED
from electera.components.data_processing.data_loader import DataLoader, DataUtils
from electera.components.modelling.benchmark_models import (
    LinearModel,
    TrivialModel1,
    TrivialModel2,
)
from electera.components.modelling.boosting.boosting import BoostingModel
from electera.components.modelling.data_split_pl import get_Xy_pl
from electera.components.modelling.election_predictor import ElectionPredictor
from electera.components.modelling.evaluation import ModelEvaluator
from electera.components.modelling.meta_booster import (
    MetaBooster,
    MetaBoosterMultipleElections,
    _safe_predict,
)
from electera.components.utils.config import BackTesterConfig
from electera.components.utils.read_config import ConfigReader

optuna.logging.set_verbosity(optuna.logging.WARNING)

S3_SAVE = True
MODELS = {
    "trivial_1": TrivialModel1,
    "trivial_2": TrivialModel2,
    "linear": LinearModel,
    "boosting": BoostingModel,
    "meta_boosting": MetaBooster,
    "meta_boosting_multiple": MetaBoosterMultipleElections,
}
MODEL_ARGS = {
    "trivial_1": {},
    "trivial_2": {},
    "linear": {"linear_model": LinearRegression},
    "boosting": {
        "n_splits_outer": 5,
        "n_splits_inner": 3,
        "n_trials": 25,
        "early_stopping_rounds": 30,
        "max_n_estimators": 1500,
        "use_gpu": False,
    },
    "meta_boosting": {
        "method": "xgboost",
        "objective_metric": mean_absolute_error,
        "weighting": "proportional",
        "features": None,
        "n_splits_inner": 10,
        "n_splits_outer": 10,
        "n_trials": 10,
        "poll_adj": False,
        "use_gpu": False,
    },
    "meta_boosting_multiple": {
        "method": "xgboost",
        "objective_metric": mean_squared_error,
        "weighting": "proportional",
        "features": None,
        "n_splits_inner": 2,
        "n_splits_outer": 10,
        "n_trials": 2,
        "ponderation": [0.7, 0.3],
        "use_gpu": False,
    },
}


class BackTester:
    def __init__(self, config=None, **kwargs):
        if config:
            self.config = config
        else:
            self.config = ConfigReader._read_config(
                "../config/backtester.json", BackTesterConfig
            )
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        self.models = {}
        self.data = {}
        self.X = {}
        self.y = {}
        self.feature_names = {}
        self.results = {}
        self.results_in_sample = {}
        self.baseline_results = {}
        self.constant_results = {}
        self.features_after_selection = {}

        # ML Flow
        if self.config.use_mlflow:
            if getattr(self.config, "mlflow_tracking_uri", None):
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            experiment_base = getattr(
                self.config,
                "mlflow_experiment",
                "ElectionBacktests",
            )
            if experiment_base == "ElectionBacktests":
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                mlflow.set_experiment(f"{experiment_base}_{timestamp}")
            else:
                mlflow.set_experiment(experiment_base)

    def _optimize_inner_fold_xgb(
        self,
        X_train_outer: np.ndarray,
        y_train_outer: np.ndarray,
        n_inner_splits: int = 3,
        n_trials: int = 25,
        use_gpu: bool = False,
        early_stopping_rounds: int = 30,
        max_n_estimators: int = 1500,
    ) -> Dict[str, Any]:
        """Inner CV loop using Optuna to tune XGBoost hyperparameters."""
        inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=42)
        device = "cuda" if use_gpu else "cpu"

        def objective(trial):
            params = {
                "n_estimators": max_n_estimators,
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=True
                ),
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

            for in_train_idx, in_val_idx in inner_cv.split(
                X_train_outer, y_train_outer
            ):
                X_in_tr, X_in_val = (
                    X_train_outer[in_train_idx],
                    X_train_outer[in_val_idx],
                )
                y_in_tr, y_in_val = (
                    y_train_outer[in_train_idx],
                    y_train_outer[in_val_idx],
                )

                regressor = xgb.XGBRegressor(**params)
                regressor.fit(
                    X_in_tr,
                    y_in_tr,
                    eval_set=[(X_in_val, y_in_val)],
                    verbose=False,
                )

                preds = regressor.predict(X_in_val)
                fold_losses.append(mean_squared_error(y_in_val, preds))
                best_iterations.append(regressor.best_iteration)

            trial.set_user_attr(
                "mean_best_n_estimators", int(np.mean(best_iterations)) + 1
            )
            return float(np.mean(fold_losses))

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params["n_estimators"] = study.best_trial.user_attrs[
            "mean_best_n_estimators"
        ]
        return best_params

    def _run_nested_cv_xgb(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_outer_splits: int = 5,
        n_inner_splits: int = 3,
        n_trials: int = 25,
        use_gpu: bool = False,
    ) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
        """Executes nested cross-validation and trains the final XGBoost model on full dataset."""
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

        outer_cv = KFold(n_splits=n_outer_splits, shuffle=True, random_state=123)
        outer_scores = []
        device = "cuda" if use_gpu else "cpu"

        logger.info(
            f"Starting Nested CV (device: {device.upper()}): {n_outer_splits} Outer Folds x {n_inner_splits} Inner Folds"
        )

        for outer_fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(X_arr, y_arr)
        ):
            X_tr_out, X_te_out = X_arr[train_idx], X_arr[test_idx]
            y_tr_out, y_te_out = y_arr[train_idx], y_arr[test_idx]

            best_params = self._optimize_inner_fold_xgb(
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

            fold_model = xgb.XGBRegressor(**final_model_params)
            fold_model.fit(X_tr_out, y_tr_out, verbose=False)

            preds = fold_model.predict(X_te_out)
            mse = mean_squared_error(y_te_out, preds)
            mae = mean_absolute_error(y_te_out, preds)
            outer_scores.append({"mse": mse, "mae": mae})

            logger.info(
                f"Outer Fold {outer_fold + 1}/{n_outer_splits} | MSE: {mse:.4f} | MAE: {mae:.4f} | Trees: {best_params['n_estimators']} | Depth: {best_params['max_depth']}"
            )

        mean_mse = float(np.mean([s["mse"] for s in outer_scores]))
        mean_mae = float(np.mean([s["mae"] for s in outer_scores]))
        logger.info(
            f"Nested CV Summary | Mean MSE: {mean_mse:.4f} | Mean MAE: {mean_mae:.4f}"
        )

        # Final fit on all training data using hyperparameter optimization
        logger.info("Tuning on entire dataset for final model deployment...")
        final_best_params = self._optimize_inner_fold_xgb(
            X_arr,
            y_arr,
            n_inner_splits=n_inner_splits,
            n_trials=n_trials,
            use_gpu=use_gpu,
        )
        final_model = xgb.XGBRegressor(
            **final_best_params,
            tree_method="hist",
            device=device,
            eval_metric="rmse",
            random_state=42,
        )
        if not use_gpu:
            final_model.set_params(n_jobs=-1)

        final_model.fit(X_arr, y_arr, verbose=False)
        return final_model, {"nested_cv_mse": mean_mse, "nested_cv_mae": mean_mae}

    def process_and_split_dataset(self, data, k_year, k_political_trends):
        """Split the dataset into training, validation, and test sets."""
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
        for name in container_names:
            setattr(self, name, {})

        self.feature_names = {}

        for trend in k_political_trends:
            values = get_Xy_pl(
                data,
                vote_variable=f"pvote{trend}",
                year=k_year,
                election_type=self.k_type_full,
                predict_delta=self.config.predict_delta,
                predict_perc=self.config.predict_percentile,
                selected_groups=[],
                selected_features=CHAMPION_FEATURES_EXTENDED,
                split_method_way="last-try-seq",
            )

            for name, value in zip(container_names, values):
                getattr(self, name)[trend] = value

            self.feature_names[trend] = self.X_train[trend].columns.tolist()
            logger.debug(
                f"Features used for trend {trend} : {self.feature_names[trend]}"
            )

    def organize_vote(self, k_year, k_type, k_political_trends, model_name):
        election_type = self.k_type_full
        election_type_code = k_type
        ground_truth_data_path = (
            self.config.data_path
            + f"raw/elections/{election_type}/{k_year}/{election_type_code}{k_year}_csv/{election_type_code}{k_year}comm.parquet"
        )
        if "CGCCD" in k_political_trends or "tauCGCCD" in k_political_trends:
            political_trends = ["par", "CG", "C", "G", "D", "CD"]
            X_true = DataLoader.load_dataset(ground_truth_data_path)[
                ["codecommune", "nomcommune", "inscrits", "votants", "exprimes"]
                + [
                    f"vote{t.replace('tau', '')}"
                    for t in political_trends
                    if t.replace("tau", "") != "par"
                ]
                + [
                    f"pvote{t.replace('tau', '')}"
                    for t in political_trends
                    if t.replace("tau", "") != "par"
                ]
                + ["ppar"]
            ]
            X_true = X_true.copy()
            X_true["voteCGCCD"] = X_true["voteCG"] + X_true["voteC"] + X_true["voteCD"]
            X_true["pvoteCGCCD"] = (
                X_true["pvoteCG"] + X_true["pvoteC"] + X_true["pvoteCD"]
            )
        else:
            X_true = DataLoader.load_dataset(ground_truth_data_path)[
                ["codecommune", "nomcommune", "inscrits", "votants", "exprimes"]
                + [
                    f"vote{t.replace('tau', '')}"
                    for t in k_political_trends
                    if t.replace("tau", "") != "par"
                ]
                + [
                    f"pvote{t.replace('tau', '')}"
                    for t in k_political_trends
                    if t.replace("tau", "") != "par"
                ]
                + ["ppar"]
            ]
        X_true = X_true.dropna()
        str_cols = ["codecommune", "nomcommune"]
        float_cols = [
            f"pvote{t.replace('tau', '')}"
            for t in k_political_trends
            if t.replace("tau", "") != "par"
        ] + ["ppar"]
        int_cols = [
            f"vote{t.replace('tau', '')}"
            for t in k_political_trends
            if t.replace("tau", "") != "par"
        ] + ["inscrits", "votants", "exprimes"]
        X_true[str_cols] = X_true[str_cols].astype(str)
        X_true[int_cols] = X_true[int_cols].astype(int)
        X_true[float_cols] = X_true[float_cols].astype(float)
        exprimes_ = X_true[
            [
                f"vote{t.replace('tau', '')}"
                for t in k_political_trends
                if t.replace("tau", "") != "par"
            ]
        ].sum(axis=1)

        X_true = X_true.loc[
            ~(
                X_true["codecommune"].isin(
                    ["75056"]
                    + [f"1320{i}" for i in range(1, 9 + 1)]
                    + [f"132{i}" for i in range(10, 16 + 1)]
                    + [f"69328{i}" for i in range(1, 9 + 1)]
                )
            ),
            :,
        ]

        data = DataLoader.load_dataset(
            self.config.data_path + self.config.dataset_path, engine="polars"
        )
        data_election = (
            data.filter(pl.col("annee") == k_year)
            .filter(pl.col("election_type") == self.k_type_full)
            .to_pandas()
        )

        X_pred = self.election_predictor.predict_votes(
            data_election,
            self.config.predict_delta,
            infer_multiple=(model_name == "meta_boosting_multiple"),
        )

        agg_results = self.election_predictor.predict_votes(
            data_election,
            self.config.predict_delta,
            infer_multiple=(model_name == "meta_boosting_multiple"),
            agg=True,
        )

        agg_results_show = {
            key: value
            for key, value in agg_results.items()
            if key
            in [
                f"tot_pvote{t.replace('tau', '')}"
                for t in k_political_trends
                if t != "par"
            ]
        }
        logger.success(
            f"Total participation predicted {agg_results['tot_ppar'] * 100:.3f}% vs. result {(X_true['votants'].sum() / X_true['inscrits'].sum()) * 100:.3f}%."
        )
        for trend in k_political_trends:
            trend_clean = trend.replace("tau", "")
            if trend_clean == "par":
                continue
            logger.success(
                f"Prediction for {trend_clean}: {agg_results_show[f'tot_pvote{trend_clean}'] * 100:.3f}%. Result for {trend_clean}: {(X_true[f'vote{trend_clean}'].sum() / exprimes_.sum()) * 100:.3f}%."
            )

        return X_pred, X_true

    def add_poll_predictions(
        self, result_synthetic, k_year, k_type, k_political_trends
    ):
        result_synthetic = result_synthetic.copy().set_index("index")
        election_type = "presidentiel" if k_type == "pres" else "legislative"
        poll_data_path = (
            self.config.data_path + f"polls/{election_type}/{k_year}/polls_t1.parquet"
        )
        if DataUtils._exists(
            poll_data_path,
            fs=DataUtils._create_fs() if DataUtils._detect_s3(poll_data_path) else None,
        ):
            if "CGCCD" in k_political_trends or "tauCGCCD" in k_political_trends:
                political_trends = ["par", "CG", "C", "G", "D", "CD"]
                X_poll = DataLoader.load_dataset(poll_data_path)[
                    [
                        t.replace("vote", "").replace("tau", "")
                        for t in political_trends
                        if t.replace("tau", "") != "par"
                    ]
                ].copy()
                X_poll["CGCCD"] = X_poll["CG"] + X_poll["C"] + X_poll["CD"]
            else:
                X_poll = DataLoader.load_dataset(poll_data_path)[
                    [
                        t.replace("vote", "").replace("tau", "")
                        for t in k_political_trends
                        if t.replace("tau", "") != "par"
                    ]
                ]
            poll_results = X_poll.mean()
            for trend in k_political_trends:
                trend_clean = trend.replace("tau", "")
                if trend_clean != "par":
                    result_synthetic.loc[
                        "pvote" + trend_clean, f"{k_year}_{k_type}_poll"
                    ] = round(poll_results[trend_clean], 2)
        result_synthetic["index"] = result_synthetic.index
        return result_synthetic.reset_index(drop=True)

    def _get_output_paths(self):
        if not DataUtils._detect_s3(self.config.data_path):
            path = (
                Path(self.config.data_path) / "output"
                if (self.config.data_path and os.path.isabs(self.config.data_path))
                else Path.cwd() / "output"
            )
            result_dir_path = str(path / "results") + "/"
            model_dir_path = str(path / "models") + "/"
            os.makedirs(result_dir_path, exist_ok=True)
            os.makedirs(model_dir_path, exist_ok=True)
        else:
            base_s3 = self.config.data_path.rstrip("/") + "/"
            result_dir_path = f"{base_s3}output/results/"
            model_dir_path = f"{base_s3}output/models/"
        return result_dir_path, model_dir_path

    def _find_model(self, model_dir_path: str, base_name: str) -> Optional[str]:
        fs = DataUtils._create_fs() if DataUtils._detect_s3(model_dir_path) else None
        candidates = [
            f"{model_dir_path}{base_name}.joblib",
            f"{model_dir_path}{base_name}.pkl",
        ]
        if not DataUtils._detect_s3(model_dir_path):
            data_output = Path(self.config.data_path) / "output" / "models"
            candidates.extend(
                [
                    str(data_output / f"{base_name}.joblib"),
                    str(data_output / f"{base_name}.pkl"),
                ]
            )
        for path_cand in candidates:
            if DataUtils._exists(path_cand, fs=fs):
                return path_cand
        return None

    def save_results(self, model, result, k_year, k_type, k_political_trends):
        result_dir_path, model_dir_path = self._get_output_paths()
        result_all, result_synthetic = result
        result_synthetic = self.add_poll_predictions(
            result_synthetic, k_year, k_type, k_political_trends
        )

        k_political_trends.sort()
        vars_ = "_".join(k_political_trends)

        DataLoader.write_dataset(
            result_all,
            result_dir_path
            + f"results_full_{k_year}_{k_type}_{vars_}_{self.config.version}.parquet",
        )
        DataLoader.write_dataset(
            result_synthetic,
            result_dir_path
            + f"results_synth_{k_year}_{k_type}_{vars_}_{self.config.version}.parquet",
        )
        DataLoader.dump_joblib(
            model,
            model_dir_path
            + f"model_{k_year}_{k_type}_{vars_}_{self.config.version}.joblib",
            compress="lzma",
        )

        if DataUtils._detect_s3(result_dir_path):
            logger.info(f"Results and models saved to S3: {result_dir_path}")
        else:
            logger.info(f"Results and models saved locally: {result_dir_path}")

    def run_backtest(
        self, data, k_year, k_type, k_political_trends, model, model_args, model_name
    ):
        self.k_type_full = "presidentiel" if k_type == "pres" else "legislative"
        k_political_trends.sort()
        vars_ = "_".join(k_political_trends)

        result_dir_path, model_dir_path = self._get_output_paths()
        full_model_base = f"model_{k_year}_{k_type}_{vars_}_{self.config.version}"
        full_model_path = self._find_model(model_dir_path, full_model_base)

        fs = DataUtils._create_fs() if DataUtils._detect_s3(result_dir_path) else None
        s_path = f"{result_dir_path}results_synth_{k_year}_{k_type}_{vars_}_{self.config.version}.parquet"
        f_path = f"{result_dir_path}results_full_{k_year}_{k_type}_{vars_}_{self.config.version}.parquet"
        if full_model_path and (
            DataUtils._exists(s_path, fs=fs) and DataUtils._exists(f_path, fs=fs)
        ):
            loc = "on S3" if DataUtils._detect_s3(result_dir_path) else "locally"
            logger.info(
                f"Model and results for {k_year}_{k_type}_{vars_}_{self.config.version} already exist ({loc}). Skipping."
            )
            return

        with mlf_utils.mlflow_tracker(
            enabled=self.config.use_mlflow, run_name=f"{model_name}_{k_type}_{k_year}"
        ):
            if full_model_path:
                self.election_predictor = DataLoader.load_joblib(full_model_path)
            else:
                self.election_predictor = ElectionPredictor(trends=k_political_trends)
                self.process_and_split_dataset(data, k_year, k_political_trends)

                if self.config.use_mlflow:
                    mlflow.log_params(
                        {
                            "model": model_name,
                            "year": k_year,
                            "election_type": k_type,
                            "version": self.config.version,
                            "trends": ",".join(k_political_trends),
                        }
                    )

                for trend in k_political_trends:
                    trend_base = f"best_model_{model_name}_{k_year}_{k_type}_{trend}_{self.config.version}"
                    trend_model_path = self._find_model(model_dir_path, trend_base)

                    if trend_model_path:
                        instance_model = DataLoader.load_joblib(trend_model_path)
                    else:
                        logger.info(f"Training model for trend: {trend}")

                        if model_name == "boosting":
                            use_gpu = model_args.get(
                                "use_gpu", getattr(self.config, "use_gpu", False)
                            )
                            xgb_model, cv_metrics = self._run_nested_cv_xgb(
                                X=self.X_train[trend],
                                y=self.y_train[trend],
                                n_outer_splits=model_args.get("n_splits_outer", 5),
                                n_inner_splits=model_args.get("n_splits_inner", 3),
                                n_trials=model_args.get("n_trials", 25),
                                use_gpu=use_gpu,
                            )
                            # Wrap into BoostingModel wrapper
                            instance_model = BoostingModel()
                            instance_model.model = xgb_model
                            instance_model.best_models = [xgb_model]
                            instance_model.infer = (
                                lambda X_eval, m=xgb_model: _safe_predict(m, X_eval)
                            )
                            if self.config.use_mlflow:
                                mlflow.log_metrics(
                                    {
                                        f"{trend}_nested_cv_mse": cv_metrics[
                                            "nested_cv_mse"
                                        ]
                                    }
                                )
                        else:
                            if (
                                model_name
                                in ("meta_boosting", "meta_boosting_multiple")
                                and "use_gpu" not in model_args
                            ):
                                model_args["use_gpu"] = getattr(
                                    self.config, "use_gpu", False
                                )
                            instance_model = model(**model_args)
                            if model_name == "trivial_1":
                                instance_model.train(
                                    self.X_train[trend],
                                    self.y_train[trend],
                                    y_prev=self.y_prev[trend],
                                )
                            elif model_name == "meta_boosting":
                                instance_model.train(
                                    self.X_train[trend],
                                    self.y_train[trend],
                                    val_set=(self.X_val[trend], self.y_val[trend]),
                                )
                            elif model_name == "meta_boosting_multiple":
                                instance_model.train_multiple(
                                    election_datasets=[
                                        (self.X_train[trend], self.y_train[trend]),
                                        (self.X_val[trend], self.y_val[trend]),
                                    ]
                                )
                            else:
                                instance_model.train(
                                    self.X_train[trend], self.y_train[trend]
                                )

                        save_trend_file = f"{model_dir_path}{trend_base}.joblib"
                        DataLoader.dump_joblib(
                            instance_model, save_trend_file, compress="lzma"
                        )

                    predictions = (
                        instance_model.infer_multiple(self.X_test[trend])
                        if model_name == "meta_boosting_multiple"
                        else instance_model.infer(self.X_test[trend])
                    )
                    predictions_in_sample = (
                        instance_model.infer_multiple(self.X_train[trend])
                        if model_name == "meta_boosting_multiple"
                        else instance_model.infer(self.X_train[trend])
                    )

                    self.results[model_name] = ModelEvaluator.evaluate(
                        self.y_test[trend], predictions, model_name, extended=True
                    )
                    self.results_in_sample[model_name] = ModelEvaluator.evaluate(
                        self.y_train[trend],
                        predictions_in_sample,
                        model_name,
                        extended=True,
                    )
                    self.baseline_results[model_name] = ModelEvaluator.evaluate(
                        self.y_test[trend],
                        self.y_prev[trend].fillna(self.y_prev[trend].mean()),
                        model_name,
                        extended=True,
                    )
                    self.constant_results[model_name] = ModelEvaluator.evaluate(
                        self.y_test[trend],
                        self.y_test[trend] * 0.0 + self.y_train[trend].mean(),
                        model_name,
                        extended=False,
                    )

                    if self.config.use_mlflow:
                        logging_metric_dict = {
                            "ML": self.results,
                            "in_sample": self.results_in_sample,
                            "previous_baseline": self.baseline_results,
                            "constant_basline": self.constant_results,
                        }
                        for suffix, results in logging_metric_dict.items():
                            mlf_utils._log_numeric_metrics(
                                trend=trend,
                                values=results[model_name],
                                model_name=model_name,
                                suffix=suffix,
                            )
                        parts = [
                            self.meta_test[trend].reset_index(drop=True),
                            self.y_test[trend].reset_index(drop=True).rename("y_true"),
                            self.y_prev[trend].reset_index(drop=True).rename("y_prev"),
                            pd.Series(np.asarray(np.ravel(predictions)), name="y_pred")
                            .reset_index(drop=True)
                            .rename("y_pred"),
                        ]

                        out = pd.concat(parts, axis=1)
                        with tempfile.TemporaryDirectory() as tmpdir:
                            csv_path = os.path.join(tmpdir, f"predictions_{trend}.csv")
                            out.to_csv(csv_path, index=False)
                            mlflow.log_artifact(csv_path, artifact_path="predictions")

                        # Log feature list
                        mlflow.log_param(
                            f"{trend}_n_features", len(self.feature_names[trend])
                        )
                        mlflow.log_dict(
                            {
                                "trend": trend,
                                "feature_names": self.feature_names[trend],
                            },
                            f"features/{trend}_feature_names.json",
                        )

                        # Log params and feature importance of all boosters
                        if model_name == "boosting":
                            instance_model.best_models = [instance_model.model]

                        if (model_name == "meta_boosting") or (
                            model_name == "boosting"
                        ):
                            importance_types = [
                                "weight",
                                "gain",
                                "cover",
                                "total_gain",
                                "total_cover",
                            ]
                            all_models_importance = {}
                            all_models_params = {}

                            with tempfile.TemporaryDirectory() as tmpdir:
                                for model_idx, boosting_model in enumerate(
                                    instance_model.best_models
                                ):
                                    model_key = f"model_{model_idx}"

                                    model_importance, model_params = (
                                        mlf_utils._collect_model_importance_and_params(
                                            boosting_model=boosting_model,
                                            feature_names=self.feature_names[trend],
                                            importance_types=importance_types,
                                        )
                                    )

                                    all_models_importance[model_key] = model_importance
                                    all_models_params[model_key] = model_params

                                    mlf_utils._log_scalar_params_to_mlflow(
                                        prefix=f"{trend}_{model_key}",
                                        params=model_params["xgb_params"],
                                    )

                                    plot_path = mlf_utils._plot_importance_types(
                                        model_key=model_key,
                                        trend=trend,
                                        model_importance=model_importance,
                                        importance_types=importance_types,
                                        out_dir=tmpdir,
                                        top_k=10,
                                    )

                                    mlflow.log_artifact(
                                        str(plot_path),
                                        artifact_path=f"feature_importance/plots/{trend}",
                                    )

                                    importance_df = pd.DataFrame(
                                        model_importance
                                    ).reset_index()
                                    importance_path = (
                                        Path(tmpdir)
                                        / f"{trend}_{model_key}_importance.csv"
                                    )
                                    importance_df.to_csv(importance_path, index=False)

                                    mlflow.log_artifact(
                                        str(importance_path),
                                        artifact_path=f"feature_importance/data/{trend}",
                                    )

                    self.election_predictor.add_model(
                        trend.replace("tau", ""),
                        instance_model,
                        features=self.feature_names[trend],
                    )
                    self.election_predictor.sign_model(
                        trend,
                        self.config.data_path + self.config.dataset_path,
                        sample=self.X_train[trend].sample(5),
                    )

            if self.config.organize_vote:
                X_pred, X_true = self.organize_vote(
                    k_year, k_type, k_political_trends, model_name
                )
                X_result = self.election_predictor.evaluate_predictions(X_pred, X_true)
                X_synthetic = self.election_predictor.compute_agg_results(
                    X_result,
                    blocs=[
                        t.replace("tau", "")
                        for t in k_political_trends
                        if t.replace("tau", "") != "par"
                    ],
                    election_code=f"{k_year}_{k_type}",
                )
                self.save_results(
                    model=self.election_predictor,
                    result=(X_result, X_synthetic),
                    k_year=k_year,
                    k_type=k_type,
                    k_political_trends=k_political_trends,
                )

                # Log results file
                if self.config.use_mlflow:
                    result_path = Path.cwd() / "mlflow_results"
                    result_path.mkdir(exist_ok=True)

                    synthetic_file = result_path / f"synthetic_{k_year}_{k_type}.csv"
                    detailed_file = result_path / f"detailed_{k_year}_{k_type}.csv"

                    X_synthetic.to_csv(
                        synthetic_file,
                        index=False,
                    )
                    X_result.to_csv(
                        detailed_file,
                        index=False,
                    )

                    mlflow.log_artifact(str(synthetic_file))
                    mlflow.log_artifact(str(detailed_file))

    def run(self):
        models = self.config.models
        k_years = self.config.k_year
        k_types = self.config.k_type
        k_political_trends = self.config.political_trends

        data = DataLoader.load_dataset(
            self.config.data_path + self.config.dataset_path,
            engine="polars",
            hive_partitioning=True,
        )
        for model_name in models:
            model = MODELS[model_name]
            model_args = copy.deepcopy(MODEL_ARGS[model_name])
            if "use_gpu" in model_args or model_name in (
                "boosting",
                "meta_boosting",
                "meta_boosting_multiple",
            ):
                model_args["use_gpu"] = getattr(self.config, "use_gpu", False)

            for political_trends in k_political_trends:
                for type_ in k_types:
                    for year in k_years[type_]:
                        self.run_backtest(
                            data=data,
                            k_year=year,
                            k_type=type_,
                            k_political_trends=political_trends,
                            model=model,
                            model_args=model_args,
                            model_name=model_name,
                        )


if __name__ == "__main__":
    BackTester().run()
