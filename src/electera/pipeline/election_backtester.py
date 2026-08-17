"""
Election Backtester
=========================================

# Implement a backtesting logic.

# We train models over presidential election and legislative election (1er tour).

# The model is trained on all elections. One election is excluded from testing and taken for test.
# Hyperparameters are optimized on the whole training dataset


# The model works as followed : it consist of five model (one for each political trend)
# For a given election, for each commune
# and for each political trend the model outputs a prediction.
# We gather all predictions and adapt them so that they match 100%
# We then give the voice of the commune to the political trend according to the predictions
# Based on this we compute the final result and compare it with the actual results
"""

import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import electera.components.mlflow.mlflow_utils as mlf_utils
from assets.delta_pred_features import (
    BASE_FEATURES,
    # CHAMPION_FEATURES,
    CHAMPION_FEATURES_EXTENDED,
)
from electera.components.data_processing.data_loader import DataLoader, DataUtils
from electera.components.modelling.benchmark_models import (
    LinearModel,
    TrivialModel1,
    TrivialModel2,
)
from electera.components.modelling.boosting.boosting import BoostingModel

# from electera.components.modelling.benchmark_models import TrivialModel2
from electera.components.modelling.data_split_pl import get_Xy_pl
from electera.components.modelling.election_predictor import ElectionPredictor
from electera.components.modelling.evaluation import ModelEvaluator
from electera.components.modelling.meta_booster import (
    MetaBooster,
    MetaBoosterMultipleElections,
)
from electera.components.utils.config import BackTesterConfig
from electera.components.utils.read_config import ConfigReader

# TODO:
# - Modèle pour les votes blancs? Fix: Predire pexpr plutôt que ppar.
# FEATURES = list(set(make_features("raw")))


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
        "parameters": {
            "subsample": 0.8,
            "min_child_weight": 1,
            "n_estimators": 5000,
            "max_depth": 8,
            "objective": "reg:squarederror",
            "colsample_bytree": 0.8,
            "colsample_bylevel": 0.8,
            "colsample_bynode": 0.8,
            "learning_rate": 0.005,
            "gamma": 1,
            "alpha": 1,
            "lambda": 1,
            "early_stopping_rounds": 25,
        }
    },
    "meta_boosting": {
        "method": "xgboost",
        "objective_metric": mean_squared_error,
        "weighting": "sqrt",
        "features": BASE_FEATURES,
        "n_splits_inner": 2,
        "n_splits_outer": 1,
        "n_trials": 1,
        "poll_adj": True,
    },
    "meta_boosting_multiple": {
        "method": "xgboost",
        "objective_metric": mean_squared_error,
        "weighting": "proportional",
        "features": BASE_FEATURES,
        "n_splits_inner": 2,
        "n_splits_outer": 10,
        "n_trials": 2,
        "ponderation": [0.7, 0.3],
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
            mlflow.set_tracking_uri(
                "https://user-arthurmanceau-mlflow.user.lab.sspcloud.fr"
            )
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

    def process_and_split_dataset(self, data, k_year, k_political_trends):
        """
        Split the dataset into training, validation, and test sets.
        """
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

        # Reset feature names for this run
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
        # Simulate a production setting

        # Ground truth - reload
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
                    f"vote{trend.replace('tau', '')}"
                    for trend in political_trends
                    if trend.replace("tau", "") != "par"
                ]
                + [
                    f"pvote{trend.replace('tau', '')}"
                    for trend in political_trends
                    if trend.replace("tau", "") != "par"
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
                    f"vote{trend.replace('tau', '')}"
                    for trend in k_political_trends
                    if trend.replace("tau", "") != "par"
                ]
                + [
                    f"pvote{trend.replace('tau', '')}"
                    for trend in k_political_trends
                    if trend.replace("tau", "") != "par"
                ]
                + ["ppar"]
            ]
        X_true = X_true.dropna()
        str_cols = ["codecommune", "nomcommune"]
        float_cols = [
            f"pvote{trend.replace('tau', '')}"
            for trend in k_political_trends
            if trend.replace("tau", "") != "par"
        ] + ["ppar"]
        int_cols = [
            f"vote{trend.replace('tau', '')}"
            for trend in k_political_trends
            if trend.replace("tau", "") != "par"
        ] + [
            "inscrits",
            "votants",
            "exprimes",
        ]
        X_true[str_cols] = X_true[str_cols].astype(str)
        X_true[int_cols] = X_true[int_cols].astype(int)
        X_true[float_cols] = X_true[float_cols].astype(float)
        exprimes_ = X_true[
            [
                f"vote{trend.replace('tau', '')}"
                for trend in k_political_trends
                if trend.replace("tau", "") != "par"
            ]
        ].sum(axis=1)

        # Remove Paris and Lyon, Marseille arrondissement (avoid duplicates)
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

        # Predictions
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

        logger.info(
            f"We computed predictions for all the communes that were in the raw result data, except: {list(set(X_true['codecommune'].to_list()) - set(data_election['codecommune'].to_list()))}"
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
                f"tot_pvote{trend.replace('tau', '')}"
                for trend in k_political_trends
                if trend != "par"
            ]
        }
        logger.success(
            f"Total participation predicted {agg_results['tot_ppar'] * 100:.3f}% vs. result {(X_true['votants'].sum() / X_true['inscrits'].sum()) * 100:.3f}%. Diff: {np.abs(agg_results['tot_ppar'] - (X_true['votants'].sum() / X_true['inscrits'].sum())) * 100:.3f}%"
        )
        for trend in k_political_trends:
            trend = trend.replace("tau", "")
            if trend == "par":
                continue
            logger.success(
                f"Prediction for {trend}: {agg_results_show[f'tot_pvote{trend}'] * 100:.3f}%. Result for {trend}:  {(X_true[f'vote{trend}'].sum() / exprimes_.sum()) * 100:.3f}%. Diff: {np.abs(agg_results_show[f'tot_pvote{trend}'] - (X_true[f'vote{trend}'].sum() / exprimes_.sum())) * 100:.3f}%"
            )

        return X_pred, X_true

    def add_poll_predictions(
        self, result_synthetic, k_year, k_type, k_political_trends
    ):
        # Try adding polling results if possible
        result_synthetic = result_synthetic.copy()
        result_synthetic = result_synthetic.set_index("index")
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
                        trend.replace("vote", "").replace("tau", "")
                        for trend in political_trends
                        if trend.replace("tau", "") != "par"
                    ]
                ]
                X_poll = X_poll.copy()
                X_poll["CGCCD"] = X_poll["CG"] + X_poll["C"] + X_poll["CD"]
            else:
                X_poll = DataLoader.load_dataset(poll_data_path)[
                    [
                        trend.replace("vote", "").replace("tau", "")
                        for trend in k_political_trends
                        if trend.replace("tau", "") != "par"
                    ]
                ]
            poll_results = X_poll.mean()  # Could be a better formula to aggregate polls
            for trend in k_political_trends:
                trend = trend.replace("tau", "")
                if trend != "par":
                    result_synthetic.loc["pvote" + trend, f"{k_year}_{k_type}_poll"] = (
                        round(poll_results[trend], 2)
                    )
        else:
            logger.warning("No poll data for this election, skipping")

        result_synthetic["index"] = result_synthetic.index
        result_synthetic.reset_index(drop=True)
        return result_synthetic

    def save_results(self, model, result, k_year, k_type, k_political_trends):
        # Create directories
        if not DataUtils._detect_s3(self.config.data_path):
            path = Path.cwd() / "output/"
            result_dir_path = str(path) + "/results/"
            model_dir_path = str(path) + "/models/"
            os.makedirs(result_dir_path, exist_ok=True)
            os.makedirs(model_dir_path, exist_ok=True)
            logger.info(f"Output saved locally: {path}")
        else:
            result_dir_path = self.config.data_path + "output/" + "results/"
            model_dir_path = self.config.data_path + "output/" + "models/"
            logger.info(f"Output saved to S3: {self.config.data_path + 'output/'}")

        # Post-treatment
        result_all, result_synthetic = result
        result_synthetic = self.add_poll_predictions(
            result_synthetic, k_year, k_type, k_political_trends
        )

        # Alphabetic sort
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
        DataLoader.dump_pickle(
            object_to_pickle=model,
            file_path=model_dir_path
            + f"model_{k_year}_{k_type}_{vars_}_{self.config.version}.pkl",
        )

    def run_backtest(
        self, data, k_year, k_type, k_political_trends, model, model_args, model_name
    ):
        """
        Run the backtesting process.
        """
        self.k_type_full = "presidentiel" if k_type == "pres" else "legislative"

        # optional mlflow tracker
        with mlf_utils.mlflow_tracker(
            enabled=self.config.use_mlflow, run_name=f"{model_name}_{k_type}_{k_year}"
        ):
            # For now only one backtesting model
            self.election_predictor = ElectionPredictor(trends=k_political_trends)

            # 2. Test and split
            self.process_and_split_dataset(data, k_year, k_political_trends)

            # Log parameters of the run
            if self.config.use_mlflow:
                mlflow.log_params(
                    {
                        "model": model_name,
                        "year": k_year,
                        "election_type": k_type,
                        "predict_delta": self.config.predict_delta,
                        "predict_percentile": self.config.predict_percentile,
                        "version": self.config.version,
                        "trends": ",".join(k_political_trends),
                    }
                )

            # 3. Train model
            for trend in k_political_trends:
                logger.info(f"Training model for trend: {trend}")

                # For trivial model (same as previous election)
                if model_name == "trivial_1":
                    model_args["y_prev"] = self.y_prev[trend]

                # Adding a base score
                # if model_name == 'boosting':
                #     logger.debug(f"Base score: {self.y_train[trend].mean()} ({trend})")
                #     model_args['parameters']['base_score'] = self.y_train[trend].mean()

                instance_model = model(**model_args)

                trainings = {
                    "trivial_1": lambda: instance_model.train(
                        self.X_train[trend], self.y_train[trend]
                    ),
                    "trivial_2": lambda: instance_model.train(
                        self.X_train[trend], self.y_train[trend]
                    ),
                    "boosting": lambda: instance_model.train(
                        self.X_train[trend],
                        self.y_train[trend],
                        self.X_val[trend],
                        self.y_val[trend],
                        weighting="proportional",
                        feature_selection_method="none",
                        nb_features="relative",
                        param_search_method="optuna",
                        meta_train=self.meta_train[trend],
                    ),
                    "linear": lambda: instance_model.train(
                        self.X_train[trend], self.y_train[trend]
                    ),
                    "meta_boosting": lambda: instance_model.train(
                        self.X_train[trend],
                        self.y_train[trend],
                        use_feature_selection=False,
                        val_set=(self.X_val[trend], self.y_val[trend]),
                    ),
                    "meta_boosting_multiple": lambda: instance_model.train_multiple(
                        election_datasets=[
                            (self.X_train[trend], self.y_train[trend]),
                            (self.X_val[trend], self.y_val[trend]),
                        ],
                        use_feature_selection=True,
                    ),
                }

                # Log model hyperparameters
                if self.config.use_mlflow:
                    mlf_utils._log_scalar_params_to_mlflow(
                        prefix=f"{trend}_{model_name}_global_params",
                        params=model_args,
                    )

                trainings[model_name]()

                # Evaluate the model on the test election
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

                logger.info("Predictions evaluation (ML)")
                self.results[model_name] = ModelEvaluator.evaluate(
                    self.y_test[trend], predictions, model_name, extended=True
                )
                logger.info("Predictions (in-sample)")
                self.results_in_sample[model_name] = ModelEvaluator.evaluate(
                    self.y_train[trend],
                    predictions_in_sample,
                    model_name,
                    extended=True,
                )
                logger.info(
                    "Baseline predictions (same as previous election of the same type)"
                )
                self.baseline_results[model_name] = ModelEvaluator.evaluate(
                    self.y_test[trend],
                    self.y_prev[trend].fillna(self.y_prev[trend].mean()),
                    model_name,
                    extended=True,
                )
                logger.info("Baseline predictions (constant)")
                # Adjust to the problem
                self.constant_results[model_name] = ModelEvaluator.evaluate(
                    self.y_test[trend],
                    self.y_test[trend] * 0.0 + self.y_train[trend].mean(),
                    model_name,
                    extended=False,
                )

                # Log metric
                if self.config.use_mlflow:
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

                    mlf_utils._log_numeric_metrics(
                        trend=trend,
                        values=self.results[model_name],
                        model_name=model_name,
                        suffix="ML",
                    )
                    mlf_utils._log_numeric_metrics(
                        trend=trend,
                        values=self.results_in_sample[model_name],
                        model_name=model_name,
                        suffix="in_sample",
                    )
                    mlf_utils._log_numeric_metrics(
                        trend=trend,
                        values=self.baseline_results[model_name],
                        model_name=model_name,
                        suffix="baseline_previous",
                    )
                    mlf_utils._log_numeric_metrics(
                        trend=trend,
                        values=self.constant_results[model_name],
                        model_name=model_name,
                        suffix="baseline_random",
                    )

                    # Log feature list
                    mlflow.log_param(
                        f"{trend}_n_features", len(self.feature_names[trend])
                    )
                    mlflow.log_dict(
                        {"trend": trend, "feature_names": self.feature_names[trend]},
                        f"features/{trend}_feature_names.json",
                    )

                    # Log params and feature importance of all boosters
                    if model_name == "boosting":
                        instance_model.best_models = [instance_model.model]

                    if (model_name == "meta_boosting") or (model_name == "boosting"):
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
                                    Path(tmpdir) / f"{trend}_{model_key}_importance.csv"
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

            if self.config.use_mlflow:
                with tempfile.TemporaryDirectory() as tmpdir:
                    pickle_path = f"{tmpdir}/model.pkl"
                    with open(pickle_path, "wb") as f:
                        pickle.dump(
                            self.election_predictor, f, protocol=pickle.HIGHEST_PROTOCOL
                        )
                    mlflow.log_artifact(pickle_path, artifact_path="model")

            if self.config.organize_vote:
                # 4. Predict
                X_pred, X_true = self.organize_vote(
                    k_year, k_type, k_political_trends, model_name
                )

                # 5. Evaluate vote
                X_result = self.election_predictor.evaluate_predictions(X_pred, X_true)
                X_synthetic = self.election_predictor.compute_agg_results(
                    X_result,
                    blocs=[
                        trend.replace("tau", "")
                        for trend in k_political_trends
                        if trend.replace("tau", "") != "par"
                    ],
                    election_code=f"{k_year}_{k_type}",
                )
                # Log into mlflow aggregated metrics
                for trend in k_political_trends:
                    synthetic_log_mlflow = (
                        X_synthetic[
                            X_synthetic["index"] == f"pvote{trend.replace('tau', '')}"
                        ]
                        .iloc[0]
                        .to_dict()
                    )
                    clean_synthetic_log_mlflow = {
                        k.split("_", 2)[-1] if k.count("_") >= 2 else k: v
                        for k, v in synthetic_log_mlflow.items()
                    }

                    mlf_utils._log_numeric_metrics(
                        trend=trend,
                        values=clean_synthetic_log_mlflow,
                        model_name=model_name,
                        suffix="",
                    )

                winner_pred = self.election_predictor.get_winner(
                    X_pred, self.k_type_full
                )
                winner_true = self.election_predictor.get_winner(
                    X_true, self.k_type_full
                )
                logger.success(
                    f"Winner predicted : {winner_pred} | Winner true : {winner_true}"
                )

                # 6. Save results (S3 - for app)
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
        """Runs the full backtesting pipeline."""
        models = self.config.models
        k_years = self.config.k_year
        k_types = self.config.k_type
        k_political_trends = self.config.political_trends

        # 1. Load all dataset
        data = DataLoader.load_dataset(
            self.config.data_path + self.config.dataset_path,
            engine="polars",
            hive_partitioning=True,
        )
        for model_name in models:
            logger.info(f"Model: {model_name}")
            model, model_args = (
                MODELS[model_name],
                MODEL_ARGS[model_name],
            )
            for political_trends in k_political_trends:
                for type_ in k_types:
                    for year in k_years[type_]:
                        logger.info(
                            f"Running backtest for year: {year}, type: {type_}, political_trends: {political_trends}"
                        )
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
