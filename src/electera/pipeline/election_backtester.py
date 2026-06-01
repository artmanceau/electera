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

import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from electera.components.data_processing.data_loader import DataLoader, DataUtils
from electera.components.modelling.benchmark_models import (
    LinearModel,
    TrivialModel1,
    TrivialModel2,
)
from electera.components.modelling.boosting.boosting import BoostingModel

# from electera.components.modelling.benchmark_models import TrivialModel2
from electera.components.modelling.data_split import Splitter
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

MLFOW_TRACKING = True
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
    "boosting": {},
    "meta_boosting": {
        "method": "xgboost",
        "objective_metric": mean_squared_error,
        "weighting": "equiproportional",
        "features": None,
        "n_splits_inner": 2,
        "n_splits_outer": 2,
        "n_trials": 2,
        "poll_adj": False,
    },
    "meta_boosting_multiple": {
        "method": "xgboost",
        "objective_metric": mean_absolute_error,
        "weighting": "proportional",
        "features": None,
        "n_splits_inner": 2,
        "n_splits_outer": 2,
        "n_trials": 2,
        "ponderation": [0.7, 0.3],
    },
}


class BackTester:
    def __init__(self):
        """ """
        self.config = ConfigReader._read_config(
            "../config/backtester.json", BackTesterConfig
        )
        self.models = {}
        self.data = {}
        self.X = {}
        self.y = {}
        self.feature_names = {}
        self.results = {}
        self.features_after_selection = {}

        # ML Flow
        if MLFOW_TRACKING:
            mlflow.set_tracking_uri(
                "https://user-arthurmanceau-mlflow.user.lab.sspcloud.fr"
            )
            mlflow.set_experiment(
                getattr(
                    self.config,
                    "mlflow_experiment",
                    "ElectionBacktests",
                )
            )

    def process_and_split_dataset(
        self, data, k_year, k_political_trends, predict_delta
    ):
        """
        Split the dataset into training, validation, and test sets.
        """
        # From this with will retrieve in the each dataset (one for political trend)
        self.X_train = {trend: [] for trend in k_political_trends}
        self.X_val = {trend: [] for trend in k_political_trends}
        self.X_test = {trend: [] for trend in k_political_trends}
        self.y_train = {trend: [] for trend in k_political_trends}
        self.y_val = {trend: [] for trend in k_political_trends}
        self.y_test = {trend: [] for trend in k_political_trends}

        # Retrieve the rows matching the years for each dataset
        for trend in k_political_trends:
            st = Splitter(trend)
            split_method = f"{k_year}_{self.k_t}"
            X, y, y_split = st.get_Xy(data, predict_delta=predict_delta)
            (
                self.X_train[trend],
                self.X_val[trend],
                self.X_test[trend],
                self.y_train[trend],
                self.y_val[trend],
                self.y_test[trend],
            ) = st.split(
                X,
                y_split,
                split_method=split_method,
            )
            (self.X_train[trend], self.X_val[trend], self.X_test[trend]) = (
                st.clean_features_list(
                    self.X_train[trend], self.X_val[trend], self.X_test[trend]
                )
            )
            self.feature_names[trend] = self.X_train[trend].columns.tolist()

    def organize_vote(self, k_year, k_political_trends, predict_delta):
        # Ground truth
        election_type = "presidentiel" if self.k_t == 0 else "legislative"
        election_type_code = "pres" if self.k_t == 0 else "leg"
        ground_truth_data_path = (
            self.config.data_path
            + f"raw/elections/{election_type}/{k_year}/{election_type_code}{k_year}_csv/{election_type_code}{k_year}comm.parquet"
        )
        X_true = DataLoader.load_dataset(ground_truth_data_path)[
            ["codecommune", "nomcommune", "inscrits", "votants", "exprimes"]
            + [f"vote{trend}" for trend in k_political_trends if trend != "par"]
            + [f"pvote{trend}" for trend in k_political_trends if trend != "par"]
            + ["ppar"]
        ]
        X_true = X_true.dropna()
        str_cols = ["codecommune", "nomcommune"]
        float_cols = [
            f"pvote{trend}" for trend in k_political_trends if trend != "par"
        ] + ["ppar"]
        int_cols = [
            f"vote{trend}" for trend in k_political_trends if trend != "par"
        ] + [
            "inscrits",
            "votants",
            "exprimes",
        ]
        X_true[str_cols] = X_true[str_cols].astype(str)
        X_true[int_cols] = X_true[int_cols].astype(int)
        X_true[float_cols] = X_true[float_cols].astype(float)
        exprimes_ = X_true[
            [f"vote{trend}" for trend in k_political_trends if trend != "par"]
        ].sum(axis=1)

        # Predictions
        data = DataLoader.load_dataset(self.config.data_path + self.config.dataset_path)
        data_election = data[
            (data["annee"].astype(int) == int(k_year))
            & (data["type"].astype(int) == int(self.k_t))
        ]
        X_pred = self.election_predictor.predict_votes(
            data_election,
            predict_delta,
            infer_multiple=(self.config.model == "meta_boosting_multiple"),
        )

        logger.info(
            f"We computed predictions for all the communes that were in the raw result data, except: {list(set(X_true['codecommune'].to_list()) - set(data_election['codecommune'].to_list()))}"
        )

        agg_results = self.election_predictor.predict_votes(
            data_election,
            predict_delta,
            infer_multiple=(self.config.model == "meta_boosting_multiple"),
            agg=True,
        )
        agg_results_show = {
            key: value
            for key, value in agg_results.items()
            if key
            in [f"tot_pvote{trend}" for trend in k_political_trends if trend != "par"]
        }
        logger.success(
            f"Total participation predicted {agg_results['tot_ppar'] * 100:.3f}% vs. result {(X_true['votants'].sum() / X_true['inscrits'].sum()) * 100:.3f}%. Diff: {np.abs(agg_results['tot_ppar'] - (X_true['votants'].sum() / X_true['inscrits'].sum())) * 100:.3f}%"
        )
        for trend in k_political_trends:
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
            X_poll = DataLoader.load_dataset(poll_data_path)[
                [
                    trend.replace("vote", "")
                    for trend in k_political_trends
                    if trend != "par"
                ]
            ]
            poll_results = X_poll.mean()  # Could be a better formula to aggregate polls
            for trend in k_political_trends:
                if trend != "par":
                    letter = trend.replace("vote", "")
                    result_synthetic.loc["p" + trend, f"{k_year}_{k_type}_poll"] = (
                        round(poll_results[letter], 2)
                    )
        else:
            logger.warning("No poll data for this election, skipping")

        result_synthetic["index"] = result_synthetic.index
        result_synthetic.reset_index(drop=True)

        return result_synthetic

    def save_results(self, model, result, k_year, k_type, k_political_trends, version):
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
        vars_ = k_political_trends

        DataLoader.write_dataset(
            result_all,
            result_dir_path
            + f"results_full_{k_year}_{k_type}_{vars_}_{version}.parquet",
        )
        DataLoader.write_dataset(
            result_synthetic,
            result_dir_path
            + f"results_synth_{k_year}_{k_type}_{vars_}_{version}.parquet",
        )
        DataLoader.dump_pickle(
            object_to_pickle=model,
            file_path=model_dir_path + f"model_{k_year}_{k_type}_{vars_}_{version}.pkl",
        )

    def run_backtest(
        self,
        k_year,
        k_type,
        k_political_trends,
        model,
        model_args,
        predict_delta,
        version,
    ):
        """
        Run the backtesting process.
        """
        self.k_t = 0 if k_type == "pres" else 1

        # For now only one backtesting model
        self.election_predictor = ElectionPredictor(trends=k_political_trends)

        # 1. Load all dataset
        data = DataLoader.load_dataset(self.config.data_path + self.config.dataset_path)

        # 2. Test and split
        self.process_and_split_dataset(data, k_year, k_political_trends, predict_delta)

        # 3. Train model
        for trend in k_political_trends:
            logger.info(f"Training model for trend: {trend}")

            # For trivial model (same as previous election)
            if self.config.model == "trivial_1":
                model_args["y_prev"] = self.X_test[trend][f"previouspvote{trend}"]

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
                ),
                "linear": lambda: instance_model.train(
                    self.X_train[trend], self.y_train[trend]
                ),
                "meta_boosting": lambda: instance_model.train(
                    self.X_train[trend],
                    self.y_train[trend],
                    use_feature_selection=True,
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

            trainings[self.config.model]()

            # Evaluate the model on the test election
            predictions = (
                instance_model.infer_multiple(self.X_test[trend])
                if self.config.model == "meta_boosting_multiple"
                else instance_model.infer(self.X_test[trend])
            )

            self.results[self.config.model] = ModelEvaluator.evaluate(
                self.y_test[trend],
                predictions,
                self.config.model,
            )

            self.election_predictor.add_model(
                trend, instance_model, features=self.feature_names[trend]
            )
            self.election_predictor.sign_model(
                trend,
                self.config.data_path + self.config.dataset_path,
                sample=self.X_train[trend].sample(5),
            )

        # 4. Predict
        X_pred, X_true = self.organize_vote(k_year, k_political_trends, predict_delta)

        # 5. Evaluate vote
        X_result = self.election_predictor.evaluate_predictions(X_pred, X_true)
        X_synthetic = self.election_predictor.compute_agg_results(
            X_result,
            blocs=[trend for trend in k_political_trends if trend != "par"],
            election_code=f"{k_year}_{k_type}",
        )

        winner_pred = self.election_predictor.get_winner(X_pred, self.k_t)
        winner_true = self.election_predictor.get_winner(X_true, self.k_t)
        logger.success(
            f"Winner predicted : {winner_pred} | Winner true : {winner_true}"
        )

        # 6. Save results
        self.save_results(
            model=self.election_predictor,
            result=(X_result, X_synthetic),
            k_year=k_year,
            k_type=k_type,
            k_political_trends=k_political_trends,
            version=version,
        )

    def run_backtest_with_MLFLOW(
        self,
        k_year,
        k_type,
        k_political_trends,
        model,
        model_args,
        predict_delta,
        version,
    ):
        """
        Run the backtesting process.
        """
        self.k_t = 0 if k_type == "pres" else 1

        run_name = f"{self.config.model}_{k_type}_{k_year}"

        with mlflow.start_run(run_name=run_name):
            # ==========================================================
            # RUN PARAMETERS
            # ==========================================================
            mlflow.log_params(
                {
                    "model": self.config.model,
                    "year": k_year,
                    "election_type": k_type,
                    "predict_delta": predict_delta,
                    "version": version,
                    "trends": ",".join(k_political_trends),
                }
            )

            # Log model hyperparameters
            for k, v in model_args.items():
                if isinstance(v, (str, int, float, bool)):
                    mlflow.log_param(k, v)

            self.election_predictor = ElectionPredictor(trends=k_political_trends)

            # ==========================================================
            # LOAD DATA
            # ==========================================================
            data = DataLoader.load_dataset(
                self.config.data_path + self.config.dataset_path
            )

            self.process_and_split_dataset(
                data,
                k_year,
                k_political_trends,
                predict_delta,
            )

            # ==========================================================
            # TRAIN ONE MODEL PER TREND
            # ==========================================================
            for trend in k_political_trends:
                logger.info(f"Training model for trend: {trend}")

                if self.config.model == "trivial_1":
                    model_args["y_prev"] = self.X_test[trend][f"previouspvote{trend}"]

                instance_model = model(**model_args)

                trainings = {
                    "trivial_1": lambda: instance_model.train(
                        self.X_train[trend],
                        self.y_train[trend],
                    ),
                    "trivial_2": lambda: instance_model.train(
                        self.X_train[trend],
                        self.y_train[trend],
                    ),
                    "boosting": lambda: instance_model.train(
                        self.X_train[trend],
                        self.y_train[trend],
                        self.X_val[trend],
                        self.y_val[trend],
                    ),
                    "linear": lambda: instance_model.train(
                        self.X_train[trend],
                        self.y_train[trend],
                    ),
                    "meta_boosting": lambda: instance_model.train(
                        self.X_train[trend],
                        self.y_train[trend],
                        use_feature_selection=True,
                        val_set=(
                            self.X_val[trend],
                            self.y_val[trend],
                        ),
                    ),
                    "meta_boosting_multiple": lambda: instance_model.train_multiple(
                        election_datasets=[
                            (
                                self.X_train[trend],
                                self.y_train[trend],
                            ),
                            (
                                self.X_val[trend],
                                self.y_val[trend],
                            ),
                        ],
                        use_feature_selection=True,
                    ),
                }

                trainings[self.config.model]()

                predictions = (
                    instance_model.infer_multiple(self.X_test[trend])
                    if self.config.model == "meta_boosting_multiple"
                    else instance_model.infer(self.X_test[trend])
                )

                metrics = ModelEvaluator.evaluate(
                    self.y_test[trend],
                    predictions,
                    self.config.model,
                )

                self.results[trend] = metrics

                # ======================================================
                # LOG ARTEFACTS
                # ======================================================
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        if isinstance(
                            metric_value, (int, float, np.integer, np.floating)
                        ):
                            mlflow.log_metric(
                                f"{trend}_{metric_name}", float(metric_value)
                            )

                # -------- Feature metadata as params --------
                mlflow.log_param(
                    f"{trend}_n_clean_features", len(self.feature_names[trend])
                )
                mlflow.log_param(
                    f"{trend}_n_selected_features", len(instance_model.features)
                )
                mlflow.log_param(f"{trend}_n_models", len(instance_model.best_models))

                with tempfile.TemporaryDirectory() as _tmp:
                    tmp_dir = Path(_tmp)

                    # -------- Feature lists --------
                    clean_feature_file = tmp_dir / f"{trend}_clean_features.txt"
                    selected_feature_file = tmp_dir / f"{trend}_selected_features.txt"

                    with open(clean_feature_file, "w") as f:
                        f.write("\n".join(self.feature_names[trend]))

                    with open(selected_feature_file, "w") as f:
                        f.write("\n".join(instance_model.features))

                    mlflow.log_artifact(
                        str(clean_feature_file), artifact_path=f"features/{trend}"
                    )
                    mlflow.log_artifact(
                        str(selected_feature_file), artifact_path=f"features/{trend}"
                    )

                    # Also log feature names as JSON (easier to load later)
                    feature_names_file = tmp_dir / f"{trend}_feature_names.json"
                    with open(feature_names_file, "w") as f:
                        json.dump(
                            {
                                "clean_features": self.feature_names[trend],
                                "selected_features": instance_model.features,
                                "n_clean": len(self.feature_names[trend]),
                                "n_selected": len(instance_model.features),
                            },
                            f,
                            indent=2,
                        )
                    mlflow.log_artifact(
                        str(feature_names_file), artifact_path=f"features/{trend}"
                    )

                    # -------- Per-model loop --------
                    importance_types = [
                        "weight",
                        "gain",
                        "cover",
                        "total_gain",
                        "total_cover",
                    ]
                    all_models_importance = {}
                    all_models_params = {}

                    for model_idx, boosting_model in enumerate(
                        instance_model.best_models
                    ):
                        booster = boosting_model.get_booster()
                        model_key = f"model_{model_idx}"

                        # -- Importance --
                        model_importance = {}
                        for importance_type in importance_types:
                            raw_importance = booster.get_score(
                                importance_type=importance_type
                            )
                            total = sum(raw_importance.values())
                            pct_importance = {}
                            for feat, score in raw_importance.items():
                                # XGBoost feature ids when trained on np.array: f0, f1, ...
                                import re

                                m = re.fullmatch(r"f(\d+)", str(feat))
                                if m:
                                    idx = int(m.group(1))
                                    mapped_feat = (
                                        instance_model.features[idx]
                                        if 0 <= idx < len(instance_model.features)
                                        else feat
                                    )
                                else:
                                    mapped_feat = feat

                                pct_importance[mapped_feat] = pct_importance.get(
                                    mapped_feat, 0.0
                                ) + (100.0 * score / total)
                            model_importance[importance_type] = dict(
                                sorted(
                                    pct_importance.items(),
                                    key=lambda x: x[1],
                                    reverse=True,
                                )
                            )

                        all_models_importance[model_key] = model_importance

                        # -- Parameters --
                        wrapper_params = boosting_model.get_params(deep=True)
                        xgb_params = (
                            boosting_model.get_xgb_params()
                            if hasattr(boosting_model, "get_xgb_params")
                            else {}
                        )
                        booster_attrs = booster.attributes() or {}

                        all_models_params[model_key] = {
                            "wrapper_params": wrapper_params,
                            "xgb_params": xgb_params,
                            "booster_attrs": booster_attrs,
                        }

                        # Log scalar params to MLflow
                        for k, v in xgb_params.items():
                            if isinstance(v, (int, float, str, bool)):
                                mlflow.log_param(f"{trend}_{model_key}_{k}", v)

                        # -- Plot all importance types in subplots --
                        n_types = len(importance_types)
                        fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 8))

                        if n_types == 1:
                            axes = [axes]

                        for ax, importance_type in zip(axes, importance_types):
                            top_10 = dict(
                                list(model_importance[importance_type].items())[:10]
                            )
                            ax.barh(
                                list(top_10.keys()),
                                list(top_10.values()),
                                color="steelblue",
                            )
                            ax.set_xlabel(f"{importance_type} (%)")
                            ax.set_title(f"{importance_type}")
                            ax.invert_yaxis()

                        fig.suptitle(
                            f"{model_key} - Top 10 Features per Importance Type ({trend})",
                            fontsize=14,
                        )
                        plt.tight_layout()

                        plot_path = (
                            tmp_dir / f"{trend}_{model_key}_all_importance_types.png"
                        )
                        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
                        plt.close()
                        mlflow.log_artifact(
                            str(plot_path),
                            artifact_path=f"feature_importance/plots/{trend}",
                        )

                        # -- Importance JSON per model --
                        imp_path = tmp_dir / f"{trend}_{model_key}_importance.json"
                        with open(imp_path, "w") as f:
                            json.dump(model_importance, f, indent=2)
                        mlflow.log_artifact(
                            str(imp_path),
                            artifact_path=f"feature_importance/json/{trend}",
                        )

                        # -- Params JSON per model --
                        params_path = tmp_dir / f"{trend}_{model_key}_params.json"
                        with open(params_path, "w") as f:
                            json.dump(all_models_params[model_key], f, indent=2)
                        mlflow.log_artifact(
                            str(params_path), artifact_path=f"parameters/{trend}"
                        )

                        # -- Native booster model --
                        booster_path = tmp_dir / f"{trend}_{model_key}_booster.json"
                        booster.save_model(str(booster_path))
                        mlflow.log_artifact(
                            str(booster_path), artifact_path=f"boosters/{trend}"
                        )

                        # -- Full booster config --
                        config_path = (
                            tmp_dir / f"{trend}_{model_key}_booster_config.json"
                        )
                        with open(config_path, "w") as f:
                            f.write(booster.save_config())
                        mlflow.log_artifact(
                            str(config_path), artifact_path=f"booster_configs/{trend}"
                        )

                    # -------- Aggregated JSONs --------
                    all_imp_path = (
                        tmp_dir / f"{trend}_all_models_feature_importance_pct.json"
                    )
                    with open(all_imp_path, "w") as f:
                        json.dump(all_models_importance, f, indent=2)
                    mlflow.log_artifact(
                        str(all_imp_path),
                        artifact_path=f"feature_importance/json/{trend}",
                    )

                    all_params_path = tmp_dir / f"{trend}_all_models_parameters.json"
                    with open(all_params_path, "w") as f:
                        json.dump(all_models_params, f, indent=2)
                    mlflow.log_artifact(
                        str(all_params_path), artifact_path=f"parameters/{trend}"
                    )

                self.election_predictor.add_model(
                    trend,
                    instance_model,
                    features=instance_model.features,
                )

                self.election_predictor.sign_model(
                    trend,
                    self.config.data_path + self.config.dataset_path,
                    sample=self.X_train[trend][instance_model.features].sample(5),
                )

            # ==========================================================
            # ELECTION PREDICTION
            # ==========================================================
            X_pred, X_true = self.organize_vote(
                k_year,
                k_political_trends,
                predict_delta,
            )

            X_result = self.election_predictor.evaluate_predictions(
                X_pred,
                X_true,
            )

            X_synthetic = self.election_predictor.compute_agg_results(
                X_result,
                blocs=[trend for trend in k_political_trends if trend != "par"],
                election_code=f"{k_year}_{k_type}",
            )

            winner_pred = self.election_predictor.get_winner(
                X_pred,
                self.k_t,
            )

            winner_true = self.election_predictor.get_winner(
                X_true,
                self.k_t,
            )

            logger.success(
                f"Winner predicted : {winner_pred} | Winner true : {winner_true}"
            )

            # ==========================================================
            # ELECTION-LEVEL METRICS
            # ==========================================================
            mlflow.log_param(
                "winner_pred",
                str(winner_pred),
            )

            mlflow.log_param(
                "winner_true",
                str(winner_true),
            )

            mlflow.log_metric(
                "winner_correct",
                int(winner_pred == winner_true),
            )

            # ==========================================================
            # ARTIFACTS
            # ==========================================================
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

            # ==========================================================
            # SAVE LEGACY OUTPUTS
            # ==========================================================
            self.save_results(
                model=self.election_predictor,
                result=(X_result, X_synthetic),
                k_year=k_year,
                k_type=k_type,
                k_political_trends=k_political_trends,
                version=version,
            )


def main():
    backtester = BackTester()
    model, model_args = (
        MODELS[backtester.config.model],
        MODEL_ARGS[backtester.config.model],
    )
    version = backtester.config.version
    k_year = backtester.config.k_year
    k_type = backtester.config.k_type
    k_political_trends = backtester.config.political_trends
    predict_delta = backtester.config.predict_delta

    logger.info(f"Model: {model}")

    for political_trends in k_political_trends:
        for year in k_year:
            for type_ in k_type:
                logger.info(
                    f"Running backtest for year: {year}, type: {type_}, political_trends: {political_trends} (delta: {predict_delta})"
                )
                backtest_function = (
                    backtester.run_backtest_with_MLFLOW
                    if MLFOW_TRACKING
                    else backtester.run_backtest
                )
                backtest_function(
                    k_year=year,
                    k_type=type_,
                    k_political_trends=political_trends,
                    model=model,
                    model_args=model_args,
                    predict_delta=predict_delta,
                    version=version,
                )


if __name__ == "__main__":
    dataset = main()
