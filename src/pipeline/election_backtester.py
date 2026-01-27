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
from datetime import datetime
from pathlib import Path

from loguru import logger
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.components.data_processing.data_loader import DataLoader, DataUtils
from src.components.modelling.benchmark_models import (
    LinearModel,
    TrivialModel1,
    TrivialModel2,
)
from src.components.modelling.boosting.boosting import BoostingModel

# from src.components.modelling.benchmark_models import TrivialModel2
from src.components.modelling.data_split import Splitter
from src.components.modelling.election_predictor import ElectionPredictor
from src.components.modelling.evaluation import ModelEvaluator
from src.components.modelling.meta_booster import (
    MetaBooster,
    MetaBoosterMultipleElections,
)
from src.components.utils.config import BackTesterConfig
from src.components.utils.read_config import ConfigReader

# TODO:
# - Modèle pour les votes blancs? Pour l'instant on considère que votants = exprimés,
# i.e. on ignore le vote blanc pour l'instant.

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
        "weighting": "proportional",
        "features": None,
        "n_splits_inner": 3,
        "n_splits_outer": 3,
        "n_trials": 3,
    },
    "meta_boosting_multiple": {
        "method": "catboost",
        "objective_metric": mean_absolute_error,
        "weighting": "proportional",
        "features": None,
        "n_splits_inner": 3,
        "n_splits_outer": 3,
        "n_trials": 3,
        "ponderation": [0.7, 0.3],
    },
}


class BackTester:

    def __init__(self):
        """ """
        self.config = ConfigReader._read_config(
            "config/backtester.json", BackTesterConfig
        )
        self.models = {}
        self.data = {}
        self.X = {}
        self.y = {}
        self.feature_names = {}
        self.results = {}
        self.features_after_selection = {}

    def process_and_split_dataset(self, data, k_year):
        """
        Split the dataset into training, validation, and test sets.
        """
        # From this with will retrieve in the each dataset (one for political trend)
        self.X_train = {trend: [] for trend in self.config.political_trends}
        self.X_val = {trend: [] for trend in self.config.political_trends}
        self.X_test = {trend: [] for trend in self.config.political_trends}
        self.y_train = {trend: [] for trend in self.config.political_trends}
        self.y_val = {trend: [] for trend in self.config.political_trends}
        self.y_test = {trend: [] for trend in self.config.political_trends}

        # Retrieve the rows matching the years for each dataset
        for trend in self.config.political_trends:
            st = Splitter("p" + trend)
            split_method = f"{k_year}_{self.k_t}"
            X, y, y_split = st.get_Xy(data)
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

    def organize_vote(self, k_year):
        # Ground truth
        ground_truth_data_path = (
            self.config.data_path
            + f"raw/elections/presidentiel/{k_year}/pres{k_year}_csv/pres{k_year}comm.parquet"
        )
        X_true = DataLoader.load_dataset(ground_truth_data_path)[
            ["codecommune", "nomcommune", "inscrits", "votants", "exprimes"]
            + [trend for trend in self.config.political_trends if trend != "par"]
            + ["p" + trend for trend in self.config.political_trends]
        ]
        str_cols = ["codecommune", "nomcommune"]
        float_cols = ["p" + trend for trend in self.config.political_trends]
        int_cols = [
            trend for trend in self.config.political_trends if trend != "par"
        ] + ["inscrits", "votants", "exprimes"]
        X_true[str_cols] = X_true[str_cols].astype(str)
        X_true[int_cols] = X_true[int_cols].astype(int)
        X_true[float_cols] = X_true[float_cols].astype(float)
        # Predictions
        data = DataLoader.load_dataset(self.config.data_path + self.config.dataset_path)
        data_election = data[
            (data["annee"].astype(int) == int(k_year))
            & (data["type"].astype(int) == int(self.k_t))
        ]
        X_pred = self.election_predictor.predict_votes(data_election)

        agg_results = self.election_predictor.predict_votes(data_election, agg=True)
        agg_results_show = {
            key: value
            for key, value in agg_results.items()
            if key
            in [
                f"tot_p{trend}"
                for trend in self.config.political_trends
                if trend != "par"
            ]
        }
        logger.success(
            f"Total participation : {agg_results['tot_ppar']}. Results : {agg_results_show}"
        )
        return X_pred, X_true

    def save_results(self, model, result, k_year, k_type):
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
        result_synthetic = result_synthetic.T.reset_index()  # pivot
        result_synthetic.columns = [
            "var",
            k_year,
        ]  # Rename column for easier comparison

        # Save (local or S3)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use version for model lineage rather than timestamp

        DataLoader.write_dataset(
            result_all, result_dir_path + f"results_full_{k_year}_{k_type}_{ts}.parquet"
        )
        DataLoader.write_dataset(
            result_synthetic,
            result_dir_path + f"results_synth_{k_year}_{k_type}_{ts}.parquet",
        )
        DataLoader.dump_pickle(
            object_to_pickle=model,
            file_path=model_dir_path + f"model_{k_year}_{k_type}_{ts}.pkl",
        )

    def run_backtest(self, k_year, k_type, model, model_args):
        """
        Run the backtesting process.
        """
        self.k_t = 0 if k_type == "pres" else 1

        # For now only one backtesting model
        self.election_predictor = ElectionPredictor(trends=self.config.political_trends)

        # 1. Load all dataset
        data = DataLoader.load_dataset(self.config.data_path + self.config.dataset_path)

        # 2. Test and split
        self.process_and_split_dataset(data, k_year)

        # 3. Train model
        for trend in self.config.political_trends:
            logger.info(f"Training model for trend: {trend}")
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
                    self.X_train[trend], self.y_train[trend]
                ),
                "meta_boosting_multiple": lambda: instance_model.train(
                    election_datasets=[
                        (self.X_train[trend], self.y_train[trend]),
                        (self.X_val[trend], self.y_val[trend]),
                    ]
                ),
            }

            trainings[self.config.model]()

            # Evaluate the model on the test election
            self.results[self.config.model] = ModelEvaluator.evaluate(
                self.y_test[trend],
                instance_model.infer(self.X_test[trend]),
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
        X_pred, X_true = self.organize_vote(k_year)

        # 5. Evaluate vote
        X_result, X_synthetic = self.election_predictor.evaluate_predictions(
            X_pred, X_true
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
        )


def main():
    backtester = BackTester()
    model, model_args = (
        MODELS[backtester.config.model],
        MODEL_ARGS[backtester.config.model],
    )
    k_year = backtester.config.k_year
    k_type = backtester.config.k_type
    for year in k_year:
        for type_ in k_type:
            logger.info(f"Running backtest for year: {year} (type: {type_})")
            backtester.run_backtest(
                k_year=year, k_type=type_, model=model, model_args=model_args
            )


if __name__ == "__main__":
    dataset = main()
