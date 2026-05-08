"""
Pipeline 2: Model Training and Evaluation
=========================================
This module handles model training, evaluation, and comparison for the election modeling project.
"""

import argparse
import re
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_squared_error

from src.components.data_processing.data_loader import DataLoader
from src.components.modelling.benchmark_models import BenchmarkModels
from src.components.modelling.boosting.boosting import BoostingModel
from src.components.modelling.data_split import Splitter
from src.components.modelling.evaluation import ModelEvaluator
from src.components.modelling.meta_booster import (
    MetaBooster,
    MetaBoosterMultipleElections,
)
from src.components.utils.config import TrainModelsConfig
from src.components.utils.read_config import ConfigReader

# Try some other targert encoding / year, type encoding...
# Revoir le storage des modèles et effectuer un lineage direct avec le backtester


class ElectionModelTrainer:
    """Class to handle model training and evaluation pipeline"""

    def __init__(self):
        """
        Initialize the model trainer

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """

        self.config = ConfigReader._read_config(
            "config/train_models.json", TrainModelsConfig
        )
        self.var = self.config.vote_variable

        # Model storage
        self.models = {}
        self.results = {}
        self.model_data = {}
        self.input_examples = {}

    def data_processing(self, data):
        """Prepare training and testing data"""
        logger.info("Preparing data splits...")

        # Split target and features.
        # Remove index key codecommune
        splitter = Splitter(self.var)
        X, y, y_split = splitter.get_Xy(data)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = (
            splitter.split(
                X,
                y_split,
                split_method=self.config.split_method,
                test_size=self.config.test_size,
                val_size=self.config.val_size,
                random_seed=self.config.random_state,
            )
        )

        # Get y_prev from self.vote_variableprevious
        if f"pvoteprevious{self.var}" in X.columns:
            if self.config.predict_delta:
                self.y_prev = pd.Series(0.0, index=self.X_test.index)
            else:
                self.y_prev = self.X_test[f"pvoteprevious{self.var}"]

        if self.config.remove_previous_features:
            self.X_train = splitter.remove_previous_features(self.X_train)
            self.X_val = splitter.remove_previous_features(self.X_val)
            self.X_test = splitter.remove_previous_features(self.X_test)

        # Clean the list of features for the three datasets
        self.X_train, self.X_val, self.X_test = splitter.clean_features_list(
            self.X_train, self.X_val, self.X_test
        )

        self.feature_names = list(self.X_train.columns)

        logger.info(
            f"Data prepared: Train {self.X_train.shape}, Test {self.X_test.shape}"
        )

    def compare_models(self):
        """Compare all trained models"""
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
            {
                "Model": model_names,
                "MSE": mse_scores,
                "MAE": mae_scores,
                "R²": r2_scores,
            }
        )

        return comparison_df

    def save_results(self, experiment_name=None, sample_size=10):
        """Save all model results using MLflow"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name is None:
            experiment_name = f"pipeline_train_models_{timestamp}"

        try:
            # Set experiment
            mlflow.set_experiment(experiment_name)
            logger.info(f"Starting MLflow experiment: {experiment_name}")

            with mlflow.start_run():
                # Log config as parameters and artifact
                if hasattr(self, "config") and self.config is not None:
                    logger.info("Logging configuration")

                    # Log individual config parameters
                    for key, value in self.config.items():
                        try:
                            # MLflow parameters must be strings and <= 500 chars
                            param_value = (
                                str(value)[:500]
                                if len(str(value)) > 500
                                else str(value)
                            )
                            mlflow.log_param(f"config_{key}", param_value)
                        except Exception as e:
                            logger.warning(f"Could not log config parameter {key}: {e}")

                else:
                    logger.warning("No config available to log")

                # Log trained models and their results
                logger.info(f"Logging {len(self.models)} trained models")
                for model_name, model in self.models.items():
                    model_name_ = re.sub(":", "_", model_name)
                    mlflow.sklearn.log_model(
                        model,
                        f"{model_name_}",
                        registered_model_name=f"{model_name_}",
                        input_example=self.input_examples[model_name],
                    )
                    logger.info(f"Successfully logged model: {model_name_}")

                    # Log individual model results as metrics
                    if (
                        hasattr(self, "results")
                        and self.results is not None
                        and model_name in self.results
                    ):
                        model_results = self.results[model_name]
                        logger.info(f"Logging results for model: {model_name}")

                        # Log metrics if they exist
                        for metric_name, metric_value in model_results.items():
                            try:
                                metric_name = metric_name.split("_").pop()
                                # Only log numeric values as metrics
                                if isinstance(metric_value, (int, float)):
                                    mlflow.set_tag("model_name", model_name)
                                    mlflow.log_metric(metric_name, metric_value)
                                else:
                                    # Log non-numeric as parameters (converted to string)
                                    mlflow.log_param(
                                        metric_name,
                                        str(metric_value),
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Could not log {metric_name} for {model_name}: {e}"
                                )

                        logger.info(
                            f"Successfully logged results for model: {model_name}"
                        )
                    else:
                        logger.warning(f"No results found for model: {model_name}")

                # Log additional parameters/tags
                mlflow.log_param("timestamp", timestamp)
                mlflow.log_param("num_models", len(self.models))
                mlflow.set_tag("experiment_type", "model_training")

                run_id = mlflow.active_run().info.run_id
                logger.info(f"Results logged to MLflow run: {run_id}")

        except Exception as e:
            logger.error(f"Failed to save results to MLflow: {e}")
            raise


def main():
    """Main function to run the model training pipeline"""

    # Initialize trainer
    trainer = ElectionModelTrainer()

    # Load dataset (after running the data preprocessing pipeline)
    data = DataLoader.load_dataset(trainer.config.dataset_path)

    # Process data
    trainer.data_processing(data)

    # Trivial model 1 : same as previous election
    if "trivial_1" in trainer.config.models:
        y_1 = BenchmarkModels.train_trivial_1(trainer.y_prev, trainer.y_test)
        trainer.results["trivial_previous"] = ModelEvaluator.evaluate(
            trainer.y_test, y_1, "trivial_previous"
        )

    # Trivial model 2 : mean
    if "trivial_2" in trainer.config.models:
        y_2 = BenchmarkModels.train_trivial_2(trainer.y_train, trainer.X_test)
        trainer.results["trivial_mean"] = ModelEvaluator.evaluate(
            trainer.y_test, y_2, "trivial_mean"
        )

    # Linear model 1 : Linear model
    if "linear_reg" in trainer.config.models:
        y_3 = BenchmarkModels.train_linear_model(
            trainer.X_train,
            trainer.y_train,
            trainer.X_test,
            linear_model=LinearRegression,
        )
        trainer.results["linear_regression"] = ModelEvaluator.evaluate(
            trainer.y_test, y_3, "linear_regression"
        )

    # Linear model 2 : Elastic net
    if "elastic_net" in trainer.config.models:
        y_3 = BenchmarkModels.train_linear_model(
            trainer.X_train, trainer.y_train, trainer.X_test, linear_model=ElasticNetCV
        )
        trainer.results["elastic_net_regression"] = ModelEvaluator.evaluate(
            trainer.y_test, y_3, "elastic_net_regression"
        )

    if "boosting" in trainer.config.models:
        # boosting
        for param_search_method in (
            trainer.config.param_search_methods
        ):  # List of hyperparameter tuning methods
            for feature_selection_method in (
                trainer.config.feature_selection_methods
            ):  # List of feature selection methods
                for boosting_method in trainer.config.boosting_methods:
                    logger.info(
                        f"Running pipeline with feature selection: {feature_selection_method}, parameters search: {param_search_method}"
                    )

                    # 0. Boosting algorithm
                    boosting_model = BoostingModel()
                    boosting_model.set_boosting_method(boosting_method)

                    # 1. Feature selection
                    boosting_model.feature_selection(
                        feature_selection_method,
                        trainer.config.top_n_features,
                        X_val=trainer.X_val,
                        y_val=trainer.y_val,
                    )

                    # 2. Grid search to tune hyperparameters
                    boosting_model.parameter_search(
                        param_search_method, X_val=trainer.X_val, y_val=trainer.y_val
                    )

                    # 3. Train
                    model, signature = boosting_model.train(
                        X_train=trainer.X_train,
                        y_train=trainer.y_train,
                        X_val=trainer.X_val,
                        y_val=trainer.y_val,
                    )
                    model_name = boosting_model.get_model_name()
                    logger.info(f"Boosting model trained {model_name}...")

                    trainer.models[model_name] = model
                    trainer.input_examples[model_name] = signature

                    # 4. Evaluate
                    trainer.results[model_name] = ModelEvaluator.evaluate(
                        trainer.y_test, boosting_model.infer(trainer.X_test), model_name
                    )

    if "meta_boosting" in trainer.config.models:
        # meta-boosting
        for feature_selection_method in trainer.config.feature_selection_methods:
            for method in trainer.config.boosting_methods:
                meta_booster = MetaBooster(
                    method=method,
                    objective_metric=mean_squared_error,
                    weighting="proportional",
                    features=None,
                    n_splits_outer=2,
                    n_splits_inner=2,
                    n_trials=2,
                )
                meta_booster.train(
                    trainer.X_train,
                    trainer.y_train,
                    use_feature_selection=(feature_selection_method == "gain"),
                )
                y_pred = meta_booster.infer(trainer.X_test)
                trainer.results[
                    f"meta_booster_{method}_featselect:{feature_selection_method}"
                ] = ModelEvaluator.evaluate(
                    trainer.y_test,
                    y_pred,
                    f"meta_booster_{method}_featselect:{feature_selection_method}",
                )

    if "meta_boosting_multiple" in trainer.config.models:
        # meta-boosting using average predictions over multiple elections used for training
        for feature_selection_method in trainer.config.feature_selection_methods:
            for method in trainer.config.boosting_methods:
                meta_booster_multiple = MetaBoosterMultipleElections(
                    method=method,
                    objective_metric=mean_squared_error,
                    weighting="proportional",
                    features=None,
                    n_splits_outer=2,
                    n_splits_inner=2,
                    n_trials=2,
                    ponderation=[0.7, 0.3],
                )
                meta_booster_multiple.train_multiple(
                    election_datasets=[
                        (trainer.X_train, trainer.y_train),
                        (trainer.X_val, trainer.y_val),
                    ],
                    use_feature_selection=(feature_selection_method == "gain"),
                )
                y_pred = meta_booster_multiple.infer_multiple(trainer.X_test)
                trainer.results[
                    f"meta_booster_multiple_{method}_featselect:{feature_selection_method}"
                ] = ModelEvaluator.evaluate(
                    trainer.y_test,
                    y_pred,
                    f"meta_booster_multiple_{method}_featselect:{feature_selection_method}",
                )

    # Compare models
    comparison_df = trainer.compare_models()
    logger.success("\nModel Comparison:")
    logger.info(comparison_df.to_string(index=False))

    # MLFLOW
    if trainer.config.use_MLFlow:
        logger.info("Saving into MLFlow...")
        parser = argparse.ArgumentParser(description="Train election prediction models")
        parser.add_argument(
            "--experiment-name",
            type=str,
            default=None,
            help="MLflow experiment name (default: election_modeling)",
        )

        args = parser.parse_args()
        trainer.save_results(experiment_name=args.experiment_name)

    logger.success("Model training pipeline completed!")

    return trainer


if __name__ == "__main__":
    trainer = main()
