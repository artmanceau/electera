"""
Pipeline 2: Model Training and Evaluation
=========================================
This module handles model training, evaluation, and comparison for the election modeling project.
"""

import argparse
import re
from datetime import datetime
import tempfile
import shutil
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_squared_error

from electera.components.data_processing.data_loader import DataLoader
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
            "../config/train_models.json", TrainModelsConfig
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

        # Reset feature names for this run
        self.feature_names = {}

        values = get_Xy_pl(
            data,
            vote_variable=f"pvote{self.var}",
            year=2022,
            election_type="presidentiel",
            predict_delta=self.config.predict_delta,
            predict_perc=self.config.predict_perc,
            selected_groups=[
                "rank",
                "pct_change",
                "raw",
                "delta",
                "lag",
                "other",
                "meta",
                "geo",
                "previous_vote",
            ],
        )

        for name, value in zip(container_names, values):
            setattr(self, name, value)

        # Fix
        features_to_remove = list(
            set(self.X_train.columns[self.X_train.isnull().mean() > 0])
            - set(
                [
                    "inscrits",
                    "distanceparis",
                    "previouspvotepar",
                    "previouspreviouspvotepar",
                ]
            )
        )
        self.X_train = self.X_train.drop(columns=features_to_remove)
        self.X_test = self.X_test.drop(columns=features_to_remove)
        self.X_val = self.X_val.drop(columns=features_to_remove)

        self.feature_names = self.X_train.columns.tolist()

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

    def save_results(self, experiment_name=None):
        """Save all model results and artifacts to MLflow."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name is None:
            experiment_name = f"pipeline_train_models_{timestamp}"

        mlflow.set_experiment(experiment_name)
        logger.info(f"Starting MLflow experiment: {experiment_name}")

        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=f"{model_name}_{timestamp}"):
                # Log config
                if hasattr(self, "config") and self.config is not None:
                    for key, value in self.config.model_dump().items():
                        mlflow.log_param(f"config_{key}", str(value)[:500])

                # Log exactly this model
                self._log_model_to_mlflow(model_name, model)

                # Per-model metadata
                mlflow.log_param("timestamp", timestamp)
                mlflow.log_param("model_name", model_name)
                mlflow.set_tag("experiment_type", "model_training")
                mlflow.set_tag("framework", "scikit-learn")

                run_id = mlflow.active_run().info.run_id
                logger.info(f"Model '{model_name}' logged to MLflow run: {run_id}")

    def _log_model_to_mlflow(self, model_name: str, model) -> None:
        """Log individual model with type-specific artifacts."""
        model_name_safe = re.sub(r"[:\s]", "_", model_name)
        model_results = self.results.get(model_name, {})
        input_example = self.input_examples.get(model_name)

        # Ensure input_example is 2D
        if input_example is not None and hasattr(input_example, "ndim"):
            if input_example.ndim == 1:
                input_example = input_example.reshape(1, -1)

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
            else:
                mlflow.log_param(
                    f"param_{metric_name}_{model_name}", str(metric_value)[:500]
                )

        self._log_model_artifacts(model_name_safe, model, model_results)
        logger.info(f"Successfully logged model: {model_name}")

    def _log_model_artifacts(
        self, model_name_safe: str, model, model_results: dict
    ) -> None:
        """Log type-specific artifacts (feature importance, coefficients, etc.)."""
        artifacts_dir = tempfile.mkdtemp()
        try:
            if hasattr(model, "feature_importances_"):
                importance_df = pd.DataFrame(
                    {
                        "feature": model.feature_names_in_,
                        "importance": model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                importance_path = os.path.join(artifacts_dir, "feature_importance.csv")
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(
                    importance_path, artifact_path=f"{model_name_safe}/artifacts"
                )

            if hasattr(model, "get_booster"):
                try:
                    booster = model.get_booster()
                    for importance_type in [
                        "weight",
                        "gain",
                        "cover",
                        "total_gain",
                        "total_cover",
                    ]:
                        importance_dict = booster.get_score(
                            importance_type=importance_type
                        )
                        if importance_dict:
                            total_importance = sum(importance_dict.values())
                            importance_data = [
                                {
                                    "feature": feat,
                                    "importance": score,
                                    "importance_pct": (
                                        (score / total_importance * 100)
                                        if total_importance > 0
                                        else 0
                                    ),
                                }
                                for feat, score in importance_dict.items()
                            ]
                            importance_df = pd.DataFrame(importance_data).sort_values(
                                "importance", ascending=False
                            )
                            importance_path = os.path.join(
                                artifacts_dir,
                                f"feature_importance_{importance_type}.csv",
                            )
                            importance_df.to_csv(importance_path, index=False)
                            mlflow.log_artifact(
                                importance_path,
                                artifact_path=f"{model_name_safe}/artifacts",
                            )
                except Exception as e:
                    logger.warning(
                        f"Could not extract booster importances for {model_name_safe}: {e}"
                    )

            if hasattr(model, "coef_"):
                coef_df = pd.DataFrame(
                    {
                        "feature": self.feature_names,
                        "coefficient": model.coef_,
                    }
                ).sort_values("coefficient", key=abs, ascending=False)
                coef_path = os.path.join(artifacts_dir, "coefficients.csv")
                coef_df.to_csv(coef_path, index=False)
                mlflow.log_artifact(
                    coef_path, artifact_path=f"{model_name_safe}/artifacts"
                )

            if "predictions" in model_results:
                preds = model_results["predictions"]
                pred_path = os.path.join(artifacts_dir, "predictions.csv")
                if isinstance(preds, pd.Series):
                    preds.to_csv(pred_path)
                elif isinstance(preds, pd.DataFrame):
                    preds[0].to_csv(pred_path)
                else:
                    pd.Series(preds).to_csv(pred_path)
                mlflow.log_artifact(
                    pred_path, artifact_path=f"{model_name_safe}/artifacts"
                )
        finally:
            shutil.rmtree(artifacts_dir, ignore_errors=True)


def main():
    """Main function to run the model training pipeline"""

    # Initialize trainer
    trainer = ElectionModelTrainer()

    # Load dataset (after running the data preprocessing pipeline)
    data = DataLoader.load_dataset(trainer.config.dataset_path, engine="polars")

    # Process data
    trainer.data_processing(data)

    # Trivial model 1 : same as previous election
    if "trivial_1" in trainer.config.models:
        bm = BenchmarkModels()
        y_1 = bm.train_trivial_1(trainer.y_prev, trainer.y_test)
        trainer.results["trivial_1"] = ModelEvaluator.evaluate(
            trainer.y_test, y_1, "trivial_1", extended=True
        )
        trainer.models["trivial_1"] = bm.get_model()

    # Trivial model 2 : mean
    if "trivial_2" in trainer.config.models:
        bm = BenchmarkModels()
        y_2 = bm.train_trivial_2(trainer.y_train, trainer.X_test)
        trainer.results["trivial_2"] = ModelEvaluator.evaluate(
            trainer.y_test, y_2, "trivial_2", extended=True
        )
        trainer.models["trivial_2"] = bm.get_model()

    # Linear model 1 : Linear model
    if "linear_reg" in trainer.config.models:
        bm = BenchmarkModels()
        y_3 = bm.train_linear_model(
            trainer.X_train,
            trainer.y_train,
            trainer.X_test,
            linear_model=LinearRegression,
        )
        trainer.results["linear_regression"] = ModelEvaluator.evaluate(
            trainer.y_test, y_3, "linear_regression", extended=True
        )
        trainer.models["linear_reg"] = bm.get_model()

    # Linear model 2 : Elastic net
    if "elastic_net" in trainer.config.models:
        bm = BenchmarkModels()
        y_3 = bm.train_linear_model(
            trainer.X_train, trainer.y_train, trainer.X_test, linear_model=ElasticNetCV
        )
        trainer.results["elastic_net"] = ModelEvaluator.evaluate(
            trainer.y_test, y_3, "elastic_net", extended=True
        )
        trainer.models["elastic_net"] = bm.get_model()

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
                        trainer.y_test,
                        boosting_model.infer(trainer.X_test),
                        model_name,
                        extended=True,
                    )

    if "meta_boosting" in trainer.config.models:
        for feature_selection_method in trainer.config.feature_selection_methods:
            for method in trainer.config.boosting_methods:
                meta_booster = MetaBooster(
                    method=method,
                    objective_metric=mean_squared_error,
                    weighting="sqrt",
                    features=None,
                    n_splits_outer=3,
                    n_splits_inner=3,
                    n_trials=3,
                )
                meta_booster.train(
                    trainer.X_train,
                    trainer.y_train,
                    use_feature_selection=(feature_selection_method != "none"),
                    feature_selection_method=feature_selection_method,
                )
                y_pred = meta_booster.infer(trainer.X_test)
                trainer.results[
                    f"meta_booster_{method}_featselect:{feature_selection_method}_{k}"
                ] = ModelEvaluator.evaluate(
                    trainer.y_test,
                    y_pred,
                    f"meta_booster_{method}_featselect:{feature_selection_method}",
                    extended=True,
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
                    extended=True,
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
