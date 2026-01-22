"""Boosting model components"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from gpboost import GPBoostRegressor
from loguru import logger
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from xgboost import XGBRegressor

BASE_PARAMS = {
    "xgboost": {
        "subsample": 0.75,  # the ratio of the training instances used
        "n_estimators": 150,
        "min_child_weight": 50,  # the minimum sum of instance weight needed in a leaf
        "max_depth": 8,
        "colsample_bytree": 0.8,  # the ratio of features used by tree
        "colsample_bylevel": 0.8,  # the ratio of features used by level
        "colsample_bynode": 0.8,  # the ratio of features used by node
        "learning_rate": 0.01,  # the learning rate of our GBM
        # (i.e. how much we update our prediction with each successive tree)
        "min_split_loss": 0.5,  # the minimum loss reduction required to make a further split
        "early_stopping_rounds": 20,
        # "objective": BoostingCustomLoss.spatial_loss(lambd=0.5, L=mean_squared_error)
        # # Spatial loss
    },
    "catboost": {"iterations": 500, "learning_rate": 0.1, "depth": 6},
    "gpboost": {},
}


class BoostingModel:
    """_summary_"""

    def __init__(self):
        """_summary_"""
        # Future attributes
        self.boosting_method = None
        self.method = None
        self.features_selected = None
        self.parameters = None
        self.model = None
        self.signature = None

        self.feature_selection_method = "none"
        self.param_search_method = "none"

    def set_boosting_method(self, method="xgboost") -> object:
        """_summary_

        Args:
            method (str, optional): xgboost or catboost. Defaults to 'xgboost'.

        Raises:
            ValueError: not valid model name

        Returns:
            object: boosting method
        """
        if method == "xgboost":
            self.boosting_method = XGBRegressor
        elif method == "catboost":
            self.boosting_method = CatBoostRegressor
        elif method == "gpboost":
            self.boosting_method = GPBoostRegressor
        else:
            raise ValueError("Method not recognized")
        self.method = method
        return self.boosting_method

    def get_model(self):
        return self.model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
        **kwargs,
    ):
        """Train XGBoost model"""

        if self.boosting_method is None:
            self.boosting_method = XGBRegressor
            self.method = "xgboost"
            logger.info(f"Boosting method selected: {self.method}")

        if self.features_selected is None:
            self.features_selected = X_train.columns.to_list()
            logger.debug(
                f"With {len(self.features_selected)}/{X_train.shape[1]} features."
            )

        if self.parameters is None:
            self.params = BASE_PARAMS[self.method]
            logger.debug(f"With parameters: {self.params}")

        # Apply feature selection
        X_train_boosting = X_train[self.features_selected].copy(deep=True)
        X_val_boosting = X_val[self.features_selected].copy(deep=True)

        # Apply selected parameters
        self.model = self.boosting_method(**self.params)

        # Fit
        if (X_val is not None) and (y_val is not None):
            self.model.fit(
                X_train_boosting,
                y_train,
                eval_set=[(X_train_boosting, y_train), (X_val_boosting, y_val)],
            )
        else:
            self.model.fit(X_train_boosting, y_train)

        self.signature = X_train_boosting.iloc[:5]

        return self.model, self.signature

    def get_model_name(self):
        if self.model is None:
            raise ValueError("Model not trained")
        else:
            self.model_name = f"{self.method}_FeatSelect:{self.feature_selection_method}_Hyperparam:{self.param_search_method}"
            return self.model_name

    def infer(self, X_test):
        if self.model is None:
            raise ValueError("Model not trained")
        else:
            if not self.features_selected:
                self.features_selected = self.X_train.columns.to_list()

            X_test_boosting = X_test[self.features_selected].copy(deep=True)
            return self.model.predict(X_test_boosting)

    def parameter_search(
        self, param_search_method: str, X_val: pd.DataFrame, y_val: pd.DataFrame
    ):
        """
        Perform parameter search for hyperparameter tuning in two stages:
        1. Optimize learning rate using GridSearchCV.
        2. Optimize other parameters using the chosen method (Bayesian, Random, or Grid Search).
        """
        logger.info(f"Starting parameter search (method: {param_search_method})...")
        self.param_search_method = param_search_method

        if param_search_method == "none":
            pass

        else:
            # Stage 1: Optimize learning rate using GridSearchCV
            logger.debug("Stage 1: Optimizing learning rate...")
            learning_rate_space = {
                "learning_rate": np.linspace(0.0001, 1.0, 10).tolist()
            }

            random_search = GridSearchCV(
                estimator=self.boosting_method(),
                param_grid=learning_rate_space,
                cv=3,  # Cross-validation folds
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )

            # Fit the model to find the best learning rate
            random_search.fit(X_val, y_val)
            best_learning_rate = random_search.best_params_["learning_rate"]
            logger.success(f"Best learning rate found: {best_learning_rate}")

            # Stage 2: Optimize other parameters using the chosen method
            logger.debug(
                f"Stage 2: Optimizing other parameters using {param_search_method} search..."
            )
            param_space = {
                "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "min_child_weight": [1, 4, 5, 6, 7],
                "subsample": [0.5, 0.6, 0.7],
                "colsample_bytree": [0.85, 0.9, 0.95],
                "n_estimators": [100, 400, 800, 1500],
            }

            if param_search_method == "bayesian":
                search_object = BayesSearchCV(
                    estimator=self.boosting_method(
                        learning_rate=best_learning_rate,
                    ),
                    search_spaces={
                        "max_depth": (1, 15),
                        "min_child_weight": (1, 10),
                        "subsample": (0.01, 1.0),
                        "colsample_bytree": (0.01, 1.0),
                        "n_estimators": (100, 2000),
                    },
                    n_iter=20,  # Number of iterations for Bayesian optimization
                    cv=3,  # Cross-validation folds
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )

            elif param_search_method == "random":
                search_object = RandomizedSearchCV(
                    estimator=XGBRegressor(
                        learning_rate=best_learning_rate,
                        random_state=self.config["random_state"],
                    ),
                    param_distributions=param_space,
                    n_iter=1,  # Number of random configurations to try
                    cv=3,  # Cross-validation folds
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )

            elif param_search_method == "grid":
                search_object = GridSearchCV(
                    estimator=self.boosting_method(
                        learning_rate=best_learning_rate,
                    ),
                    param_grid=param_space,
                    cv=3,  # Cross-validation folds
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )

            else:
                raise ValueError(f"Unknown search method: {param_search_method}")

            # Fit the model to find the best parameters
            search_object.fit(X_val, y_val)
            best_params = search_object.best_params_
            best_params["learning_rate"] = (
                best_learning_rate  # Include the best learning rate from Stage 1
            )

            logger.success(f"Best parameters found: {best_params}")
            self.params = best_params
            return self.params

    def feature_selection(self, feature_selection_method, nb_features, X_val, y_val):
        """
        Perform feature selection based on the specified method.
        """
        logger.info(
            f"Performing feature selection using method: {feature_selection_method}..."
        )
        self.feature_selection_method = feature_selection_method

        if feature_selection_method == "none":
            self.features_selected = X_val.columns.to_list()

        else:
            # Train a first XGboost model
            model_0 = self.boosting_method()
            model_0.fit(X_val, y_val)
            # Feature
            self.importance_df = pd.DataFrame(
                {
                    "Feature": X_val.columns.to_list(),
                    "Importance (default gain)": model_0.feature_importances_,
                }
            )

            if feature_selection_method == "gain":
                # Top features based on XGBoost feature importance (gain)
                self.importance_df["Gain"] = model_0.feature_importances_
                self.importance_df = self.importance_df.sort_values(
                    by="Gain", ascending=False
                )
                self.features_selected = self.importance_df.head(nb_features)[
                    "Feature"
                ].tolist()

            elif feature_selection_method == "weight":
                # Top features based on XGBoost feature importance (weight)
                self.importance_df["Weight"] = model_0.get_booster().get_score(
                    importance_type="weight"
                )
                self.importance_df = self.importance_df.sort_values(
                    by="Weight", ascending=False
                )
                self.features_selected = self.importance_df.head(nb_features)[
                    "Feature"
                ].tolist()

            elif feature_selection_method == "coverage":
                # Top features based on coverage
                self.importance_df["Coverage"] = model_0.get_booster().get_score(
                    importance_type="cover"
                )
                self.importance_df = self.importance_df.sort_values(
                    by="Coverage", ascending=False
                )
                self.features_selected = self.importance_df.head(nb_features)[
                    "Feature"
                ].tolist()

            elif feature_selection_method == "permutation":
                # Top features based on permutation importance
                model = model_0
                perm_importance = permutation_importance(
                    model,
                    self.X_val,
                    self.y_val,
                    n_repeats=3,
                    random_state=self.config["random_state"],
                )
                self.importance_df["Permutation"] = perm_importance.importances_mean
                self.importance_df = self.importance_df.sort_values(
                    by="Permutation", ascending=False
                )
                self.features_selected = self.importance_df.head(nb_features)[
                    "Feature"
                ].tolist()

            elif feature_selection_method == "RFE":
                # Top features using Recursive Feature Elimination
                rfe = RFE(
                    estimator=XGBRegressor(random_state=self.config["random_state"]),
                    n_features_to_select=nb_features,
                )
                rfe.fit(self.X_val, self.y_val)
                self.importance_df["RFE"] = rfe.support_
                self.features_selected = np.array(self.feature_names)[
                    rfe.support_
                ].tolist()

            else:
                raise ValueError(
                    f"Unknown feature selection method: {feature_selection_method}"
                )

        return self.features_selected
