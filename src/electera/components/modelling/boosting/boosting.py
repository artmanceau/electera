"""Boosting model components"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from gpboost import GPBoostRegressor
from loguru import logger
import optuna
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

BASE_PARAMS = {
    "xgboost": {
        "n_estimators": 1000,
        "min_child_weight": 250,  # the minimum sum of instance weight needed in a leaf
        "max_depth": 8,
        # "colsample_bytree": 0.8,  # the ratio of features used by tree
        # "colsample_bylevel": 0.8,  # the ratio of features used by level
        # "colsample_bynode": 0.8,  # the ratio of features used by node
        "learning_rate": 0.1,  # the learning rate of our GBM
        # (i.e. how much we update our prediction with each successive tree)
        "early_stopping_rounds": 150,
        "lambda": 5,
        "alpha": 5,
        "gamma": 5,
        # "objective": BoostingCustomLoss.spatial_loss(lambd=0.5, L=mean_squared_error)
        # # Spatial loss
    },
    "catboost": {"iterations": 500, "learning_rate": 0.1, "depth": 6},
    "gpboost": {},
}


class BoostingModel:
    """_summary_"""

    def __init__(self, parameters=None):
        """_summary_"""
        # Future attributes
        self.boosting_method = None
        self.method = None
        self.features_selected = None
        self.parameters = parameters
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

    def _compute_weights(self, X, y, weighting):
        inscrits = X["inscrits"].to_numpy().flatten()
        y = np.array(y)
        weighting_ = {
            "equiproportional": np.ones_like(y),
            "proportional": inscrits,
            "proportional_squared": inscrits**2,
            "sqrt": np.sqrt(inscrits),
            "log": np.log(inscrits + 1),
            "inverse": 1.0 / (inscrits + 1e-6),
            "inverse_y": 1.0 / (y.flatten() + 1e-6),
        }
        weights = weighting_[weighting]
        weights /= np.mean(weights)
        return weights

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
        weighting: str = "equiproportional",
        feature_selection_method: str = "none",
        nb_features: int = 'relative',
        param_search_method="none",
        **kwargs,
    ):
        """Train XGBoost model"""

        # Boosting method
        if self.boosting_method is None:
            self.boosting_method = XGBRegressor
            self.method = "xgboost"
            logger.info(f"Boosting method selected: {self.method}")

        # Weights
        weights = self._compute_weights(X_train, y_train, weighting=weighting)

        # Features
        self.feature_selection(feature_selection_method, nb_features=nb_features, X_val=X_train, y_val=y_train)

        logger.debug(
            f"With {len(self.features_selected)}/{X_train.shape[1]} features."
        )
        logger.debug(f'Feature selected: {self.features_selected}')

        # Parameters
        if self.parameters is None:
            self.params = BASE_PARAMS[self.method]
        else:
            self.params = self.parameters

        # Parameter search
        best_params = self.parameter_search(param_search_method=param_search_method, X_val=X_train, y_val=y_train)

        self.params.update(best_params)

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
                sample_weight=weights,
                eval_set=[(X_train_boosting, y_train), (X_val_boosting, y_val)],
            )
        else:
            self.model.fit(X_train_boosting, y_train, sample_weight=weights)

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

        elif param_search_method == "optuna":
            # Optimize other parameters using the chosen method
            logger.debug(
                f"Optimizing parameters using {param_search_method} search..."
            )
            _, X, _, y = train_test_split(
                X_val, y_val, test_size=0.33, random_state=42
            )

            def objective(trial):
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
                    "max_depth": trial.suggest_int("max_depth", 5, 12),
                    "min_child_weight": trial.suggest_int("min_child_weight", 25, 350),
                    "alpha": trial.suggest_float("alpha", 1e0, 1e2, log=True),
                    "gamma": trial.suggest_float("gamma", 1e0, 1e2, log=True),
                    "lambda": trial.suggest_float("lambda", 1e0, 1e2, log=True),
                    "eval_metric": "mae"
                }

                model = self.boosting_method(**params)
                cv = KFold(n_splits=3, shuffle=True, random_state=42)

                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                )
                return -scores.mean()

            '''
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
                        random_state=42,
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
            '''
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=10)
            best_params = study.best_params

        else:
            raise ValueError(f"Unknown search method: {param_search_method}")

            # Fit the model to find the best parameters
            # search_object.fit(X_val, y_val)
            # best_params = search_object.best_params_
            # best_params["learning_rate"] = (
            #    best_learning_rate  # Include the best learning rate from Stage 1
            # )

        logger.success(f"Best parameters found: {best_params}")
        return best_params

    def feature_selection(self, feature_selection_method='none', nb_features='relative', X_val=None, y_val=None):
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
                if nb_features == 'relative':
                    gain = self.importance_df["Gain"]
                    total_gain = gain.sum()
                    cum_ratio = gain.cumsum() / total_gain
                    n_features_80 = min(len(self.importance_df), int((cum_ratio < 0.90).sum() + 1))
                    self.features_selected = self.importance_df.head(n_features_80)["Feature"].tolist()
                else:
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
                    X_val,
                    y_val,
                    n_repeats=3,
                    random_state=42,
                )
                self.importance_df["Permutation"] = perm_importance.importances_mean
                self.importance_df = self.importance_df.sort_values(
                    by="Permutation", ascending=False
                )

                if nb_features == 'relative':
                    logger.debug('Keeping features larger than twice the median importance')
                    # Keep features larger than twice the median importance
                    median_importance = self.importance_df["Permutation"].median()
                    threshold = 2 * median_importance
                    self.features_selected = self.importance_df[
                        self.importance_df["Permutation"] > threshold
                    ]["Feature"].tolist()
                else:
                    self.features_selected = self.importance_df.head(nb_features)[
                        "Feature"
                    ].tolist()

            elif feature_selection_method == "RFE":
                # Top features using Recursive Feature Elimination
                rfe = RFE(
                    estimator=XGBRegressor(random_state=42),
                    n_features_to_select=nb_features,
                )
                rfe.fit(X_val, y_val)
                self.importance_df["RFE"] = rfe.support_
                self.features_selected = np.array(self.feature_names)[
                    rfe.support_
                ].tolist()

            else:
                raise ValueError(
                    f"Unknown feature selection method: {feature_selection_method}"
                )

        return self.features_selected
