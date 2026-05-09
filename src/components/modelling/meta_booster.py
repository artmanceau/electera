""" "
Meta-Booster Model : models that builds a meta-boosting model from simple boosting model
The model is build using nested-cross-validation paramaters optimization
"""

from multiprocessing import Pool

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.components.explanability.feature_importance import FeatureImportance

optuna.logging.set_verbosity(optuna.logging.WARNING)

USE_GPU = False
if USE_GPU:
    import cupy as cp

USE_MP = False

BOOSTING_ALG = {"xgboost": XGBRegressor, "catboost": CatBoostRegressor}

BOOSTING_PARAM = {
    "xgboost": lambda trial: {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
        "gamma": trial.suggest_float("gamma", 0.5, 5),
        "alpha": trial.suggest_float("alpha", 0.5, 5),
        'lambda': trial.suggest_float('lambda', 0.5, 5),
        "min_split_loss": trial.suggest_float("min_split_loss", 0.5, 1.0),
    },
    "catboost": lambda trial: {
        "iterations": trial.suggest_int("iterations", 200, 1500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        # "rsm": trial.suggest_float("rsm", 0.5, 1.0), # Not supported on GPU
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli"]
        ),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise"]
        ),
        "leaf_estimation_method": trial.suggest_categorical(
            "leaf_estimation_method", ["Newton", "Gradient"]
        ),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations", 1, 10
        ),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        "od_wait": trial.suggest_int("od_wait", 20, 80),
        "verbose": False,
    },
}

GPU_PARAM = {
    "xgboost": {"device": "cuda"},
    "catboost": {"task_type": "GPU", "devices": "0"},
}


class MetaBooster:
    """The MetaBooster is trained on input data."""

    def __init__(
        self,
        method="xgboost",
        objective_metric=mean_squared_error,
        weighting="equiproportional",
        features=None,
        n_splits_outer=2,
        n_splits_inner=2,
        n_trials=2,
        poll_adj=False,
    ):
        self.method = method
        self.boosting_method = BOOSTING_ALG[self.method]
        self.objective_metric = objective_metric
        self.weighting = weighting
        self.n_splits_outer = n_trials
        self.n_splits_inner = n_splits_inner
        self.n_trials = n_trials
        self.features = features
        self.poll_adj = poll_adj

        # Container
        self.best_models = None
        self.best_params_list_outer = None
        self.best_test_scores_list_outer = None
        self.adjustment_model = None
        self.poll_feature = None

    def _check_feature_consistency(self, X, features=None):
        if features is None:
            features = X.columns

        X_train = X[list(set(features) & set(X.columns))]

        if (set(features) & set(X.columns)) != set(features):
            logger.warning(
                f"The following features are not in X: {set(features) - (set(features) & set(X.columns))}"
            )

        return X_train

    def train(self, X, y, use_feature_selection=False, val_set=None):
        logger.info("Training meta-booster model")
        weights = self._compute_weights(X, y)

        if use_feature_selection:
            X_val, y_val = (X, y) if val_set is None else val_set
            self.feature_selection(X_val, y_val)

        X = self._check_feature_consistency(X, self.features)

        (
            self.best_models,
            self.best_params_list_outer,
            self.best_test_scores_list_outer,
        ) = self._perform_nested_cv(
            X,
            y,
            weights=weights,
            n_splits_outer=self.n_splits_outer,
            n_splits_inner=self.n_splits_inner,
            n_trials=self.n_trials,
        )
        logger.info(
            f"Fit completed. Meta-Booster consist of {len(self.best_models)} models."
        )

        # Train an adjusted model f(x_i) = alpha * b_i + beta * p_i + gamma.
        if self.poll_adj:
            preds = self.infer(X, with_adjustment=False)

            features_list = X.columns.to_list()
            poll_features = [item for item in features_list if "poll" in item]
            if len(poll_features) == 1:
                self.poll_feature = [item for item in features_list if "poll" in item][
                    0
                ]
                polls = np.array(X[self.poll_feature]) / 100

                X_linear = np.column_stack([preds, polls])
                y_linear = np.array(y)

                self.adjustment_model = LinearRegression()
                self.adjustment_model.fit(X_linear, y_linear)
                logger.info(
                    "Linear model adjusted with polling data and model prediction"
                )
                logger.debug(f"Coefficients: {self.adjustment_model.coef_}")
                logger.debug(f"Intercept: {self.adjustment_model.intercept_}")
                logger.debug(
                    f"R^2 score: {self.adjustment_model.score(X_linear, y_linear)}"
                )
            else:
                logger.warning("No poll feature for this election/political trend")

    def get_features(self):
        return self.features

    def infer(self, X, with_adjustment=True):
        if self.best_models is None:
            raise ValueError("Train method not ran before")
        else:
            # Predictions are the average predictions of the best models
            X = self._check_feature_consistency(X, self.features)
            X = np.array(X)
            preds = np.zeros(len(X))
            n_models = len(self.best_models)
            for k in range(n_models):
                model = self.best_models[k]
                preds += model.predict(X)
            preds /= n_models

            if with_adjustment & self.poll_adj & (self.adjustment_model is not None):
                if self.poll_feature in X:
                    polls = np.array(X[self.poll_feature]) / 100

                    X = np.column_stack([preds, polls])
                    adj_preds = self.adjustment_model.predict(X)
                    return adj_preds

            return preds

    def feature_selection(self, X, y, threshold=0.8, method="permuation", nb_feature=150):
        logger.info(
            "Performing feature selection. Method: threshold best features in gain"
        )
        sample_model = BOOSTING_ALG[self.method]()
        sample_model.fit(X=X, y=y)

        if method == "total_gain":
            features_imp_df = FeatureImportance.compute_importance(
                models=[sample_model],
                features=X.columns,
                _get_importance_method=lambda x: x.feature_importances_,
            )
            self.features = features_imp_df[
                features_imp_df.cumsum()["Importance"] < threshold
            ]["Feature"].to_list()

        elif method == "permuation":
            perm = permutation_importance(
                sample_model, X, y, n_repeats=10, random_state=0
            )
            self.features = X.columns[
                perm.importances_mean.argsort()[::-1][:nb_feature]
            ].to_list()

        else:
            raise Exception("Feature selection method not implemented")

        logger.success(f"{len(self.features)} features are selected: {self.features}")

    def _compute_weights(self, X, y):
        inscrits = X["inscrits"].to_numpy().flatten()
        y = np.array(y)
        weighting_ = {
            "equiproportional": np.ones_like(y),
            "proportional": inscrits,
            "proportional_squared": inscrits**2,
            "sqrt": np.sqrt(inscrits),
            "inverse": 1.0 / (inscrits + 1e-6),
            "inverse_y": 1.0 / (y.flatten() + 1e-6),
        }
        weights = weighting_[self.weighting]
        weights /= np.mean(weights)
        return weights

    def _instantiate_model(self, param, gpu):
        if gpu:
            param.update(GPU_PARAM[self.method])

        return self.boosting_method(**param)

    def _perform_nested_cv(
        self, X, y, weights, n_splits_outer=3, n_splits_inner=3, n_trials=3
    ):
        xp = cp if ((USE_GPU) and (self.method == "xgboost")) else np
        X = xp.array(X)
        y = xp.array(y)
        weights = xp.array(weights)

        kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        kf_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=24)

        best_params_list_outer = []
        best_test_scores_list_outer = []

        args_list = [
            (
                fold_outer,
                train_index_outer,
                test_index_outer,
                X,
                y,
                weights,
                kf_inner,
                n_trials,
                USE_GPU,
            )
            for fold_outer, (train_index_outer, test_index_outer) in enumerate(
                kf_outer.split(X), start=1
            )
        ]

        if USE_MP:
            with Pool() as pool:
                results = pool.map(self._process_fold, args_list)

            for best_params, best_test_score in results:
                best_params_list_outer.append(best_params)
                best_test_scores_list_outer.append(best_test_score)
        else:
            for args in args_list:
                best_params, best_test_score = self._process_fold(args)
                best_params_list_outer.append(best_params)
                best_test_scores_list_outer.append(best_test_score)

        # Finally, train the best models on the entire dataset
        logger.debug("Training best models on the entire dataset")
        best_models = []
        for param in best_params_list_outer:
            boosting_model = self._instantiate_model(param=param, gpu=USE_GPU)
            boosting_model.fit(X, y, sample_weight=weights)
            best_models.append(boosting_model)

        return best_models, best_params_list_outer, best_test_scores_list_outer

    def _process_fold(self, args):
        (
            fold_outer,
            train_index_outer,
            test_index_outer,
            X,
            y,
            weights,
            kf_inner,
            n_trials,
            USE_GPU,
        ) = args
        return self._outer_cv(
            X,
            y,
            weights,
            kf_inner,
            fold_outer,
            n_trials,
            train_index_outer,
            test_index_outer,
            USE_GPU,
        )

    def _inner_cv(
        self,
        X_train_outer,
        y_train_outer,
        weights,
        n_trials,
        train_index,
        val_index,
        USE_GPU,
    ):
        X_train, X_val = X_train_outer[train_index], X_train_outer[val_index]
        y_train, y_val = y_train_outer[train_index], y_train_outer[val_index]
        weights_train = weights[train_index]

        if (USE_GPU) and (self.method == "xgboost"):
            y_val = y_val.get()

        def objective(trial):
            param = BOOSTING_PARAM[self.method](trial)

            # Set parameters
            boosting_model = self._instantiate_model(param=param, gpu=USE_GPU)
            # Train
            boosting_model.fit(X_train, y_train, sample_weight=weights_train)

            # Evaluate model on validation set
            y_pred = (
                cp.array(boosting_model.predict(X_val)).get()
                if USE_GPU
                else boosting_model.predict(X_val)
            )
            val_score = self.objective_metric(y_val.flatten(), y_pred.flatten())

            return val_score

        # Create and run the optimization process with 100 trials
        study = optuna.create_study(
            study_name="example_xgboost_study",
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=2)

        # Retrieve the best parameter values
        best_params = study.best_params
        logger.debug(f"\n best parameters inner: {best_params}")
        return best_params

    def _outer_cv(
        self,
        X,
        y,
        weights,
        kf_inner,
        fold_outer,
        n_trials,
        train_index_outer,
        test_index_outer,
        USE_GPU,
    ):
        logger.debug(f"Fold outer {fold_outer}")
        X_train_outer, X_test = X[train_index_outer], X[test_index_outer]
        y_train_outer, y_test = y[train_index_outer], y[test_index_outer]

        if (USE_GPU) and (self.method == "xgboost"):
            y_test = y_test.get()

        best_params_list = []
        fold_inner = 0
        for train_index, val_index in kf_inner.split(X_train_outer):
            fold_inner += 1
            logger.debug(f"Fold outer {fold_outer} fold inner {fold_inner}")
            best_params_list.append(
                self._inner_cv(
                    X_train_outer,
                    y_train_outer,
                    weights,
                    n_trials,
                    train_index,
                    val_index,
                    USE_GPU,
                )
            )

        # Evaluate the best models from inner folds on the outer test set
        logger.debug(" evaluating best models from inner folds on the full inner set")
        test_scores = []
        for i, param in enumerate(best_params_list):
            boosting_model = self._instantiate_model(param=param, gpu=USE_GPU)
            boosting_model.fit(
                X_train_outer,
                y_train_outer,
                sample_weight=weights[train_index_outer],
            )
            y_pred = boosting_model.predict(X_test)
            val_score = self.objective_metric(y_test.flatten(), y_pred.flatten())
            test_scores.append(val_score)
        idx_best = np.argmin(test_scores)
        logger.debug(
            f" best model is model {idx_best} with params {best_params_list[idx_best]} and test score {test_scores[idx_best]}"
        )
        return best_params_list[idx_best], test_scores[idx_best]


class MetaBoosterMultipleElections(MetaBooster):
    """Allows to average predictions over multiple elections used for training"""

    def __init__(
        self,
        method=XGBRegressor,
        objective_metric=mean_squared_error,
        weighting="equiproportional",
        features=None,
        n_splits_outer=2,
        n_splits_inner=2,
        n_trials=2,
        ponderation=None,
    ):
        super().__init__(
            method=method,
            objective_metric=objective_metric,
            weighting=weighting,
            features=features,
            n_splits_outer=n_splits_outer,
            n_splits_inner=n_splits_inner,
            n_trials=n_trials,
        )
        self.ponderation = ponderation
        self.N = len(self.ponderation)
        self.all_best_models = [None] * self.N
        self.all_features = [None] * self.N

    def train_multiple(
        self,
        election_datasets: list[tuple[pd.DataFrame, pd.Series]],
        use_feature_selection: str = False,
    ):
        logger.info(
            f"Training meta-booster model with {len(election_datasets)} elections"
        )
        for i, election in enumerate(election_datasets):
            X, y = election
            self.train(X, y, use_feature_selection=use_feature_selection)
            self.all_best_models[i] = self.best_models
            self.all_features[i] = self.get_features()

    def infer_multiple(self, X):
        preds = np.zeros(len(X))
        for i in range(self.N):
            self.best_models = self.all_best_models[i]
            self.features = self.all_features[i]
            preds += self.ponderation[i] * self.infer(X)
        return preds
