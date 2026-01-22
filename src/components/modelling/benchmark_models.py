import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression


class TrivialModel1:

    def __init__(self, y_prev):
        self.y_prev = y_prev

    def train(self, X_train, y_train):
        self.mean = y_train.mean()

    def infer(self, X_test):
        if self.y_prev is not None:
            y_prev_imp = self.y_prev.copy(deep=True)
            y_prev_imp.fillna(self.mean, inplace=True)
            return y_prev_imp
        else:
            raise Exception("Fit model before predict")


class TrivialModel2:

    def __init__(self):
        self.mean = None

    def train(self, X_train, y_train):
        self.mean = y_train.mean()

    def infer(self, X_test):
        if self.mean is not None:
            y_pred = pd.DataFrame([self.mean] * len(X_test.index), index=X_test.index)
            return y_pred
        else:
            raise Exception("Fit model before predict")


class LinearModel:

    def __init__(self, linear_model=LinearRegression):
        self.linear_model = linear_model
        self.model = None

    @staticmethod
    def _linear_pre_process(X):
        X = X.copy(deep=True)

        # Linear model don't handle missing values
        # They will be imputed by the mean
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean().fillna(0.0))

        assert not X.fillna(X.mean().fillna(0.0)).isna().any().any()

        return X

    def train(self, X_train, y_train):
        X_train_linear = self._linear_pre_process(X_train)
        self.model = self.linear_model()
        self.model.fit(X_train_linear, y_train)
        self.features = list(X_train_linear.columns)

    def infer(self, X_test):
        X_test_linear = self._linear_pre_process(X_test[self.features])
        if self.model is not None:
            y_pred = self.model.predict(X_test_linear)
            return y_pred
        else:
            raise Exception("Fit model before predict")


class BenchmarkModels:

    @staticmethod
    def train_trivial_1(y_prev, y_train):
        """Train trivial baseline models"""
        logger.info(
            "Training trivial models 1 : same vote rate than previous election..."
        )

        model = TrivialModel1(y_prev=y_prev)
        model.train(X_train=None, y_train=y_train)
        return model.infer(X_test=None)

    @staticmethod
    def train_trivial_2(y_train, X_test):
        logger.info(
            "Training trivial models 2 : average vote rate than previous election..."
        )
        model = TrivialModel2()
        model.train(X_train=None, y_train=y_train)
        return model.infer(X_test=X_test)

    @staticmethod
    def train_linear_model(X_train, y_train, X_test, linear_model=LinearRegression):
        """Train linear regressions model"""
        logger.info("Training linear regression model...")
        model = LinearModel(linear_model=linear_model)
        model.train(X_train=X_train, y_train=y_train)
        return model.infer(X_test=X_test)
