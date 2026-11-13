from tabicl import TabICLRegressor


class FoundationTabularModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        self.model = TabICLRegressor(n_estimators=8, device="cpu")
        self.model.fit(X_train, y_train.values.ravel())
        return self

    def infer(self, X_test):
        return self.model.predict(X_test[self.model.feature_names_in_])
