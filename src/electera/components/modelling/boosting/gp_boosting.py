# Object containing a gp_model
from gpboost import GPBoostRegressor, GPModel


class GPBoostRegressor_:
    """Wrapper object for the Gaussian Process Boosting model that estimates the gp_model and integrates it in the fit/predict"""

    def __init__(self, X, cluster, long, lat):
        self.boosting = GPBoostRegressor
        self.gp_model = GPModel(
            likelihood="gaussian",
            group_data=X[cluster],
            gp_coords=X[[long, lat]],
            cov_function="exponential",
        )

    def fit(self, X, y):
        self.boosting.fit(X, y, gp_model=self.gp_model)

    def predict(self, X, y):
        self.boosting.predict(X, y)
