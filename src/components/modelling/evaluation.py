"""Module with evaluation functionalities"""

from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    @staticmethod
    def evaluate(y_test, y_pred, model_name):
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        result = {"mse": mse, "mae": mae, "r2": r2, "predictions": y_pred}
        logger.success(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return result
