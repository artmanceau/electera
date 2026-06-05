"""Module with evaluation functionalities"""

from loguru import logger
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)


class ModelEvaluator:
    @staticmethod
    def evaluate(y_test, y_pred, model_name, extended=False):
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        result = {"mse": mse, "rmse": rmse, "mae": mae}
        logger.success(
            f"{model_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, "
        )

        if extended:
            mape = mean_absolute_percentage_error(y_test, y_pred)
            medae = median_absolute_error(y_test, y_pred)
            evs = explained_variance_score(y_test, y_pred)
            maxerr = max_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            spearman_corr, spearman_p = spearmanr(y_test, y_pred)
            kendall_corr, kendall_p = kendalltau(y_test, y_pred)

            extended_result = {
                "mape": mape,
                "median_ae": medae,
                "explained_variance": evs,
                "max_error": maxerr,
                "r2": r2,
                "predictions": y_pred,
                "spearman_corr": spearman_corr,
                "kendall_corr": kendall_corr,
            }

            logger.success(
                f"MAPE: {mape:.4f}, MedAE: {medae:.4f}, EVS: {evs:.4f}, "
                f"MaxErr: {maxerr:.4f}, R²: {r2:.4f}"
                f"Spearman: {spearman_corr:.4f}, Kendall: {kendall_corr:.4F}"
            )

            return {**result, **extended_result}

        return result
