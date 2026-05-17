import numpy as np
import pandas as pd


class FeatureImportance:
    @staticmethod
    def _create_importance_df(features, importance):
        return (
            pd.DataFrame({"Feature": features, "Importance": importance})
            .sort_values(by="Importance", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def compute_importance(models, features, _get_importance_method):
        n_features = len(features)
        n_models = len(models)
        feature_importance = np.zeros((n_features))
        for k in range(n_models):
            feature_importance += _get_importance_method(models[k]) / n_models
        return FeatureImportance._create_importance_df(features, feature_importance)
