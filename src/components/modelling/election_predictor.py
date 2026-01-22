import numpy as np
import pandas as pd
from loguru import logger


class ElectionPredictor:
    """This object is able to predict the vote in a commune.
    It consists of 5 models, one for each trend.
    The submodels should be any object with train/infer/get_features methods.
    """

    def __init__(self, trends):
        self.trends = trends
        self.models = {trend: None for trend in trends}
        self.data_paths = {trend: None for trend in trends}
        self.signatures = {trend: None for trend in trends}
        self.features = {trend: None for trend in trends}

    def _check_complete(self):
        pass

    def predict_votes(self, X, Insc=None, agg=False):
        """Feed with data (codecommune)"""
        if "codecommune" not in X.columns:
            raise Exception("Invalid input data")

        X_result = X[["codecommune"]].astype(str).copy(deep=True)

        for trend in self.trends:
            X_result["p" + trend] = self.models[trend].infer(X[self.features[trend]])

        # We have to readjust
        # as X_result[['p'+trend for trend in self.trends if trend!='par']].sum(axis=1) == 1
        trend_w_o_ppar = ["p" + trend for trend in self.trends if trend != "par"]
        n = len(trend_w_o_ppar)
        delta = (1 - X_result[trend_w_o_ppar].sum(axis=1)) / n
        delta_df = pd.concat(
            [delta.rename(trend_w_o_ppar[i]) for i in range(n)], axis=1
        )
        for trend in trend_w_o_ppar:
            X_result[trend] += delta_df[trend]

        assert (
            np.abs(
                X_result[["p" + trend for trend in self.trends if trend != "par"]].sum(
                    axis=1
                )
                - 1
            ).mean()
            < 0.001
        )

        if "inscrits" in X.columns:
            X_result["votants"] = (X_result["ppar"] * X["inscrits"]).astype(int)
            for trend in self.trends:
                if trend != "par":
                    X_result[trend] = (
                        X_result["p" + trend] * X_result["votants"]
                    ).astype(int)

        if agg:
            agg_results = {}
            agg_results["tot_par"] = X_result["votants"].sum(axis=0)
            agg_results["tot_ppar"] = agg_results["tot_par"] / X["inscrits"].sum()
            for trend in self.trends:
                if trend != "par":
                    agg_results[f"tot_{trend}"] = X_result[trend].sum()
                    agg_results[f"tot_p{trend}"] = (
                        X_result[trend].sum()
                        / X[
                            "inscrits"
                        ].sum()  # Should be exprimes but we don't take vote blanc into consideration
                    )

            return agg_results

        return X_result

    def compute_votes_per_circo(self, X):
        circo_mapping = pd.read_csv("data/raw/insee_geo/circo.csv")[
            ["COMMUNE_RESID", "circo"]
        ]
        X_merged = X.merge(
            circo_mapping,
            left_on="codecommune",
            right_on="COMMUNE_RESID",
            how="left",
            validate="1:m",
        )
        trends = [trend for trend in self.trends if trend != "par"] + ["votants"]
        X_circo = X_merged.groupby("circo")[trends].sum().reset_index()
        return X_circo

    def get_winner(self, X, k_type):
        """Determine trend that gets more voice"""
        if k_type == 1:
            winner = (
                X[[trend for trend in self.trends if trend != "par"]]
                .sum(axis=0)
                .idxmax()
            )
        else:
            # Legislative election
            X_circo = self.compute_votes_per_circo(X)
            assemblee = (
                X_circo[[trend for trend in self.trends if trend != "par"]]
                .idxmax(axis=1)
                .value_counts()
            )
            winner = assemblee.idxmax()

        return winner

    def evaluate_predictions(self, X_pred, X_true):
        """Compute the L1 norm for two system of prediction according to axis.
        Axis can be either tendance, commune, all.
        X_pred and X_true should contain code commune and same columns"""
        cols_to_compute = list(X_pred.columns)
        cols_to_compute = list(
            set(cols_to_compute)
            - set(["exprimes", "codecommune", "nomcommune", "inscrits"])
        )

        X_merged = X_pred.merge(
            X_true,
            on="codecommune",
            how="inner",
            suffixes=("_pred", "_true"),
            validate="1:1",
        )
        for col in cols_to_compute:
            X_merged[f"{col}_diff"] = np.abs(
                X_merged[f"{col}_pred"] - X_merged[f"{col}_true"]
            )

        diff_cols = [col for col in X_merged.columns if "_diff" in col]
        diff_cols_p = [col for col in diff_cols if "p" in col]
        # diff_cols_vote = list(set(diff_cols) - set(diff_cols_p))
        avg_error_per_trend = X_merged[diff_cols].mean(axis=0)
        avg_error_per_commune_p = X_merged[diff_cols_p].mean(axis=1)
        # avg_error_per_commune_vote = (
        #     X_merged[diff_cols_vote].mean(axis=1) / X_merged["inscrits"]
        # )
        avg_error_tot_1 = avg_error_per_trend[diff_cols_p].mean()
        avg_error_tot_2 = avg_error_per_commune_p.mean()
        assert np.abs(avg_error_tot_1 - avg_error_tot_2) < 0.1

        std_error_tot = avg_error_per_commune_p.std()

        X_synthetic = (
            pd.Series(
                [avg_error_tot_1, std_error_tot]
                + [X_merged[f"p{trend}_diff"].mean(axis=0) for trend in self.trends]
                + [X_merged[f"p{trend}_diff"].std(axis=0) for trend in self.trends],
                index=["avg_error_tot", "std_error_tot"]
                + [f"error_p{trend}" for trend in self.trends]
                + [f"std_p{trend}" for trend in self.trends],
                name="value",
            )
            .to_frame()
            .T
        )

        logger.success(f"Evaluation completed! Mean error is {avg_error_tot_1}")

        return X_merged, X_synthetic

    def add_model(self, trend, model, **kwargs):
        self.models[trend] = model
        if "features" in kwargs:
            self.features[trend] = kwargs["features"]
        else:
            self.features[trend] = model.get_features()

    def sign_model(self, trend, data_path, sample):
        self.signatures[trend] = sample
        self.data_paths[trend] = data_path
