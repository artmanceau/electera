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

    @staticmethod
    def compute_agg_results(X: pd.DataFrame, blocs: list, election_code: str):
        index = (
            ["inscrits", "votants", "exprimes", "ppar", "pexpr"]
            + blocs
            + [f"p{b}" for b in blocs]
        )
        result = pd.DataFrame(
            index=index,
            columns=[
                f"{election_code}_true",
                f"{election_code}_pred",
                f"{election_code}_diff",
                f"{election_code}_std",
                f"{election_code}_diff_agg",
            ],
        )

        result.loc["inscrits", f"{election_code}_true"] = X["inscrits"].sum()
        result.loc["exprimes", f"{election_code}_true"] = X["exprimes"].sum()

        for m in ["pred", "true"]:
            result.loc["votants", f"{election_code}_{m}"] = X[f"votants_{m}"].sum()
            result.loc["ppar", f"{election_code}_{m}"] = round(
                result.loc["votants", f"{election_code}_{m}"]
                / result.loc["inscrits", f"{election_code}_true"]
                * 100,
                2,
            )
        for x in ["votants", "ppar"]:
            result.loc[x, f"{election_code}_diff_agg"] = round(
                result.loc[x, f"{election_code}_true"]
                - result.loc[x, f"{election_code}_pred"],
                2,
            )
            result.loc[x, f"{election_code}_diff"] = X[f"{x}_diff"].mean()
            result.loc[x, f"{election_code}_std"] = X[f"{x}_diff"].std()
        for b in blocs:
            for m in ["pred", "true"]:
                result.loc[b, f"{election_code}_{m}"] = X[f"{b}_{m}"].sum()
                result.loc[b, f"{election_code}_{m}"] = round(
                    result.loc[b, f"{election_code}_{m}"]
                    / result.loc["exprimes", f"{election_code}_true"]
                    * 100,
                    2,
                )

            result.loc[b, f"{election_code}_diff_agg"] = (
                result.loc[b, f"{election_code}_true"]
                - result.loc[b, f"{election_code}_pred"]
            )
            result.loc[b, f"{election_code}_diff"] = X[f"{b}_diff"].mean()
            result.loc[b, f"{election_code}_std"] = X[f"{b}_diff"].std()
            result.loc[f"p{b}", f"{election_code}_diff_agg"] = round(
                result.loc[f"p{b}", f"{election_code}_true"]
                - result.loc[f"p{b}", f"{election_code}_pred"],
                2,
            )
            result.loc[f"p{b}", f"{election_code}_diff"] = X[f"p{b}_diff"].mean()
            result.loc[f"p{b}", f"{election_code}_std"] = X[f"p{b}_diff"].std()
        return result

    def compute_votes_per_circo(self, X):
        circo_mapping = pd.read_csv("config/mappings/circo_mapping.csv")[
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

        diff_cols = [
            col for col in X_merged.columns if ("_diff" in col) and ("p" in col)
        ]
        mean_error = X_merged[diff_cols].mean(axis=0).mean(axis=0)

        logger.success(f"Evaluation completed! Mean error is {mean_error}")

        return X_merged

    def add_model(self, trend, model, **kwargs):
        self.models[trend] = model
        if "features" in kwargs:
            self.features[trend] = kwargs["features"]
        else:
            self.features[trend] = model.get_features()

    def sign_model(self, trend, data_path, sample):
        self.signatures[trend] = sample
        self.data_paths[trend] = data_path
