import numpy as np
import pandas as pd
from loguru import logger
from src.components.data_processing.data_loader import DataLoader


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

    def predict_votes(self, X, predict_delta=False, infer_multiple=False, agg=False):
        """Feed with data (codecommune)"""
        if "codecommune" not in X.columns:
            raise Exception("Invalid input data")
        
        X_result = X[["codecommune"]].astype(str).copy(deep=True)
        X_result['inscrits'] = X['inscrits']

        def prediction_fn(model, X, predict_delta, infer_multiple):
            preds = model.infer_multiple(X) if infer_multiple else model.infer(X)
            if predict_delta:
                return preds + X[f'previouspvote{trend}'].fillna(X[f'previouspvote{trend}'].mean())
            else:
                return preds

        for trend in self.trends:
            X_result["pvote" + trend] = prediction_fn(self.models[trend], X[self.features[trend]], predict_delta, infer_multiple)

        # We have to readjust
        # as X_result[['p'+trend for trend in self.trends if trend!='par']].sum(axis=1) == 1
        trend_w_o_ppar = ["pvote" + trend for trend in self.trends if trend != "par"]
        n = len(trend_w_o_ppar)
        delta = (1 - X_result[trend_w_o_ppar].sum(axis=1)) / n
        delta_df = pd.concat([delta.rename(trend_w_o_ppar[i]) for i in range(n)], axis=1)
        for trend in trend_w_o_ppar:
            X_result[trend] += delta_df[trend]

        assert (
            np.abs(
                X_result[["pvote" + trend for trend in self.trends if trend != "par"]].sum(
                    axis=1
                )
                - 1
            ).mean()
            < 0.001
        )

        if "inscrits" in X.columns:
            X_result["votants"] = (X_result["pvotepar"] * X_result["inscrits"]).astype(int)
            for trend in self.trends:
                if trend != "par":
                    X_result[f'vote{trend}'] = round(X_result["pvote" + trend] * X_result["votants"], 0).astype(int)

        assert np.abs(X_result[[f'vote{trend}' for trend in self.trends if trend != 'par']].sum(axis=1) - X_result['votants']).mean() < 0.5

        if agg:
            agg_results = {}
            agg_results["tot_par"] = X_result["votants"].sum(axis=0)
            agg_results["tot_ppar"] = agg_results["tot_par"] / X_result["inscrits"].sum()
            for trend in self.trends:
                if trend != "par":
                    agg_results[f"tot_vote{trend}"] = X_result[f'vote{trend}'].sum()
                    agg_results[f"tot_pvote{trend}"] = (
                        X_result[f'vote{trend}'].sum()
                        / X_result[
                            "votants"
                        ].sum()
                    )

            return agg_results

        return X_result

    @staticmethod
    def compute_agg_results(X: pd.DataFrame, blocs: list, election_code: str):
        index = (
            ["inscrits", "votants", "exprimes", "exprimes_", "pvotepar", "pvoteexpr"]
            + [f"vote{b}" for b in blocs]
            + [f"pvote{b}" for b in blocs]
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

        result.loc["inscrits", f"{election_code}_true"] = X["inscrits_true"].sum()
        result.loc["inscrits", f"{election_code}_pred"] = X["inscrits_true"].sum()
        
        result.loc["exprimes", f"{election_code}_true"] = X["exprimes"].sum()
        result.loc["exprimes_", f"{election_code}_true"] = X[[f'vote{trend}_true' for trend in blocs]].sum().sum()
        result.loc["pvoteexpr", f"{election_code}_true"] = round(X["exprimes"].sum() / X["inscrits_true"].sum()*100, 2)
        
        # Start here
        for m in ["pred", "true"]:
            result.loc["votants", f"{election_code}_{m}"] = X[f"votants_{m}"].sum()

            result.loc["pvotepar", f"{election_code}_{m}"] = round(
                result.loc["votants", f"{election_code}_{m}"]
                / result.loc["inscrits", f"{election_code}_true"]
                * 100,
                2,
            )

        result.loc["exprimes", f"{election_code}_pred"] = X["votants_pred"].sum()
        result.loc["exprimes_", f"{election_code}_pred"] = X["votants_pred"].sum()
        result.loc["pvoteexpr", f"{election_code}_pred"] = result.loc["pvotepar", f"{election_code}_pred"]

        for x in ["votants", "pvotepar", 'pvoteexpr', 'inscrits', 'exprimes', 'exprimes_']:
            result.loc[x, f"{election_code}_diff_agg"] = round(
                result.loc[x, f"{election_code}_true"]
                - result.loc[x, f"{election_code}_pred"],
                2,
            )
            if x == 'votants':
                result.loc[x, f"{election_code}_diff"] = round(X[f"{x}_diff"].mean()*100, 0)
                result.loc[x, f"{election_code}_std"] = round(X[f"{x}_diff"].std()*100, 0)
            if x == 'pvotepar':
                result.loc[x, f"{election_code}_diff"] = round(X[f"{x}_diff"].mean()*100, 2)
                result.loc[x, f"{election_code}_std"] = round(X[f"{x}_diff"].std()*100, 2)

        result.loc['inscrits', f"{election_code}_diff"] = 0
        result.loc['inscrits', f"{election_code}_std"] = 0
        for x in ['exprimes', 'exprimes_']:
            result.loc[x, f"{election_code}_diff"] = result.loc['votants', f"{election_code}_diff"]
            result.loc[x, f"{election_code}_std"] = result.loc['votants', f"{election_code}_diff"]

        result.loc['pvoteexpr', f"{election_code}_diff"] = result.loc['pvotepar', f"{election_code}_diff"]
        result.loc['pvoteexpr', f"{election_code}_std"] = result.loc['pvotepar', f"{election_code}_diff"]

        for b in blocs:
            for m in ["pred", "true"]:
                result.loc[f'vote{b}', f"{election_code}_{m}"] = X[f"vote{b}_{m}"].sum()

        for b in blocs:
            for m in ["pred", "true"]:
                if m == 'pred':
                    result.loc[f"pvote{b}", f"{election_code}_{m}"] = round(
                        result.loc[f'vote{b}', f"{election_code}_{m}"]
                        / result.loc["votants", f"{election_code}_pred"]
                        * 100,
                        2,
                    )
                else:
                    result.loc[f"pvote{b}", f"{election_code}_{m}"] = round(
                        result.loc[f'vote{b}', f"{election_code}_{m}"]
                        / result.loc["exprimes_", f"{election_code}_true"]
                        * 100,
                        2,
                    )

        for b in blocs:
            result.loc[f"vote{b}", f"{election_code}_diff_agg"] = (
                result.loc[f"vote{b}", f"{election_code}_true"]
                - result.loc[f"vote{b}", f"{election_code}_pred"]
            )
            result.loc[f"vote{b}", f"{election_code}_diff"] = round(X[f"vote{b}_diff"].mean(), 0)
            result.loc[f"vote{b}", f"{election_code}_std"] = round(X[f"vote{b}_diff"].std(), 0)
            result.loc[f"pvote{b}", f"{election_code}_diff_agg"] = round(
                result.loc[f"pvote{b}", f"{election_code}_true"]
                - result.loc[f"pvote{b}", f"{election_code}_pred"],
                2,
            )
            result.loc[f"pvote{b}", f"{election_code}_diff"] = round(X[f"pvote{b}_diff"].mean()*100, 2)
            result.loc[f"pvote{b}", f"{election_code}_std"] = round(X[f"pvote{b}_diff"].std()*100, 2)

        result = result.reset_index()

        assert result.isna().astype(int).sum().sum() == 0

        return result

    def compute_votes_per_circo(self, X):
        # circo of 2022, have to find another circo mapping for older circonscriptions (not found online ?)
        circo_mapping = DataLoader.load_dataset("s3://arthurmanceau/election_modeling_uhcp/data/raw/insee_geo/circo_composition_2022_.csv", formate='csv')
        circo_mapping["COMMUNE_RESID"] = (
            circo_mapping["COMMUNE_RESID"]
            .astype(str)
            .str.zfill(5)
        )
        # Problem for PLM — multiple communes and multiple circonscriptions with some overlap
        # Solution : Create a sythetic PARIS-LYON-MARSEILLE and match for all circo
        plm = pd.DataFrame(index=['PARIS', 'LYON', 'MARSEILLE'], columns=X.columns)
        plm['codecommune'] = ['75056', '69123', '13055']
        plm.loc['PARIS', ['inscrits', 'votants'] + [f'vote{trend}' for trend in self.trends if trend != 'par']] = X[X['codecommune'].str.startswith('751')][['inscrits', 'votants'] + [f'vote{trend}' for trend in self.trends if trend != 'par']].sum()
        plm.loc['LYON', ['inscrits', 'votants'] + [f'vote{trend}' for trend in self.trends if trend != 'par']] = X[X['codecommune'].isin(['69381', '69382', '69383', '69384', '69385', '69386', '69387', '69388', '69389'])][['inscrits', 'votants'] + [f'vote{trend}' for trend in self.trends if trend != 'par']].sum()
        plm.loc['MARSEILLE', ['inscrits', 'votants'] + [f'vote{trend}' for trend in self.trends if trend != 'par']] = X[X['codecommune'].str.startswith('132')][['inscrits', 'votants'] + [f'vote{trend}' for trend in self.trends if trend != 'par']].sum()
        plm['pvotepar'] = plm['votants'] / plm['inscrits']
        for trend in self.trends:
            if trend == 'par':
                continue
            plm[f'pvote{trend}'] = plm[f'vote{trend}'] / plm['votants']

        plm_to_add = plm.loc[~plm["codecommune"].isin(X["codecommune"])]
        X = pd.concat([X, plm_to_add], axis=0, ignore_index=True)

        X_merged = X.merge(
            circo_mapping,
            left_on="codecommune",
            right_on="COMMUNE_RESID",
            how="left",
            validate="1:m",
        )
        communes_with_no_circo = X_merged[X_merged['circo'].isna()]['codecommune'].to_list()
        communes_with_no_circo_other_than_plm = list(set(communes_with_no_circo)-set(['13201', '13202', '13203', '13204', '13205', '13206', '13207', '13208', '13209', '13210', '13211', '13212', '13213', '13214', '13215', '13216', '69380', '69381', '69382', '69383', '69384', '69385', '69386', '69387', '69388', '69389', '75101', '75102', '75103', '75104', '75105', '75106', '75107', '75108', '75109', '75110', '75111', '75112', '75113', '75114', '75115', '75116', '75117', '75118', '75119', '75120']))
        if len(communes_with_no_circo_other_than_plm)>0:
            logger.warning(f'Commune with no circo (other than PLM) : {communes_with_no_circo_other_than_plm}')


        trends = [f'vote{trend}' for trend in self.trends if trend != "par"] + ["votants"]
        X_circo = X_merged.groupby("circo")[trends].sum().reset_index()
        return X_circo

    def get_winner(self, X, k_type):
        """Determine trend that gets more voice"""
        if k_type == 0:
            winner = (
                X[[f'vote{trend}' for trend in self.trends if trend != "par"]]
                .sum(axis=0)
                .idxmax()
            )
        else:
            # Legislative election
            X_circo = self.compute_votes_per_circo(X)
            assemblee = (
                X_circo[[f'vote{trend}' for trend in self.trends if trend != "par"]]
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

        X_true = X_true.rename(columns={"ppar": "pvotepar"})

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

        logger.success(f"Evaluation completed! Mean error is {mean_error*100:.2f}%")

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
