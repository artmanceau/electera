# A pipeline to ingest new elections results — calibrated on the 2024 legislative election

# Steps
#   1. Fetch election data with the public API
#   2. Process the results to match the expected format
#   3. Apply adjustments to match the hypothesis of the previous election
#       France metropolitaine
#       Blocs politiques

import os

import pandas as pd
import requests
import numpy as np
from loguru import logger

from src.components.data_processing.data_loader import DataLoader, DataUtils

# Subtilities:
    # Municipales : 2 files (>1000 and <1000)
    # Tendances politiques 

election_to_ingest = {'2024_leg': {
        'LINK': "https://www.data.gouv.fr/api/1/datasets/r/ab337c6f-e7e8-4981-843c-45052b71096b",
        'SAVE_TO': "raw/elections/legislative/2024/leg2024_csv/leg2024comm.parquet", 
        'political_mapping': {
            "G": ["voix_EXG", "pvoix_EXG", "voix_UG", "pvoix_UG", 'pvoix_COM', "voix_COM", 'voix_FI', 'pvoix_FI'],
            "D": [
                "voix_RN",
                "pvoix_RN",
                "voix_REC",
                "pvoix_REC",
                'voix_EXD',
                'pvoix_EXD'
            ],
            "CD": ["voix_LR",
                "pvoix_LR",
                "voix_DVD",
                "pvoix_DVD",],
            "CG": ["voix_REG", "pvoix_REG", "voix_ECO", "pvoix_ECO", 'voix_SOC', 'pvoix_SOC', 'voix_DVG', 'pvoix_DVG'],
            "C": ["voix_ENS", "pvoix_ENS", "voix_DIV", "pvoix_DIV", "voix_DSV", "pvoix_DSV", 'voix_DVC', 'pvoix_DVC'],
        }
    },
    '2020_muni': {
        'LINK': "https://www.data.gouv.fr/api/1/datasets/r/4feeef01-24f7-4d5a-914f-8aa806f31ec2",
        'SAVE_TO': "s3://arthurmanceau/election_modelling_uhcp/data/raw/elections/municipales/2020/muni2020.parquet",
        'political_mapping': {
            "G": ["voix_EXG", "pvoix_EXG", "voix_UG", "pvoix_UG"],
            "D": [
                "voix_RN",
                "pvoix_RN",
                "voix_REC",
                "pvoix_REC",
                "voix_LR",
                "pvoix_LR",
                "voix_DVD",
                "pvoix_DVD",
            ],
            "CD": [],
            "CG": ["voix_REG", "pvoix_REG", "voix_ECO", "pvoix_ECO"],
            "C": ["voix_ENS", "pvoix_ENS", "voix_DIV", "pvoix_DIV", "voix_DSV", "pvoix_DSV"],
        }
    },
    '2026_muni': {
        'LINK': "",
        'SAVE_TO': "s3://arthurmanceau/election_modelling_uhcp/data/raw/elections/municipales/2026/muni2026.parquet",
        'political_mapping': {
            "G": ["voix_EXG", "pvoix_EXG", "voix_UG", "pvoix_UG"],
            "D": [
                "voix_RN",
                "pvoix_RN",
                "voix_REC",
                "pvoix_REC",
                "voix_LR",
                "pvoix_LR",
                "voix_DVD",
                "pvoix_DVD",
            ],
            "CD": [],
            "CG": ["voix_REG", "pvoix_REG", "voix_ECO", "pvoix_ECO"],
            "C": ["voix_ENS", "pvoix_ENS", "voix_DIV", "pvoix_DIV", "voix_DSV", "pvoix_DSV"],
        }
    },
    }
data_path = "s3://arthurmanceau/election_modelling_uhcp/data/"


class ElectionIngester:

    def __init__(self, election_code):
        self.access_link = election_to_ingest[election_code]['LINK']
        self.data_path = data_path
        self.output_file = election_to_ingest[election_code]['SAVE_TO']
        self.political_mapping = election_to_ingest[election_code]['political_mapping']

    def download_open_and_delete_file(
        self, url, folder_path="data/raw/temp", file_name="election_temp.xlsx"
    ):
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, file_name)

        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
            logger.info(f"File downloaded successfully and saved to {file_path}")
        else:
            logger.warning(
                f"Failed to download file. HTTP Status Code: {response.status_code}"
            )
            return None

        data = pd.read_excel(file_path)

        if os.path.exists(file_path):
            os.remove(file_path)

        return data

    def get_and_rename(
        self,
        X,
        mapping={
            "Code département": "dep",
            "Libellé département": "nomdep",
            "Code commune": "codecommune",
            "Libellé commune": "nomcommune",
            "Inscrits": "inscrits",
            "Votants": "votants",
            "Exprimés": "exprimes",
            "Blancs": "blancs",
            "Nuls": "nuls",
        },
    ):
        X_ = X[list(mapping.keys())].copy(deep=True)
        X_.rename(columns=mapping, inplace=True)
        return X_

    def pivot_data(self, data, N=204):
        pivot_data = []
        for idx, row in data.iterrows():
            codecommune = row.get("Code commune")
            for i in range(1, N + 1):
                nuance = row.get(f"Nuance candidat {i}", None)
                voix = row.get(f"Voix {i}", None)
                pvoix = row.get(f"% Voix/exprimés {i}", None)
                if pd.notna(nuance) and nuance != "":
                    voix = float(str(voix).replace(",", "."))
                    pvoix = float(str(pvoix).replace(",", ".").replace("%", ""))
                    pivot_data.append(
                        {
                            "codecommune": codecommune,
                            "trend": nuance,
                            "voix": voix,
                            "pvoix": pvoix,
                        }
                    )

        pivot_dataset = pd.DataFrame(pivot_data)

        pivot_dataset_grouped = (
            pivot_dataset.groupby(["codecommune", "trend"], dropna=False)
            .agg({"voix": "sum", "pvoix": "sum"})
            .reset_index()
        )

        # # Pivot to wide format
        X = pivot_dataset_grouped.pivot(
            index="codecommune", columns="trend", values=["voix", "pvoix"]
        )

        # # Flatten the columns
        X.columns = [f"{stat}_{nuance}" for stat, nuance in X.columns]
        X = X.reset_index()
        X = self.apply_political_adjustments(X)
        return X

    def apply_political_adjustments(self, X):
        X = X.copy()
        trends_kept = (
            X.isna().astype(int).sum(axis=0).sort_values()[1:23].index.to_list()
        )
        trends_dropped = X.isna().astype(int).sum(axis=0).sort_values()[23:].index.to_list()

        trends_kept_p = [col for col in trends_kept if "p" in col]
        trends_kept_v = set(trends_kept) - set(trends_kept_p)
        assert len(trends_kept_p) == len(trends_kept_v)

        trends_dropped_p = [col for col in trends_dropped if 'p' in col]
        trends_dropped_v = set(trends_dropped) - set(trends_dropped_p)
        assert len(trends_kept_p) == len(trends_kept_v)

        X['voix_AUTRES'] = X[list(trends_dropped_v)].sum(axis=1)
        X['pvoix_AUTRES'] = X[trends_dropped_p].sum(axis=1)

        X = X[["codecommune"] + trends_kept+['voix_AUTRES', 'pvoix_AUTRES']]
        X = X.fillna(0.0)
        X[trends_kept+['voix_AUTRES', 'pvoix_AUTRES']] = X[trends_kept+['voix_AUTRES', 'pvoix_AUTRES']].astype(float)

        X = X.loc[~(X[trends_kept_p+['pvoix_AUTRES']].sum(axis=1) == 0), :]

        assert (~(np.abs(X[trends_kept_p+['pvoix_AUTRES']].sum(axis=1) - 100) < 2)).astype(int).sum() == 0

        for pol_trend in ["G", "D", "CD", "CG", "C"]:
            partis = [col.split("_")[1] for col in self.political_mapping[pol_trend]]
            X[f"voix_{pol_trend}"] = X[
                [col for col in trends_kept_v if col.split("_")[1] in partis]
            ].sum(axis=1)
            X[f"pvoix_{pol_trend}"] = X[
                [col for col in trends_kept_p if col.split("_")[1] in partis]
            ].sum(axis=1)

        return X

    def save_processed_election(self, X):
        if not DataUtils._detect_s3(self.data_path):
            os.makedirs(self.output_file, exist_ok=True)
            DataLoader.write_dataset(X, self.data_path + self.output_file)
        else:
            DataLoader.write_dataset(X, self.data_path + self.output_file)

    def run(self):
        # Get data from source
        data = self.download_open_and_delete_file(self.access_link)

        # Ids
        data_i = self.get_and_rename(data)

        # Vote
        data_j = self.pivot_data(data)

        # Merge
        data_merged = data_i.merge(data_j, on="codecommune", how="inner")
        data_merged = data_merged.copy()
        data_merged["ppar"] = data_merged["votants"] / data_merged["inscrits"]

        cols_to_string = ['dep', 'nomdep', 'codecommune', 'nomcommune']
        data_merged[cols_to_string] = data_merged[cols_to_string].astype(str)

        self.save_processed_election(data_merged)


if __name__ == "__main__":
    for election_code in election_to_ingest.keys():
        logger.info(f'Ingesting: {election_code}')
        ingester = ElectionIngester(election_code)
        ingester.run()
