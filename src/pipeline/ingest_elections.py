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
# Commune cleaning PLM - OK except 2026
# Political mapping + NB list à revoir

election_to_ingest = {'2024_leg': {
        'LINK': "https://www.data.gouv.fr/api/1/datasets/r/ab337c6f-e7e8-4981-843c-45052b71096b",
        'SAVE_TO': "raw/elections/legislative/2024/leg2024_csv/leg2024comm.parquet",
        'FILE_FORMAT': ".xlsx",
        'NB_OF_FILE': 1,
        'SOURCE': 'API',
        'NUANCE_ID': 'candidat',
        'political_mapping': {
            "G": ["vote_EXG", "pvote_EXG", "vote_UG", "pvote_UG", 'pvote_COM', "vote_COM", 'vote_FI', 'pvote_FI'],
            "D": [
                "vote_RN",
                "pvote_RN",
                "vote_REC",
                "pvote_REC",
                'vote_EXD',
                'pvote_EXD'
            ],
            "CD": ["vote_LR",
                "pvote_LR",
                "vote_DVD",
                "pvote_DVD",],
            "CG": ["vote_REG", "pvote_REG", "vote_ECO", "pvote_ECO", 'vote_SOC', 'pvote_SOC', 'vote_DVG', 'pvote_DVG'],
            "C": ["vote_ENS", "pvote_ENS", "vote_DIV", "pvote_DIV", "vote_DSV", "pvote_DSV", 'vote_DVC', 'pvote_DVC'],
        }
    },
    '2020_muni': {
        'LINK': {0: "ingestion/municipales/2020/2020_1000_plus.csv", 1: "ingestion/municipales/2020/2020_1000_moins.csv"},
        'SAVE_TO': "raw/elections/municipales/2020/muni2020_csv/muni2020comm.parquet",
        'FILE_FORMAT': ".csv",
        'NB_OF_FILE': 2,
        'SOURCE': 'ingestion',
        'NUANCE_ID': 'liste',
        'political_mapping': {
            "G": ['vote_LUG', 'pvote_LUG', 'vote_LEXG', 'pvote_LEXG',  'vote_LFI', 'pvote_LFI', 'pvote_LCOM', 'vote_LCOM'],
            "D": ['vote_LRN', 'pvote_LRN', 'vote_LEXD', 'pvote_LEXD'],
            "CD": ['vote_LDVD', 'pvote_LDVD', 'pvote_LLR', 'vote_LLR',  'pvote_LUD', 'vote_LUD', 'vote_LUDI', 'pvote_LUDI'],
            "CG": ['vote_LDVG', 'pvote_LDVG', 'pvote_LSOC', 'vote_LSOC', 'vote_LECO', 'pvote_LECO'],
            "C": ['pvote_LDVC', 'vote_LDVC', 'pvote_LREM', 'vote_LREM', 'vote_LUC', 'pvote_LUC'],
        }
    },
    '2026_muni': {
        'LINK': {0: "ingestion/municipales/2026/2026.csv", 1: "ingestion/municipales/2026/2026_plm.csv"},
        'SAVE_TO': "raw/elections/municipales/2026/muni2026_csv/muni2026comm.parquet",
        'FILE_FORMAT': ".csv",
        'NB_OF_FILE': 2,
        'SOURCE': 'ingestion',
        'NUANCE_ID': 'liste',
         "political_mapping": {
            "G": [
                'vote_LEXG', 'pvote_LEXG', 'vote_LFI', 'pvote_LFI', 'vote_LCOM', 'pvote_LCOM',
            ],
            "D": [
                'vote_LRN', 'pvote_LRN', 'vote_LREC', 'pvote_LREC', 'vote_LUXD', 'pvote_LUXD',
                'vote_LEXD', 'pvote_LEXD', 'vote_LUDR', 'pvote_LUDR'
            ],
            "CD": [
                'vote_LDVD', 'pvote_LDVD', 'vote_LLR', 'pvote_LLR', 'vote_LUD', 'pvote_LUD',
            ],
            "CG": [
                'vote_LDVG', 'pvote_LDVG', 'vote_LECO', 'pvote_LECO', 'vote_LSOC', 'pvote_LSOC', 'vote_LVEC', 'pvote_LVEC', 'vote_LUG', 'pvote_LUG'
            ],
            "C": [
                'vote_LDVC', 'pvote_LDVC', 'vote_LUC', 'pvote_LUC', 'vote_LREN', 'pvote_LREN',
                'vote_LMDM', 'pvote_LMDM', 'vote_LHOR', 'pvote_LHOR', 'vote_LUDI', 'pvote_LUDI'
            ]
        }
    },
    '2026_muni_t2': {
        'LINK': {0: "ingestion/municipales/2026/2026_t2.csv", 1: "ingestion/municipales/2026/2026_plm_t2.csv"},
        'SAVE_TO': "raw/elections/municipales/2026/muni2026_csv/muni2026comm_t2.parquet",
        'FILE_FORMAT': ".csv",
        'NB_OF_FILE': 2,
        'SOURCE': 'ingestion',
        'NUANCE_ID': 'liste',
        "political_mapping": {
            "G": [
                'vote_LEXG', 'pvote_LEXG', 'vote_LFI', 'pvote_LFI', 'vote_LCOM', 'pvote_LCOM',
            ],
            "D": [
                'vote_LRN', 'pvote_LRN', 'vote_LUXD', 'pvote_LUXD',
                'vote_LEXD', 'pvote_LEXD', 'vote_LUDR', 'pvote_LUDR'
            ],
            "CD": [
                'vote_LDVD', 'pvote_LDVD', 'vote_LLR', 'pvote_LLR', 'vote_LUD', 'pvote_LUD',
            ],
            "CG": [
                'vote_LDVG', 'pvote_LDVG', 'vote_LECO', 'pvote_LECO', 'vote_LSOC', 'pvote_LSOC', 'vote_LVEC', 'pvote_LVEC', 'vote_LUG', 'pvote_LUG'
            ],
            "C": [
                'vote_LDVC', 'pvote_LDVC', 'vote_LUC', 'pvote_LUC',
                'vote_LHOR', 'pvote_LHOR', 'vote_LUDI', 'pvote_LUDI'
            ]
        }
    },
    '2014_muni': {
        'LINK': {0: "ingestion/municipales/2014/2014_1000_moins.csv", 1: "ingestion/municipales/2014/2014_1000_plus.csv"},
        'SAVE_TO': "raw/elections/municipales/2014/muni2014_csv/muni2014comm.parquet",
        'FILE_FORMAT': "csv",
        'NB_OF_FILE': 2,
        'SOURCE': 'ingestion',
        'NUANCE_ID': 'liste',
        'political_mapping' : {
            "G": [
                'vote_LFG', 'pvote_LFG', 'vote_LCOM', 'pvote_LCOM',  'vote_LEXG', 'pvote_LEXG', 'vote_LEXG', 'pvote_LEXG',
            ],
            "D": [
                'vote_LFN', 'pvote_LFN', 'vote_LEXD', 'pvote_LEXD', 'vote_LEXD', 'pvote_LEXD'
            ],
            "CD": [
                'vote_LDVD', 'pvote_LDVD', 'vote_LUMP', 'pvote_LUMP', 'vote_LUD', 'pvote_LUD',
                'vote_LUDI', 'pvote_LUDI', 'vote_LUDI', 'pvote_LUDI',
                'vote_LUD', 'pvote_LUD', 'vote_LDVD', 'pvote_LDVD', 'vote_LUMP', 'pvote_LUMP'
            ],
            "CG": [
                'vote_LUG', 'pvote_LUG', 'vote_LDVG', 'pvote_LDVG', 'vote_LSOC', 'pvote_LSOC', 'vote_LVEC', 'pvote_LVEC',
                'vote_LPG', 'pvote_LPG',
            ],
            "C": [
                'vote_LMDM', 'pvote_LMDM', 'vote_LUC', 'pvote_LUC', 'vote_LUC', 'pvote_LUC'
            ],
        },
    },
    '2014_muni_t2': {
        'LINK': "ingestion/municipales/2014/2014_1000_plus_t2.csv",
        'SAVE_TO': "raw/elections/municipales/2014/muni2014_csv/muni2014comm_t2.parquet",
        'FILE_FORMAT': "csv",
        'NB_OF_FILE': 1,
        'SOURCE': 'ingestion',
        'NUANCE_ID': 'liste',
        'political_mapping' : {
            "G": [
                'vote_LFG', 'pvote_LFG', 'vote_LCOM', 'pvote_LCOM',  'vote_LEXG', 'pvote_LEXG', 'vote_LEXG', 'pvote_LEXG',
            ],
            "D": [
                'vote_LFN', 'pvote_LFN', 'vote_LEXD', 'pvote_LEXD', 'vote_LEXD', 'pvote_LEXD'
            ],
            "CD": [
                'vote_LDVD', 'pvote_LDVD', 'vote_LUMP', 'pvote_LUMP', 'vote_LUD', 'pvote_LUD',
                'vote_LUDI', 'pvote_LUDI', 'vote_LUDI', 'pvote_LUDI',
                'vote_LUD', 'pvote_LUD', 'vote_LDVD', 'pvote_LDVD', 'vote_LUMP', 'pvote_LUMP'
            ],
            "CG": [
                'vote_LUG', 'pvote_LUG', 'vote_LDVG', 'pvote_LDVG', 'vote_LSOC', 'pvote_LSOC', 'vote_LVEC', 'pvote_LVEC',
                'vote_LPG', 'pvote_LPG',
            ],
            "C": [
                'vote_LMDM', 'pvote_LMDM', 'vote_LUC', 'pvote_LUC', 'vote_LUC', 'pvote_LUC'
            ],
        },
    },
    '2020_muni_t2': {
        'LINK': {0: "ingestion/municipales/2020/2020_1000_plus_t2.csv", 1:"ingestion/municipales/2020/2020_1000_moins_t2.csv"},
        'SAVE_TO': "raw/elections/municipales/2020/muni2020_csv/muni2020comm_t2.parquet",
        'FILE_FORMAT': "csv",
        'NB_OF_FILE': 2,
        'SOURCE': 'ingestion',
        'NUANCE_ID': 'liste',
        'political_mapping': {
            "G": ['vote_LUG', 'pvote_LUG', 'vote_LEXG', 'pvote_LEXG',  'vote_LFI', 'pvote_LFI', 'pvote_LCOM', 'vote_LCOM'],
            "D": ['vote_LRN', 'pvote_LRN', 'vote_LEXD', 'pvote_LEXD'],
            "CD": ['vote_LDVD', 'pvote_LDVD', 'pvote_LLR', 'vote_LLR',  'pvote_LUD', 'vote_LUD', 'vote_LUDI', 'pvote_LUDI'],
            "CG": ['vote_LDVG', 'pvote_LDVG', 'pvote_LSOC', 'vote_LSOC', 'vote_LECO', 'pvote_LECO'],
            "C": ['pvote_LDVC', 'vote_LDVC', 'pvote_LREM', 'vote_LREM', 'vote_LUC', 'pvote_LUC'],
        }
    },
    }
data_path = "s3://arthurmanceau/election_modeling_uhcp/data/"


class ElectionIngester:

    def __init__(self, election_code):
        self.access_link = election_to_ingest[election_code]['LINK']
        self.data_path = data_path
        self.output_file = election_to_ingest[election_code]['SAVE_TO']
        self.political_mapping = election_to_ingest[election_code]['political_mapping']
        self.file_format = election_to_ingest[election_code]['FILE_FORMAT']
        self.nb_file = election_to_ingest[election_code]['NB_OF_FILE']
        self.nuance_id = election_to_ingest[election_code]['NUANCE_ID']
        self.source = election_to_ingest[election_code]['SOURCE']

    def download_open_and_delete_file(
        self, url, folder_path="data/raw/temp", file_name="election_temp",
    ):
        if self.source == 'API':
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, file_name+self.file_format)

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

            if self.file_format == '.xlsx':
                data = pd.read_excel(file_path)
            else:
                data = pd.read_csv(file_path, sep=';', low_memory=False)

            if os.path.exists(file_path):
                os.remove(file_path)
        else:
            data = DataLoader.load_dataset(self.data_path + url, formate='csv')

        return data

    def get_and_rename(
        self,
        X,
        mapping={
            'select_unique': 'select_unique',
            'plm': 'plm',
            "Code département": "dep",
            "Libellé département": "nomdep",
            "Code commune": "codecommune",
            "Libellé commune": "nomcommune",
            "Inscrits": "inscrits",
            "Abstentions": "abstentions",
            "Votants": "votants",
            "Exprimés": "exprimes",
            "Blancs": "blancs",
            "Nuls": "nuls",
            "% Abstentions": "% Abstentions",
            "% Vot/Ins": "% Votants",
            "% Votants": "% Votants",
            "% Nuls/inscrits": "% Nuls/inscrits",
            "% Nuls/votants": "% Nuls/votants",
            "% Blancs/inscrits": "% Blancs/inscrits",
            "% Blancs/votants": "% Blancs/votants"
        },
    ):
        cols = list(set(mapping.keys()).intersection(set(X.columns)))
        X_ = X[cols].copy(deep=True)
        X_.rename(columns=mapping, inplace=True)
        return X_

    def pivot_data(self, data):
        pivot_data = []
        N = [int(s) for s in data.columns.to_list()[-1].split() if s.isdigit()][0]
        for idx, row in data.iterrows():
            nuance_seen = []
            codecommune = row.get("Code commune")
            j = 0
            for i in range(1, N + 1):
                nuance = row.get(f"Nuance {self.nuance_id} {i}", None)
                vote = row.get(f"Voix {i}", None)
                pvote = row.get(f"% Voix/exprimés {i}", None)
                if pd.isna(nuance) and float(str(vote).replace(",", ".")) > 0:
                    # Nuance should be taken into consideration
                    nuance = f'AUTRES_{j}'
                    j += 1
                if pd.notna(nuance) and nuance!='nan':
                    if nuance in nuance_seen:
                        matches = [
                            item for item in pivot_data
                            if item["codecommune"] == codecommune and item["trend"] == nuance
                        ]
                        assert len(matches) == 1
                        element = matches[0]
                        vote_ = element['vote']
                        pvote_ = element['pvote']
                        pivot_data.remove(element)
                        vote = float(str(vote).replace(",", "."))
                        pvote = float(str(pvote).replace(",", ".").replace("%", ""))
                        pivot_data.append(
                                {
                                    "codecommune": codecommune,
                                    "trend": nuance,
                                    "vote": vote+vote_,
                                    "pvote": pvote+pvote_,
                                }
                            )

                    else:
                        nuance_seen.append(nuance)
                        vote = float(str(vote).replace(",", "."))
                        pvote = float(str(pvote).replace(",", ".").replace("%", ""))
                        pivot_data.append(
                                {
                                    "codecommune": codecommune,
                                    "trend": nuance,
                                    "vote": vote,
                                    "pvote": pvote,
                                }
                            )

        pivot_dataset = pd.DataFrame(pivot_data)

        pivot_dataset_grouped = (
            pivot_dataset.groupby(["codecommune", "trend"], dropna=False)
            .agg({"vote": "sum", "pvote": "sum"})
            .reset_index()
        )

        # # Pivot to wide format
        X = pivot_dataset_grouped.pivot(
            index="codecommune", columns="trend", values=["vote", "pvote"]
        )

        # # Flatten the columns
        X.columns = [f"{stat}_{nuance}" for stat, nuance in X.columns]
        X = X.reset_index()
        X = self.apply_political_adjustments(X)
        return X

    def apply_political_adjustments(self, X):
        X = X.copy()
        trends = (
            X.isna().astype(int).sum(axis=0).sort_values().index.to_list()
        )
        trends.remove('codecommune')
        
        # Tendances AUTRES or not in mapping
        trends_in_mapping = set()
        for key in self.political_mapping.keys():
            trends_in_mapping = trends_in_mapping.union(self.political_mapping[key])
        trends_in_mapping = list(trends_in_mapping)

        trends_kept = trends_in_mapping
        trends_dropped = list(set(trends) - set(trends_kept))

        trends_kept_p = [col for col in trends_kept if "p" in col]
        trends_kept_v = set(trends_kept) - set(trends_kept_p)
        assert len(trends_kept_p) == len(trends_kept_v)

        trends_dropped_p = [col for col in trends_dropped if 'p' in col]
        trends_dropped_v = set(trends_dropped) - set(trends_dropped_p)
        assert len(trends_kept_p) == len(trends_kept_v)

        vote_nan_cols = [col for col in X.columns if 'vote_nan' in col]
        X['nb_list'] = X[list(set(X.columns)-set(['codecommune']+vote_nan_cols))].notna().sum(axis=1) / 2

        X['vote_AUTRES'] = X[list(trends_dropped_v)].sum(axis=1)
        X['pvote_AUTRES'] = X[trends_dropped_p].sum(axis=1)

        X = X[["codecommune"] + trends_kept + ['vote_AUTRES', 'pvote_AUTRES', 'nb_list']]
        X = X.fillna(0.0)
        X[trends_kept+['vote_AUTRES', 'pvote_AUTRES']] = X[trends_kept+['vote_AUTRES', 'pvote_AUTRES']].astype(float)

        X = X.loc[~(X[trends_kept_p+['pvote_AUTRES']].sum(axis=1) == 0), :]

        if (~(np.abs(X[trends_kept_p+['pvote_AUTRES']].sum(axis=1) - 100) < 2)).astype(int).sum() != 0:
            logger.warning(f"These communes are outliers: {X.loc[(~(np.abs(X[trends_kept_p+['pvote_AUTRES']].sum(axis=1) - 100) < 2)), :]['codecommune'].to_list()}")
            X = X.loc[(np.abs(X[trends_kept_p+['pvote_AUTRES']].sum(axis=1) - 100) < 2), :]

        assert (~(np.abs(X[trends_kept_p+['pvote_AUTRES']].sum(axis=1) - 100) < 2)).astype(int).sum() == 0

        for pol_trend in ["G", "D", "CD", "CG", "C"]:
            partis = [col.split("_")[1] for col in self.political_mapping[pol_trend]]
            X[f"vote{pol_trend}"] = X[
                [col for col in trends_kept_v if col.split("_")[1] in partis]
            ].sum(axis=1) 
            X[f"pvote{pol_trend}"] = X[
                [col for col in trends_kept_p if col.split("_")[1] in partis]
            ].sum(axis=1) / 100

        for p_ in ['vote', 'pvote']:
            X[f"{p_}GCG"] = X[f"{p_}G"] + X[f"{p_}CG"]
            X[f"{p_}DCD"] = X[f"{p_}D"] + X[f"{p_}CD"]
            X[f"{p_}TG"] = X[f"{p_}G"] + X[f"{p_}CG"] + X[f"{p_}C"] / 2
            X[f"{p_}TD"] = X[f"{p_}D"] + X[f"{p_}CD"] + X[f"{p_}C"] / 2

        return X

    def save_processed_election(self, X):
        if not DataUtils._detect_s3(self.data_path):
            os.makedirs(self.output_file, exist_ok=True)
            DataLoader.write_dataset(X, self.data_path + self.output_file)
        else:
            DataLoader.write_dataset(X, self.data_path + self.output_file)

    def run(self):
        # Get data from source
        if self.nb_file == 1:
            data = self.download_open_and_delete_file(self.access_link)
        else:
            data_dict = {}
            for i in range(self.nb_file):
                data_i = self.download_open_and_delete_file(self.access_link[i])
                data_dict[i] = data_i
            data = pd.concat([data_dict[0], data_dict[1]], axis=0)

        # Ids
        data_i = self.get_and_rename(data)

        # Vote
        data_j = self.pivot_data(data)

        # Merge
        data_merged = data_i.merge(data_j, on="codecommune", how="inner")
        data_merged = data_merged.copy()
        cols_to_string = ['dep', 'nomdep', 'codecommune', 'nomcommune']
        data_merged[cols_to_string] = data_merged[cols_to_string].astype(str)

        cols_to_int = ["inscrits", "abstentions", "votants", "exprimes", "blancs", "nuls"]
        data_merged[cols_to_int] = data_merged[cols_to_int].astype(int)

        cols_to_float = ["% Abstentions", "% Votants", "% Nuls/inscrits", "% Nuls/votants", "% Blancs/inscrits", "% Blancs/votants"]
        data_merged[cols_to_float] = data_merged[cols_to_float].astype(str).replace({',': '.', r'%$': ''}, regex=True).astype(float)

        data_merged["ppar"] = data_merged['% Votants'] / 100
        data_merged["pabs"] = data_merged['% Abstentions'] / 100
        data_merged["pnulsi"] = data_merged["% Nuls/inscrits"] / 100
        data_merged["pnulsv"] = data_merged["% Nuls/votants"] / 100
        data_merged["pblancsi"] = data_merged["% Blancs/inscrits"] / 100
        data_merged["pblancsv"] = data_merged["% Blancs/votants"] / 100

        assert (~(np.abs(data_merged["votants"] / data_merged["inscrits"] - data_merged["ppar"]) < 0.01)).astype(int).sum() == 0
        assert (~(np.abs(data_merged["abstentions"] / data_merged["inscrits"] - data_merged["pabs"]) < 0.01)).astype(int).sum() == 0
        assert (~(np.abs(data_merged["ppar"] + data_merged["pabs"] - 1) < 0.01)).astype(int).sum() == 0
      
        self.save_processed_election(data_merged)


if __name__ == "__main__":
    for election_code in list(election_to_ingest.keys())[::-1]:
        logger.info(f'Ingesting: {election_code}')
        ingester = ElectionIngester(election_code)
        ingester.run()
