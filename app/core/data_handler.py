from typing import List, Literal, Optional, Tuple, Union
import s3fs
from loguru import logger
import pandas as pd
import polars as pl
from electera.components.data_processing.data_loader import DataLoader
from concurrent.futures import ThreadPoolExecutor

class FileSystem:
    instance = None

    def __new__(cls, client_kwargs, key, secret):
        """Create or return existing singleton instance."""
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self, client_kwargs, key, secret):
        if not hasattr(self, "initialized"):
            self.fs = s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": client_kwargs},
                key=key,
                secret=secret,
            )
            self.initialized = True

    @classmethod
    def is_initialized(cls):
        return cls.instance is not None and hasattr(cls.instance, "initialized")


def load_fs(client_kwargs, key, secret):
    return FileSystem(client_kwargs, key, secret)


def get_fs():
    if not FileSystem.is_initialized():
        raise ValueError("Call load_fs first!")

    return FileSystem.instance


def _convert_to_pandas(X: Union[pl.DataFrame, pd.DataFrame]):
    if isinstance(X, pl.DataFrame):
        return X.to_pandas()
    else:
        return X


class AppData:

    def __init__(self, data_path, version, tau):
        self.tau = tau
        self.data_path = data_path
        self.version = version
        self.container = {}

    def load_communes_list(self):
        commmunes_list = pd.read_csv('app/asset/communes2022.csv')
        self.container['communes_list'] = commmunes_list
        
    def _load_trend(self, fs, version, asset: str, trend: str, trends_: list, year: int, election_type: str, columns, filters):
        """Load a single trend dataset."""
        file_path = f"{self.data_path}/output/explain/{asset}_{trends_}_{trend}_{year}_{election_type}_{version}.parquet"
        element = DataLoader.load_dataset(
            file_path,
            fs=fs,
            formate="parquet",
            columns=columns,
            filters=filters,
        )
        return trend, _convert_to_pandas(element)

    def load_explain(
        self,
        asset: Literal["feature_importance", "shap_values"],
        trends: List[str],
        year: int,
        election_type: Literal["leg", "pres", "ref"],
        columns: Optional[List] | None = None,
        filters: Optional[List[Tuple]] | None = None,
        asset_name: Optional[str] | None = None,
    ):
        self.container[asset] = {}

        trends_ = [f'tau{trend}' for trend in trends] if self.tau else trends

        fs = get_fs().fs

        with ThreadPoolExecutor(max_workers=len(trends)) as executor:
            futures = [
                executor.submit(self._load_trend, fs, self.version, asset, trend, trends_, year, election_type, columns, filters)
                for trend in trends
            ]
            for future in futures:
                trend, data = future.result()
                self.container[asset][trend] = data

        logger.info(f"{asset} loaded with success!")

    def load_result(
        self,
        asset: Literal["result_full", "result_synth"],
        trends: List[str],
        year: int,
        election_type: Literal["leg", "pres", "ref"],
        columns: Optional[List] | None = None,
        filters: Optional[List[Tuple]] | None = None,
        asset_name: Optional[str] | None = None,
    ):
        if self.tau:
            trends = [f'tau{trend}' for trend in trends]

        element = DataLoader.load_dataset(
            f"{self.data_path}/output/results/{asset}_{year}_{election_type}_{trends}_{self.version}.parquet",
            fs=get_fs().fs,
            formate="parquet",
            columns=columns,
            filters=filters,
        )
        logger.info(f"{asset} loaded with success!")

        asset_name = asset_name if asset_name is not None else asset
        self.container[asset_name] = _convert_to_pandas(element)

    def load_data_sample(
        self,
        columns: Optional[List] | None = None,
        filters: Optional[List[Tuple]] | None = None,
        asset_name: Optional[str] | None = None,
    ):
        element = DataLoader.load_dataset(
            f"{self.data_path}/derived/processed/data_processed_presidentiel_legislative_from1800_to2027_20260623_174327.parquet",
            fs=get_fs().fs,
            formate="parquet",
            columns=columns,
            filters=filters,
        )
        logger.info(f"{asset_name} loaded with success!")
        asset_name = asset_name if asset_name is not None else "data"
        self.container[asset_name] = _convert_to_pandas(element)
