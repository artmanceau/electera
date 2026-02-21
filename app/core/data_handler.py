from typing import List, Literal, Optional, Tuple

import s3fs
from loguru import logger

from src.components.data_processing.data_loader import DataLoader


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


class AppData:

    def __init__(self, data_path, version):
        self.data_path = data_path
        self.version = version
        self.container = {}

    def load_explain(
        self,
        asset: Literal["feature_importance", "shap_values"],
        trends: List[str],
        year: int,
        election_type: Literal["leg", "pres", "ref"],
        filters: Optional[List[Tuple]] | None = None,
    ):
        self.container[asset] = {}
        for trend in trends:
            element = DataLoader.load_dataset(
                f"{self.data_path}/output/explain/{asset}_{trends}_{trend}_{year}_{election_type}_{self.version}.parquet",
                fs=get_fs().fs,
                formate="parquet",
                filters=filters,
            )
            self.container[asset][trend] = element
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
        element = DataLoader.load_dataset(
            f"{self.data_path}/output/results/{asset}_{year}_{election_type}_{trends}_{self.version}.parquet",
            fs=get_fs().fs,
            formate="parquet",
            columns=columns,
            filters=filters,
        )
        logger.info(f"{asset} loaded with success!")

        asset_name = asset_name if asset_name is not None else asset
        self.container[asset_name] = element
