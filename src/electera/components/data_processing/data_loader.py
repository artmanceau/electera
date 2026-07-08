"""Module with utils function to load data"""

import json
import os
import pickle
import shutil
from typing import List, Optional, Tuple, Literal

import pandas as pd
import pyarrow.dataset as ds
import polars as pl
import s3fs
from loguru import logger


def convert_filters(filters: list[tuple]) -> pl.Expr:
    op_map = {
            "==": lambda col, val: pl.col(col) == val,
            "!=": lambda col, val: pl.col(col) != val,
            ">": lambda col, val: pl.col(col) > val,
            ">=": lambda col, val: pl.col(col) >= val,
            "<":  lambda col, val: pl.col(col) < val,
            "<=": lambda col, val: pl.col(col) <= val,
            'in': lambda col, val: pl.col(col).is_in(val)
    }
    exprs = [op_map[op](col, val) for col, op, val in filters]
    return exprs


class DataUtils:
    @staticmethod
    def _detect_s3(file_path: str) -> bool:
        return file_path.startswith("s3://")

    @staticmethod
    def _create_fs():
        return s3fs.S3FileSystem()

    @staticmethod
    def _create_directory(directory_path, fs=None):
        # No action required if S3
        if fs is None:
            os.makedirs(directory_path, exist_ok=True)

    @staticmethod
    def _remove_file(file_path, fs=None):
        if DataUtils._exists(file_path, fs):
            if fs:
                fs.rm(file_path)
            else:
                os.remove(file_path)

    @staticmethod
    def _remove_dir(dir_path, fs=None):
        if DataUtils._exists(dir_path, fs):
            if fs:
                fs.rm(dir_path, recursive=True)
            else:
                shutil.rmtree(dir_path)

    @staticmethod
    def _read_parquet(
        file_path: str,
        fs: object = None,
        columns: Optional[List] | None = None,
        filters: Optional[List[Tuple]] | None = None,
        hive_partitioning: Optional[bool] = False,
        storage_otpions: Optional[dict] | None = None
    ) -> pd.DataFrame:
        """Reads a parquet file.

        Args:
            file_path (str): file path
            fs (Optional[object], optional): If in S3, precise filesystem. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        logger.debug(f"Loading dataset from {file_path}...")
        if fs is None:
            data = pd.read_parquet(file_path, filters=filters)
        else:
            data = pd.read_parquet(
                file_path, filesystem=fs, columns=columns, filters=filters
            )
        return data

    @staticmethod
    def _read_parquet_pl(
        file_path: str,
        fs: object = None,
        columns: Optional[List] | None = None,
        filters: Optional[List[Tuple]] | None = None,
        hive_partitioning: Optional[bool] = False,
        storage_options: Optional[dict] | None = None
    ) -> pd.DataFrame:
        logger.debug(f"Loading dataset from {file_path}...")
        if fs is None:
            lf = pl.scan_parquet(file_path)
        else:
            lf = pl.scan_parquet(
                file_path,
                storage_options={
                    "aws_endpoint_url": "https://minio.lab.sspcloud.fr",
                    "aws_region": "us-east-1",
                },
                credential_provider=pl.CredentialProviderAWS(
                    profile_name="default",
                    region_name="us-east-1",
                ),
            )
            if filters:
                exprs = convert_filters(filters)
                lf = lf.filter(*exprs)
            if columns:
                lf = lf.select(columns)
        return lf.collect(streaming=True)

    def _read_parquet_pl_pyarrow(
        file_path: str,
        fs: object = None,
        columns: Optional[List] | None = None,
        filters: Optional[List[Tuple]] | None = None,
        hive_partitioning: Optional[bool] = False,
        storage_options: Optional[dict] | None = None
    ) -> pd.DataFrame:
        logger.debug(f"Loading dataset from {file_path}...")
        if fs is None:
            lf = pl.scan_parquet(file_path)
        else:
            if hive_partitioning:
                dset = ds.dataset(file_path, filesystem=fs, format="parquet", partitioning="hive")
            else:
                dset = ds.dataset(file_path, filesystem=fs, format="parquet")
            lf = pl.scan_pyarrow_dataset(dset)
            if filters:
                exprs = convert_filters(filters)
                lf = lf.filter(*exprs)
            if columns:
                lf = lf.select(columns)
        return lf.collect(streaming=True)


    def _read_csv(
        file_path: str,
        fs: object = None,
        columns: Optional[List] | None = None,
        filters: Optional[List[Tuple]] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Reads a parquet file.

        Args:
            file_path (str): file path
            fs (Optional[object], optional): If in S3, precise filesystem. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        logger.debug(f"Loading dataset from {file_path}...")
        if fs is None:
            data = pd.read_csv(file_path)
        else:
            data = pd.read_csv(fs.open(file_path, mode="rb"), low_memory=False)
        return data

    def _to_parquet(df: pd.DataFrame, file_path: str, fs=None) -> None:
        logger.debug(f"Writing dataset at {file_path}...")
        if fs is None:
            df.to_parquet(file_path, index=False)
        else:
            df.to_parquet(file_path, index=False, filesystem=fs)

    def _exists(file_path: str, fs=None) -> bool:
        """Checks if a file exists

        Args:
            file_path (str): _description_
            fs (_type_, optional): _description_. Defaults to None.

        Raises:
            FileNotFoundError: _description_
            FileNotFoundError: _description_

        Returns:
            bool: _description_
        """
        checker = os.path if fs is None else fs
        return checker.exists(file_path)

    @staticmethod
    def path_helper(start_path: str, end_path: str) -> str:
        """
        Merge two paths by taking everything from start_path at start,
        everything from end_path at end, and only one iteration of duplicated elements
        in the middle.

        Args:
            start_path (str): The starting path
            end_path (str): The ending path

        Returns:
            str: Merged path with duplicates removed from the middle
        """
        # Handle protocol prefix (s3://, http://, etc.)
        prefix = ""
        if "://" in start_path:
            prefix, start_path = start_path.split("://", 1)
            prefix += "://"

        # Remove protocol from end_path if present
        if "://" in end_path:
            _, end_path = end_path.split("://", 1)

        # Split paths into parts, removing empty parts
        prev_parts = [p for p in start_path.replace("\\", "/").split("/") if p]
        end_parts = [p for p in end_path.replace("\\", "/").split("/") if p]

        if not prev_parts:
            return prefix + "/".join(end_parts)
        if not end_parts:
            return prefix + "/".join(prev_parts)

        # Find the longest overlap from the end of prev_parts and start of end_parts
        max_overlap = 0
        overlap_len = min(len(prev_parts), len(end_parts))

        for i in range(1, overlap_len + 1):
            if prev_parts[-i:] == end_parts[:i]:
                max_overlap = i

        # Merge: prev_parts + end_parts[overlap:]
        if max_overlap > 0:
            merged_parts = prev_parts + end_parts[max_overlap:]
        else:
            merged_parts = prev_parts + end_parts

        return prefix + "/".join(merged_parts)

    @staticmethod
    def _read_geojson(geo_data_path: str, fs=None) -> pd.DataFrame:
        if fs is None:
            with open(geo_data_path) as f:
                communes_geojson = json.load(f)
        else:
            with fs.open(geo_data_path) as f:
                communes_geojson = json.load(f)
        return communes_geojson


class DataLoader:
    @staticmethod
    def load_dataset(
        file_path: str,
        fs: Optional[object] | None = None,
        formate: str = "parquet",
        columns: Optional[List] | None = None,
        filters: Optional[List[Tuple]] | None = None,
        engine: Literal['pandas', 'polars'] = "pandas",
        hive_partitioning: Optional[bool] = False,
        storage_options: Optional[dict] | None = None
    ) -> pd.DataFrame:
        """Loads a dataset either locally or in S3 depending on the file_path

        Args:
            file_path (str): file_path

        Raises:
            FileNotFoundError: If the file is not found

        Returns:
            pd.DataFrame: dataframe with the content of the dataset
        """
        read_method = {
            "pandas": {"csv": DataUtils._read_csv, "parquet": DataUtils._read_parquet},
            "polars": {"parquet": DataUtils._read_parquet_pl},
            'polars-pyarrow': {'parquet': DataUtils._read_parquet_pl_pyarrow}
        }
        # S3 path - starts with s3://
        if not fs:
            fs = DataUtils._create_fs() if DataUtils._detect_s3(file_path) else None
        if not DataUtils._exists(file_path, fs):
            raise FileNotFoundError(f"The file {file_path} can't be found.")
        data = read_method[engine][formate](file_path, fs, columns, filters, hive_partitioning, storage_options)
        logger.debug(f"Dataset loaded: {data.shape}")
        return data

    @staticmethod
    def load_geojson(
        geo_data_path: str, fs: Optional[object] | None = None
    ) -> pd.DataFrame:
        if not fs:
            fs = DataUtils._create_fs() if DataUtils._detect_s3(geo_data_path) else None
        if not DataUtils._exists(geo_data_path, fs):
            raise FileNotFoundError(f"The file {geo_data_path} can't be found.")
        geojson = DataUtils._read_geojson(geo_data_path, fs)
        logger.debug("Geojson loaded")
        return geojson

    @staticmethod
    def write_dataset(data: pd.DataFrame, file_path: str) -> None:
        """"""
        fs = DataUtils._create_fs() if DataUtils._detect_s3(file_path) else None
        DataUtils._to_parquet(data, file_path, fs)
        logger.debug("Dataset saved")

    @staticmethod
    def dump_pickle(object_to_pickle: object, file_path: str) -> None:
        fs = DataUtils._create_fs() if DataUtils._detect_s3(file_path) else None
        if fs:
            with fs.open(file_path, "wb") as file:
                pickle.dump(object_to_pickle, file)
        else:
            with open(file_path, "wb") as file:
                pickle.dump(object_to_pickle, file)

    @staticmethod
    def load_pickle(file_path: str, fs: Optional[object] | None = None) -> None:
        if not fs:
            fs = DataUtils._create_fs() if DataUtils._detect_s3(file_path) else None
        if fs:
            with fs.open(file_path, "rb") as file:
                return pickle.load(file)
        else:
            with open(file_path, "rb") as file:
                return pickle.load(file)
