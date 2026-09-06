from typing import List, Literal, Optional, Tuple, Union
import s3fs
import requests
from loguru import logger
import pandas as pd

import polars as pl
from asset.definitions import API_URL
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
            self.storage_options = {
                "aws_access_key_id": key,
                "aws_secret_access_key": secret,
                "aws_region": "us-east-1",
            }
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


def get_storage_options():
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
        commmunes_list = pd.read_csv("app/asset/communes2022.csv")
        self.container["communes_list"] = commmunes_list

    def _make_api_request(self, endpoint: str, payload: dict):
        url = f"{API_URL}{endpoint}"
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def _load_result_s3(self, asset, year, election_type, trends, columns, filters):
        file_path = f"{self.data_path}/output/results/{asset}_{year}_{election_type}_{'_'.join(trends)}_{self.version}.parquet"
        element = DataLoader.load_dataset(
            file_path,
            fs=get_fs().fs,
            formate="parquet",
            columns=columns,
            filters=filters,
            engine="polars-pyarrow",
        )
        return _convert_to_pandas(element)

    def _load_explain_s3(self, asset, trends, year, election_type, columns, filters):
        self.container[asset] = {}
        trends_ = [f"tau{trend}" for trend in trends] if self.tau else trends
        fs = get_fs().fs
        with ThreadPoolExecutor(max_workers=len(trends)) as executor:
            futures = [
                executor.submit(
                    self._load_trend,
                    fs,
                    self.version,
                    asset,
                    trend,
                    trends_,
                    year,
                    election_type,
                    columns,
                    filters,
                )
                for trend in trends
            ]
            for future in futures:
                trend, data = future.result()
                self.container[asset][trend] = data
        return self.container[asset]

    def _load_data_sample_s3(self, columns, filters):
        element = DataLoader.load_dataset(
            f"{self.data_path}/derived/processed/data_processed_presidentiel_legislative_from1800_to2027_20260707_143756.parquet/",
            fs=get_fs().fs,
            formate="parquet",
            columns=columns,
            filters=filters,
            engine="polars-pyarrow",
            hive_partitioning=True,
        )
        return _convert_to_pandas(element)

    def _build_pres_table_s3(
        self, df: pd.DataFrame, years: list, parties: list
    ) -> pd.DataFrame:
        result_cols = {}
        for year in years:
            d = df.loc[df["annee"] == year].copy()
            if d.empty:
                continue
            pred = {"pvotepar": d["pvotepar_pred"].sum()}
            true = {"pvotepar": d["pvotepar_true"].sum()}
            for p in parties:
                pred[f"pvote{p}"] = d[f"pvote{p}_pred"].sum()
                true[f"pvote{p}"] = d[f"pvote{p}_true"].sum()
            result_cols[f"{year}_pres_pred"] = pd.Series(pred)
            result_cols[f"{year}_pres_true"] = pd.Series(true)
        return pd.DataFrame(result_cols)

    def _load_results_over_time_s3(
        self, years, asset, election_type, trends, columns, filters, codecommune
    ):
        results = []
        for year in years:
            try:
                res = self._load_result_s3(
                    asset, year, election_type, trends, columns, filters
                )
                if "annee" not in res.columns:
                    res = res.copy()
                    res["annee"] = year
                results.append(res)
            except FileNotFoundError:
                continue
        return pd.concat(results, axis=0 if codecommune else 1) if results else None

    def load_results_backtest(
        self,
        years: List[int],
        asset: str,
        election_type: str,
        trends: List[str],
        columns: Optional[List] = None,
        filters: Optional[List[Tuple]] = None,
        codecommune: Optional[str] = None,
    ):
        try:
            payload = {
                "years": years,
                "asset": asset,
                "election_type": election_type,
                "trends": trends,
                "columns": columns,
                "filters": filters,
            }
            data = self._make_api_request("/data/results/backtest", payload)
            element = pd.DataFrame(data)
            logger.info("Backtest results loaded with success from API!")
        except Exception as e:
            logger.warning(
                f"Failed to load backtest from API: {e}. Falling back to S3."
            )
            # Fallback: Load over time and then aggregate if not in commune mode
            df_over_time = self._load_results_over_time_s3(
                years, asset, election_type, trends, columns, filters, codecommune
            )
            if df_over_time is None:
                element = None
            elif codecommune:
                element = df_over_time
            else:
                element = self._build_pres_table_s3(df_over_time, years, trends)
            logger.info("Backtest results loaded with success from S3!")

        self.container["backtest_results"] = element

    def _load_trend(
        self,
        fs,
        version,
        asset: str,
        trend: str,
        trends_: list,
        year: int,
        election_type: str,
        columns,
        filters,
    ):
        """Load a single trend dataset."""
        file_path = f"{self.data_path}/output/explain/{asset}_{trends_}_{trend}_{year}_{election_type}_{version}.parquet"
        element = DataLoader.load_dataset(
            file_path,
            fs=fs,
            formate="parquet",
            columns=columns,
            filters=filters,
            engine="polars-pyarrow",
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
        try:
            payload = {
                "asset": asset,
                "trends": trends,
                "year": year,
                "election_type": election_type,
                "columns": columns,
                "filters": filters,
            }
            data = self._make_api_request("/data/explain", payload)
            # The API returns a dict of lists (JSON), we need to convert them to DataFrames
            processed_data = {trend: pd.DataFrame(df) for trend, df in data.items()}
            self.container[asset] = processed_data
            logger.info(f"{asset} loaded with success from API!")
        except Exception as e:
            logger.warning(f"Failed to load {asset} from API: {e}. Falling back to S3.")
            self._load_explain_s3(asset, trends, year, election_type, columns, filters)
            logger.info(f"{asset} loaded with success from S3!")

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
            trends = [f"tau{trend}" for trend in trends]

        try:
            payload = {
                "asset": asset,
                "year": year,
                "election_type": election_type,
                "trends": trends,
                "columns": columns,
                "filters": filters,
            }
            data = self._make_api_request("/data/results", payload)
            element = pd.DataFrame(data)
            logger.info(f"{asset} loaded with success from API!")
        except Exception as e:
            logger.warning(f"Failed to load {asset} from API: {e}. Falling back to S3.")
            element = self._load_result_s3(
                asset, year, election_type, trends, columns, filters
            )
            logger.info(f"{asset} loaded with success from S3!")

        asset_name = asset_name if asset_name is not None else asset
        self.container[asset_name] = element

    def load_data_sample(
        self,
        columns: Optional[List] | None = None,
        filters: Optional[List[Tuple]] | None = None,
        asset_name: Optional[str] | None = None,
        use_api: bool = True,
    ):
        if use_api:
            try:
                url = API_URL + "/data/sample"
                payload = {
                    "columns": list(columns) if isinstance(columns, set) else columns,
                    "filters": filters,
                    "asset_name": asset_name if asset_name is not None else "data",
                }
                response = requests.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                element = pd.DataFrame(data)
                logger.info(f"{asset_name} loaded with success from API!")
            except Exception as e:
                logger.warning(
                    f"Failed to load {asset_name} from API: {e}. Falling back to S3."
                )
                element = self._load_data_sample_s3(columns, filters)
                logger.info(f"{asset_name} loaded with success from S3!")
        else:
            element = self._load_data_sample_s3(columns, filters)
            logger.info(f"{asset_name} loaded with success from S3!")

        asset_name = asset_name if asset_name is not None else "data"
        self.container[asset_name] = element
