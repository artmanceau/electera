from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional, Any
import uvicorn
from loguru import logger
from electera.components.data_processing.data_loader import DataLoader
import os
import s3fs
from dotenv import load_dotenv
from pathlib import Path

API_DIR = Path(__file__).resolve().parent
ENV_FILE = API_DIR / ".env"

load_dotenv(ENV_FILE, override=False)
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "https://minio.lab.sspcloud.fr")
DATA_PATH = "s3://arthurmanceau/election_modeling_uhcp/data"
MODEL_VERSION = "4.5.0"

app = FastAPI(title="Electera Data API")


def get_s3_fs():
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": S3_ENDPOINT},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


class DataRequest(BaseModel):
    columns: Optional[List[str]] = None
    filters: Optional[List[Tuple[str, str, Any]]] = None
    asset_name: Optional[str] = "data"


class ResultsRequest(BaseModel):
    asset: str
    year: int
    election_type: str
    trends: List[str]
    columns: Optional[List[str]] = None
    filters: Optional[List[Tuple[str, str, Any]]] = None


class BacktestRequest(BaseModel):
    years: List[int]
    asset: str
    election_type: str
    trends: List[str]
    columns: Optional[List[str]] = None
    filters: Optional[List[Tuple[str, str, Any]]] = None


class ExplainRequest(BaseModel):
    asset: str
    trends: List[str]
    year: int
    election_type: str
    columns: Optional[List[str]] = None
    filters: Optional[List[Tuple[str, str, Any]]] = None


# Hardcoded path to the sample data as found in data_handler.py
SAMPLE_DATA_PATH = "s3://arthurmanceau/election_modeling_uhcp/data/derived/processed/data_processed_presidentiel_legislative_from1800_to2027_20260707_143756.parquet/"


@app.post("/data/sample")
async def load_data_sample(request: DataRequest):
    try:
        df = DataLoader.load_dataset(
            file_path=SAMPLE_DATA_PATH,
            fs=get_s3_fs(),
            columns=request.columns,
            filters=request.filters,
            engine="polars-pyarrow",
            hive_partitioning=True,
        )
        return df.to_dicts()
    except Exception as e:
        logger.error(f"Error loading data sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/results")
async def load_results(request: ResultsRequest):
    try:
        file_path = f"{DATA_PATH}/output/results/{request.asset}_{request.year}_{request.election_type}_{'_'.join(request.trends)}_{MODEL_VERSION}.parquet"
        df = DataLoader.load_dataset(
            file_path,
            fs=get_s3_fs(),
            formate="parquet",
            columns=request.columns,
            filters=request.filters,
            engine="polars-pyarrow",
        )
        return df.to_dicts()
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/explain")
async def load_explain(request: ExplainRequest):
    try:
        results = {}
        trends_processed = [f"tau{t}" for t in request.trends]
        trends_combined = "_".join(trends_processed)

        for trend in request.trends:
            trend_processed = f"tau{trend}"
            file_path = f"{DATA_PATH}/output/explain/{request.asset}_{trends_combined}_{trend_processed}_{request.year}_{request.election_type}_{MODEL_VERSION}.parquet"
            df = DataLoader.load_dataset(
                file_path,
                fs=get_s3_fs(),
                formate="parquet",
                columns=request.columns,
                filters=request.filters,
                engine="polars-pyarrow",
            )
            results[trend] = df.to_dicts()

        return results
    except Exception as e:
        logger.error(f"Error loading explain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/results/backtest")
async def load_results_backtest(request: BacktestRequest):
    try:
        all_results = {}
        for year in request.years:
            file_path = f"{DATA_PATH}/output/results/{request.asset}_{year}_{request.election_type}_{'_'.join(request.trends)}_{MODEL_VERSION}.parquet"
            df_pl = DataLoader.load_dataset(
                file_path,
                fs=get_s3_fs(),
                formate="parquet",
                columns=request.columns,
                filters=request.filters,
                engine="polars-pyarrow",
            )
            df = df_pl.to_pandas()

            # Aggregation logic similar to build_pres_table
            # Note: The a-priori assume "pres" as the fixed type for aggregation in build_pres_table
            # but we should use request.election_type or a reasonable default.
            # Original build_pres_table used "pres" in result_cols keys.

            pred = {"pvotepar": df["pvotepar_pred"].sum()}
            true = {"pvotepar": df["pvotepar_true"].sum()}

            for p in request.trends:
                pred[f"pvote{p}"] = df[f"pvote{p}_pred"].sum()
                true[f"pvote{p}"] = df[f"pvote{p}_true"].sum()

            all_results[f"{year}_{request.election_type}_pred"] = pred
            all_results[f"{year}_{request.election_type}_true"] = true

        # Return as a list of dicts (like a DataFrame.to_dicts())
        # The expected output of build_pres_table is a DataFrame where each column is a year_type_pred/true
        # And indices are the metrics (pvotepar, pvoteX).
        # To match this, we can return it as a list containing one dict.
        return [all_results]

    except Exception as e:
        logger.error(f"Error loading backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
