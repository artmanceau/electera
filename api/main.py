from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional, Any
import uvicorn
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from electera.components.data_processing.data_loader import DataLoader
import os
import s3fs
import pandas as pd


S3_ENDPOINT = "https://minio.lab.sspcloud.fr"


app = FastAPI(title="Electera Data API")


def build_pres_table(df, election_type, years, parties):
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
        result_cols[f"{year}_{election_type}_pred"] = pd.Series(pred)
        result_cols[f"{year}_{election_type}_true"] = pd.Series(true)
    return pd.DataFrame(result_cols)


def get_s3_fs():
    client_kwargs = {}
    if S3_ENDPOINT:
        client_kwargs["endpoint_url"] = S3_ENDPOINT
    return s3fs.S3FileSystem(
        client_kwargs=client_kwargs,
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


class DataRequest(BaseModel):
    columns: Optional[List[str]] = None
    filters: Optional[List[Tuple[str, str, Any]]] = None
    asset_name: Optional[str] = "data"
    sample_data_path: str


class ResultsRequest(BaseModel):
    asset: str
    year: int
    election_type: str
    trends: List[str]
    data_path: str
    model_version: str
    columns: Optional[List[str]] = None
    filters: Optional[List[Tuple[str, str, Any]]] = None


class BacktestRequest(BaseModel):
    years: List[int]
    election_type: str
    trends: List[str]
    data_path: str
    model_version: str
    codecommune: Optional[str] = None


class ExplainRequest(BaseModel):
    asset: str
    trends: List[str]
    year: int
    election_type: str
    data_path: str
    model_version: str
    columns: Optional[List[str]] = None
    filters: Optional[List[Tuple[str, str, Any]]] = None


@app.post("/data/sample")
async def load_data_sample(request: DataRequest):
    try:
        df = DataLoader.load_dataset(
            file_path=request.sample_data_path,
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
        file_path = f"{request.data_path}/output/results/{request.asset}_{request.year}_{request.election_type}_{'_'.join(request.trends)}_{request.model_version}.parquet"
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
        trends_combined = str(trends_processed)

        for trend in request.trends:
            file_path = f"{request.data_path}/output/explain/{request.asset}_{trends_combined}_{trend}_{request.year}_{request.election_type}_{request.model_version}.parquet"
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


@app.post("/data/backtest")
async def load_backtest(request: BacktestRequest):
    try:
        codecommune = request.codecommune
        years = request.years
        election_type = request.election_type
        trends = request.trends

        if codecommune:
            asset = "results_full"
            filters = [("codecommune", "==", codecommune)]
            columns = None
            axis = 0
        else:
            asset = "results_synth"
            columns = {
                year: [
                    "index",
                    f"{year}_{election_type}_pred",
                    f"{year}_{election_type}_true",
                ]
                for year in years
            }
            filters = None
            axis = 1

        def _load_one(year):
            file_path = (
                f"{request.data_path}/output/results/"
                f"{asset}_{year}_{election_type}_{'_'.join(trends)}_{request.model_version}.parquet"
            )
            element = DataLoader.load_dataset(
                file_path,
                fs=get_s3_fs(),
                formate="parquet",
                columns=(columns.get(year) if isinstance(columns, dict) else columns),
                filters=filters,
                engine="polars-pyarrow",
            )
            X = element.to_pandas() if hasattr(element, "to_pandas") else element
            if columns:
                X = X.set_index("index")
            if "annee" not in X.columns:
                X = X.copy()
                X["annee"] = year
            return X

        results = []
        with ThreadPoolExecutor(max_workers=max(1, len(years))) as executor:
            futures = [executor.submit(_load_one, year) for year in years]
            results = [f.result() for f in futures if f.result() is not None]

        if not results:
            return []

        df = pd.concat(results, axis=axis)

        if codecommune:
            # trends are tau-prefixed already; strip prefix for aggregation parties
            parties = [
                t.replace("tau", "") if t.startswith("tau") else t for t in trends
            ]
            df = build_pres_table(df, election_type, years, parties)

        # Ensure a serializable index column
        out = df.reset_index()
        if "index" not in out.columns:
            out = out.rename(columns={out.columns[0]: "index"})
        return out.to_dict(orient="records")

    except Exception as e:
        logger.error(f"Error loading backtest data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
