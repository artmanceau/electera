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

load_dotenv(ENV_FILE, override=True)

app = FastAPI(title="Electera Data API")


class DataRequest(BaseModel):
    columns: Optional[List[str]] = None
    filters: Optional[List[Tuple[str, str, Any]]] = None
    asset_name: Optional[str] = "data"


# Hardcoded path to the sample data as found in data_handler.py
SAMPLE_DATA_PATH = "s3://arthurmanceau/election_modeling_uhcp/data/derived/processed/data_processed_presidentiel_legislative_from1800_to2027_20260707_143756.parquet/"


@app.post("/data/sample")
async def load_data_sample(request: DataRequest):
    try:
        # Use the existing DataLoader to fetch data from S3
        # DataLoader.load_dataset handles S3 detection and filesystem creation internally
        df = DataLoader.load_dataset(
            file_path=SAMPLE_DATA_PATH,
            fs=s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": os.environ["CLIENT_KWARGS"]},
                key=os.environ["AWS_ACCESS_KEY_ID"],
                secret=os.environ["AWS_SECRET_ACCESS_KEY"],
            ),
            columns=request.columns,
            filters=request.filters,
            engine="polars-pyarrow",
            hive_partitioning=True,
        )

        # Convert to JSON for the response
        return df.to_dicts()

    except Exception as e:
        logger.error(f"Error loading data sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
