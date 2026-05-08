import sys

from dotenv import load_dotenv

project_root = "/home/onyxia/work/election_modeling_uhcp"
sys.path.insert(0, project_root)
from src.components.data_processing.data_loader import DataLoader

load_dotenv()

if __name__ == "__main__":
    dataset = DataLoader().load_dataset(
        "s3://arthurmanceau/election_modeling_uhcp/data/derived/processed/data_ppar_pvoteD_pvoteG_pvoteCG_pvoteCD_pvoteC_pvoteTD_pvoteTG_pvoteGCG_pvoteDCD_1958_presidentiel_legislative_20260202_110718.parquet"
    )
    breakpoint()
