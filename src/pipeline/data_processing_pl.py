import gc
import os
import re
from datetime import datetime
import polars as pl
import pandas as pd
import polars as pl
from loguru import logger
from shapely.geometry import shape

from src.components.data_processing.data_loader import DataLoader, DataUtils
from src.components.data_processing.projections_module import (
    ProjectionModel,
    ProjectionUtils,
)
from src.components.utils.config import DataProcessingConfig
from src.components.utils.read_config import ConfigReader


"""
Objective: improve the data quality by re-designing the pipeline
- Reduce errors
- Be flexible for the integration of new features
- Be faster using polars

The main challenge is that we have communes that change through time.
We make the choice to only consider the communes actives in 2026 (this can be re-run to adapt to the new commune).

TABLE1: commune. This table list all the communes active in 2026, it is the ground truth for code commune, commune name and geographic information. Everything will be a left join from this table
TABLE2: election results. 
"""


class ElectionDataProcessor2:
    """Class to handle election data processing pipeline.
    
    Three building blocks:
        1. Election
        2. Socio-economic variables
        3. Merge
    """

    def __init__(self):
        """
        Initialize the data processor using the configuration file.
        """
        self.config = ConfigReader._read_config(
            "config/data_processing.json", DataProcessingConfig
        )

    def load_electoral_data(self):
        folder_path = os.path.join(self.config.data_path + "raw/", "elections")
        election_included = []
        election_datasets = {}
        breakpoint()
        xs = (
            DataUtils._create_fs()
            if DataUtils._detect_s3(self.config.data_path)
            else os
        )
        for root, dirs, files in xs.walk(folder_path):
            for file in files:
                if (
                    file.endswith(".parquet")
                    and (not file.startswith("."))
                    and file not in self.config.elections_to_exclude
                ):
                    path_element = os.path.relpath(root, folder_path).split(os.sep)[
                        -3:
                    ]  # ['legislative', '1848', 'leg1848_csv']
                    election_type = path_element[0]  # 'legislative'
                    year = path_element[1]  # '1848'
                    code = path_element[2].split('_')[0] # leg1848
                    election_included.append(code)
                    logger.info(f"Processing election : {(election_type, year)}")

                    file_path = DataUtils.path_helper(
                        folder_path, os.path.join(root, file)
                    )
                    df = DataLoader.load_dataset(file_path=file_path)
                    df["codecommune"] = df["codecommune"].astype(str)
                    breakpoint()
                    election_datasets[code] = df[["codecommune", "inscrits"] + self.config.vote_variables]

        return election_datasets
    


def main():
    """Main function to run the data processing pipeline"""
    # Initialize processor
    processor = ElectionDataProcessor2()

    processor.load_electoral_data()

    return processor


if __name__ == "__main__":
    dataset = main()
