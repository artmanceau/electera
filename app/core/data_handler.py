from typing import List, Tuple

import s3fs
import streamlit as st
from loguru import logger

from src.components.data_processing.data_loader import DataLoader


class FileSystem:

    def __init__(self, client_kwargs, key, secret):
        if not hasattr(self, "initialized"):
            self.fs = s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": client_kwargs},
                key=key,
                secret=secret,
            )
            self.initialized = True

    def is_initialized(self):
        return cls.instance is not None and hasattr(cls.instance, "initialized")


def load_fs(client_kwargs, key, secret):
    return FileSystem(client_kwargs, key, secret)


def get_fs():
    if not FileSystem.is_initialized():
        raise ValueError("FileSystem is not initialized. Call load_fs() first.")

    return FileSystem.instance


class AppData:

    def __init__(self, data_path, version):
        self.data_path = data_path
        self.version = version
        st.session_state["data"] = {}

    def load_element(
        self,
        asset: Literal[
            "result_full", "result_synth", "feature_importance", "shap_values"
        ],
        year: int,
        election_type: str,
        filters: List[Tuple] = None,
    ):
        sub_folder = {
            "feature_importance": "explain",
            "shap_values": "explain",
            "result_full": "results",
            "result_synth": "results",
        }
        # modify to load asset explain correctly
        element = DataLoader.load_dataset(
            f"{self.data_path}/output/{sub_folder[asset]}/{asset}_{year}_{election_type}_{self.version}.parquet",
            fs=get_fs().fs,
            filters=None,
        )
        st.session_state["data"]['results'][asset] = element
        logger.info(f"{asset} loaded with success!")
