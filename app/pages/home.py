import streamlit as st

from src.components.data_processing.data_loader import DataLoader
from src.components.streamlit_utils.utils import trends
from src.components.utils.config import AppConfig
from src.components.utils.read_config import ConfigReader

st.divider()


@st.cache_data
def load_config():
    config = ConfigReader._read_config("config/app_config.json", AppConfig)

    return config


st.session_state["config"] = load_config()
st.session_state["ELECTION_YEAR"] = st.selectbox(
    "Election year", st.session_state["config"].years_to_display, index=0
)
st.session_state["ELECTION_TYPE"] = st.selectbox(
    "Election type", st.session_state["config"].types_to_display, index=0
)


def load_assets(config):
    # Will load the relevant assets
    #   - Results for recent elections (2017, 2022, 2027 pres & leg)
    #   - (The model for these elections?)
    #   - Shap values
    #   - Explain assets

    st.session_state["data"] = {}
    version = st.session_state["config"].model_version
    year = st.session_state["ELECTION_YEAR"]
    election_type = st.session_state["ELECTION_TYPE"]
    data_path = config.data_path

    results_assets_to_load = [
        "results_synth",
        "results_full",
    ]  # , 'shap_values', 'feature_importance']
    explain_assets_to_load = ["feature_importance", "shap_values"]
    for asset in results_assets_to_load:
        st.session_state["data"][asset] = DataLoader.load_dataset(
            f"{data_path}/output/results/{asset}_{year}_{election_type}_{version}.parquet"
        )

    for asset in explain_assets_to_load:
        st.session_state["data"][asset] = {}
        for b in trends:
            st.session_state["data"][asset][b] = DataLoader.load_dataset(
                f"{data_path}/output/explain/{asset}_{b}_{year}_{election_type}_{version}.parquet"
            )
            st.session_state["data"][asset][b] = DataLoader.load_dataset(
                f"{data_path}/output/explain/{asset}_{b}_{year}_{election_type}_{version}.parquet"
            )


if st.button("Charger l'application"):
    with st.spinner(text="Chargement en cours...", show_time=True, width="content"):
        load_assets(st.session_state["config"])

if "data" not in st.session_state:
    st.warning("Application non chargée")
else:
    st.success("Appliction chargée")
