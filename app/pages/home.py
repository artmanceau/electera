import streamlit as st

from src.components.data_processing.data_loader import DataLoader
from core.utils import trends
from src.components.utils.config import AppConfig
from src.components.utils.read_config import ConfigReader
from core.data_handler import FileSystem

st.divider()

# The home page presents the project, load key elements (fs & config) and show an image of the back-testing (maybe interactive graph later on)
# Later the pages are set onto 2027

@st.cache_data
def load_config():
    return ConfigReader._read_config("config/app_config.json", AppConfig)


st.markdown(
    """
    Cette application propose de prédire les comportements electoraux à l'aide d'un modèle de Machine Learning.
    L'entrainement se fait à partir des données socio-économiques et des résultats des élections précédentes.
    On vise à prédire les résultats dans chaque commune.
    """
)

st.markdown(
    """
    Auteur et créateur : Arthur Manceau

    Contact : art.manceau [at] gmail [.] com
    """
)


st.session_state["config"] = load_config()

# Not needed - move to the head of pages
st.session_state["ELECTION_YEAR"] = st.selectbox(
    "Election year", st.session_state["config"].years_to_display, index=0
)
st.session_state["ELECTION_TYPE"] = st.selectbox(
    "Election type", st.session_state["config"].types_to_display, index=0
)

# Instantiate fs — cache it (with cache ressources)
fs = FileSystem(client_kwargs='https://'+'minio.lab.sspcloud.fr', key=st.secrets["AWS_ACCESS_KEY_ID"], secret=st.secrets["AWS_SECRET_ACCESS_KEY"])
fs.load_fs()


# Two elements :
# 1. global results already aggregated (result synth should be computed from the aggregate result function)
# 2. detailed results queried with pandas parquet filters not to load everything
# 3. File feature importance global (fast to load) — definition of features to store in features
# 4. Shap values for commune same + a global shap file where we query ~20% of the data - needs calibration on average prediction
# 5. Map : maybe let the page load at

# Clarify the paths that should be on


# Include help (identify partis politiques)


## Loading should be done in each pages
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
            f"{data_path}/output/results/{asset}_{year}_{election_type}_{version}.parquet", fs=fs.get_fs()
        )

    for asset in explain_assets_to_load:
        st.session_state["data"][asset] = {}
        for b in trends:
            st.session_state["data"][asset][b] = DataLoader.load_dataset(
                f"{data_path}/output/explain/{asset}_{b}_{year}_{election_type}_{version}.parquet", fs=fs.get_fs()
            )
            st.session_state["data"][asset][b] = DataLoader.load_dataset(
                f"{data_path}/output/explain/{asset}_{b}_{year}_{election_type}_{version}.parquet", fs=fs.get_fs()
            )

# Not needed anymore — will be loaded in app
if st.button("Charger l'application (environ 60 secondes)"):
    with st.spinner(text="Chargement en cours...", show_time=True, width="content"):
        load_assets(st.session_state["config"])

if "data" not in st.session_state:
    st.warning("Application non chargée")
else:
    st.success("Appliction chargée")
