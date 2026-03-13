import streamlit as st
from asset.definitions import client_kwargs
from core.data_handler import AppData, load_fs

from src.components.utils.config import AppConfig
from src.components.utils.read_config import ConfigReader
from core.state_handler import SessionHandler

st.divider()


@st.cache_data
def load_config():
    return ConfigReader._read_config("config/app_config.json", AppConfig)


def instantiate_filesystem():
    return load_fs(
        client_kwargs=client_kwargs,
        key=st.secrets["AWS_ACCESS_KEY_ID"],
        secret=st.secrets["AWS_SECRET_ACCESS_KEY"],
    )


def instantiate_session_state():
    st.session_state["home_run"] = True
    st.session_state["config"] = load_config()
    st.session_state["data"] = AppData(
        st.session_state["config"].data_path, st.session_state["config"].model_version
    )
    st.session_state['state'] = SessionHandler()


instantiate_filesystem()
instantiate_session_state()


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

st.divider()


pages = [
    {
        "name": "Résultat et prédictions",
        "link": "pages/election_results.py",
        "desc": "Analysez les résultats du modèle à l'echelle nationale",
        "icon": "🇫🇷",
    },
    {
        "name": "Communes",
        "link": "pages/commune_explorer.py",
        "desc": "Analysez les predictions du modèle commune par commune",
        "icon": "🔍",
    },
    {
        "name": "Cartes",
        "link": "pages/map_explorer.py",
        "desc": "Comparer les prédictions du modèle aux résultats sur une carte de la France",
        "icon": "🌍",
    },
    {
        "name": "Back-Testing",
        "link": "pages/back_testing.py",
        "desc": "Analysez les performances du modèle sur plusieurs années",
        "icon": "📈",
    },
]

for page in pages:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.page_link(page["link"], label=page["name"], icon=page["icon"])
    with col2:
        st.markdown(page["desc"])
    st.markdown("---")
