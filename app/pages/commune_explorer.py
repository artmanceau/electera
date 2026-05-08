import streamlit as st
from core.utils import check_home_run, present_results, show_shap_values

# from src.pipeline.counterfactuals import CounterfactualPipeline
# from core.data_handler import get_fs


@st.cache_data
def load_results():
    st.session_state["data"].load_result(
        asset="results_full",
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
        year=st.session_state["state"].year,
        election_type=st.session_state["state"].get_type(as_type="code"),
        columns=None,
        filters=[("codecommune", "==", st.session_state["state"].codecommune)],
        asset_name="results_commune_selected",
    )


@st.cache_data
def load_shap_values():
    st.session_state["data"].load_explain(
        asset="shap_values",
        year=st.session_state["state"].year,
        election_type=st.session_state["state"].get_type(as_type="code"),
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
        columns=None,
        filters=[("codecommune", "==", st.session_state["state"].codecommune)],
        asset_name="results_commune_selected",
    )


@st.cache_data
def load_communes_list():
    st.session_state["data"].load_result(
        asset="results_full",
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
        year=st.session_state["state"].year,
        election_type=st.session_state["state"].get_type(as_type="code"),
        columns=["codecommune", "nomcommune"],
        filters=None,
        asset_name="communes_list",
    )


check_home_run()

st.session_state["state"].selection_box()

st.header("Resultat au niveau de chaque commune")

load_communes_list()
st.session_state["state"].commune_selector()
st.write(
    f"Commune sélectionnée : {st.session_state['state'].commune} ({st.session_state['state'].codecommune})"
)

st.divider()

load_results()

present_results(
    st.session_state["data"].container["results_commune_selected"],
    year=st.session_state["state"].year,
    t=st.session_state["state"].get_type(as_type="code"),
    blocs=st.session_state["state"].get_blocs(as_type="code", order="political"),
    scale="local",
)

st.divider()

load_shap_values()

show_shap_values(
    st.session_state["data"].container["shap_values"],
    BLOCS=st.session_state["state"].get_blocs(as_type="code", order="political"),
    selection_code_commune=st.session_state["state"].codecommune,
)

# st.divider()

# VAR = st.selectbox(
#         "Tendance politique", BLOCS, index=0
#     )

# delta = st.slider(
#         "Changer le score de la tendance politique de x%",
#         -0.5,
#         0.5,
#     )

# Donner l'accès au service account à tout data
# cf_generator = CounterfactualPipeline(VAR, YEAR, TYPE, BLOCS, st.session_state["config"].model_version, st.session_state["config"].data_path+'/', fs=get_fs().fs)
# counterfactuals_list = cf_generator.run(codecommune=selection_code_commune, variation=delta)
# Presentation of counterfactuals
