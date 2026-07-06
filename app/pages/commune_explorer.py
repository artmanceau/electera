import streamlit as st
from core.utils import check_home_run, present_results, show_shap_values

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


def load_data(features, selection_code_commune):
    filters = [
        ("annee", "==", int(st.session_state["state"].year)),
        ("type", "==", int(st.session_state["state"].get_type(as_type="number"))),
        ("codecommune", "==", str(selection_code_commune))
    ]
    st.session_state["data"].load_data_sample(
        columns=features, filters=filters, asset_name="data_sample_commune"
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

if st.button("Compute shap values"):
    st.session_state.show_shap_values = True

    load_shap_values()

    features = set()
    for df in st.session_state["data"].container["shap_values"].values():
        features.update(df.columns)
    features.discard("base_value")

    load_data(features=features.union(['codecommune']), selection_code_commune=st.session_state['state'].codecommune)

if st.session_state.show_shap_values and 'data_sample_commune' in st.session_state["data"].container:
    show_shap_values(
        shap_df=st.session_state["data"].container["shap_values"],
        data_sample=st.session_state["data"].container["data_sample_commune"],
        BLOCS=st.session_state["state"].get_blocs(as_type="code", order="political"),
        selection_code_commune=st.session_state["state"].codecommune,
    )
