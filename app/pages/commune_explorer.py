import streamlit as st
from core.utils import (
    check_home_run,
    present_results,
    show_shap_values,
    get_asset,
    make_slider,
)
from asset.features import CHAMPION_FEATURES_EXTENDED

check_home_run()

st.session_state["state"].selection_box(multiple_years=False)

st.header("Résultat au niveau de chaque commune")

communes_list = get_asset(st.session_state["dh"].load_communes_list)
st.session_state["state"].commune_selector(communes_list)
st.write(
    f"Commune sélectionnée : {st.session_state['state'].commune} ({st.session_state['state'].codecommune})"
)

st.divider()

present_results(
    get_asset(
        st.session_state["dh"].load_result,
        asset_name="results_full",
        year=st.session_state["state"].year,
        election_type=st.session_state["state"].get_type(as_type="code"),
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
        filters=[("codecommune", "==", st.session_state["state"].codecommune)],
    ),
    year=st.session_state["state"].year,
    t=st.session_state["state"].get_type(as_type="code"),
    blocs=st.session_state["state"].get_blocs(as_type="code", order="political"),
    scale="local",
)

st.divider()

nb_feat_shap = make_slider("nb_feat_shap")

show_shap_values(
    shap_df=get_asset(
        st.session_state["dh"].load_explain,
        asset_name="shap_values",
        year=st.session_state["state"].year,
        election_type=st.session_state["state"].get_type(as_type="code"),
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
        filters=[("codecommune", "==", st.session_state["state"].codecommune)],
    ),
    data_sample=get_asset(
        st.session_state["dh"].load_data_sample,
        columns=CHAMPION_FEATURES_EXTENDED + ["codecommune"],
        filters=[
            ("annee", "==", int(st.session_state["state"].year)),
            (
                "election_type",
                "==",
                str(st.session_state["state"].get_type(as_type="code_full")),
            ),
            ("codecommune", "==", st.session_state["state"].codecommune),
        ],
    ),
    BLOCS=st.session_state["state"].get_blocs(as_type="code", order="political"),
    selection_code_commune=st.session_state["state"].codecommune,
    nb_feat=nb_feat_shap,
)
