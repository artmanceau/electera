import streamlit as st
from asset.definitions import blocs, trad, display_config_converter, convert, reverse_convert
from core.utils import (
    present_results,
    show_feature_importance,
    check_home_run,
    diff_show
)

check_home_run()


# User selections
col1, col2, col3 = st.columns(3)
with col1:
    YEAR = st.selectbox(
        "Année électorale", st.session_state["config"].years_to_display, index=0
    )
with col2:
    t = st.selectbox(
        "Type d'élection", [convert('type', el) for el in st.session_state["config"].types_to_display], index=0
    )
with col3:
    b = st.selectbox(
        "Division politique", [convert('political_division', el) for el in st.session_state['config'].political_divisions_to_dislay], index=0
    )

TYPE, BLOCS = reverse_convert('type', t), reverse_convert('political_division', b)


st.header("Resultat au niveau de chaque commune")

st.session_state['data'].load_result(asset="results_full", trends=BLOCS, year=YEAR, election_type=TYPE, columns=['codecommune', 'nomcommune'], filters=None, asset_name='communes_list')
communes_list = st.session_state["data"].container['communes_list']

communes_list["nomcommune"] = (
    communes_list["nomcommune"]
    .str.replace("Ã", "É", regex=False)
    .str.replace("Ã", "Â", regex=False)
    .str.replace("Ã", "È", regex=False)
    .str.replace("Ã", "Ô", regex=False)
    .str.replace("Ã", "Ê", regex=False)
    .str.replace("Ã", "À", regex=False)
)
communes = communes_list["nomcommune"]

selection = st.selectbox("Selectionnez une commune", [""] + communes)
if selection and selection != "-- none --":
    st.write("Commune selectionée : ", selection)
    if len(communes_list[communes_list["nomcommune"] == selection]) > 1:
        arrondissements = communes_list[communes_list["nomcommune"] == selection]["codecommune"]
        selection_code_commune = st.selectbox(
            "Selectionnez une arrondissement", [""] + arrondissements
        )
    else:
        selection_code_commune = communes_list[communes_list["nomcommune"] == selection]["codecommune"].iloc[0]

st.session_state['data'].load_result(asset="results_full", trends=BLOCS, year=YEAR, election_type=TYPE, columns=None, filters=[('codecommune', '==', selection_code_commune)], asset_name='results_commune_selected')
data_line = st.session_state["data"].container['results_commune_selected']

st.divider()

present_results(data_line, year=YEAR, t=TYPE, scale='local')

st.divider()

# show_shap_values(
#     st.session_state["data"]["shap_values"],
#     selection_code_commune=selection_code_commune,
# )
