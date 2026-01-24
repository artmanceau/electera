import streamlit as st

from src.components.streamlit_utils.utils import (
    blocs,
    present_results,
    show_shap_values,
)

if "data" in st.session_state:
    X = st.session_state["data"]["results_full"]
else:
    st.warning("Visit the home page!")
    st.stop()

st.title("Resultat au niveau de chaque commune")

X["nomcommune"] = (
    X["nomcommune"]
    .str.replace("Ã", "É", regex=False)
    .str.replace("Ã", "Â", regex=False)
    .str.replace("Ã", "È", regex=False)
    .str.replace("Ã", "Ô", regex=False)
    .str.replace("Ã", "Ê", regex=False)
    .str.replace("Ã", "À", regex=False)
)
communes = X["nomcommune"]

selection = st.selectbox("Selectionnez une commune", [""] + communes)
if selection and selection != "-- none --":
    st.write("Commune selectionée : ", selection)
    if len(X[X["nomcommune"] == selection]) > 1:
        arrondissements = X[X["nomcommune"] == selection]["codecommune"]
        selection_code_commune = st.selectbox(
            "Selectionnez une arrondissement", [""] + arrondissements
        )
    else:
        selection_code_commune = X[X["nomcommune"] == selection]["codecommune"].iloc[0]

data_line = X[X["codecommune"] == selection_code_commune].copy(deep=True)

pct_cols = (
    ["ppar_true"]
    + [f"pvote{b}_true" for b in blocs]
    + [f"pvote{b}_pred" for b in blocs]
    + [f"pvote{b}_diff" for b in blocs]
    + ["ppar_diff"]
)
for col in pct_cols:
    data_line[col] = round(data_line[col] * 100, 2)

st.divider()

present_results(data_line)

st.divider()

show_shap_values(
    st.session_state["data"]["shap_values"],
    selection_code_commune=selection_code_commune,
)
