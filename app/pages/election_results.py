import streamlit as st
from core.data_handler import AppData
from core.utils import (
    blocs,
    compute_agg_results,
    present_results,
    show_feature_importance,
    show_shap_values,
    trad,
    type_trad,
)

YEAR = st.selectbox(
    "Election year", st.session_state["config"].years_to_display, index=0
)
TYPE = st.selectbox(
    "Election type", st.session_state["config"].types_to_display, index=0
)

AppData(
    st.session_state["config"].data_path, st.session_state["config"].model_version
).load_element(asset="results_synth", year=YEAR, type=TYPE)
AppData(
    st.session_state["config"].data_path, st.session_state["config"].model_version
).load_element(asset="feature_importance", year=YEAR, type=TYPE)
results = st.session_state["data"]["result_synth"]
feature_importance = st.session_state["data"]["feature_importance"]

st.header(f"Résultat des élections {TYPE} de {YEAR}")

present_results(page_data)

st.divider()

st.header("Erreur du modèle")


mean = round(
    y[y["var"] == "avg_error_tot"][str(YEAR)].values[0] * 100,
    2,
)
std = round(
    y[y["var"] == "std_error_tot"][str(YEAR)].values[0] * 100,
    2,
)
st.write(f"Erreur moyenne des prédictions: {mean}%")
st.write(f"Ecart type de l'erreur de prédiction: {std}%")

with st.expander("Erreurs moyenne des prédictions"):
    variable_error = [f"error_pvote{bloc}" for bloc in blocs] + ["error_ppar"]
    col_config_error = {f"error_pvote{bloc}": f"Vote {trad[bloc]}" for bloc in blocs}
    col_config_error["error_ppar"] = "Participation"
    st.dataframe(
        y[y["var"].isin(variable_error)][[str(YEAR)]].T * 100,
        column_config=col_config_error,
        hide_index=True,
    )

with st.expander("Ecart type de l'erreur de prédiction"):
    variable_std = [f"std_pvote{bloc}" for bloc in blocs] + ["std_ppar"]
    col_config_std = {f"std_pvote{bloc}": f"Vote {trad[bloc]}" for bloc in blocs}
    col_config_std["std_ppar"] = "Participation"
    st.dataframe(
        y[y["var"].isin(variable_std)][[str(YEAR)]].T * 100,
        column_config=col_config_std,
        hide_index=True,
    )

st.divider()

show_feature_importance(feature_importance)

# st.divider()

# show_shap_values(st.session_state["data"]["shap_values"])
