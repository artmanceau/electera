import streamlit as st

from src.components.streamlit_utils.utils import (
    blocs,
    compute_agg_results,
    present_results,
    show_feature_importance,
    show_shap_values,
    trad,
    type_trad,
)

# Check that assets are loaded
if (
    "data" in st.session_state
    and "ELECTION_YEAR" in st.session_state
    and "ELECTION_TYPE" in st.session_state
):
    X = st.session_state["data"]["results_full"]
    y = st.session_state["data"]["results_synth"]
    y.index = y["var"]
    YEAR = st.session_state["ELECTION_YEAR"]
    TYPE = type_trad[st.session_state["ELECTION_TYPE"]]
else:
    st.warning("Visit the home page!")
    st.stop()

# Compute aggregated results
data_line = compute_agg_results(X)

st.header(f"Résultat des élections {TYPE} de {YEAR}")

present_results(data_line.to_frame().T)

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

show_feature_importance(st.session_state["data"]["feature_importance"])

st.divider()

show_shap_values(st.session_state["data"]["shap_values"])
