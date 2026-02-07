import streamlit as st
from asset.definitions import blocs, trad, display_config_converter, convert, reverse_convert
from core.utils import (
    present_results,
    show_feature_importance,
    check_home_run
)

check_home_run()
st.write( st.session_state["config"])

YEAR = st.selectbox(
    "Election year", st.session_state["config"].years_to_display, index=0
)
t = st.selectbox(
    "Election type", [convert('type', el) for el in st.session_state["config"].types_to_display], index=0
)
b = st.selectbox(
    "Division du spectre politique", [convert('political_division', el) for el in st.session_state['config'].political_divisions_to_dislay], index=0
)
TYPE, BLOCS = reverse_convert('type', t), reverse_convert('political_division', b)

st.session_state['data'].load_result(asset="results_synth", year=YEAR, election_type=TYPE, trends=BLOCS)
results = st.session_state["data"].container['results_synth']

#st.session_state['data'].load_explain(asset="feature_importance", trend='voteG', year=YEAR, election_type=TYPE)


#feature_importance = st.session_state["data"]["feature_importance"]

st.header(f"Résultat des {t} de {YEAR} ({b})")

present_results(results)

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
