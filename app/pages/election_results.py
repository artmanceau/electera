import time

import streamlit as st
from asset.definitions import convert, political_align, reverse_convert, trad
from core.utils import (
    check_home_run,
    diff_show,
    present_results,
    show_feature_importance,
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
        "Type d'élection",
        [convert("type", el) for el in st.session_state["config"].types_to_display],
        index=0,
    )
with col3:
    b = st.selectbox(
        "Division politique",
        [
            convert("political_division", el)
            for el in st.session_state["config"].political_divisions_to_dislay
        ],
        index=0,
    )

TYPE, BLOCS = reverse_convert("type", t), reverse_convert("political_division", b)

st.session_state["data"].load_result(
    asset="results_synth", year=YEAR, election_type=TYPE, trends=BLOCS
)
results = st.session_state["data"].container["results_synth"].set_index("index")

st.session_state["data"].load_explain(
    asset="feature_importance", year=YEAR, election_type=TYPE, trends=BLOCS
)
feature_importance = st.session_state["data"].container["feature_importance"]

st.header(f"Résultat des {t} de {YEAR} ({b})")

present_results(
    results, year=YEAR, t=TYPE, blocs=political_align(BLOCS), scale="global"
)

st.divider()

st.header("Erreur du modèle")

# Create trad adapté à bloc
mean_error = (
    results.loc[[f"p{b}" for b in BLOCS], f"{YEAR}_{TYPE}_diff"].values.mean()
) * 100
std_error = (
    results.loc[[f"p{b}" for b in BLOCS], f"{YEAR}_{TYPE}_diff"].values.std()
) * 100
st.write(f"Erreur moyenne des prédictions: {mean_error:1f}%")
st.write(f"Ecart type de l'erreur de prédiction: {std_error:1f}%")

with st.expander("Erreurs moyenne des prédictions (sur l'ensemble des communes)"):
    diff_show(results, BLOCS, trad, "diff", "error", YEAR, TYPE)

with st.expander("Ecart type de l'erreur de prédiction (sur l'ensemble des communes)"):
    diff_show(results, BLOCS, trad, "std", "ecart_type", YEAR, TYPE)

st.divider()

time.sleep(2)

show_feature_importance(feature_importance, political_align(BLOCS))

st.divider()

# show_shap_values(st.session_state["data"]["shap_values"])
