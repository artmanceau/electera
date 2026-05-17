import pandas as pd
import streamlit as st
from core.utils import check_home_run, plot_backtest


@st.cache_data
def load_results_over_time():
    all_results = []
    for year in st.session_state["state"].get_years():
        try:
            st.session_state["data"].load_result(
                asset="results_synth",
                year=year,
                election_type=st.session_state["state"].get_type(as_type="code"),
                trends=st.session_state["state"].get_blocs(
                    as_type="code", order="alpha"
                ),
                columns=[
                    "index",
                    f"{year}_{st.session_state['state'].get_type(as_type='code')}_pred",
                    f"{year}_{st.session_state['state'].get_type(as_type='code')}_true",
                ],
            )
            all_results.append(
                st.session_state["data"].container["results_synth"].set_index("index")
            )
        except:
            continue

    if len(all_results) == 0:
        return None
    else:
        return pd.concat(all_results, axis=1)


check_home_run()

st.header("Back-testing du modèle")
st.markdown("Analysez l'évolution des résultats et prédictions sur plusieurs années.")

st.divider()

st.session_state["state"].selection_box(multiple_years=True)

st.divider()

# Load data for selected years
temporal_data = load_results_over_time()
if temporal_data is None:
    st.warning("No election data fetched!")
    st.stop()

st.subheader("📊 Évolution de la Participation")

plot_backtest(temporal_data, "pvotepar", years=st.session_state["state"].get_years())

st.divider()
st.subheader("🗳️ Évolution des Votes par Bloc Politique")


selected_blocs = st.multiselect(
    "Sélectionnez les tendances politiques à inclure dans le graphique",
    st.session_state["state"].get_blocs(
        as_type="code", order="political", prefix="pvote"
    ),
    default=st.session_state["state"].get_blocs(
        as_type="code", order="political", prefix="pvote"
    ),
)

plot_backtest(
    temporal_data,
    selected_blocs,
    years=st.session_state["state"].get_years(),
    yaxis_title="Taux de vote (%)",
)
