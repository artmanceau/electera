import streamlit as st
from core.utils import check_home_run, plot_backtest


@st.cache_data
def load_backtest_data():
    st.session_state["data"].load_results_backtest(
        years=st.session_state["state"].get_years(),
        asset="results_full"
        if st.session_state["state"].codecommune
        else "results_synth",
        election_type=st.session_state["state"].get_type(as_type="code"),
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
        codecommune=st.session_state["state"].codecommune,
    )


@st.cache_data
def load_communes_list():
    st.session_state["data"].load_communes_list()


check_home_run()

st.header("Back-testing du modèle")
st.markdown("Analysez l'évolution des résultats et prédictions sur plusieurs années.")

st.divider()
st.session_state["state"].selection_box(multiple_years=True, clear_cache_on_rerun=False)

tab1, tab2 = st.columns(2)
with tab1:
    mode_choice = st.radio(
        "Mode", options=["France Entière", "Commune"], on_change=st.cache_data.clear()
    )

if mode_choice == "Commune":
    with tab2:
        load_communes_list()
        st.session_state["state"].commune_selector()
        st.write(
            f"Commune sélectionnée : {st.session_state['state'].commune} ({st.session_state['state'].codecommune})"
        )

load_backtest_data()
temporal_data = st.session_state["data"].container.get("backtest_results")

st.divider()

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
