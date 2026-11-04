import pandas as pd
import streamlit as st
from core.utils import check_home_run, plot_backtest, get_asset
import numpy as np


@st.cache_data
def load_temporal_data(
    years: list, election_type: str, trends: list, codecommune: str | None
) -> pd.DataFrame | None:
    return st.session_state["dh"].load_backtest(
        years=years,
        election_type=election_type,
        trends=trends,
        codecommune=codecommune,
    )


@st.cache_data
def load_communes_list():
    return st.session_state["dh"].load_communes_list()


check_home_run()

st.header("Back-testing du modèle")
st.markdown("Analysez l'évolution des résultats et prédictions sur plusieurs années.")

st.divider()

st.session_state["state"].selection_box(multiple_years=True)

tab1, tab2 = st.columns(2)
with tab1:
    mode_choice = st.radio(
        "Mode", options=["France Entière", "Commune"], on_change=st.cache_data.clear()
    )

if mode_choice == "France Entière":
    st.session_state["state"].codecommune = None

if mode_choice == "Commune":
    with tab2:
        communes_list = get_asset(st.session_state["dh"].load_communes_list)
        st.session_state["state"].commune_selector(communes_list)
        st.write(
            f"Commune sélectionnée : {st.session_state['state'].commune} ({st.session_state['state'].codecommune})"
        )
    temporal_data = load_temporal_data(
        years=st.session_state["state"].get_years(),
        election_type=st.session_state["state"].get_type(as_type="code"),
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
        codecommune=st.session_state["state"].codecommune,
    )
    temporal_data["2027_pres_true"] = np.nan


if mode_choice == "France Entière":
    temporal_data = load_temporal_data(
        years=st.session_state["state"].get_years(),
        election_type=st.session_state["state"].get_type(as_type="code"),
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
        codecommune=None,
    )
    temporal_data["2027_pres_true"] = np.nan

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
# temporal_data.to_csv(f'notebooks/model/results_ts/backtest_{st.session_state["state"].get_blocs(as_type="code", order="political", prefix="pvote")}_{st.session_state["state"].get_type(as_type="code")}.csv')
plot_backtest(
    temporal_data,
    selected_blocs,
    years=st.session_state["state"].get_years(),
    yaxis_title="Taux de vote (%)",
)
