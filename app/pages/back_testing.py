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


@st.cache_data
def load_communes_list():
    st.session_state["data"].load_result(
        asset="results_full",
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
        year=2022,
        election_type=st.session_state["state"].get_type(as_type="code"),
        columns=["codecommune", "nomcommune"],
        filters=None,
        asset_name="communes_list",
    )


def build_pres_table(df: pd.DataFrame, years: list, parties: list) -> pd.DataFrame:
    result_cols = {}

    for year in years:
        d = df.loc[df["annee"] == year].copy()
        if d.empty:
            continue

        # Aggregate base rows
        pred = {
            "pvotepar": d["pvotepar_pred"].sum(),
        }
        true = {
            "pvotepar": d['pvotepar_true'].sum(),
        }
        # Votes per bloc
        for p in parties:
            pred[f'pvote{p}'] = d[f"pvote{p}_pred"].sum()
            true[f'pvote{p}'] = d[f"pvote{p}_true"].sum()

        result_cols[f"{year}_pres_pred"] = pd.Series(pred)
        result_cols[f"{year}_pres_true"] = pd.Series(true)

    return pd.DataFrame(result_cols)


@st.cache_data
def load_results_local_over_time(selected_commune):
    all_results = []
    for year in st.session_state["state"].get_years():
        st.session_state["data"].load_result(
                asset="results_full",
                year=year,
                election_type=st.session_state["state"].get_type(as_type="code"),
                trends=st.session_state["state"].get_blocs(
                    as_type="code", order="alpha"
                ),
                filters=[("codecommune", "==", st.session_state["state"].codecommune)]
            )
        X = st.session_state["data"].container["results_full"]
        X['annee'] = year
        all_results.append(
            X
        )
    if len(all_results) == 0:
        return None
    else:
        return pd.concat(all_results, axis=0)


check_home_run()


st.header("Back-testing du modèle")
st.markdown("Analysez l'évolution des résultats et prédictions sur plusieurs années.")

st.divider()

st.session_state["state"].selection_box(multiple_years=True)

tab1, tab2 = st.columns(2)
with tab1:
    mode_choice = st.radio('Mode', options=['France Entière', 'Commune'], on_change=st.cache_data.clear())

if mode_choice == 'Commune':
    with tab2:
        load_communes_list()
        st.session_state["state"].commune_selector()
        st.write(
            f"Commune sélectionnée : {st.session_state['state'].commune} ({st.session_state['state'].codecommune})"
        )

    data = load_results_local_over_time(st.session_state['state'].codecommune)
    temporal_data = build_pres_table(load_results_local_over_time(st.session_state['state'].codecommune), years=st.session_state["state"].get_years(), parties=st.session_state["state"].get_blocs(as_type="code", order="alpha"))


if mode_choice == 'France Entière':
    temporal_data = load_results_over_time()


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
