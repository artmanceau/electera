import pandas as pd
import streamlit as st
from core.utils import check_home_run, plot_backtest
from loguru import logger
from concurrent.futures import ThreadPoolExecutor


def _load_result_for_year(
    year: int,
    asset: str,
    election_type: str,
    trends: list,
    columns: list | None,
    filters: list | None,
    data_loader,  # Pass the loader instance directly
) -> pd.DataFrame | None:
    """Load result for a single year (thread-safe: no session_state access)."""
    try:
        data_loader.load_result(
            asset=asset,
            year=year,
            election_type=election_type,
            trends=trends,
            columns=columns,
            filters=filters,
        )
        X = data_loader.container[asset]
        if columns:
            X = X.set_index("index")
        if "annee" not in X.columns:
            X = X.copy()
            X["annee"] = year
        return X
    except FileNotFoundError:
        logger.warning(f"Year {year} ({election_type}) not available")
        return None


@st.cache_data
def load_results_over_time() -> pd.DataFrame | None:
    """Concatenate results across years."""
    data_loader = st.session_state["data"]  # Access ONCE in main thread
    years = st.session_state["state"].get_years()
    election_type = st.session_state["state"].get_type(as_type="code")
    trends = st.session_state["state"].get_blocs(as_type="code", order="alpha")
    codecommune = st.session_state["state"].codecommune

    # Determine mode parameters
    if codecommune:
        asset, columns, filters, axis = "results_full", None, [("codecommune", "==", codecommune)], 0
    else:
        asset = "results_synth"
        columns = {year: ["index", f"{year}_{election_type}_pred", f"{year}_{election_type}_true"] for year in years}
        filters, axis = None, 1

    # Thread the data loading
    results = []
    with ThreadPoolExecutor(max_workers=len(years)) as executor:
        futures = [
            executor.submit(
                _load_result_for_year,
                year,
                asset,
                election_type,
                trends,
                columns.get(year) if isinstance(columns, dict) else columns,
                filters,
                data_loader,  # Pass loader to worker
            )
            for year in years
        ]
        results = [f.result() for f in futures if f.result() is not None]

    return pd.concat(results, axis=axis) if results else None

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


check_home_run()


st.header("Back-testing du modèle")
st.markdown("Analysez l'évolution des résultats et prédictions sur plusieurs années.")

st.divider()

st.session_state["state"].selection_box(multiple_years=True, clear_cache_on_rerun=False)

tab1, tab2 = st.columns(2)
with tab1:
    mode_choice = st.radio('Mode', options=['France Entière', 'Commune'], on_change=st.cache_data.clear())

if mode_choice == 'France Entière':
    st.session_state['state'].codecommune = None

if mode_choice == 'Commune':
    with tab2:
        load_communes_list()
        st.session_state["state"].commune_selector()
        st.write(
            f"Commune sélectionnée : {st.session_state['state'].commune} ({st.session_state['state'].codecommune})"
        )

    data = load_results_over_time()
    temporal_data = build_pres_table(data, years=st.session_state["state"].get_years(), parties=st.session_state["state"].get_blocs(as_type="code", order="alpha"))


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
