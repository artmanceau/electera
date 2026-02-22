import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from asset.definitions import convert, reverse_convert, political_align, colors_dict, trad
from core.utils import check_home_run

check_home_run()

st.header("Back-testing du modèle")
st.markdown("Analysez l'évolution des résultats et prédictions sur plusieurs années.")

st.divider()

# User selections
col1, col2 = st.columns(2)
with col1:
    t = st.selectbox(
        "Type d'élection",
        [convert("type", el) for el in st.session_state["config"].types_to_display],
        index=0,
    )
with col2:
    b = st.selectbox(
        "Division politique",
        [
            convert("political_division", el)
            for el in st.session_state["config"].political_divisions_to_dislay
        ],
        index=0,
    )

TYPE = reverse_convert("type", t)
BLOCS = reverse_convert("political_division", b)

# Extract bloc names
current_blocs = political_align(BLOCS) #[bloc.replace("vote", "") for bloc in BLOCS if bloc != "par"]

# Multi-select for years
selected_years = st.multiselect(
    "Sélectionnez les années à analyser",
    st.session_state["config"].years_to_display,
    default=st.session_state["config"].years_to_display[
        : min(3, len(st.session_state["config"].years_to_display))
    ],
)

st.divider()

# Load data for selected years
with st.spinner("Chargement des données temporelles..."):
    all_results = []
    for year in selected_years:
        st.session_state["data"].load_result(
            asset="results_synth",
            year=year,
            election_type=TYPE,
            trends=BLOCS,
            columns=["index", f"{year}_{TYPE}_pred", f"{year}_{TYPE}_true"],
        )
        all_results.append(
            st.session_state["data"].container["results_synth"].set_index("index")
        )

temporal_data = pd.concat(all_results, axis=1)


def plot_participation(
    df,
    variables,
    years,
    true_suffix="_true",
    pred_suffix="_pred",
    yaxis_title="Taux de participation (%)",
):
    """
    Plot true and predicted values for one or more variables over years using Plotly in Streamlit.
    Each variable uses its own color for both lines (solid for true, dashed for pred).
    Args:
        df (pd.DataFrame): DataFrame with index as variable names and columns as '{year}_{TYPE}true' and '{year}_{TYPE}pred'.
        variables (list or str): List of variables or a single variable to plot.
        years (list): List of years to plot.
        true_suffix (str): Suffix for true columns.
        pred_suffix (str): Suffix for predicted columns.
        yaxis_title (str): Y-axis label.
    """
    if isinstance(variables, str):
        variables = [variables]
    years_sorted = sorted(years)

    fig = go.Figure()

    for idx, variable in enumerate(variables):
        color = "#008000" if (variable=='ppar') else colors_dict[variable.replace('p', '')]
        true_vals = [
            (
                df.loc[variable, f"{year}_{TYPE}{true_suffix}"]
                if f"{year}_{TYPE}{true_suffix}" in df.columns
                else None
            )
            for year in years_sorted
        ]
        pred_vals = [
            (
                df.loc[variable, f"{year}_{TYPE}{pred_suffix}"]
                if f"{year}_{TYPE}{pred_suffix}" in df.columns
                else None
            )
            for year in years_sorted
        ]
        fig.add_trace(
            go.Scatter(
                x=years_sorted,
                y=true_vals,
                mode="lines+markers",
                name=f"{trad[variable[1:]]} - Réel" if (variable=='ppar') else f"Vote {trad[variable[1:]]} - Réel",
                line=dict(color=color, width=3, dash="solid"),
                marker=dict(size=10, color=color, symbol='triangle-down')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=years_sorted,
                y=pred_vals,
                mode="lines+markers",
                name=f"{trad[variable[1:]]} - Prédiction" if (variable=='ppar') else f"Vote {trad[variable[1:]]} - Prédiction",
                line=dict(color=color, width=3, dash="dot"),
                marker=dict(size=10, color=color, symbol='triangle-up'),
            )
        )

    fig.update_layout(
        xaxis_title="Année",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


st.subheader("📊 Évolution de la Participation")

plot_participation(temporal_data, "ppar", years=selected_years)

st.divider()
st.subheader("🗳️ Évolution des Votes par Bloc Politique")

pblocs = ['p'+b for b in current_blocs]
selected_blocs = st.multiselect(
    "Sélectionnez les tendances politiques à inclure dans le graphique",
    pblocs,
    default=pblocs,
)

plot_participation(temporal_data, selected_blocs, years=selected_years)
