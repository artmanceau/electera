import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from asset.definitions import convert, reverse_convert
from core.utils import colors, trad, check_home_run

check_home_run()

st.header("Perspective Temporelle des Résultats Électoraux")
st.markdown("Analysez l'évolution des résultats et prédictions sur plusieurs années.")

st.divider()

# User selections
col1, col2 = st.columns(2)
with col1:
    t = st.selectbox(
        "Type d'élection",
        [convert('type', el) for el in st.session_state["config"].types_to_display],
        index=0
    )
with col2:
    b = st.selectbox(
        "Division politique",
        [convert('political_division', el) for el in st.session_state['config'].political_divisions_to_dislay],
        index=0
    )

TYPE = reverse_convert('type', t)
BLOCS = reverse_convert('political_division', b)

# Extract bloc names
current_blocs = [bloc.replace('vote', '') for bloc in BLOCS if bloc != 'par']

# Multi-select for years
selected_years = st.multiselect(
    "Sélectionnez les années à analyser",
    st.session_state["config"].years_to_display,
    default=st.session_state["config"].years_to_display[:min(3, len(st.session_state["config"].years_to_display))]
)

st.divider()

# Load data for selected years
with st.spinner("Chargement des données temporelles..."):
    all_results = {}
    for year in selected_years:
        st.session_state['data'].load_result(asset="results_synth", year=year, election_type=TYPE, trends=BLOCS)
        all_results[year] = st.session_state["data"].container['results_synth'].set_index('index')

temporal_data = pd.DataFrame()


st.subheader("📊 Évolution de la Participation")

# Create participation plot
fig_participation = go.Figure()

years_sorted = sorted(selected_years)
participation_true = []
participation_pred = []

for year in years_sorted:
    year_data = temporal_data[temporal_data['year'] == year]
    if len(year_data) > 0:
        col_name = f'{year}_{TYPE}_true'
        if 'ppar' in year_data.index:
            participation_true.append(year_data.loc['ppar', col_name] * 100)
        else:
            participation_true.append(None)

        col_name_pred = f'{year}_{TYPE}_pred'
        if 'ppar' in year_data.index:
            participation_pred.append(year_data.loc['ppar', col_name_pred] * 100)
        else:
            participation_pred.append(None)

fig_participation.add_trace(go.Scatter(
    x=years_sorted,
    y=participation_true,
    mode='lines+markers',
    name='Résultats réels',
    line=dict(color='darkgreen', width=3),
    marker=dict(size=10)
))

fig_participation.add_trace(go.Scatter(
    x=years_sorted,
    y=participation_pred,
    mode='lines+markers',
    name='Prédictions',
    line=dict(color='steelblue', width=3, dash='dash'),
    marker=dict(size=10)
))

fig_participation.update_layout(
    xaxis_title="Année",
    yaxis_title="Taux de participation (%)",
    hovermode='x unified',
    height=400,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_participation, use_container_width=True)


st.divider()
st.subheader("🗳️ Évolution des Votes par Bloc Politique")

# Create subplot for each bloc
num_blocs = len(current_blocs)
fig_blocs = make_subplots(
    rows=1,
    cols=num_blocs,
    subplot_titles=[f"Vote {trad.get(bloc, bloc)}" for bloc in current_blocs],
    shared_yaxes=True
)

for idx, bloc in enumerate(current_blocs):
    votes_true = []
    votes_pred = []

    for year in years_sorted:
        year_data = temporal_data[temporal_data['year'] == year]
        if len(year_data) > 0:
            col_name = f'{year}_{TYPE}_true'
            index_name = f'pvote{bloc}'

            if index_name in year_data.index:
                votes_true.append(year_data.loc[index_name, col_name] * 100)
                votes_pred.append(year_data.loc[index_name, f'{year}_{TYPE}_pred'] * 100)
            else:
                votes_true.append(None)
                votes_pred.append(None)

    # Add traces
    fig_blocs.add_trace(
        go.Scatter(
            x=years_sorted,
            y=votes_true,
            mode='lines+markers',
            name='Réel',
            line=dict(color=colors[idx], width=2),
            marker=dict(size=8),
            showlegend=(idx == 0)
        ),
        row=1, col=idx+1
    )

    fig_blocs.add_trace(
        go.Scatter(
            x=years_sorted,
            y=votes_pred,
            mode='lines+markers',
            name='Prédit',
            line=dict(color=colors[idx], width=2, dash='dash'),
            marker=dict(size=8),
            showlegend=(idx == 0)
        ),
        row=1, col=idx+1
    )

fig_blocs.update_xaxes(title_text="Année")
fig_blocs.update_yaxes(title_text="% des suffrages", row=1, col=1)

fig_blocs.update_layout(
    height=400,
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_blocs, use_container_width=True)

st.divider()
st.subheader("📈 Vue Combinée - Tous les Blocs")

fig_combined = go.Figure()

for idx, bloc in enumerate(current_blocs):
    votes_true = []
    votes_pred = []

    for year in years_sorted:
        year_data = temporal_data[temporal_data['year'] == year]
        if len(year_data) > 0:
            col_name = f'{year}_{TYPE}_true'
            index_name = f'pvote{bloc}'

            if index_name in year_data.index:
                votes_true.append(year_data.loc[index_name, col_name] * 100)
                votes_pred.append(year_data.loc[index_name, f'{year}_{TYPE}_pred'] * 100)
            else:
                votes_true.append(None)
                votes_pred.append(None)

    bloc_label = trad.get(bloc, bloc)

    # True results
    fig_combined.add_trace(go.Scatter(
        x=years_sorted,
        y=votes_true,
        mode='lines+markers',
        name=f'{bloc_label} (réel)',
        line=dict(color=colors[idx], width=2),
        marker=dict(size=8)
    ))

    # Predictions
    fig_combined.add_trace(go.Scatter(
        x=years_sorted,
        y=votes_pred,
        mode='lines+markers',
        name=f'{bloc_label} (prédit)',
        line=dict(color=colors[idx], width=2, dash='dash'),
        marker=dict(size=8),
        opacity=0.7
    ))

fig_combined.update_layout(
    xaxis_title="Année",
    yaxis_title="% des suffrages",
    hovermode='x unified',
    height=500,
    showlegend=True,
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
)

st.plotly_chart(fig_combined, use_container_width=True)


st.divider()
st.subheader("⚠️ Évolution de l'Erreur de Prédiction")

# Calculate errors over time
fig_errors = go.Figure()

for idx, bloc in enumerate(current_blocs):
    errors = []

    for year in years_sorted:
        year_data = temporal_data[temporal_data['year'] == year]
        if len(year_data) > 0:
            col_name = f'{year}_{TYPE}_diff'
            index_name = f'pvote{bloc}'

            if index_name in year_data.index:
                errors.append(abs(year_data.loc[index_name, col_name] * 100))
            else:
                errors.append(None)

    bloc_label = trad.get(bloc, bloc)

    fig_errors.add_trace(go.Bar(
        x=years_sorted,
        y=errors,
        name=bloc_label,
        marker_color=colors[idx]
    ))

fig_errors.update_layout(
    xaxis_title="Année",
    yaxis_title="Erreur absolue moyenne (%)",
    barmode='group',
    height=400,
    hovermode='x unified',
    showlegend=True
)

st.plotly_chart(fig_errors, use_container_width=True)

st.divider()
st.subheader("📊 Statistiques Récapitulatives")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Années analysées", len(selected_years))

with col2:
    # Calculate average accuracy across all years
    all_errors = []
    for year in years_sorted:
        year_data = temporal_data[temporal_data['year'] == year]
        if len(year_data) > 0:
            col_name = f'{year}_{TYPE}_diff'
            for bloc in current_blocs:
                index_name = f'pvote{bloc}'
                if index_name in year_data.index:
                    all_errors.append(abs(year_data.loc[index_name, col_name] * 100))

    if all_errors:
        avg_error = sum(all_errors) / len(all_errors)
        st.metric("Erreur moyenne globale", f"{avg_error:.2f}%")
    else:
        st.metric("Erreur moyenne globale", "N/A")

with col3:
    # Find best year (lowest error)
    year_errors = {}
    for year in years_sorted:
        year_data = temporal_data[temporal_data['year'] == year]
        if len(year_data) > 0:
            col_name = f'{year}_{TYPE}_diff'
            year_error = []
            for bloc in current_blocs:
                index_name = f'pvote{bloc}'
                if index_name in year_data.index:
                    year_error.append(abs(year_data.loc[index_name, col_name] * 100))
            if year_error:
                year_errors[year] = sum(year_error) / len(year_error)

    if year_errors:
        best_year = min(year_errors, key=year_errors.get)
        st.metric("Meilleure prédiction", str(best_year))
    else:
        st.metric("Meilleure prédiction", "N/A")

# Detailed table
with st.expander("📋 Tableau détaillé des résultats"):
    # Create summary table
    summary_data = []

    for year in years_sorted:
        year_data = temporal_data[temporal_data['year'] == year]
        if len(year_data) > 0:
            row = {"Année": year}

            # Participation
            if 'ppar' in year_data.index:
                col_true = f'{year}_{TYPE}_true'
                col_pred = f'{year}_{TYPE}_pred'
                row["Participation (réel)"] = f"{year_data.loc['ppar', col_true] * 100:.2f}%"
                row["Participation (prédit)"] = f"{year_data.loc['ppar', col_pred] * 100:.2f}%"

            # Each bloc
            for bloc in current_blocs:
                index_name = f'pvote{bloc}'
                if index_name in year_data.index:
                    col_true = f'{year}_{TYPE}_true'
                    col_pred = f'{year}_{TYPE}_pred'
                    bloc_label = trad.get(bloc, bloc)
                    row[f"{bloc_label} (réel)"] = f"{year_data.loc[index_name, col_true] * 100:.2f}%"
                    row[f"{bloc_label} (prédit)"] = f"{year_data.loc[index_name, col_pred] * 100:.2f}%"

            summary_data.append(row)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
