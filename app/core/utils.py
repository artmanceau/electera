import altair as alt
import matplotlib.pyplot as plt
import shap
import streamlit as st
from asset.definitions import FEATURES_DICT, colors_dict, get_colors, trad
import plotly.graph_objects as go

def check_home_run():
    if "home_run" in st.session_state:
        if st.session_state["home_run"]:
            return None
    st.switch_page("pages/home.py")


def diff_show(results, blocs, trad, label, label_show, year, t):
    col_config_error = {
        f"{label_show}_pvote{bloc}": f"Vote {trad[bloc]}" for bloc in blocs
    }
    st.dataframe(
        results.loc[[f"p{b}" for b in blocs], f"{year}_{t}_{label}"].to_frame().T * 100,
        column_config=col_config_error,
        hide_index=True,
    )


def results_loc(data_line, year_type, blocs, label, p=""):
    ind = [f"ppar_{label}"] if p == "p" else [f"votants_{label}", "exprimes"]
    st.dataframe(
        data_line[ind].reset_index(drop=True),
        hide_index=True,
        column_config={
            "votants_true": "Nombre de votants",
            "exprimes": "Nombre de suffrage exprimés",
        },
    )
    st.dataframe(
        data_line[[f"{p}{b}_{label}" for b in blocs]].reset_index(drop=True),
        hide_index=True,
        column_config={f"{b}_{label}": f"Nombre de vote {trad[b]}" for b in blocs},
    )
    st.bar_chart(
        data=(data_line[[f"{p}{b}_{label}" for b in blocs]].reset_index(drop=True)),
        color=get_colors(blocs, colors_dict),
        horizontal=True,
    )


def results_glob(data_line, year_type, blocs, label, p=""):
    ind = ["ppar"] if p == "p" else ["votants", "exprimes"]
    col = f"{year_type}_{label}"
    st.dataframe(
        data_line.loc[ind, col].to_frame().T,
        hide_index=True,
        column_config={
            "votants": "Nombre de votants",
            "exprimes": "Nombre de suffrage exprimés",
        },
    )
    col_config = {}
    for b in blocs:
        col_config[f"{p}{b}"] = st.column_config.NumberColumn(f"Nombre de vote {trad[b]}", format="%.1f%%" if p=="p" else None)
    
    st.dataframe(
        data_line.loc[[f"{p}{b}" for b in blocs], col].to_frame().T,
        hide_index=True,
        column_config=col_config,
    )
    st.bar_chart(
        data=(data_line.loc[[f"{p}{b}" for b in blocs], col].to_frame().T),
        color=get_colors(blocs, colors_dict),
        horizontal=True,
    )


def present_results(data_line, year, t, blocs, scale):
    result_func = results_glob if scale == "global" else results_loc


    tab1, tab2 = st.tabs(["Pourcentage des suffrages", "Nombre de vote"])

    with tab2:

        with st.expander("Résultats", expanded=True):
            st.write(
            """
                Résultats
            """
            )
            result_func(
                data_line, year_type=f"{year}_{t}", blocs=blocs, label="true", p=""
            )

        with st.expander("Prédictions", expanded=True):
            st.write(
            """
                Prédictions du modèle pour l'élection
            """
            )
            result_func(
                data_line, year_type=f"{year}_{t}", blocs=blocs, label="pred", p=""
            )

        with st.expander("Erreur", expanded=True):
            st.write(
            """
                Erreur de la prédictions du modèle pour l'élection
            """
            )
            if scale == "local":
                col_config = {
                    f"vote{b}": f"Différence avec la prédiction du vote {trad[b]}"
                    for b in blocs
                }
                col_config["votants_diff"] = (
                    "Différence avec la prédiction pour la participation"
                )

                data_element = data_line[
                    [f"{b}_diff" for b in blocs] + ["votants_diff"]
                ].reset_index(drop=True)
            else:
                data_element = (
                    data_line.loc[
                        [f"{b}" for b in blocs] + ["votants"],
                        f"{year}_{t}_diff_agg",
                    ]
                    .to_frame()
                    .T
                )

                col_config = {
                    f"{b}": f"Différence avec la prédiction du vote {trad[b]}"
                    for b in blocs
                }
                col_config["votants"] = (
                    "Différence avec la prédiction pour la participation"
                )

            st.dataframe(
                data_element,
                hide_index=True,
                column_config=col_config,
            )
            st.bar_chart(data=(data_element).T, sort=False)


    with tab1:

        with st.expander("Résultats", expanded=True):
            st.write(
            """
                Résultats
            """
            )
            result_func(
                data_line, year_type=f"{year}_{t}", blocs=blocs, label="true", p="p"
            )

        with st.expander("Prédictions", expanded=True):
            st.write(
                """
                Prédictions du modèle pour l'élection
            """
            )
            result_func(
                data_line, year_type=f"{year}_{t}", blocs=blocs, label="pred", p="p"
            )

        if f"{year}_{t}_poll" in data_line.columns:
            with st.expander("Sondages", expanded=True):
                st.write(
                """
                    Prédictions du modèle pour l'élection
                """
                )
                result_func(
                    data_line, year_type=f"{year}_{t}", blocs=blocs, label="poll", p="p"
                )


        with st.expander("Erreur", expanded=True):
            st.write(
                """
                Erreur de la prédictions du modèle pour l'élection
            """
            )
            if scale == "local":
                data_element = data_line[
                    [f"p{b}_diff" for b in blocs] + ["ppar_diff"]
                ].reset_index(drop=True)

                col_config = {
                    f"p{b}": f"Différence avec la prédiction du vote {trad[b]}"
                    for b in blocs
                }
                col_config["votants_diff"] = (
                    "Différence avec la prédiction pour la participation"
                )
            else:

                data_element = (
                    data_line.loc[
                        [f"p{b}" for b in blocs] + ["ppar"],
                        f"{year}_{t}_diff_agg",
                    ]
                    .to_frame()
                    .T
                )

                col_config = {
                    f"p{b}": f"Différence avec la prédiction du vote {trad[b]}"
                    for b in blocs
                }
                col_config["votants"] = (
                    "Différence avec la prédiction pour la participation"
                )

            st.dataframe(
                data_element,
                hide_index=True,
                column_config=col_config,
            )
            st.bar_chart(data=(data_element).T, sort=False)


def show_feature_importance(importance_df, blocs):
    st.header(
        "Déterminants socio-économiques les plus importants dans le modèle de prédiction"
    )

    nb_feat = st.slider(
        "Selectionnez un nombre de variable pour visualiser l'importance des variables socio-économiques",
        5,
        30,
    )
    trends = ["par"] + [f"{b}" for b in blocs]
    tabs = st.tabs(["Participation"] + [f"Vote {trad[b]}" for b in blocs])
    for i, tab in enumerate(tabs):
        with tab:
            df = importance_df[trends[i]].copy()
            df["feature_desc"] = df["Feature_gain"].map(FEATURES_DICT)
            st.write("Importance en gain total")
            top_gain = df.nlargest(nb_feat, "Importance_gain")[
                ["Feature_gain", "Importance_gain", "feature_desc"]
            ]
            top_gain = top_gain.sort_values("Importance_gain", ascending=False)
            chart = (
                alt.Chart(top_gain)
                .mark_bar()
                .encode(
                    x=alt.X("Feature_gain:N", title="Feature"),
                    y=alt.Y("Importance_gain:Q", title="Importance"),
                    tooltip=["Feature_gain", "Importance_gain", "feature_desc"],
                )
                .properties(width=600, height=400)
            )
            st.altair_chart(chart)


def show_shap_values(shap_df, BLOCS, selection_code_commune=None):
    st.header("Shap values")

    st.write(
        "Les valeurs de shap quantifient à quel point une variable socio-économique permet en moyenne d'augmenter (ou de diminuer) la valeur de la prédiction par rapport à la prédiction moyenne."
    )

    nb_feat_shap = st.slider(
        "Selectionnez un nombre de variable pour visualiser les valeurs de shap", 5, 30
    )

    tabs = st.tabs(["Participation"] + [f" Vote {trad[b]}" for b in BLOCS])
    for i, tab in enumerate(tabs):
        with tab:
            trends = ["par"] + [f"{b}" for b in BLOCS]
            shap_values_df = shap_df[trends[i]].copy()
            if selection_code_commune is not None:
                shap_commune = shap_values_df[
                    shap_values_df["codecommune"].astype(str)
                    == str(selection_code_commune)
                ]
                if len(shap_commune) == 0:
                    st.warning("Pas de valeurs de Shap pour cette commune")
                    st.stop()

            shap_values_df_wo_cc = shap_values_df.copy()
            if "codecommune" in shap_values_df_wo_cc.columns:
                shap_values_df_wo_cc = shap_values_df_wo_cc.drop(
                    columns=["codecommune"]
                )
            shap_values_df_wo_cc = shap_values_df_wo_cc.astype(float)

            if selection_code_commune is not None:
                row = (
                    shap_commune.drop(columns=["codecommune"], errors="ignore")
                    .iloc[0]
                    .astype(float)
                )
                row_values = row.values
            
                # Get expected values
                st.session_state["data"].load_result(
                    asset="results_full",
                    year=st.session_state['state'].year,
                    election_type=st.session_state['state'].get_type(as_type='code'),
                    trends=st.session_state['state'].get_blocs(as_type='code', order='alpha'),
                    columns=[f'p{trends[i]}_pred'],
                    filters=None,
                    asset_name="result_trend_i",
                )

                expl = shap.Explanation(
                    values=row_values,
                    data=row_values,
                    base_values=st.session_state["data"].container["result_trend_i"].mean(),
                    feature_names=shap_values_df_wo_cc.columns,
                )
                shap.plots.bar(expl, max_display=nb_feat_shap)

            else:
                # Maybe sub-sample if too much values
                expl = shap.Explanation(
                    values=shap_values_df_wo_cc.values, # Get the real values
                    data=shap_values_df_wo_cc.values,
                    base_values=st.session_state["data"].container["result_trend_i"].mean(),
                    feature_names=shap_values_df_wo_cc.columns
                   
                )
                shap.plots.beeswarm(expl, max_display=nb_feat_shap)
            st.pyplot(plt.gcf())
            plt.clf()



def plot_backtest(
    df,
    variables,
    years,
    true_suffix="_true",
    pred_suffix="_pred",
    yaxis_title="Taux de participation (%)",
):
    if isinstance(variables, str):
        variables = [variables]
    years_sorted = sorted(years)

    fig = go.Figure()

    for idx, variable in enumerate(variables):
        color = "#008000" if (variable=='ppar') else colors_dict[variable.replace('p', '')]
        true_vals = [
            (
                df.loc[variable, f"{year}_{st.session_state['state'].get_type(as_type='code')}{true_suffix}"]
                if f"{year}_{st.session_state['state'].get_type(as_type='code')}{true_suffix}" in df.columns
                else None
            )
            for year in years_sorted
        ]
        pred_vals = [
            (
                df.loc[variable, f"{year}_{st.session_state['state'].get_type(as_type='code')}{pred_suffix}"]
                if f"{year}_{st.session_state['state'].get_type(as_type='code')}{pred_suffix}" in df.columns
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
