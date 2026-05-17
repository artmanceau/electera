from typing import Optional

import altair as alt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
import streamlit as st
from asset.definitions import (
    candidats_2022_mapping,
    colors_dict,
    get_colors,
    trad,
)
from asset.features import FEATURES_MAP, FEATURE_AUG


def check_home_run():
    if "home_run" in st.session_state:
        if st.session_state["home_run"]:
            return None
    st.switch_page("pages/home.py")


def diff_show(results, blocs, trad, label, label_show, year, t):
    col_config_error = {}
    for bloc in blocs:
        col_config_error[f"pvote{bloc}"] = st.column_config.NumberColumn(
            f"Vote {trad[bloc]}", format="%.1f%%"
        )
    st.dataframe(
        results.loc[[f"pvote{b}" for b in blocs], f"{year}_{t}_{label}"].to_frame().T,
        column_config=col_config_error,
        hide_index=True,
    )


def results_loc(data_line, year_type, blocs, label, p=""):
    ind = [f"pvotepar_{label}"] if p == "pvote" else [f"votants_{label}", "exprimes"]
    st.dataframe(
        data_line[ind].reset_index(drop=True),
        hide_index=True,
        column_config={
            "votants_true": st.column_config.NumberColumn(
                "Nombre de votants",
                format="%.1f%%" if p == "pvote" else "%,.0f",
            ),
            "votants_pred": st.column_config.NumberColumn(
                "Nombre de votants",
                format="%.1f%%" if p == "pvote" else "%,.0f",
            ),
            "exprimes": st.column_config.NumberColumn(
                "Nombre de suffrage exprimés",
                help="Le modèle ne prend pas en compte les votes blancs et nuls et considère que ce nombre est égale au nombre de votants.",
                format="%.1f%%" if p == "pvote" else "%,.0f",
            ),
            "pvotepar_true": st.column_config.NumberColumn(
                "Taux de participation",
                help="Le taux de participation est le nombre de votants sur le nombre d'inscrits",
                format="%.1f%%" if p == "pvote" else "%,.0f",
            ),
            "pvotepar_pred": st.column_config.NumberColumn(
                "Taux de participation",
                help="Le taux de participation est le nombre de votants sur le nombre d'inscrits",
                format="%.1f%%" if p == "pvote" else "%,.0f",
            ),
        },
    )
    col_config = {}
    for b in blocs:
        col_config[f"{p}{b}_{label}"] = st.column_config.NumberColumn(
            f"Vote {trad[b]}" if p == "pvote" else f"Nombre de vote {trad[b]}",
            help=f"Le vote {trad[b]} correspond à un vote pour un candidat d'une nuance politique appartenant à ce bloc politique. \
            Pour plus de détail sur la classification des blocs politique, cf. https://www.unehistoireduconflitpolitique.fr/. \
            En 2022, le vote {trad[b]} correspondait à un vote pour les candidats : {', '.join(candidats_2022_mapping[b])}. \
            Lorsqu'un candidat ne peut être classé dans un bloc, ses voix sont répartis à part égal entre l'ensemble des autres blocs",
            format="%.1f%%" if p == "pvote" else "%,.0f",
        )

    st.dataframe(
        data_line[[f"{p}{b}_{label}" for b in blocs]].reset_index(drop=True),
        hide_index=True,
        column_config=col_config,
    )
    df = data_line[[f"{p}{b}_{label}" for b in blocs]].reset_index(drop=True)
    df.columns = [f"Vote {trad.get(b, b)}" for b in blocs]
    st.bar_chart(
        data=df,
        color=get_colors(blocs, colors_dict),
        horizontal=True,
    )


def results_glob(data_line, year_type, blocs, label, p=""):
    ind = ["pvotepar", "pvoteexpr"] if p == "pvote" else ["votants", "exprimes"]
    col = f"{year_type}_{label}"

    if label != "poll":
        st.dataframe(
            data_line.loc[ind, col].to_frame().T,
            hide_index=True,
            column_config={
                "votants": st.column_config.NumberColumn(
                    "Nombre de votants",
                    format="%,.0f",
                ),
                "exprimes": st.column_config.NumberColumn(
                    "Nombre de suffrage exprimés",
                    help="Le modèle ne prend pas en compte les votes blancs et nuls et considère que ce nombre est égale au nombre de votants.",
                    format="%,.0f",
                ),
                "pvotepar": st.column_config.NumberColumn(
                    "Taux de participation",
                    help="Le taux de participation est le nombre de votants sur le nombre d'inscrits",
                    format="%.1f%%",
                ),
                "pvoteexpr": st.column_config.NumberColumn(
                    "Taux de suffrage exprimés",
                    help="Le taux de suffrage exprimés est le nombre de votes enregistré en retirant les bulletins blancs et nuls. \
                    Le modèle ne prend pas en compte les votes blancs et nuls et considère que ce taux est égale au taux de participation.",
                    format="%.1f%%",
                ),
            },
        )
    col_config = {}
    for b in blocs:
        col_config[f"{p}{b}"] = st.column_config.NumberColumn(
            f"Vote {trad[b]}" if p in ["p", "pvote"] else f"Nombre de vote {trad[b]}",
            help=f"Le vote {trad[b]} correspond à un vote pour un candidat d'une nuance politique appartenant à ce bloc politique. \
            Pour plus de détail sur la classification des blocs politique, cf. https://www.unehistoireduconflitpolitique.fr/. \
            En 2022, le vote {trad[b]} correspondait à un vote pour les candidats : {', '.join(candidats_2022_mapping[b])}. \
            Lorsqu'un candidat ne peut être classé dans un bloc, ses voix sont répartis à part égal entre l'ensemble des autres blocs",
            format="%.1f%%" if p in ["p", "pvote"] else "%,.0f",
        )

    st.dataframe(
        data_line.loc[[f"{p}{b}" for b in blocs], col].to_frame().T,
        hide_index=True,
        column_config=col_config,
    )
    df = data_line.loc[[f"{p}{b}" for b in blocs], col].to_frame().T
    df.columns = [f"Vote {trad.get(b, b)}" for b in blocs]
    st.bar_chart(
        data=df,
        color=get_colors(blocs, colors_dict),
        horizontal=True,
    )


def present_results(data_line, year, t, blocs, scale):
    if scale == "global":
        result_func = results_glob
    else:
        result_func = results_loc
        pvote_cols = [c for c in data_line.columns if c.startswith("pvote")]
        data_line[pvote_cols] = data_line[pvote_cols] * 100

    tab1, tab2 = st.tabs(["Pourcentage des suffrages", "Nombre de vote"])

    with tab2:
        with st.expander("Résultats", expanded=True):
            st.write(
                """
                Résultats
            """
            )
            result_func(
                data_line, year_type=f"{year}_{t}", blocs=blocs, label="true", p="vote"
            )

        with st.expander("Prédictions", expanded=True):
            st.write(
                """
                Prédictions du modèle pour l'élection
            """
            )
            result_func(
                data_line, year_type=f"{year}_{t}", blocs=blocs, label="pred", p="vote"
            )

        with st.expander("Erreur", expanded=True):
            st.write(
                """
                Erreur de la prédiction du modèle pour l'élection
            """
            )
            if scale == "local":
                col_config = {}
                for b in blocs:
                    col_config[f"vote{b}_diff"] = st.column_config.NumberColumn(
                        f"Différence avec la prédiction du vote {trad[b]}",
                        format="%,.0f",
                    )
                col_config["votants_diff"] = st.column_config.NumberColumn(
                    "Différence avec la prédiction pour le taux de participation",
                    format="%,.0f",
                )
                data_element = data_line[
                    [f"vote{b}_diff" for b in blocs] + ["votants_diff"]
                ].reset_index(drop=True)
            else:
                data_element = (
                    data_line.loc[
                        [f"vote{b}" for b in blocs] + ["votants"],
                        f"{year}_{t}_diff_agg",
                    ]
                    .to_frame()
                    .T
                )
                col_config = {}
                for b in blocs:
                    col_config[f"vote{b}"] = st.column_config.NumberColumn(
                        f"Différence avec la prédiction du vote {trad[b]}",
                        format="%,.0f",
                    )
                col_config["votants"] = st.column_config.NumberColumn(
                    "Différence avec la prédiction pour la participation",
                    format="%,.0f",
                )
            st.dataframe(
                data_element,
                hide_index=True,
                column_config=col_config,
            )
            data_plot = data_element.T.copy()
            trad_ = trad
            trad_["votants"] = "participation"
            data_plot.index = [
                f"Vote {trad[c.replace('vote', '').replace('_diff', '')]}"
                for c in data_plot.index
            ]
            st.bar_chart(data=data_plot, sort=False)

    with tab1:
        with st.expander("Résultats", expanded=True):
            st.write(
                """
                Résultats
            """
            )
            result_func(
                data_line, year_type=f"{year}_{t}", blocs=blocs, label="true", p="pvote"
            )

        with st.expander("Prédictions", expanded=True):
            st.write(
                """
                Prédictions du modèle pour l'élection
            """
            )
            result_func(
                data_line, year_type=f"{year}_{t}", blocs=blocs, label="pred", p="pvote"
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
                Erreur de la prédiction du modèle pour l'élection
            """
            )
            if scale == "local":
                data_element = data_line[
                    [f"pvote{b}_diff" for b in blocs] + ["pvotepar_diff"]
                ].reset_index(drop=True)
                col_config = {}
                for b in blocs:
                    col_config[f"pvote{b}_diff"] = st.column_config.NumberColumn(
                        f"Différence avec la prédiction du vote {trad[b]}",
                        help="L'erreur est ici calculée comme la différence entre le résultat réel et la prédiction à l'echelle agrégée.",
                        format="%.1f%%",
                    )
                col_config["pvotepar_diff"] = st.column_config.NumberColumn(
                    "Différence avec la prédiction pour la participation",
                    help="L'erreur est ici calculée comme la différence entre le résultat réel et la prédiction à l'echelle agrégée.",
                    format="%.1f%%",
                )
            else:
                data_element = (
                    data_line.loc[
                        [f"pvote{b}" for b in blocs] + ["pvotepar"],
                        f"{year}_{t}_diff_agg",
                    ]
                    .to_frame()
                    .T
                )
                col_config = {}
                for b in blocs:
                    col_config[f"pvote{b}"] = st.column_config.NumberColumn(
                        f"Différence avec la prédiction du vote {trad[b]}",
                        help="L'erreur est ici calculée comme la différence entre le résultat réel et la prédiction à l'echelle agrégée.",
                        format="%.1f%%",
                    )
                col_config["pvotepar"] = st.column_config.NumberColumn(
                    "Différence avec la prédiction pour la participation",
                    help="L'erreur est ici calculée comme la différence entre le résultat réel et la prédiction à l'echelle agrégée.",
                    format="%.1f%%",
                )

            st.dataframe(
                data_element,
                hide_index=True,
                column_config=col_config,
            )
            data_plot = data_element.T.copy()
            data_plot.index = [
                f"Vote {trad[c.replace('pvote', '').replace('_diff', '')]}"
                for c in data_plot.index
            ]
            st.bar_chart(data=data_plot, sort=False)


def show_feature_importance(importance_df, blocs):
    st.header(
        "Déterminants socio-économiques les plus importants dans le modèle de prédiction"
    )

    nb_feat = st.slider(
        "Selectionnez un nombre de variable pour visualiser l'importance des variables socio-économiques",
        5,
        50,
    )
    trends = ["par"] + [f"{b}" for b in blocs]
    tabs = st.tabs(["Participation"] + [f"Vote {trad[b]}" for b in blocs])
    for i, tab in enumerate(tabs):
        with tab:
            df = importance_df[trends[i]].copy()
            
            df["feature_name"] = [
                FEATURE_AUG.get((f.removeprefix("F_") if f.startswith("F_") else f).split('_')[0], '') + FEATURES_MAP.get((f.removeprefix("F_") if f.startswith("F_") else f).split("_")[-1], (f.removeprefix("F_") if f.startswith("F_") else f).split("_")[-1])
                for f in df["Feature_gain"].values
            ]
            with st.expander('Feature utilisés'):
                st.write(f'{len(df["feature_name"].to_list())} features utilisés (sélection par permutaiion feature importance)')
                st.info(", ".join(df["feature_name"].to_list()))

            st.write("Importance en gain total")
            top_gain = df.nlargest(nb_feat, "Importance_gain")[
                ["Feature_gain", "Importance_gain", "feature_name"]
            ]
            top_gain = top_gain.sort_values("Importance_gain", ascending=False)
            chart = (
                alt.Chart(top_gain)
                .mark_bar()
                .encode(
                    x=alt.X("feature_name:N", title="Feature"),
                    y=alt.Y("Importance_gain:Q", title="Importance", axis=alt.Axis(format=".0%")),
                    tooltip=["Feature_gain", "Importance_gain", "feature_name"],
                )
                .properties(width=600, height=400)
            )
            st.altair_chart(chart)


@st.cache_data
def load_data_sample(codecommune=None):
    if codecommune is None:
        st.session_state["data"].load_data_sample(
            columns=[f"p{trend}_pred"],
            filters=None,
            asset_name="result_trend_i",
        )
    else:
        pass


@st.cache_data
def load_base_values(trend):
    try:
        st.session_state["data"].load_result(
            asset="results_full",
            year=st.session_state["state"].year - 5,
            election_type=st.session_state["state"].get_type(as_type="code"),
            trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
            columns=[f"p{trend}_true"],
            filters=None,
            asset_name="result_trend_i",
        )
        return st.session_state["data"].container["result_trend_i"].mean().iloc[0]
    # Should be handle better in case no previous election available (no shap)?
    except:
        return 0.0


@st.cache_data
def load_data(features, selection_code_commune: Optional[str] | None = None):
    filters = [
        ("annee", "==", float(st.session_state["state"].year)),
        ("type", "==", int(st.session_state["state"].get_type(as_type="number"))),
    ]
    if selection_code_commune:
        filters.append(("codecommune", "==", selection_code_commune))
        name = "data_sample_commune"
    else:
        name = "data_sample_all"

    st.session_state["data"].load_data_sample(
        columns=features, filters=filters, asset_name=name
    )


def show_shap_values(shap_df, BLOCS, selection_code_commune=None):
    st.header("Shap values")

    # Get data values
    all_columns = set()
    for df in shap_df.values():
        all_columns.update(df.columns)

    load_data(all_columns, selection_code_commune)

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
            features = list(set(shap_values_df.columns) - set(["codecommune"]))
            if selection_code_commune is not None:
                shap_commune = shap_values_df[
                    shap_values_df["codecommune"].astype(str)
                    == str(selection_code_commune)
                ]
                if len(shap_commune) == 0:
                    st.warning("Pas de valeurs de Shap pour cette commune")
                    st.stop()

            # Get expected values
            mv = load_base_values(trends[i])

            if selection_code_commune is not None:
                expl = shap.Explanation(
                    values=shap_commune[features].iloc[0].astype(float).values,
                    data=st.session_state["data"]
                    .container["data_sample_all"]
                    .loc[
                        st.session_state["data"].container["data_sample_all"][
                            "codecommune"
                        ]
                        == selection_code_commune,
                        features,
                    ]
                    .iloc[0]
                    .astype(float)
                    .values,
                    base_values=float(mv),
                    feature_names=features,
                )
                shap.plots.waterfall(expl, max_display=nb_feat_shap)

            else:
                communes_communes = list(
                    set(shap_values_df["codecommune"]).intersection(
                        set(
                            st.session_state["data"].container["data_sample_all"][
                                "codecommune"
                            ]
                        )
                    )
                )

                expl = shap.Explanation(
                    values=shap_values_df.loc[
                        shap_values_df["codecommune"].isin(communes_communes), features
                    ]
                    .astype(float)
                    .values,
                    data=st.session_state["data"]
                    .container["data_sample_all"]
                    .loc[
                        st.session_state["data"]
                        .container["data_sample_all"]["codecommune"]
                        .isin(communes_communes),
                        features,
                    ],
                    base_values=float(mv),
                    feature_names=features,
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
        color = (
            "#008000"
            if (variable == "pvotepar")
            else colors_dict[variable.replace("pvote", "")]
        )
        true_vals = [
            (
                df.loc[
                    variable,
                    f"{year}_{st.session_state['state'].get_type(as_type='code')}{true_suffix}",
                ]
                if f"{year}_{st.session_state['state'].get_type(as_type='code')}{true_suffix}"
                in df.columns
                else None
            )
            for year in years_sorted
        ]
        pred_vals = [
            (
                df.loc[
                    variable,
                    f"{year}_{st.session_state['state'].get_type(as_type='code')}{pred_suffix}",
                ]
                if f"{year}_{st.session_state['state'].get_type(as_type='code')}{pred_suffix}"
                in df.columns
                else None
            )
            for year in years_sorted
        ]
        fig.add_trace(
            go.Scatter(
                x=years_sorted,
                y=true_vals,
                mode="lines+markers",
                name=(
                    f"{trad[variable.replace('pvote', '')]} - Réel"
                    if (variable == "pvotepar")
                    else f"Vote {trad[variable.replace('pvote', '')]} - Réel"
                ),
                line=dict(color=color, width=3, dash="solid"),
                marker=dict(size=10, color=color, symbol="triangle-down"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=years_sorted,
                y=pred_vals,
                mode="lines+markers",
                name=(
                    f"{trad[variable.replace('pvote', '')]} - Prédiction"
                    if (variable == "pvotepar")
                    else f"Vote {trad[variable.replace('pvote', '')]} - Prédiction"
                ),
                line=dict(color=color, width=3, dash="dot"),
                marker=dict(size=10, color=color, symbol="triangle-up"),
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

    st.plotly_chart(fig, width="stretch")
