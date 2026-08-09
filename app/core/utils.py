from typing import Dict, List, Optional
import altair as alt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
import pandas as pd
import numpy as np
import streamlit as st
from asset.definitions import candidats_2022_mapping, colors_dict, get_colors, trad
from asset.features import FEATURE_AUG, get_feature_desc

# Constants
DEFAULT_NB_FEAT = 10
MIN_NB_FEAT = 5
MAX_NB_FEAT = 100
BASE_COLUMNS = {"codecommune", "base_value"}
PLOT_TYPES_LOCAL = ["Waterfall", "Force"]
PLOT_TYPES_GLOBAL = ["Beeswarm", "Bar", "Scatter"]


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
            if int(year) < 2026:
                result_func(
                    data_line,
                    year_type=f"{year}_{t}",
                    blocs=blocs,
                    label="true",
                    p="vote",
                )
            else:
                st.write("Election qui n'a pas encore eu lieu")

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
            if int(year) < 2026:
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
            else:
                st.write("Election qui n'a pas encore eu lieu")

    with tab1:
        with st.expander("Résultats", expanded=True):
            st.write(
                """
                Résultats
            """
            )
            if int(year) < 2026:
                result_func(
                    data_line,
                    year_type=f"{year}_{t}",
                    blocs=blocs,
                    label="true",
                    p="pvote",
                )
            else:
                st.write("Election qui n'a pas encore eu lieu")

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
                    data_line,
                    year_type=f"{year}_{t}",
                    blocs=blocs,
                    label="poll",
                    p="pvote",
                )

        with st.expander("Erreur", expanded=True):
            st.write(
                """
                Erreur de la prédiction du modèle pour l'élection
            """
            )
            if int(year) < 2026:
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
            else:
                st.write("Election qui n'a pas encore eu lieu")


def show_feature_importance(importance_df, blocs):
    st.header(
        "Déterminants socio-économiques les plus importants dans le modèle de prédiction"
    )

    tab1, tab2 = st.columns([3, 1])
    with tab1:
        nb_feat = st.slider(
            "Selectionnez un nombre de variable pour visualiser l'importance des variables socio-économiques",
            5,
            50,
        )
    with tab2:
        importance_type = st.selectbox(
            "Importance type",
            options=[
                "gain",
                "shap",
                "cover",
                "weight",
                "permutation",
                "total_gain",
                "total_cover",
            ],
        )

    trends = ["par"] + [f"{b.replace('tau', '')}" for b in blocs]
    tabs = st.tabs(["Participation"] + [f"Vote {trad[b]}" for b in blocs])
    for i, tab in enumerate(tabs):
        with tab:
            df = importance_df[trends[i]].copy()

            df["feature_name"] = [
                FEATURE_AUG.get(
                    (f.removeprefix("F_") if f.startswith("F_") else f).split("_")[0],
                    "",
                )
                + get_feature_desc(
                    (f.removeprefix("F_") if f.startswith("F_") else f).split("_")[-1]
                )
                for f in df["Feature"].values
            ]
            with st.expander("Feature utilisés"):
                st.write(
                    f"{len(df['feature_name'].to_list())} features utilisés (sélection par permutation feature importance)"
                )
                st.info(", ".join(df["feature_name"].to_list()))

            st.write(f"Importance (impotance type: {importance_type})")
            top_gain = df.nlargest(nb_feat, importance_type)[
                ["Feature", importance_type, "feature_name"]
            ]
            top_gain = top_gain.sort_values(importance_type, ascending=False)
            chart = (
                alt.Chart(top_gain)
                .mark_bar()
                .encode(
                    x=alt.X("feature_name:N", title="Feature"),
                    y=alt.Y(
                        f"{importance_type}:Q",
                        title="Importance",
                        axis=alt.Axis(),
                    ),
                    tooltip=["Feature", importance_type, "feature_name"],
                )
                .properties(width=600, height=400)
            )
            st.altair_chart(chart)


def format_feature(feature: str) -> str:
    """Format a feature name for display."""
    feature = feature.removeprefix("F_") if feature.startswith("F_") else feature
    parts = feature.split("_")
    return FEATURE_AUG.get(parts[0], "") + get_feature_desc(parts[-1])


def generate_local_plot(
    expl: shap.Explanation,
    plot_type: str,
    nb_feat: int,
    feature_names: List[str],
) -> None:
    """Generate a local SHAP plot (Waterfall, Force, or Decision)."""
    plt.close("all")
    plt.figure()
    plt.rcParams["font.size"] = 8  # Smaller feature names
    if plot_type == "Waterfall":
        shap.plots.waterfall(
            expl,
            max_display=nb_feat,
            show=False,
        )
    elif plot_type == "Force":
        shap.plots.force(
            expl,
            matplotlib=True,
            show=False,
            feature_names=feature_names,
        )
    st.pyplot(plt.gcf())
    plt.close()


def generate_global_plot(
    values: np.ndarray,
    data: np.ndarray,
    feature_names: List[str],
    plot_type: str,
    nb_feat: int,
    x_feat: Optional[str] = None,
    y_feat: Optional[str] = None,
) -> None:
    """Generate a global SHAP plot (Beeswarm, Bar, Scatter, etc.)."""
    plt.close("all")
    plt.figure()

    if plot_type == "Beeswarm":
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(values).mean(axis=0)
        # Get indices of top nb_feat features
        top_indices = np.argsort(mean_abs_shap)[-nb_feat:][::-1]  # Descending order
        # Slice values and data to include only top features
        values_subset = values[:, top_indices]
        data_subset = data[:, top_indices] if data is not None else None
        feature_names_subset = [feature_names[i] for i in top_indices]
        # Generate the plot
        shap.summary_plot(
            values_subset,
            data_subset,
            feature_names=feature_names_subset,
            show=False,
            color_bar=True,
            plot_size=(10, 6),
        )
    elif plot_type == "Bar":
        mean_abs = np.abs(values).mean(axis=0)
        idx = np.argsort(mean_abs)[-nb_feat:]
        plt.barh(np.array(feature_names)[idx], mean_abs[idx])
        plt.title("Mean Absolute SHAP Values")

    elif plot_type == "Scatter":
        if values.shape[1] >= 2 and x_feat and y_feat:
            explanation = shap.Explanation(
                values=values,
                data=data,
                feature_names=feature_names,
            )

            ix = feature_names.index(x_feat)
            iy = feature_names.index(y_feat)

            shap.plots.scatter(
                explanation[:, ix],
                color=explanation[:, iy],
                show=False,
            )

    st.pyplot(plt.gcf())
    plt.close()


def show_shap_values(
    shap_df: Dict[str, pd.DataFrame],
    data_sample: pd.DataFrame,
    BLOCS: List[str],
    selection_code_commune: Optional[str] = None,
) -> None:
    """
    Display SHAP values for a given dataset and model.

    Args:
        shap_df: Dictionary of SHAP DataFrames, keyed by trend (e.g., "par", bloc names).
        data_sample: DataFrame containing the raw data for the samples.
        BLOCS: List of bloc names (e.g., ["LREM", "RN"]).
        selection_code_commune: Optional commune code for local explanations.
    """
    st.header("SHAP Values Analysis")

    # Get all columns (excluding base_value)
    all_columns = set()
    for df in shap_df.values():
        all_columns.update(df.columns)
    all_columns.discard("base_value")

    nb_feat = st.slider(
        "Number of features to display",
        MIN_NB_FEAT,
        MAX_NB_FEAT,
        DEFAULT_NB_FEAT,
        key="shap_nb_feat",
    )

    trends = ["par"] + [f"{b}" for b in BLOCS]
    tab_labels = ["Participation"] + [f"Vote {trad.get(b, b)}" for b in BLOCS]
    tabs = st.tabs(tab_labels)

    for i, tab in enumerate(tabs):
        with tab:
            trend = trends[i]
            df = shap_df.get(trend)
            if df is None or df.empty:
                st.warning(f"No SHAP values for {tab_labels[i]}")
                continue

            # Extract base columns and raw features
            base_cols = BASE_COLUMNS
            raw_features = [c for c in df.columns if c not in base_cols]
            feature_names = [format_feature(f) for f in raw_features]

            # Filter data for the current trend
            mask = df["codecommune"].isin(data_sample["codecommune"].values)
            if not mask.any():
                st.warning(f"No matching communes for {tab_labels[i]}")
                continue

            values = df.loc[mask, raw_features].astype(float).to_numpy()
            data = (
                data_sample.loc[
                    data_sample["codecommune"].isin(df.loc[mask, "codecommune"]),
                    raw_features,
                ]
                .astype(float)
                .to_numpy()
            )
            base_value = float(df.loc[mask, "base_value"].iloc[0])

            # Local mode (single commune)
            if selection_code_commune is not None:
                local_mask = df["codecommune"].astype(str) == str(
                    selection_code_commune
                )
                if not local_mask.any():
                    st.warning(f"No SHAP values for commune {selection_code_commune}")
                    continue

                local_df = df[local_mask]
                x = local_df[raw_features].iloc[0].astype(float).values
                x_data = (
                    data_sample.loc[
                        data_sample["codecommune"] == selection_code_commune,
                        raw_features,
                    ]
                    .iloc[0]
                    .astype(float)
                    .values
                )

                expl = shap.Explanation(
                    values=x,
                    data=x_data,
                    base_values=base_value,
                    feature_names=feature_names,
                )
                local_plot_type = st.selectbox(
                    "Local Plot Type",
                    PLOT_TYPES_LOCAL,
                    key=f"local_plot_{i}",
                )
                generate_local_plot(expl, local_plot_type, nb_feat, feature_names)

            # Global mode (all communes)
            else:
                global_plot_type = st.selectbox(
                    "Global Plot Type",
                    PLOT_TYPES_GLOBAL,
                    key=f"global_plot_{i}",
                )

                if global_plot_type == "Scatter":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_feat = st.selectbox(
                            "Shap values depending on feature",
                            feature_names,
                            key=f"x_feat_{i}",
                        )
                    with col2:
                        y_feat = st.selectbox(
                            "Interaction with feature",
                            feature_names,
                            key=f"y_feat_{i}",
                        )
                    generate_global_plot(
                        values,
                        data,
                        feature_names,
                        global_plot_type,
                        nb_feat,
                        x_feat,
                        y_feat,
                    )
                else:
                    generate_global_plot(
                        values,
                        data,
                        feature_names,
                        global_plot_type,
                        nb_feat,
                    )


# def show_shap_values(shap_df, BLOCS, selection_code_commune=None):
#     st.header("Shap values")

#     all_columns = set()
#     for df in shap_df.values():
#         all_columns.update(df.columns)
#     all_columns.discard("base_value")

#     load_data(all_columns, selection_code_commune)

#     def format_feature(f):
#         f = f.removeprefix("F_") if f.startswith("F_") else f
#         parts = f.split("_")
#         return (
#             FEATURE_AUG.get(parts[0], "")
#             + get_feature_desc(parts[-1])
#         )

#     st.write(
#         "Les valeurs de shap quantifient à quel point une variable socio-économique influence la prédiction."
#     )

#     nb_feat_shap = st.slider(
#         "Selectionnez un nombre de variables pour visualiser les valeurs de shap", 5, 30
#     )

#     tabs = st.tabs(["Participation"] + [f" Vote {trad[b]}" for b in BLOCS])

#     for i, tab in enumerate(tabs):
#         with tab:
#             trends = ["par"] + [f"{b}" for b in BLOCS]
#             shap_values_df = shap_df[trends[i]].copy()

#             base_cols = {"codecommune", "base_value"}
#             raw_features = [c for c in shap_values_df.columns if c not in base_cols]

#             feature_names = [format_feature(f) for f in raw_features]

#             if selection_code_commune is not None:
#                 shap_commune = shap_values_df[
#                     shap_values_df["codecommune"].astype(str) == str(selection_code_commune)
#                 ]

#                 if len(shap_commune) == 0:
#                     st.warning("Pas de valeurs de Shap pour cette commune")
#                     st.stop()

#                 base_value = float(shap_commune["base_value"].iloc[0])

#                 row_values = shap_commune[raw_features].iloc[0].astype(float).values
#                 row_data = st.session_state["data"].container["data_sample_all"].loc[
#                     st.session_state["data"].container["data_sample_all"]["codecommune"]
#                     == selection_code_commune,
#                     raw_features,
#                 ].iloc[0].astype(float).values

#                 expl = shap.Explanation(
#                     values=row_values,
#                     data=row_data,
#                     base_values=base_value,
#                     feature_names=feature_names,
#                 )

#                 shap.plots.waterfall(expl, max_display=nb_feat_shap)

#             else:
#                 communes_communes = list(
#                     set(shap_values_df["codecommune"]).intersection(
#                         set(st.session_state["data"].container["data_sample_all"]["codecommune"])
#                     )
#                 )

#                 mask = shap_values_df["codecommune"].isin(
#                     st.session_state["data"].container["data_sample_all"]["codecommune"].values
#                 )

#                 values = shap_values_df.loc[mask, raw_features].astype(float).to_numpy()

#                 data = st.session_state["data"].container["data_sample_all"].loc[
#                     st.session_state["data"].container["data_sample_all"]["codecommune"].isin(
#                         shap_values_df.loc[mask, "codecommune"]
#                     ),
#                     raw_features,
#                 ].astype(float).to_numpy()

#                 base_value = shap_values_df.loc[mask, "base_value"].iloc[0]

#                 st.write("BASE VALUE:", base_value)
#                 st.write("SHAP CHECK SUM:", values[0].sum())

#                 shap.summary_plot(
#                     values,
#                     data,
#                     feature_names=feature_names,
#                     show=False
#                 )

#             st.pyplot(plt.gcf())
#             plt.clf()


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
