from typing import Dict, List, Optional
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go
import shap
import urllib.parse
import pandas as pd
import numpy as np
from scipy.special import expit
import streamlit as st
import s3fs
import polars as pl
from asset.definitions import candidats_2022_mapping, colors_dict, get_colors, trad
from asset.features import FEATURE_AUG, get_feature_desc


@st.cache_data
def get_asset(_load_func, asset_name=None, **kwargs):
    """
    Universal cached loader
    """
    if asset_name is not None:
        kwargs["asset"] = asset_name
    return _load_func(**kwargs)


def check_home_run():
    # Check home page has been run
    if "home_run" not in st.session_state:
        st.switch_page("pages/home.py")


SLIDER_TEXT = "Selectionnez un nombre de features"
DEFAULT_NB_FEAT = 15
MIN_NB_FEAT = 5
MAX_NB_FEAT = 60


def make_slider(key):
    return st.slider(
        SLIDER_TEXT,
        MIN_NB_FEAT,
        MAX_NB_FEAT,
        DEFAULT_NB_FEAT,
        key=key,
    )


# Constants

BASE_COLUMNS = {"codecommune", "base_value"}
PLOT_TYPES_LOCAL = ["Waterfall", "Force", "Decision"]
PLOT_TYPES_GLOBAL = ["Beeswarm", "Scatter", "Decision"]


def latest_poll_refresh(year, election_type, trends):
    # Use directly the file with the utility functions
    fs = s3fs.S3FileSystem(
        anon=True, client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
    )
    if election_type != "pres":
        return None

    with fs.open(
        f"arthurmanceau/poll_tracker/polls/presidentiel/{year}/t1/polls.parquet",
        "rb",
    ) as f:
        X = (
            pl.scan_parquet(f)
            .select(
                [f"BP_{trend}" for trend in trends if trend != "par"] + ["start_date"]
            )
            .sort("start_date", descending=True)
            .head(10)
            .collect()
            .select(pl.all().mean())
        )
    return X


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
                    Résultats selon les (10) derniers [sondages](https://sondages-presidentielles.streamlit.app/)
                """
                )
                # Update data line
                refresh = latest_poll_refresh(
                    st.session_state["state"].year,
                    st.session_state["state"].get_type(as_type="code"),
                    st.session_state["state"].get_blocs(as_type="code"),
                )
                if not refresh.is_empty():
                    latest = refresh.row(0, named=True)
                    for target in latest:
                        target_data_line = f"pvote{target.replace('BP_', '')}"
                        if target_data_line in data_line.index and target in latest:
                            data_line.loc[target_data_line, "2027_pres_poll"] = latest[
                                target
                            ]

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


def show_feature_importance(importance_df, blocs, nb_feat, importance_type):
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

            # Normalize importance to % of total
            total_imp = df[importance_type].sum()
            if total_imp != 0:
                df[f"{importance_type}_pct"] = (df[importance_type] / total_imp) * 100
            else:
                df[f"{importance_type}_pct"] = 0.0

            # Use the original importance_type for sorting, but keep the original column in the slice
            top_gain = df.nlargest(nb_feat, importance_type)[
                ["Feature", importance_type, f"{importance_type}_pct", "feature_name"]
            ]
            top_gain = top_gain.sort_values(importance_type, ascending=False)
            chart = (
                alt.Chart(top_gain)
                .mark_bar()
                .encode(
                    x=alt.X("feature_name:N", title="Feature"),
                    y=alt.Y(
                        f"{importance_type}_pct:Q",
                        title="Importance (%)",
                        axis=alt.Axis(),
                    ),
                    tooltip=["Feature", f"{importance_type}_pct:Q", "feature_name"],
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
    """Generate a local SHAP plot using isolated rendering for Streamlit."""

    if plot_type == "Force":
        # Rendu HTML Natif et Interactif (JavaScript)
        shap_values_matrix = getattr(expl, "values", expl)
        feature_values_matrix = getattr(expl, "data", None)
        base_value = getattr(expl, "base_values", None)

        force_plot_html = shap.plots.force(
            base_value,
            shap_values_matrix,
            features=feature_values_matrix,
            feature_names=feature_names,
            matplotlib=False,
            link="logit",
            show=False,
        )

        # Injection du script JS d'Ollama/SHAP + Composant HTML
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
        encoded_html = urllib.parse.quote(shap_html)
        st.iframe(src=f"data:text/html;charset=utf-8,{encoded_html}", height=140)

    else:
        # Rendu Matplotlib standard mais ISOLÉ (évite les figures blanches)
        plt.close("all")

        # Ajustement de la taille de la figure selon le type de graphique
        if plot_type == "Decision":
            fig, ax = plt.subplots(figsize=(10, max(4, nb_feat * 0.35 + 2)))
        else:
            fig, ax = plt.subplots(figsize=(10, max(4, nb_feat * 0.4)))

        plt.rcParams["font.size"] = 7

        if plot_type == "Waterfall":
            shap.plots.waterfall(
                expl,
                max_display=nb_feat,
                show=False,
            )
            ax.xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda y, _: f"{expit(y):.1%}")
            )

        elif plot_type == "Decision":
            shap_values_matrix = getattr(expl, "values", expl)
            feature_values_matrix = getattr(expl, "data", None)

            # Gestion sécurisée de la matrice de données
            if feature_values_matrix is None and expl is not None:
                feature_values_matrix = expl.to_numpy()

            display_range = slice(None, -nb_feat - 1, -1)

            shap.decision_plot(
                expl.base_values,
                shap_values_matrix,
                features=feature_values_matrix,
                feature_names=feature_names,
                feature_display_range=display_range,
                highlight=0
                if (shap_values_matrix.ndim == 1 or len(shap_values_matrix) == 1)
                else None,
                show=False,
            )
            # Puisqu'on a multiplié par 100 au-dessus, l'axe X affiche directement les pourcentages réels
            plt.gca().xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, _: f"{expit(x):.1%}")
            )

        # Affichage sécurisé de la figure Matplotlib dédiée
        st.pyplot(fig)
        plt.close(fig)


def generate_global_plot(
    values: np.ndarray,
    data: np.ndarray,
    feature_names: List[str],
    plot_type: str,
    nb_feat: int,
    x_feat: Optional[str] = None,
    y_feat: Optional[str] = None,
    sample_labels: Optional[List[str]] = None,
    base_value: Optional[float] = None,
) -> None:
    """Generate a global SHAP plot (Beeswarm, Scatter, Decision, etc.)."""
    plt.close("all")
    fig = plt.figure()
    plt.rcParams["font.size"] = 7

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
            max_display=nb_feat,
            show=False,
            color_bar=True,
        )
        plt.gca().xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, _: f"{expit(x) - 0.5:.1%}")
        )
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

            fig = plt.gcf()
            fig.set_size_inches(16, 10, forward=True)
            ax = fig.axes[0]

            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda y, _: f"{expit(y) - 0.5:.1%}")
            )
            ax.tick_params(axis="both", labelsize=9)
            fig.tight_layout()
    elif plot_type == "Decision":
        if base_value is not None:
            display_range = slice(None, -nb_feat - 1, -1)
            explanation = shap.Explanation(
                values=values,
                data=data,
                feature_names=feature_names,
            )
            shap_values_matrix = getattr(explanation, "values", explanation)
            shap.decision_plot(
                base_value,
                shap_values_matrix,
                feature_names=feature_names,
                feature_display_range=display_range,
                legend_labels=list(sample_labels),
                show=False,
            )
            plt.gca().xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, _: f"{expit(x):.1%}")
            )

    st.pyplot(fig)
    plt.close(fig)


def convert_logits_to_probs(
    values: np.ndarray, base_value: float
) -> tuple[np.ndarray, float]:
    """
    Convert SHAP values from logit space to probability space using proportional redistribution.

    Args:
        values: SHAP values matrix (observations x features) in logit space.
        base_value: Base value (intercept) in logit space.

    Returns:
        Tuple of (shap_probs, prob_base).
    """
    return values, base_value
    # def sigmoid(x):
    #     return 1 / (1 + np.exp(-x))

    # # Total predicted logit per observation
    # logit_total = np.sum(values, axis=1) + base_value

    # # Base and predicted probabilities
    # prob_base = sigmoid(base_value)
    # prob_pred = sigmoid(logit_total)

    # # Total probability difference to distribute
    # delta_prob_total = prob_pred - prob_base

    # # Absolute sum of logits per observation to use as weights
    # sum_abs_logit = np.sum(np.abs(values), axis=1, keepdims=True)
    # sum_abs_logit[sum_abs_logit == 0] = 1.0  # Avoid division by zero

    # # Proportional redistribution: delta_prob * (logit_val / sum_abs_logit)
    # shap_probs = delta_prob_total[:, np.newaxis] * (values / sum_abs_logit)

    # return shap_probs, prob_base


def show_shap_values(
    shap_df: Dict[str, pd.DataFrame],
    data_sample: pd.DataFrame,
    BLOCS: List[str],
    selection_code_commune: Optional[str] = None,
    nb_feat: int = 15,
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

            shap_values_prob, prob_base = values, base_value

            # Local mode (single commune)
            if selection_code_commune is not None:
                local_mask = df["codecommune"].astype(str) == str(
                    selection_code_commune
                )
                if not local_mask.any():
                    st.warning(f"No SHAP values for commune {selection_code_commune}")
                    continue

                # Find the index of the selected commune within the probability matrix
                commune_code = df.loc[local_mask, "codecommune"].iloc[0]

                # Get the index of this commune in the 'mask' filtered view
                # mask is a boolean series of the same length as df
                global_indices = np.where(mask)[0]
                # Find which of these global indices corresponds to our local commune
                commune_global_idx = np.where(
                    df.loc[global_indices, "codecommune"] == commune_code
                )[0][0]

                x_prob = shap_values_prob[commune_global_idx]
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
                    values=x_prob,
                    data=x_data,
                    base_values=prob_base,
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
                    "Type de représentation",
                    PLOT_TYPES_GLOBAL,
                    key=f"global_plot_{i}",
                )

                if global_plot_type == "Scatter":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_feat = st.selectbox(
                            "Valeur de shap en fonction du prédicteur",
                            feature_names,
                            key=f"x_feat_{i}",
                        )
                    with col2:
                        y_feat = st.selectbox(
                            "Intéraction avec le prédicteur",
                            feature_names,
                            key=f"y_feat_{i}",
                        )
                    generate_global_plot(
                        shap_values_prob,
                        data,
                        feature_names,
                        global_plot_type,
                        nb_feat,
                        x_feat,
                        y_feat,
                    )
                elif global_plot_type == "Decision":
                    if "inscrits" in data_sample.columns:
                        top_5_indices = data_sample.nlargest(5, "inscrits").index
                        # We need indices relative to the 'mask' used for shap_values_prob
                        # mask is a boolean series of the same length as df
                        # we want to find where mask is true AND index is in top_5_indices
                        commune_codes = df.loc[mask, "codecommune"].values
                        top_5_codes = data_sample.loc[
                            top_5_indices, "codecommune"
                        ].values

                        mapping_commune = pd.read_csv("app/asset/communes2022.csv")
                        subset_labels = [
                            mapping_commune[mapping_commune["codecommune"] == code][
                                "nomcommune"
                            ].iloc[0]
                            for code in list(top_5_codes)
                        ]

                        indices = []
                        for code in top_5_codes:
                            if code in commune_codes:
                                indices.append(np.where(commune_codes == code)[0][0])

                        subset_values = shap_values_prob[indices]
                        # subset_labels = [str(code) for code in top_5_codes[0:len(indices)]]
                    else:
                        subset_values = shap_values_prob[:5]
                        subset_labels = [
                            str(c) for c in df.loc[mask, "codecommune"].values[:5]
                        ]

                    generate_global_plot(
                        subset_values,
                        data,
                        feature_names,
                        global_plot_type,
                        nb_feat,
                        sample_labels=subset_labels,
                        base_value=prob_base,
                    )
                else:
                    generate_global_plot(
                        shap_values_prob,
                        data,
                        feature_names,
                        global_plot_type,
                        nb_feat,
                    )


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
    state_code = st.session_state["state"].get_type(as_type="code")
    fig = go.Figure()

    for variable in variables:
        key = variable.replace("pvote", "")
        label = trad[key] if variable == "pvotepar" else f"Vote {trad[key]}"
        color = "#008000" if variable == "pvotepar" else colors_dict[key]
        true_vals = [
            df.loc[variable, f"{year}_{state_code}{true_suffix}"]
            if f"{year}_{state_code}{true_suffix}" in df.columns
            else np.nan
            for year in years_sorted
        ]
        pred_vals = [
            df.loc[variable, f"{year}_{state_code}{pred_suffix}"]
            if f"{year}_{state_code}{pred_suffix}" in df.columns
            else np.nan
            for year in years_sorted
        ]
        baseline_vals = [np.nan] + true_vals[:-1] if true_vals else []
        true_array = np.asarray(true_vals, dtype=float)
        pred_array = np.asarray(pred_vals, dtype=float)
        baseline_array = np.asarray(baseline_vals, dtype=float)
        valid = (
            np.isfinite(true_array)
            & np.isfinite(pred_array)
            & np.isfinite(baseline_array)
        )
        with st.expander(f"Erreur moyenne {label}"):
            if valid.any():
                errors = np.mean(np.abs(true_array[valid] - pred_array[valid]))
                errors_baseline = np.mean(
                    np.abs(true_array[valid] - baseline_array[valid])
                )
                col_model, col_baseline = st.columns(2)
                col_model.metric(
                    label="Erreur moyenne · Modèle",
                    value=f"{errors:.2f}",
                    delta=f"{errors - errors_baseline:+.2f} vs baseline",
                    delta_color="inverse",
                    border=False,
                    help="Erreur absolue moyenne. Une valeur plus faible est meilleure.",
                )
                col_baseline.metric(
                    label="Erreur moyenne · Baseline",
                    value=f"{errors_baseline:.2f}",
                    border=False,
                    help="Valeur réelle de l'année précédente, sur les mêmes années.",
                )
            else:
                st.info("Pas assez de données pour comparer le modèle à la baseline.")

        for values, name, trace_color, width, dash, size, symbol in [
            (baseline_vals, "Baseline", "#999999", 1, "dot", 5, "triangle-down"),
            (true_vals, "Réel", color, 3, "solid", 10, "triangle-down"),
            (pred_vals, "Prédiction", color, 3, "dot", 10, "triangle-up"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=years_sorted,
                    y=values,
                    mode="lines+markers",
                    name=f"{label} - {name}",
                    line=dict(color=trace_color, width=width, dash=dash),
                    marker=dict(size=size, color=trace_color, symbol=symbol),
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
