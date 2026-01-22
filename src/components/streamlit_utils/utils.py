import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# colors = {'G':'#bb1840', 'CG': '#ffc0c0', 'C':'#FED700', 'CD':'#0066cc', 'D':'#0D378A'}
colors = ["#bb1840", "#ffc0c0", "#FED700", "#0066cc", "#0D378A"]
blocs = ["G", "CG", "C", "CD", "D"]
trends = ["par"] + [f"vote{b}" for b in blocs]
trad = {
    "G": "à gauche",
    "CD": "pour le centre-droite",
    "C": "pour le centre",
    "D": "à droite",
    "CG": "pour le centre-gauche",
}
type_trad = {"pres": "présidentielles", "leg": "leglisatives"}


def compute_agg_results(X):
    data_line = pd.Series([np.nan] * len(X.columns), index=X.columns)
    data_line["inscrits"] = X["inscrits"].sum()
    data_line["exprimes"] = X["exprimes"].sum()
    for m in ["pred", "true"]:
        data_line[f"votants_{m}"] = X[f"votants_{m}"].sum()
        data_line[f"ppar_{m}"] = round(
            data_line[f"votants_{m}"] / data_line["inscrits"] * 100, 2
        )
    for x in ["votants", "ppar"]:
        data_line[f"{x}_diff"] = round(
            data_line[f"{x}_true"] - data_line[f"{x}_pred"], 2
        )
    for b in blocs:
        for m in ["pred", "true"]:
            data_line[f"vote{b}_{m}"] = X[f"vote{b}_{m}"].sum()
            data_line[f"pvote{b}_{m}"] = round(
                data_line[f"vote{b}_{m}"] / data_line["exprimes"] * 100, 2
            )
        data_line[f"vote{b}_diff"] = (
            data_line[f"vote{b}_true"] - data_line[f"vote{b}_pred"]
        )
        data_line[f"pvote{b}_diff"] = round(
            data_line[f"pvote{b}_true"] - data_line[f"pvote{b}_pred"], 2
        )
    return data_line


def present_results(data_line):

    tab1, tab2 = st.tabs(["Nombre de vote", "Pourcentage des suffrages"])

    with tab1:

        with st.expander("Résultats"):
            st.write(
                """
                Résultats de l'élection
            """
            )
            st.dataframe(
                data_line[["votants_true", "exprimes"]].reset_index(drop=True),
                hide_index=True,
                column_config={
                    "votants_true": "Nombre de votants",
                    "exprimes": "Nombre de suffrage exprimés",
                },
            )
            st.dataframe(
                data_line[[f"vote{b}_true" for b in blocs]].reset_index(drop=True),
                hide_index=True,
                column_config={
                    f"vote{b}_true": f"Nombre de vote {trad[b]}" for b in blocs
                },
            )
            st.bar_chart(
                data=(
                    data_line[[f"vote{b}_true" for b in blocs]].reset_index(drop=True)
                ),
                color=colors,
                horizontal=True,
            )

        with st.expander("Prédictions"):
            st.write(
                """
                Prédictions du modèle pour l'élection
            """
            )
            st.write(
                "On ne considère pas les votes blancs et nuls, le nombre de votants est le nombre de suffrages exprimés."
            )
            st.dataframe(
                data_line[["votants_pred"]].reset_index(drop=True),
                hide_index=True,
                column_config={"votants_pred": "Nombre de votants"},
            )
            st.dataframe(
                data_line[[f"vote{b}_pred" for b in blocs]].reset_index(drop=True),
                hide_index=True,
                column_config={
                    f"vote{b}_pred": f"Nombre de vote {trad[b]}" for b in blocs
                },
            )
            st.bar_chart(
                data=(
                    data_line[[f"vote{b}_pred" for b in blocs]].reset_index(drop=True)
                ),
                color=colors,
                horizontal=True,
            )

        with st.expander("Erreur"):
            st.write(
                """
                Erreur de la prédictions du modèle pour l'élection
            """
            )
            col_config = {
                f"vote{b}_diff": f"Différence avec la prédiction du vote {trad[b]}"
                for b in blocs
            }
            col_config["votants_diff"] = (
                "Différence avec la prédiction pour la participation"
            )
            st.dataframe(
                data_line[
                    [f"vote{b}_diff" for b in blocs] + ["votants_diff"]
                ].reset_index(drop=True),
                hide_index=True,
                column_config=col_config,
            )
            st.bar_chart(
                data=(
                    data_line[
                        [f"vote{b}_diff" for b in blocs] + ["votants_diff"]
                    ].reset_index(drop=True)
                ).T
            )

    with tab2:

        with st.expander("Résultats"):
            st.write(
                """
                Results of the election
            """
            )
            st.dataframe(
                data_line[["ppar_true"]].reset_index(drop=True), hide_index=True
            )
            st.dataframe(
                data_line[[f"pvote{b}_true" for b in blocs]].reset_index(drop=True),
                hide_index=True,
            )
            st.bar_chart(
                data=(
                    data_line[[f"pvote{b}_true" for b in blocs]].reset_index(drop=True)
                ),
                color=colors,
                horizontal=True,
            )

        with st.expander("Prédictions"):
            st.write(
                """
                Prédictions du modèle pour l'élection
            """
            )
            st.dataframe(
                data_line[["ppar_pred"]].reset_index(drop=True), hide_index=True
            )
            st.dataframe(
                data_line[[f"pvote{b}_pred" for b in blocs]].reset_index(drop=True),
                hide_index=True,
            )
            st.bar_chart(
                data=(
                    data_line[[f"pvote{b}_pred" for b in blocs]].reset_index(drop=True)
                ),
                color=colors,
                horizontal=True,
            )

        with st.expander("Erreur"):
            st.write(
                """
                Erreur de la prédictions du modèle pour l'élection
            """
            )
            st.dataframe(
                data_line[
                    [f"pvote{b}_diff" for b in blocs] + ["ppar_diff"]
                ].reset_index(drop=True),
                hide_index=True,
            )
            st.bar_chart(
                data=(
                    data_line[
                        [f"pvote{b}_diff" for b in blocs] + ["ppar_diff"]
                    ].reset_index(drop=True)
                ).T
            )


def show_feature_importance(importance_df):
    st.header(
        "Déterminants socio-économiques les plus importants dans le modèle de prédiction"
    )

    nb_feat = st.slider(
        "Selectionnez un nombre de variable pour visualiser l'importance des variables socio-économiques",
        5,
        30,
    )
    tabs = st.tabs(["Participation"] + [f"Vote {trad[b]}" for b in blocs])
    for i, tab in enumerate(tabs):
        with tab:
            df = importance_df[trends[i]].copy()
            st.write("Importance en gain total")
            top_gain = df.nlargest(nb_feat, "Importance_gain")[
                ["Feature_gain", "Importance_gain"]
            ]
            top_gain = top_gain.sort_values("Importance_gain", ascending=False)
            st.bar_chart(top_gain.set_index("Feature_gain")["Importance_gain"])
            st.write("Importance en valeur de shap")
            top_shap = df.nlargest(nb_feat, "Importance_shap")[
                ["Feature_shap", "Importance_shap"]
            ]
            top_shap = top_shap.sort_values("Importance_shap", ascending=False)
            st.bar_chart(top_shap.set_index("Feature_shap")["Importance_shap"])

            if st.button(
                f"Montrer l'importance en gain total de toutes les variables pour la prédiction de {trends[i]}"
            ):
                st.dataframe(
                    df,
                    column_config={
                        "Feature_gain": "Feature",
                        "Importance_gain": st.column_config.NumberColumn(
                            "Importance en gain (total)",
                            help="Quantifie à quel point ce feature permet de purifier l'arbre",
                            format="percent",
                        ),
                        "Feature_perm": "Feature",
                        "Importance_perm": st.column_config.NumberColumn(
                            "Importance en permutation",
                            help="",
                            format="percent",
                        ),
                        "Feature_shap": "Feature",
                        "Importance_shap": st.column_config.NumberColumn(
                            "Importance en valeur de shap",
                            help="Somme en valeur absolue des valeur de shap associé à ce feature pour chaque instance",
                            format="percent",
                        ),
                    },
                    hide_index=True,
                )


def show_shap_values(shap_df, selection_code_commune=None):
    st.header("Shap values")

    st.write(
        "Les valeurs de shap quantifient à quel point une variable socio-économique permet en moyenne d'augmenter (ou de diminuer) la valeur de la prédiction par rapport à la prédiction moyenne."
    )

    nb_feat_shap = st.slider(
        "Selectionnez un nombre de variable pour visualiser les valeurs de shap", 5, 30
    )

    tabs = st.tabs(["Participation"] + [f" Vote {trad[b]}" for b in blocs])
    for i, tab in enumerate(tabs):
        with tab:
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
                expl = shap.Explanation(
                    values=row_values,
                    data=row_values,
                    base_values=0.0,
                    feature_names=shap_values_df_wo_cc.columns,
                )

            else:
                expl = shap.Explanation(
                    values=shap_values_df_wo_cc.values,
                    data=shap_values_df_wo_cc.values,
                    feature_names=shap_values_df_wo_cc.columns,
                )
            shap.plots.beeswarm(expl, max_display=nb_feat_shap)
            st.pyplot(plt.gcf())
            plt.clf()

            if st.button(
                f"Montrer l'importance en valeur de Shap de toutes les variables pour la prédiction de {trends[i]}"
            ):
                st.dataframe(shap_values_df, hide_index=True)
