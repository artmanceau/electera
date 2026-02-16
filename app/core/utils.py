import matplotlib.pyplot as plt
import shap
import streamlit as st

# colors = {'G':'#bb1840', 'CG': '#ffc0c0', 'C':'#FED700', 'CD':'#0066cc', 'D':'#0D378A'}
colors = [
    "#bb1840",
    "#0D378A",
]  # ["#bb1840", "#ffc0c0", "#FED700", "#0066cc", "#0D378A"]
blocs = ["TG", "TD"]  # ["G", "CG", "C", "CD", "D"]
trends = ["par"] + [f"vote{b}" for b in blocs]
trad = {"TD": "à gauche", "TG": "à droite"}
# {
#     "G": "à gauche",
#     "CD": "pour le centre-droite",
#     "C": "pour le centre",
#     "D": "à droite",
#     "CG": "pour le centre-gauche",
# }
type_trad = {"pres": "présidentielles", "leg": "leglisatives"}


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


def results_loc(data_line, year_type, label, p=""):
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
        data_line[[f"{p}vote{b}_{label}" for b in blocs]].reset_index(drop=True),
        hide_index=True,
        column_config={f"vote{b}_{label}": f"Nombre de vote {trad[b]}" for b in blocs},
    )
    st.bar_chart(
        data=(data_line[[f"{p}vote{b}_{label}" for b in blocs]].reset_index(drop=True)),
        color=colors,
        horizontal=True,
    )


def results_glob(data_line, year_type, label, p=""):
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
    st.dataframe(
        data_line.loc[[f"{p}vote{b}" for b in blocs], col].to_frame().T,
        hide_index=True,
        column_config={f"{p}vote{b}": f"Nombre de vote {trad[b]}" for b in blocs},
    )
    st.bar_chart(
        data=(data_line.loc[[f"{p}vote{b}" for b in blocs], col].to_frame().T),
        color=colors,
        horizontal=True,
    )


def present_results(data_line, year, t, scale):
    result_func = results_glob if scale == "global" else results_loc

    tab1, tab2 = st.tabs(["Nombre de vote", "Pourcentage des suffrages"])

    with tab1:

        with st.expander("Résultats"):
            st.write(
                """
                Résultats de l'élection
            """
            )
            result_func(data_line, year_type=f"{year}_{t}", label="true", p="")

        with st.expander("Prédictions"):
            st.write(
                """
                Prédictions du modèle pour l'élection
            """
            )
            result_func(data_line, year_type=f"{year}_{t}", label="true", p="")

        with st.expander("Erreur"):
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

                col_config = {
                    f"vote{b}": f"Différence avec la prédiction du vote {trad[b]}"
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
            st.bar_chart(data=(data_element).T)

    with tab2:

        with st.expander("Résultats"):
            st.write(
                """
                Results of the election
            """
            )
            result_func(data_line, year_type=f"{year}_{t}", label="true", p="p")

        with st.expander("Prédictions"):
            st.write(
                """
                Prédictions du modèle pour l'élection
            """
            )
            result_func(data_line, year_type=f"{year}_{t}", label="true", p="p")

        with st.expander("Erreur"):
            st.write(
                """
                Erreur de la prédictions du modèle pour l'élection
            """
            )
            if scale == "local":
                data_element = data_line[
                    [f"pvote{b}_diff" for b in blocs] + ["votants_diff"]
                ].reset_index(drop=True)

                col_config = {
                    f"pvote{b}": f"Différence avec la prédiction du vote {trad[b]}"
                    for b in blocs
                }
                col_config["votants_diff"] = (
                    "Différence avec la prédiction pour la participation"
                )
            else:

                data_element = (
                    data_line.loc[
                        [f"pvote{b}" for b in blocs] + ["votants"],
                        f"{year}_{t}_diff_agg",
                    ]
                    .to_frame()
                    .T
                )

                col_config = {
                    f"pvote{b}": f"Différence avec la prédiction du vote {trad[b]}"
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
            st.bar_chart(data=(data_element).T)


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
