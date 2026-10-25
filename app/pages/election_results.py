import streamlit as st
from asset.definitions import trad
from asset.features import CHAMPION_FEATURES_EXTENDED
from core.utils import (
    check_home_run,
    diff_show,
    present_results,
    show_feature_importance,
    show_shap_values,
    get_asset,
    make_slider,
)

check_home_run()

st.session_state["state"].selection_box(multiple_years=False)

results = get_asset(
    st.session_state["dh"].load_result,
    asset_name="results_synth",
    year=st.session_state["state"].year,
    election_type=st.session_state["state"].get_type(as_type="code"),
    trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
).set_index("index")

st.header(
    f"Résultat des {st.session_state['state'].get_type(as_type='verbose')} de {st.session_state['state'].year} ({st.session_state['state'].get_blocs(as_type='verbose')})"
)

present_results(
    results,
    year=st.session_state["state"].year,
    t=st.session_state["state"].get_type(as_type="code"),
    blocs=st.session_state["state"].get_blocs(as_type="code", order="political"),
    scale="global",
)

st.divider()

if int(st.session_state["state"].year) < 2026:
    st.header("Erreur du modèle")

    # Create trad adapté à bloc
    mean_error = results.loc[
        [
            f"pvote{b}"
            for b in st.session_state["state"].get_blocs(as_type="code", order="alpha")
        ],
        f"{st.session_state['state'].year}_{st.session_state['state'].get_type(as_type='code')}_diff",
    ].values.mean()
    std_error = results.loc[
        [
            f"pvote{b}"
            for b in st.session_state["state"].get_blocs(as_type="code", order="alpha")
        ],
        f"{st.session_state['state'].year}_{st.session_state['state'].get_type(as_type='code')}_diff",
    ].values.std()
    st.info(
        "Les erreurs calculées ici sont les erreurs des prédictions par commune, on s'interesse à leur moyenne et écart-type. \
            Toutes les erreurs sont données en différence de pourcentage et non en pourcentage."
    )

    st.metric("Erreur moyenne des prédictions", f"{round(mean_error, 2)}%")

    st.metric("Écart type de l'erreur de prédiction", f"{round(std_error, 2)}%")

    with st.expander("Erreurs moyenne des prédictions (sur l'ensemble des communes)"):
        diff_show(
            results,
            st.session_state["state"].get_blocs(as_type="code", order="alpha"),
            trad,
            "diff",
            "error",
            st.session_state["state"].year,
            st.session_state["state"].get_type(as_type="code"),
        )

    with st.expander(
        "Ecart type de l'erreur de prédiction (sur l'ensemble des communes)"
    ):
        diff_show(
            results,
            st.session_state["state"].get_blocs(as_type="code", order="alpha"),
            trad,
            "std",
            "ecart_type",
            st.session_state["state"].year,
            st.session_state["state"].get_type(as_type="code"),
        )

st.divider()

st.header(
    "Déterminants socio-économiques les plus importants dans le modèle de prédiction"
)
tab1, tab2 = st.columns([3, 1])

with tab1:
    nb_feat_fe = make_slider("nb_feat_fe")

with tab2:
    importance_type = st.selectbox(
        "Importance type",
        options=[
            "gain",
            "shap",
            "cover",
            "weight",
            "total_gain",
            "total_cover",
        ],
    )

show_feature_importance(
    get_asset(
        st.session_state["dh"].load_explain,
        asset_name="feature_importance",
        year=st.session_state["state"].year,
        election_type=st.session_state["state"].get_type(as_type="code"),
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
    ),
    st.session_state["state"].get_blocs(
        as_type="code", order="political", prefix="tau"
    ),
    nb_feat=nb_feat_fe,
    importance_type=importance_type,
)

st.divider()

sampled_communes_codes = (
    get_asset(st.session_state["dh"].load_communes_list)
    .sample(frac=0.01, random_state=42)["codecommune"]
    .tolist()
)

nb_feat_shap = make_slider("nb_feat_shap")

show_shap_values(
    shap_df=get_asset(
        st.session_state["dh"].load_explain,
        asset_name="shap_values",
        year=st.session_state["state"].year,
        election_type=st.session_state["state"].get_type(as_type="code"),
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
        filters=[("codecommune", "in", sampled_communes_codes)],
    ),
    data_sample=get_asset(
        st.session_state["dh"].load_data_sample,
        columns=CHAMPION_FEATURES_EXTENDED + ["codecommune"],
        filters=[
            ("annee", "==", int(st.session_state["state"].year)),
            (
                "election_type",
                "==",
                str(st.session_state["state"].get_type(as_type="code_full")),
            ),
            ("codecommune", "in", sampled_communes_codes),
        ],
    ),
    BLOCS=st.session_state["state"].get_blocs(as_type="code", order="political"),
    nb_feat=nb_feat_shap,
)
