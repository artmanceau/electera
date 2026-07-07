import streamlit as st
from asset.definitions import trad
from core.utils import (
    check_home_run,
    diff_show,
    present_results,
    show_feature_importance,
    show_shap_values,
)


@st.cache_data
def load_results():
    st.session_state["data"].load_result(
        asset="results_synth",
        year=st.session_state["state"].year,
        election_type=st.session_state["state"].get_type(as_type="code"),
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
    )


@st.cache_data
def load_feature_importance():
    st.session_state["data"].load_explain(
        asset="feature_importance",
        year=st.session_state["state"].year,
        election_type=st.session_state["state"].get_type(as_type="code"),
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
    )


@st.cache_data
def load_shap_values(sampled_communes_codes):
    st.session_state["data"].load_explain(
        asset="shap_values",
        year=st.session_state["state"].year,
        election_type=st.session_state["state"].get_type(as_type="code"),
        trends=st.session_state["state"].get_blocs(as_type="code", order="alpha"),
        filters=[("codecommune", "in", sampled_communes_codes)]
    )


@st.cache_data
def sample_communes(sample_frac=None):
    st.session_state["data"].load_communes_list()

    communes = st.session_state["data"].container["communes_list"]

    if sample_frac is not None:
        sampled_communes = communes.sample(frac=sample_frac, random_state=42)
        sampled_commune_codes = sampled_communes['codecommune'].tolist()
    else:
        sampled_commune_codes = communes['codecommune'].tolist()
    return sampled_commune_codes


@st.cache_data
def load_data(features, sampled_communes_codes):
    # Define your filters
    filters = [
        ("annee", "==", int(st.session_state["state"].year)),
        ("election_type", "==", str(st.session_state["state"].get_type(as_type="code_full"))),
        ("codecommune", "in", sampled_communes_codes)
    ]
    st.session_state["data"].load_data_sample(
        columns=features,
        filters=filters,
        asset_name='data_sample_all'
    )


check_home_run()

st.session_state["state"].selection_box(multiple_years=False, clear_cache_on_rerun=True)


load_results()

results = st.session_state["data"].container["results_synth"].set_index("index")

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

if st.button("Compute feature importance"):
    st.session_state.show_feature_importance = True
    load_feature_importance()

if st.session_state.show_feature_importance:
    show_feature_importance(
        st.session_state["data"].container["feature_importance"],
        st.session_state["state"].get_blocs(as_type="code", order="political", prefix="tau"),
    )

st.divider()

if st.button("Compute shap values"):
    st.session_state.show_shap_values = True

    sampled_communes_codes = sample_communes(sample_frac=0.2)

    load_shap_values(sampled_communes_codes)

    features = set()
    for df in st.session_state["data"].container["shap_values"].values():
        features.update(df.columns)
    features.discard("base_value")

    load_data(features=features, sampled_communes_codes=sampled_communes_codes)

if st.session_state.show_shap_values:
    show_shap_values(
        shap_df=st.session_state["data"].container["shap_values"],
        data_sample=st.session_state["data"].container["data_sample_all"],
        BLOCS=st.session_state["state"].get_blocs(as_type="code", order="political"),
    )