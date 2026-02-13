import pydeck as pdk
import streamlit as st
from asset.definitions import COMMUNES_MAP_PATH, convert, reverse_convert
from core.data_handler import get_fs
from core.utils import blocs, colors, trad, check_home_run
from src.components.data_processing.data_loader import DataLoader



@st.cache_data
def load_geojson_page(path: str, _fs: object):
    return DataLoader.load_geojson(geo_data_path=path, fs=_fs)


check_home_run()

col1, col2, col3 = st.columns(3)
with col1:
    YEAR = st.selectbox(
        "Année électorale", st.session_state["config"].years_to_display, index=0
    )
with col2:
    t = st.selectbox(
        "Type d'élection", [convert('type', el) for el in st.session_state["config"].types_to_display], index=0
    )
with col3:
    b = st.selectbox(
        "Division politique", [convert('political_division', el) for el in st.session_state['config'].political_divisions_to_dislay], index=0
    )

TYPE, BLOCS = reverse_convert('type', t), reverse_convert('political_division', b)
current_blocs = [bloc.replace('vote', '') for bloc in BLOCS if bloc != 'par']

st.header("Carte des résultats électoraux")

# Load data
with st.spinner("Chargement des données..."):
    st.session_state['data'].load_result(asset="results_full", year=YEAR, election_type=TYPE, trends=BLOCS, asset_name='results_full')
    results = st.session_state["data"].container['results_full']

    communes_geojson = load_geojson_page(COMMUNES_MAP_PATH, get_fs().fs)


# Prepare results data
results_indexed = results.copy()
results_indexed.index = results_indexed["codecommune"]
winner_true = results_indexed[[f"pvote{b}_true" for b in current_blocs]].idxmax(axis=1).reset_index()
winner_pred = results_indexed[[f"pvote{b}_pred" for b in current_blocs]].idxmax(axis=1).reset_index()

# Use current_blocs and colors for consistent color mapping
label_to_color = {f"pvote{bloc}": colors[i] for i, bloc in enumerate(current_blocs)}
label_to_color["No data"] = "#dee2e6"


@st.cache_data
def _prepare_geojson_data_core(version, winner_lookup, color_cache):
    """Core data preparation without UI elements"""
    features = communes_geojson["features"]
    features_data = []

    for feature in features:
        props = feature.get("properties", {})
        code = props.get("code")

        # Fast dictionary lookup
        winner_val = winner_lookup.get(str(code)) if code else None
        if winner_val:
            try:
                label = winner_val.split("_")[0]
            except (AttributeError, IndexError):
                label = "No data"
        else:
            label = "No data"

        # Fast color lookup
        color_rgb = color_cache.get(label, color_cache["No data"])

        features_data.append(
            {
                "geometry": feature["geometry"],
                "properties": props,
                "code": code,
                "label": label,
                "color": color_rgb,
                "name": props.get("name", ""),
            }
        )

    return features_data


@st.cache_data
def prepare_geojson_data(version):
    """Main function with progress bar"""

    # Pre-compute color mapping
    color_cache = {}
    for label, hex_color in label_to_color.items():
        color_cache[label] = [
            int(hex_color[1:3], 16),
            int(hex_color[3:5], 16),
            int(hex_color[5:7], 16),
            180,
        ]
    color_cache["No data"] = [222, 226, 230, 180]  # #dee2e6

    # Prepare winner lookup
    if version == "true":
        winner_df = winner_true.copy()
    else:
        winner_df = winner_pred.copy()

    winner_df["codecommune"] = winner_df["codecommune"].astype(str)
    winner_lookup = dict(zip(winner_df["codecommune"], winner_df[0]))

    # Show progress only on first load
    if f"geojson_data_{version}" not in st.session_state:
        with st.spinner(f"Processing {version} data..."):
            features_data = _prepare_geojson_data_core(
                version, winner_lookup, color_cache
            )
            st.session_state[f"geojson_data_{version}"] = features_data

    return st.session_state[f"geojson_data_{version}"]

pred_data = prepare_geojson_data("pred")
true_data = prepare_geojson_data("true")

st.subheader("Visualisation des Cartes")

view_state = pdk.ViewState(
    latitude=46.603354, longitude=1.888334, zoom=5.5, pitch=0, bearing=0
)

col1, col2 = st.columns(2)

with col1:
        st.markdown("**Prédiction du Modèle**")

        # Model prediction layer
        pred_layer = pdk.Layer(
            "GeoJsonLayer",
            data=pred_data,
            opacity=0.8,
            stroked=True,
            filled=True,
            extruded=False,
            wireframe=False,
            get_fill_color="color",
            get_line_color=[51, 51, 51, 255],
            get_line_width=0.5,
            pickable=True,
            auto_highlight=True,
        )

        # Create prediction deck
        pred_deck = pdk.Deck(
            layers=[pred_layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>Commune:</b> {name}<br/><b>Code:</b> {code}<br/><b>Gagnant prédit:</b> {label}",
                "style": {"backgroundColor": "steelblue", "color": "white", "fontSize": "14px", "padding": "10px"},
            },
        )

        # Render prediction map
        st.pydeck_chart(pred_deck, use_container_width=True, height=500)

with col2:
        st.markdown("**Résultats Réels**")

        # True results layer
        true_layer = pdk.Layer(
            "GeoJsonLayer",
            data=true_data,
            opacity=0.8,
            stroked=True,
            filled=True,
            extruded=False,
            wireframe=False,
            get_fill_color="color",
            get_line_color=[51, 51, 51, 255],
            get_line_width=0.5,
            pickable=True,
            auto_highlight=True,
        )

        # Create true results deck
        true_deck = pdk.Deck(
            layers=[true_layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>Commune:</b> {name}<br/><b>Code:</b> {code}<br/><b>Gagnant réel:</b> {label}",
                "style": {"backgroundColor": "darkgreen", "color": "white", "fontSize": "14px", "padding": "10px"},
            },
        )

        # Render true results map
        st.pydeck_chart(true_deck, use_container_width=True, height=500)


st.divider()

legend_html = "<div style='background: white; padding: 15px; border: 1px solid #ccc; border-radius: 4px; margin-top: 10px;'>"
legend_html += "<b>Légende</b><br>"
for i, bloc in enumerate(current_blocs):
    bloc_label = trad.get(bloc, bloc)
    legend_html += f"<div style='display:flex;align-items:center;margin:5px 0;'><span style='display:inline-block;width:20px;height:20px;background:{colors[i]};border:1px solid #999;margin-right:8px;'></span><strong>Vote {bloc_label}</strong></div>"
legend_html += "</div>"
st.markdown(legend_html, unsafe_allow_html=True)

st.divider()
st.subheader("Métriques de Performance")

# Calculate accuracy
pred_winners = [item["label"] for item in pred_data if item["label"] != "No data"]
true_winners = [item["label"] for item in true_data if item["label"] != "No data"]

if len(pred_winners) > 0 and len(true_winners) > 0:
    accuracy = sum(1 for p, t in zip(pred_winners, true_winners) if p == t) / len(pred_winners)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Précision Globale", f"{accuracy:.2%}")
    with col2:
        most_pred = max(set(pred_winners), key=pred_winners.count)
        st.metric("Gagnant le plus prédit", most_pred.replace("pvote", ""))
    with col3:
        most_true = max(set(true_winners), key=true_winners.count)
        st.metric("Gagnant le plus fréquent", most_true.replace("pvote", ""))
