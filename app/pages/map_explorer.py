import pydeck as pdk
import s3fs
import streamlit as st
from asset.definitions import COMMUNES_MAP_PATH, DATA, RESULT_FULL_PATH
from core.data_handler import FileSystem
from core.utils import blocs, colors, trad

from src.components.data_processing.data_loader import DataLoader

VERSIONS = [
    ("Model prediction", "pred"),
    ("Actual result", "true"),
]


@st.cache_data
def load_geojson_page(path: str, _fs: object):
    return DataLoader.load_geojson(geo_data_path=path, fs=_fs)


if DATA in st.session_state:
    result_df = st.session_state[DATA][RESULT_FULL_PATH]
    communes_geojson = load_geojson_page(path=COMMUNES_MAP_PATH, _fs=fs.get_fs())
else:
    st.warning("Visit the home page!")
    st.stop()

result_df.index = result_df["codecommune"]
winner_true = result_df[[f"pvote{b}_true" for b in blocs]].idxmax(axis=1).reset_index()
winner_pred = result_df[[f"pvote{b}_pred" for b in blocs]].idxmax(axis=1).reset_index()

# Use blocs and colors for consistent color mapping
label_to_color = {f"pvote{bloc}": colors[i] for i, bloc in enumerate(blocs)}
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


# -------------------------
# Create both datasets
# -------------------------
pred_data = prepare_geojson_data("pred")
true_data = prepare_geojson_data("true")

# -------------------------
# Side by side maps
# -------------------------
st.subheader("Comparison des Cartes")

col1, col2 = st.columns(2)

# Define shared view state for both maps
view_state = pdk.ViewState(
    latitude=46.603354, longitude=1.888334, zoom=5.5, pitch=0, bearing=0
)

with col1:
    st.markdown("**Prédiction du Modèle**")

    # Model prediction layer
    pred_layer = pdk.Layer(
        "GeoJsonLayer",
        data=pred_data,
        opacity=0.7,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=False,
        get_fill_color="color",
        get_line_color=[51, 51, 51, 255],
        get_line_width=1,
        pickable=True,
        auto_highlight=True,
    )

    # Create prediction deck
    pred_deck = pdk.Deck(
        layers=[pred_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Code:</b> {code}<br/><b>Name:</b> {name}<br/><b>Predicted Winner:</b> {label}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        },
    )

    # Render prediction map
    st.pydeck_chart(pred_deck, use_container_width=True)

with col2:
    st.markdown("**Résultats Réels**")

    # True results layer
    true_layer = pdk.Layer(
        "GeoJsonLayer",
        data=true_data,
        opacity=0.7,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=False,
        get_fill_color="color",
        get_line_color=[51, 51, 51, 255],
        get_line_width=1,
        pickable=True,
        auto_highlight=True,
    )

    # Create true results deck
    true_deck = pdk.Deck(
        layers=[true_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Code:</b> {code}<br/><b>Name:</b> {name}<br/><b>Actual Winner:</b> {label}",
            "style": {"backgroundColor": "darkgreen", "color": "white"},
        },
    )

    # Render true results map
    st.pydeck_chart(true_deck, use_container_width=True)

# -------------------------
# Shared Legend
# -------------------------
st.markdown("---")
legend_html = "<div style='background: white; padding: 15px; border: 1px solid #ccc; border-radius: 4px; margin-top: 10px;'>"
legend_html += "<b>Légende</b><br>"
for i, bloc in enumerate(blocs):
    legend_html += f"<div style='display:flex;align-items:center;margin:5px 0;'><span style='display:inline-block;width:20px;height:20px;background:{colors[i]};border:1px solid #999;margin-right:8px;'></span><strong>Vote {trad[bloc]}</strong></div>"
legend_html += "</div>"
st.markdown(legend_html, unsafe_allow_html=True)

# -------------------------
# Optional: Add accuracy metrics
# -------------------------
st.markdown("---")
st.subheader("Performance")

# Calculate accuracy
pred_winners = [item["label"] for item in pred_data if item["label"] != "No data"]
true_winners = [item["label"] for item in true_data if item["label"] != "No data"]


accuracy = sum(1 for p, t in zip(pred_winners, true_winners) if p == t) / len(
    pred_winners
)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Précision Globale", f"{accuracy:.2%}")
with col2:
    st.metric("Vainqueur prédit", max(set(pred_winners), key=pred_winners.count))
with col3:
    st.metric("Vainqueur réel", max(set(true_winners), key=pred_winners.count))
