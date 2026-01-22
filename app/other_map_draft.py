import json
import re

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from asset.definitions import COMMUNES_MAP_PATH, DATA, RESULT_FULL_PATH
from tqdm import tqdm

from src.components.streamlit_utils.utils import (
    blocs,
    colors,
    present_results,
    show_shap_values,
    trends,
)

VERSIONS = [
    ("Model prediction", "pred"),
    ("Actual result", "true"),
]


@st.cache_data
def load_geojson(path: str):
    with open(path) as f:
        return json.load(f)


if DATA in st.session_state:
    result_df = st.session_state[DATA][RESULT_FULL_PATH]
    communes_geojson = load_geojson(COMMUNES_MAP_PATH)
else:
    st.warning("Visit the home page!")
    st.stop()


result_df.index = result_df["codecommune"]
winner_true = result_df[[f"pvote{b}_true" for b in blocs]].idxmax(axis=1).reset_index()
winner_pred = result_df[[f"pvote{b}_pred" for b in blocs]].idxmax(axis=1).reset_index()


# -------------------------
# Sidebar controls
# -------------------------
version_name, version = st.sidebar.radio(
    "Layer to view", VERSIONS, index=0, format_func=lambda x: x[0]
)


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
    color_cache["No data"] = [204, 204, 204, 180]

    # Prepare winner lookup
    if version == "true":
        winner_df = winner_true.copy()
    else:
        winner_df = winner_pred.copy()

    winner_df["codecommune"] = winner_df["codecommune"].astype(str)
    winner_lookup = dict(zip(winner_df["codecommune"], winner_df[0]))

    # Show progress
    with st.spinner("Processing commune data..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # For very large datasets, you might want to process in chunks
        # but for caching, we'll do it all at once
        status_text.text("Preparing geographical data...")
        progress_bar.progress(0.5)

        # Call the cached core function
        features_data = _prepare_geojson_data_core(version, winner_lookup, color_cache)

        progress_bar.progress(1.0)
        status_text.text(f"Completed! Processed {len(features_data)} communes.")

        # Clean up after a short delay
        import time

        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

    return features_data


st.subheader("Carte")

# Create the pydeck layer
geojson_layer = pdk.Layer(
    "GeoJsonLayer",
    data=prepare_geojson_data(version),
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

# Define the initial view state
view_state = pdk.ViewState(
    latitude=46.603354, longitude=1.888334, zoom=5.5, pitch=0, bearing=0
)

# Create the deck
deck = pdk.Deck(
    layers=[geojson_layer],
    initial_view_state=view_state,
    tooltip={
        "html": "<b>Code:</b> {code}<br/><b>Name:</b> {name}<br/><b>Winner:</b> {label}",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    },
)

# Render the map
map_data = st.pydeck_chart(deck, use_container_width=True)

# -------------------------
# Legend
# -------------------------
legend_html = "<div style='background: white; padding: 10px; border: 1px solid #ccc; border-radius: 4px; margin-top: 10px;'>"
legend_html += f"<b>Legend — {version}</b><br>"
for i, bloc in enumerate(blocs):
    legend_html += f"<div style='display:flex;align-items:center;margin:3px 0;'><span style='display:inline-block;width:16px;height:16px;background:{colors[i]};border:1px solid #999;margin-right:6px;'></span>{bloc}</div>"
legend_html += "</div>"
st.markdown(legend_html, unsafe_allow_html=True)
