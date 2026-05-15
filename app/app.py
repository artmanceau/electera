import streamlit as st

# TODO: disjonction des années en fonction des elections
# Shap + feature importance (test)

st.set_page_config(page_title="Electera", layout="wide")
st.title("Electera")

pages = [
    st.Page("pages/home.py", title="Electera", icon="🗳️"),
    st.Page("pages/election_results.py", title="Résultat et prédictions", icon="🇫🇷"),
    st.Page("pages/commune_explorer.py", title="Communes", icon="🔍"),
    st.Page("pages/map_explorer.py", title="Cartes", icon="🌍"),
    st.Page("pages/back_testing.py", title="Back-testing", icon="📈"),
]

pg = st.navigation(pages, position="top")
pg.run()
