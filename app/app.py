import streamlit as st

st.set_page_config(page_title="Electera", layout="wide")
st.title("Electera")

pages = [
    st.Page("pages/home.py", title="Electera"),
    st.Page("pages/election_results.py", title="Résultat et prédictions"),
    st.Page("pages/commune_explorer.py", title="Communes"),
    st.Page("pages/map_explorer.py", title="Cartes"),
    st.Page("pages/time_series.py", title="Perspective temporelle"),
    st.Page("pages/about.py", title="A propos"),
]

pg = st.navigation(pages, position="top")
pg.run()
