import streamlit as st

st.set_page_config(page_title="Electera", layout="wide")
st.title("Electera")

pages = [
    st.Page("pages/home.py", title="Electera", icon="🗳️"),
    st.Page("pages/election_results.py", title="Résultat et prédictions", icon="🇫🇷"),
    st.Page("pages/commune_explorer.py", title="Communes", icon='🔍'),
    st.Page("pages/map_explorer.py", title="Cartes", icon="🌍"),
]

pg = st.navigation(pages, position="top")
pg.run()
