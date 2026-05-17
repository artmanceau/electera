from typing import Literal

import streamlit as st
from asset.definitions import convert, political_align, reverse_convert
from loguru import logger


class SessionHandler:
    """A native class to handle session state attributes and return them in pratical manner"""

    def __init__(self):
        self.year = None
        self.type = None
        self.blocs = None

        self.commune = None
        self.codecommune = None

        self.years = None

    def selection_box(self, multiple_years=False):
        """Selection method for year, type and blocs"""
        # User selections
        col1, col2, col3 = st.columns(3)
        with col1:
            self.type = st.selectbox(
                "Type d'élection",
                [
                    convert("type", el)
                    for el in st.session_state["config"].types_to_display
                ],
                index=0,
                on_change=st.cache_data.clear(),
            )
        with col2:
            if multiple_years:
                self.years = st.multiselect(
                    "Sélectionnez les années à analyser",
                    st.session_state["config"].years_to_display[reverse_convert("type", self.type)],
                    default=st.session_state["config"].years_to_display[reverse_convert("type", self.type)][
                        : min(3, len(st.session_state["config"].years_to_display))
                    ],
                    on_change=st.cache_data.clear(),
                )

            else:
                self.year = st.selectbox(
                    "Année électorale",
                    st.session_state["config"].years_to_display[reverse_convert("type", self.type)],
                    index=0,
                    on_change=st.cache_data.clear(),
                )
       
        with col3:
            self.blocs = st.selectbox(
                "Division politique",
                [
                    convert("political_division", el)
                    for el in st.session_state["config"].political_divisions_to_dislay
                ],
                index=0,
                on_change=st.cache_data.clear(),
            )
        logger.debug(f"State registered: {self.type} | {self.year} | {self.blocs}")

    def get_year(self):
        return self.year

    def get_years(self):
        return self.years

    def get_type(self, as_type: Literal["verbose", "code", "number"] = "code"):
        if as_type == "verbose":
            return self.type
        else:
            code = reverse_convert("type", self.type)
            if as_type == "code":
                return code
            elif as_type == "number":
                return 0 if code == "pres" else (1 if code == "leg" else 2)
            else:
                raise Exception("Type of return not configured.")

    def get_blocs(
        self,
        as_type: Literal["verbose", "code"] = "code",
        order: Literal["alpha", "political"] | None = None,
        prefix: Literal["p"] | None = None,
    ):
        if as_type == "verbose":
            bloc_return = self.blocs
        elif as_type == "code":
            bloc_return = reverse_convert("political_division", self.blocs)
        else:
            raise Exception("Type of return not configured.")

        if order == "alpha":
            bloc_return.sort()
            return bloc_return
        elif order == "political":
            bloc_return = political_align(bloc_return)

        if prefix:
            return [f"{prefix}{b}" for b in bloc_return]
        else:
            return bloc_return

    def commune_selector(self):
        communes_list = st.session_state["data"].container["communes_list"]

        # Requires cleaning but could be improved
        communes_list["nomcommune"] = (
            communes_list["nomcommune"]
            .str.replace("Ã", "É", regex=False)
            .str.replace("Ã", "Â", regex=False)
            .str.replace("Ã", "È", regex=False)
            .str.replace("Ã", "Ô", regex=False)
            .str.replace("Ã", "Ê", regex=False)
            .str.replace("Ã", "À", regex=False)
        )
        communes = communes_list["nomcommune"].drop_duplicates()

        self.commune = st.selectbox(
            "Selectionnez une commune", [""] + communes, on_change=st.cache_data.clear()
        )

        if len(communes_list[communes_list["nomcommune"] == self.commune]) > 1:
            arrondissements = communes_list[
                communes_list["nomcommune"] == self.commune
            ]["codecommune"]
            self.codecommune = st.selectbox(
                "Selectionnez une arrondissement",
                [""] + arrondissements,
                on_change=st.cache_data.clear(),
            )
        else:
            self.codecommune = communes_list[
                communes_list["nomcommune"] == self.commune
            ]["codecommune"].iloc[0]

        logger.debug(f"Commune selected: {self.commune} ({self.codecommune})")
