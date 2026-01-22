import os
import re
from datetime import date
from io import StringIO

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

from src.components.data_processing.data_loader import DataLoader, DataUtils

# Method: scrapping wikiepdia which contains table with polling data
# Added value: consistent database (errors checking) + political trends
# Challenges: different formatting of the page

# Nice to have : source des sondages

YEARS = ["1988"]  # ["2022", '2007', '2012', '2017', '2022', '2027']

data_path = "s3://arthurmanceau/election_modeling_uhcp/polls/"

links = {
    "2027": "https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_l%27%C3%A9lection_pr%C3%A9sidentielle_fran%C3%A7aise_de_2027",
    "2022": "https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_l%27%C3%A9lection_pr%C3%A9sidentielle_fran%C3%A7aise_de_2022",
    "2017": "https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_l%27%C3%A9lection_pr%C3%A9sidentielle_fran%C3%A7aise_de_2017",
    "2012": "https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_l%27%C3%A9lection_pr%C3%A9sidentielle_fran%C3%A7aise_de_2012",
    "2007": "https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_l%27%C3%A9lection_pr%C3%A9sidentielle_fran%C3%A7aise_de_2007",
    "2002": "https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_l%27%C3%A9lection_pr%C3%A9sidentielle_fran%C3%A7aise_de_2002",
    "1995": "https://fr.wikipedia.org/wiki/%C3%89lection_pr%C3%A9sidentielle_fran%C3%A7aise_de_1995",
    "1988": "https://fr.wikipedia.org/wiki/%C3%89lection_pr%C3%A9sidentielle_fran%C3%A7aise_de_1988",
}

table_selector = {
    "2027": {
        "tour 1": {
            "start": "Année 2025",
            "end": "Année 2023",
        },
        "tour 2": {
            "element1": "Hypothèse Attal – Bardella",
            "element2": None,
        },
    },
    "2022": {
        "tour 1": {
            "start": "Sondages réalisés après la publication de la liste officielle des candidats",
            "end": "Années 2017-2020",
        },
        "tour 2": {
            "element1": "Sondages effectués après le premier tour",
            "element2": "Sondages effectués avant le premier tour",
        },
    },
    "2017": {
        "tour 1": {
            "start": "Sondages réalisés après la publication de la liste officielle des candidats",
            "end": "Sondages de 2013",
        },
        "tour 2": {
            "element1": "Intentions de vote",
            "element2": "Sondages entre Emmanuel Macron et Marine Le Pen",
        },
    },
    "2012": {
        "tour 1": {"start": "Avril 2012", "end": "Années 2007-2009"},
        "tour 2": {"element1": "2012", "element2": "2011"},
    },
    "2007": {
        "tour 1": {"start": "Avril", "end": "2005"},
        "tour 2": {"element1": "2007", "element2": "2006 (2)"},
    },
    "2002": {
        "tour 1": {"start": "Avril", "end": "1999"},
        "tour 2": {"element1": "Chirac - Le Pen", "element2": None},
    },
    "1995": {
        "tour 1": {"start": "En 1995", "end": "En 1995"},
        "tour 2": {"element1": "Balladur - Chirac", "element2": None},
    },
    "1988": {
        "tour 1": {
            "start": "Sondages concernant le premier tour (2)",
            "end": "Sondages concernant le premier tour (2)",
        },
        "tour 2": {"element1": "Mitterrand - Chirac", "element2": None},
    },
}


candidates_list = {
    "2027": [
        "Arthaud (LO)",
        "Poutou[a] (NPA)",
        "Mélenchon (LFI)",
        "Roussel (PCF)",
        "Tondelier (LÉ)",
        "Candidat PS / PP",
        "Candidat EPR",
        "Villepin (LFH)",
        "Candidat LR",
        "Dupont-Aignan (DLF)",
        "Candidat RN",
        "Zemmour[b] (REC)",
        "Autres",
        "Poutou[a] (NPA-A)",
        "Mélenchon[b] (LFI)",
        "Faure[b] (PS)",
        "Tondelier[b] (LÉ)",
        "Candidat ENS",
        "Wauquiez[b] (LR)",
        "Lassalle (RES)",
        "Le Pen[b] (RN)",
        "Zemmour (REC)",
        "Candidat PS/DVG",
        "Candidat EÉLV",
        "Wauquiez (LR)",
    ],
    "2022": [
        "Arthaud (LO)",
        "Poutou (NPA)",
        "Roussel (PCF)",
        "Mélenchon (LFI)",
        "Hidalgo (PS)",
        "Jadot (EÉLV)",
        "Macron (LREM)",
        "Pécresse (LR)",
        "Lassalle (RES)",
        "Dupont-Aignan (DLF)",
        "Le Pen (RN)",
        "Zemmour (REC)",
    ],
    "2017": [
        "Nathalie Arthaud (LO)",
        "Philippe Poutou (NPA)",
        "Jean-Luc Mélenchon (LFI)",
        "Benoît Hamon (PS)",
        "Emmanuel Macron (EM)",
        "Jean Lassalle (RES)",
        "François Fillon (LR)",
        "Nicolas Dupont-Aignan (DLF)",
        "François Asselineau (UPR)",
        "Marine Le Pen (FN)",
        "Jacques Cheminade (S&P)",
    ],
    "2012": [
        "Nathalie Arthaud (LO)",
        "Philippe Poutou (NPA)",
        "Jean-Luc Mélenchon (FG)",
        "François Hollande (PS)",
        "Eva Joly (EELV)",
        "François Bayrou (MoDem)",
        "Nicolas Sarkozy (UMP)",
        "Nicolas Dupont-Aignan (DLR)",
        "Marine Le Pen (FN)",
        "Jacques Cheminade (S&P)",
    ],
    "2007": [
        "Arlette Laguiller (LO)",
        "Olivier Besancenot (LCR)",
        "Marie-George Buffet (PCF)",
        "Gérard Schivardi (PT)",
        "Ségolène Royal (PS)",
        "José Bové (DVG)",
        "Dominique Voynet (LV)",
        "François Bayrou (UDF)",
        "Nicolas Sarkozy (UMP)",
        "Philippe de Villiers (MPF)",
        "Frédéric Nihous (CPNT)",
        "Jean-Marie Le Pen (FN)",
    ],
    "2002": [
        "Arlette Laguiller (LO)",
        "Daniel Gluckstein (POI)",
        "Olivier Besancenot (LCR)",
        "Robert Hue (PCF)",
        "Lionel Jospin (PS)",
        "Christiane Taubira (PRG)",
        "Noël Mamère (LV)",
        "Jean-Pierre Chevènement (MDC)",
        "Corinne Lepage (Cap21)",
        "François Bayrou (UDF)",
        "Christine Boutin (FRS)",
        "Jacques Chirac (RPR)",
        "Alain Madelin (DL)",
        "Jean Saint-Josse (CPNT)",
        "Jean-Marie Le Pen (FN)",
        "Bruno Mégret (MNR)",
    ],
    "1995": [
        "Arlette Laguiller (LO)",
        "Robert Hue (PCF)",
        "Lionel Jospin (PS)",
        "Dominique Voynet (Verts)",
        "Édouard Balladur (RPR)",
        "Jacques Chirac (RPR)",
        "Philippe de Villiers (MPF)",
        "Jean-Marie Le Pen (FN)",
        "Jacques Cheminade (FNS)",
    ],
    "1988": [
        "Arlette Laguiller (LO)",
        "André Lajoinie (PCF)",
        "Pierre Juquin (DVG)",
        "François Mitterrand (PS)",
        "Michel Rocard (PS)",
        "Antoine Waechter (LV)",
        "Raymond Barre (UDF)",
        "Valéry Giscard d'Estaing (UDF)",
        "François Léotard (UDF)",
        "Jacques Chirac (RPR)",
        "Jean-Marie Le Pen (FN)",
    ],
}

candidates_list_t2 = {
    "2027": None,
    "2022": ["Macron (LREM)", "Le Pen (RN)"],
    "2017": ["Macron (EM)", "Le Pen (FN)"],
    "2012": ["François Hollande (PS)", "Nicolas Sarkozy (UMP)"],
    "2007": ["Nicolas Sarkozy (UMP)", "Ségolène Royal (PS)"],
    "2002": ["Jacques Chirac (RPR)", "Jean-Marie Le Pen (FN)"],
    "1995": ["Édouard Balladur (RPR)", "Jacques Chirac (RPR)"],
    "1988": ["Jacques Chirac (RPR)", "François Mitterrand (PS)"],
}

blocs = {
    "2027": {
        "G": [
            "Arthaud (LO)",
            "Poutou[a] (NPA)",
            "Mélenchon (LFI)",
            "Roussel (PCF)",
            "Poutou[a] (NPA-A)",
            "Mélenchon[b] (LFI)",
        ],
        "CG": [
            "Tondelier (LÉ)",
            "Candidat PS / PP",
            "Faure[b] (PS)",
            "Tondelier[b] (LÉ)",
            "Candidat PS/DVG",
            "Candidat EÉLV",
        ],
        "C": ["Candidat EPR", "Candidat ENS"],
        "CD": ["Candidat LR", "Wauquiez[b] (LR)", "Wauquiez (LR)"],
        "D": [
            "Villepin (LFH)",
            "Dupont-Aignan (DLF)",
            "Candidat RN",
            "Zemmour[b] (REC)",
            "Le Pen[b] (RN)",
            "Zemmour (REC)",
        ],
    },
    "2022": {
        "G": ["Arthaud (LO)", "Poutou (NPA)", "Roussel (PCF)", "Mélenchon (LFI)"],
        "CG": [
            "Hidalgo (PS)",
            "Jadot (EÉLV)",
        ],
        "C": ["Macron (LREM)"],
        "CD": [
            "Pécresse (LR)",
        ],
        "D": ["Dupont-Aignan (DLF)", "Le Pen (RN)", "Zemmour (REC)"],
    },
    "2017": {
        "G": [
            "Nathalie Arthaud (LO)",
            "Philippe Poutou (NPA)",
            "Jean-Luc Mélenchon (LFI)",
        ],
        "CG": ["Benoît Hamon (PS)"],
        "C": ["Emmanuel Macron (EM)"],
        "CD": [
            "François Fillon (LR)",
        ],
        "D": [
            "Nicolas Dupont-Aignan (DLF)",
            "Marine Le Pen (FN)",
            "François Asselineau (UPR)",
        ],
    },
    "2012": {
        "G": [
            "Nathalie Arthaud (LO)",
            "Philippe Poutou (NPA)",
            "Jean-Luc Mélenchon (FG)",
        ],
        "CG": ["François Hollande (PS)", "Eva Joly (EELV)"],
        "C": ["François Bayrou (MoDem)"],
        "CD": ["Nicolas Sarkozy (UMP)"],
        "D": [
            "Nicolas Dupont-Aignan (DLR)",
            "Marine Le Pen (FN)",
        ],
    },
    "2007": {
        "G": [
            "Arlette Laguiller (LO)",
            "Olivier Besancenot (LCR)",
            "Marie-George Buffet (PCF)",
            "Gérard Schivardi (PT)",
        ],
        "CG": ["Ségolène Royal (PS)", "José Bové (DVG)", "Dominique Voynet (LV)"],
        "C": ["François Bayrou (UDF)"],
        "CD": ["Nicolas Sarkozy (UMP)"],
        "D": ["Philippe de Villiers (MPF)", "Jean-Marie Le Pen (FN)"],
    },
    "2002": {
        "G": [
            "Arlette Laguiller (LO)",
            "Daniel Gluckstein (POI)",
            "Olivier Besancenot (LCR)",
            "Robert Hue (PCF)",
        ],
        "CG": [
            "Lionel Jospin (PS)",
            "Christiane Taubira (PRG)",
            "Noël Mamère (LV)",
            "Jean-Pierre Chevènement (MDC)",
            "Corinne Lepage (Cap21)",
        ],
        "C": ["François Bayrou (UDF)"],
        "CD": ["Christine Boutin (FRS)", "Jacques Chirac (RPR)", "Alain Madelin (DL)"],
        "D": [
            "Jean Saint-Josse (CPNT)",
            "Jean-Marie Le Pen (FN)",
            "Bruno Mégret (MNR)",
        ],
    },
    "1995": {
        "G": ["Arlette Laguiller (LO)", "Robert Hue (PCF)"],
        "CG": [
            "Lionel Jospin (PS)",
            "Dominique Voynet (Verts)",
        ],
        "C": ["Édouard Balladur (RPR)"],
        "CD": ["Jacques Chirac (RPR)"],
        "D": [
            "Philippe de Villiers (MPF)",
            "Jean-Marie Le Pen (FN)",
        ],
    },
    "1988": {
        "G": ["Arlette Laguiller (LO)", "André Lajoinie (PCF)"],
        "CG": [
            "Pierre Juquin (DVG)",
            "François Mitterrand (PS)",
            "Michel Rocard (PS)",
            "Antoine Waechter (LV)",
        ],
        "C": ["Raymond Barre (UDF)"],
        "CD": [
            "Valéry Giscard d'Estaing (UDF)",
            "François Léotard (UDF)",
            "Jacques Chirac (RPR)",
        ],
        "D": ["Jean-Marie Le Pen (FN)"],
    },
}


blocs_level_1 = ["G", "CG", "CD", "C", "D"]
blocs_level_2 = ["GCG", "DCD", "C_"]
blocs_level_3 = ["TG", "TD"]


class FetcherUtils:

    FRENCH_MONTHS = {
        "janvier": 1,
        "janv": 1,
        "février": 2,
        "fevrier": 2,
        "fev": 2,
        "fév": 2,
        "mars": 3,
        "avril": 4,
        "avr": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7,
        "août": 8,
        "aout": 8,
        "septembre": 9,
        "octobre": 10,
        "novembre": 11,
        "décembre": 12,
        "decembre": 12,
    }

    SEP = r"\s*[-–—]\s*"

    @staticmethod
    def _norm(s: str) -> str:
        s = s.replace("\xa0", " ").replace("\u202f", " ")
        s = s.replace(".", "")
        return " ".join(s.lower().strip().split())

    @staticmethod
    def _int_day(tok: str) -> int:
        tok = re.sub(r"[^\d]", "", tok)  # remove "er" and any nondigits
        return int(tok)

    @staticmethod
    def reference_year_for_month(month: int, year: int) -> int:
        # previous heuristic: months < May -> year 2022 else 2021
        return year if month < 5 else year - 1

    @staticmethod
    def parse_french_date_range(
        s: str, default_year: int | None = None
    ) -> tuple[date, date]:
        """
        Parse French date strings and ranges returning (start_date, end_date) as datetime.date.
        Handles:
        - single:     "8 avril" or "8 avril 2022" or "1er avril"
        - same-month: "6-8 avril" or "6 - 8 avril 2022"
        - cross-month:"31 mars - 4 avril" or "31 mars 2021 - 4 avril 2022"
        If an explicit year is present in the string it is used. If not, the function
        uses `default_year` if provided, otherwise falls back to reference_year_for_month().
        """
        if s is None:
            raise ValueError("Input is None")
        s0 = FetcherUtils._norm(str(s))

        # cross-month range: "31 mars 2021 - 4 avril 2022" (years optional per side)
        m = re.match(
            rf"^(\d{{1,2}}(?:er)?)\s+([a-zéèêûàôùç]+)(?:\s+(\d{{4}}))?\s*{FetcherUtils.SEP}\s*(\d{{1,2}}(?:er)?)\s+([a-zéèêûàôùç]+)(?:\s+(\d{{4}}))?$",
            s0,
        )
        if m:
            d1, mon1, y1_str, d2, mon2, y2_str = m.groups()
            m1 = FetcherUtils.FRENCH_MONTHS.get(mon1)
            m2 = FetcherUtils.FRENCH_MONTHS.get(mon2)
            if not m1 or not m2:
                raise ValueError(f"Unknown month: {mon1} or {mon2}")
            y1 = (
                int(y1_str)
                if y1_str
                else (
                    default_year
                    if default_year is not None
                    else FetcherUtils.reference_year_for_month(m1)
                )
            )
            y2 = (
                int(y2_str)
                if y2_str
                else (
                    default_year
                    if default_year is not None
                    else FetcherUtils.reference_year_for_month(m2)
                )
            )
            # if no explicit years and month2 < month1 assume range crosses year boundary
            if (not y1_str and not y2_str) and (m2 < m1):
                # assume year2 = year1 + 1
                y2 = y1 + 1
            start = date(y1, m1, FetcherUtils._int_day(d1))
            end = date(y2, m2, FetcherUtils._int_day(d2))
            return end

        # same-month range: "6-8 avril 2022" or "6-8 avril"
        m = re.match(
            r"^(\d{{1,2}}(?:er)?)\s*[-–—]\s*(\d{{1,2}}(?:er)?)\s+([a-zéèêûàôùç]+)(?:\s+(\d{{4}}))?$",
            s0,
        )
        if m:
            d1, d2, mon, y_str = m.groups()
            mnum = FetcherUtils.FRENCH_MONTHS.get(mon)
            if not mnum:
                raise ValueError(f"Unknown month: {mon}")
            y = (
                int(y_str)
                if y_str
                else (
                    default_year
                    if default_year is not None
                    else FetcherUtils.reference_year_for_month(mnum)
                )
            )
            start = date(y, mnum, FetcherUtils._int_day(d1))
            end = date(y, mnum, FetcherUtils._int_day(d2))
            if end < start:
                # assume end in next month/year
                em = mnum + 1
                ey = y
                if em > 12:
                    em = 1
                    ey += 1
                end = date(ey, em, FetcherUtils._int_day(d2))
            return end

        # single date: "8 avril 2022" or "1er avril"
        m = re.match(r"^(\d{{1,2}}(?:er)?)\s+([a-zéèêûàôùç]+)(?:\s+(\d{{4}}))?$", s0)
        if m:
            d, mon, y_str = m.groups()
            mnum = FetcherUtils.FRENCH_MONTHS.get(mon)
            if not mnum:
                raise ValueError(f"Unknown month: {mon}")
            y = (
                int(y_str)
                if y_str
                else (
                    default_year
                    if default_year is not None
                    else FetcherUtils.reference_year_for_month(mnum)
                )
            )
            dt = date(y, mnum, FetcherUtils._int_day(d))
            return dt

        logger.warning(f"Unrecognized date format: {s}")
        return pd.NaT

    @staticmethod
    def clean_numeric_percent(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.replace("\u00A0", "", regex=False).str.strip()
        is_lt = s.str.startswith("<", na=False)
        # pct_mask = s.str.contains("%", na=False)

        # remove % and leading '<', normalize decimal comma
        t = s.str.replace("%", "", regex=False)
        t = t.str.replace(r"^[<\s]+", "", regex=True)
        t = t.str.replace(",", ".", regex=False)
        t = t.str.replace(r"[^\d\.\-]", "", regex=True)

        num = pd.to_numeric(t, errors="coerce")

        # wherever original value began with '<', divide that numeric value by 2
        num = num.where(~is_lt, num / 2.0)

        return num

    @staticmethod
    def clean_number(s):
        s = re.sub(r"\s+", " ", s.replace("\xa0", " "))
        s = re.sub(r"\[.*?\]", "", s)
        s = re.sub(r"[^\d\.\-]", "", s)
        try:
            x = int(s)
            return x
        except:
            logger.warning(f"Could not conver to valid number: {s}")
            return np.nan


class PollFetcher:

    def __init__(self):
        self.headers = {"User-Agent": "PollFetcher/1.0 (contact@example.com)"}
        self.events = []

    def fetch_page(self, url):
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        html = r.text
        tables = pd.read_html(StringIO(html))
        soup_tables = BeautifulSoup(html, "lxml").find_all("table")
        return tables, soup_tables

    def parse_page(self, tables, soup_tables):
        table_dict = {}
        for idx, (df, tag) in enumerate(zip(tables, soup_tables), start=1):
            # titre préféré : <caption>
            caption = tag.find("caption")
            if caption and caption.get_text(strip=True):
                title = caption.get_text(" ", strip=True)
            else:
                # sinon chercher le titre de section précédent (h1..h6)
                prev = tag.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
                title = prev.get_text(" ", strip=True) if prev else f"table_{idx}"
            # nettoyage simple
            title = re.sub(r"\s+", " ", title).strip()
            # garantir clé unique
            key = title
            i = 1
            while key in table_dict:
                i += 1
                key = f"{title} ({i})"

            if getattr(df.columns, "nlevels", 1) > 1:
                df.columns = df.columns.get_level_values(1)

            df = df.dropna(how="all")

            df = df.copy()
            if "Date" in df.columns:
                df.rename(columns={"Date": "Dates"}, inplace=True)

            mask_rows = df.apply(lambda r: r.dropna().nunique() == 1, axis=1)
            self.events += (
                df[mask_rows].apply(lambda r: r.dropna().iat[0], axis=1).values
            ).tolist()
            df = df[~mask_rows].reset_index(drop=True)

            table_dict[key] = df.astype(str)
        return table_dict

    def _formate(self, X, year):
        logger.debug("Reformating...")
        X = X.copy(deep=True)
        X[["Sondeur", "Dates"]] = X[["Sondeur", "Dates"]].astype(str)
        X.loc[X["Sondeur"] == "Harris Interractive", "Sondeur"] = "Harris Interactive"
        X.loc[X["Sondeur"] == "Cluster 17", "Sondeur"] = "Cluster17"
        X.loc[X["Sondeur"] == "Opinionway", "Sondeur"] = "OpinionWay"
        X.loc[:, "Dates_pd"] = X.loc[:, "Dates"].apply(
            lambda s: FetcherUtils.parse_french_date_range(s, default_year=int(year))
        )
        if "Échantillon" in X.columns:
            X.loc[0, "Échantillon"] = FetcherUtils.clean_number(X.loc[0, "Échantillon"])
            X["Échantillon"] = (
                X["Échantillon"].astype(str).str.replace("\u00A0", "", regex=False)
            )
        for col in X.columns:
            if col not in ["Sondeur", "Dates", "Échantillon", "Dates_pd"]:
                X[col] = FetcherUtils.clean_numeric_percent(X[col])
        return X

    def _restrict_candidates(self, X, year):
        logger.debug("Restricting the candidate list to the actual candidates")
        X = X.copy(deep=True)
        candidates = candidates_list[year]
        if "Abstention" in X.columns:
            X = X.drop(columns="Abstention")
        if "Indécis (échantillon)" in X.columns:
            X = X.drop(columns="Indécis (échantillon)")
        if "Indécis" in X.columns:
            X = X.drop(columns="Indécis")
        not_candidate_cols = list(
            set(X.columns)
            - set(
                candidates + ["Dates", "Dates_pd", "Échantillon", "Sondeur", "Autres"]
            )
        )
        X["Autres"] = X[not_candidate_cols].sum(axis=1)
        X.drop(columns=[c for c in not_candidate_cols if c in X.columns], inplace=True)

        mask = np.abs(X[candidates + ["Autres"]].sum(axis=1) - 100) > 10
        X.drop(index=X[mask].index, inplace=True)
        logger.warning(
            f"Dropping {mask.astype(int).sum()} because of inconsistent results"
        )

        assert np.abs(X[candidates + ["Autres"]].sum(axis=1).mean() - 100.0) < 2
        return X

    def _add_political_trend(self, X, year):
        logger.debug("Adding political groups...")
        bloc = blocs[year]
        X = X.copy(deep=True)
        for b, canditates in bloc.items():
            X[b] = X[canditates].sum(axis=1)

        X["GCG"] = X["G"] + X["CG"] / 2
        X["C_"] = X["C"] + +X["CG"] / 2 + +X["CD"] / 2
        X["DCD"] = X["D"] + X["CD"] / 2
        X["TG"] = X["G"] + X["CG"] + X["C"] / 2
        X["TD"] = X["D"] + X["CD"] + X["C"] / 2
        assert X[blocs_level_1].sum(axis=1).mean() > 0.95
        assert X[blocs_level_2].sum(axis=1).mean() > 0.95
        assert X[blocs_level_3].sum(axis=1).mean() > 0.95
        return X

    def _note_col_handling(self, X):
        X = X.copy(deep=True)
        X = X.loc[:, ~X.columns.str.startswith("Unnamed")]
        # Reconcile columns with almosth the same name
        n5_cols = X.columns[X.columns.str.contains(r"\[N 5\]")]
        for col in n5_cols:
            matching_col = re.sub(r"\s*\[N\s*5\]\s*", " ", str(col))
            print(matching_col in X.columns)
            idx = X[col].dropna().index
            X.loc[idx, matching_col] = X.loc[idx, col].values
        return X

    def concat_t2(self, year, table_dict):
        if table_selector[year]["tour 2"]["element2"] is not None:
            return pd.concat(
                [
                    table_dict[table_selector[year]["tour 2"]["element1"]],
                    table_dict[table_selector[year]["tour 2"]["element2"]],
                ],
                ignore_index=True,
            )
        else:
            return table_dict[table_selector[year]["tour 2"]["element1"]]

    def concat_t1(self, year, table_dict):
        X = pd.DataFrame()
        started = False
        for key, df in table_dict.items():
            if key != table_selector[year]["tour 1"]["start"] and (not started):
                continue
            elif key == table_selector[year]["tour 1"]["start"]:
                started = True

            X = X.loc[:, ~X.columns.duplicated()]
            df = df.loc[:, ~df.columns.duplicated()]
            X = pd.concat([X, df], ignore_index=True)

            if key == table_selector[year]["tour 1"]["end"]:
                break
        return X

    def get_polls(self, year):
        """Method to get polling data for a given year"""
        logger.info(f"Getting poll data for year: {year}")

        url = links[year]
        tables, soup_tables = self.fetch_page(url)
        table_dict = self.parse_page(tables, soup_tables)

        # Instantiate the datasets
        poll_dataset, poll_dataset_t2 = self.concat_t1(
            year, table_dict
        ), self.concat_t2(year, table_dict)

        # N5
        poll_dataset = self._note_col_handling(poll_dataset)

        # Sondeur and Date
        poll_dataset = self._formate(poll_dataset, year)
        poll_dataset_t2 = self._formate(poll_dataset_t2, year)

        # Restrict candidates
        poll_dataset = self._restrict_candidates(poll_dataset, year)

        # Add political blocs
        poll_dataset = self._add_political_trend(poll_dataset, year)

        return poll_dataset, poll_dataset_t2

    def save_s3(self, datasets, year):
        logger.debug("Saving to S3")
        t1, t2 = datasets

        if not DataUtils._detect_s3(data_path):
            os.makedirs(f"{year}/", exist_ok=True)
            os.makedirs(f"{year}/", exist_ok=True)

        DataLoader.write_dataset(
            t1,
            data_path + f"{year}/polls_t1.parquet",
        )
        DataLoader.write_dataset(
            t2,
            data_path + f"{year}/polls_t2.parquet",
        )


def main():
    fetcher = PollFetcher()
    for year in YEARS:
        datasets = fetcher.get_polls(year)
        fetcher.save_s3(datasets, year)


if __name__ == "__main__":
    trainer = main()
