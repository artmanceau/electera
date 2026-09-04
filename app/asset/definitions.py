DATA = "data"

COMMUNES_MAP_PATH = "s3://arthurmanceau/election_modeling_uhcp/data/raw/geo_data/communes-version-simplifiee.geojson"
RESULT_FULL_PATH = "results_full"


colors_dict = {
    "G": "#bb1840",
    "CG": "#ffc0c0",
    "C": "#FED700",
    "CD": "#0066cc",
    "D": "#0D378A",
    "TG": "#ffc0c0",
    "TD": "#0066cc",
    "GCG": "#ffc0c0",
    "DCD": "#0066cc",
    "CGCCD": "#FED700",
}


def get_colors(blocs, colors_dict):
    """Returns a list in the same order as bloc, with the associated elements"""
    return [colors_dict.get(bloc, None) for bloc in blocs]


candidats_2022_mapping = {
    "G": [
        "Jean-Luc Mélenchon",
        "Fabien Roussel",
        "Philippe Poutou",
        "Nathalie Arthaud",
    ],
    "CGCCD": ["Anne Hidalgo", "Yannick Jadot", "Emmanuel Macron", "Valérie Pécresse", "Nicolas Dupont-Aignan"],
    "CG": ["Anne Hidalgo", "Yannick Jadot"],
    "GCG": [
        "Anne Hidalgo",
        "Yannick Jadot",
        "Jean-Luc Mélenchon",
        "Fabien Roussel",
        "Philippe Poutou",
        "Nathalie Arthaud",
    ],
    "TG": [
        "Anne Hidalgo",
        "Yannick Jadot",
        "Jean-Luc Mélenchon",
        "Fabien Roussel",
        "Philippe Poutou",
        "Nathalie Arthaud",
        "Motié des voix pour Emmanuel Macron",
    ],
    "C": [
        "Emmanuel Macron",
    ],
    "TD": [
        "Valérie Pécresse",
        "Nicolas Dupont-Aignan",
        "Marine Le Pen",
        "Éric Zemmour",
        "Motié des voix pour Emmanuel Macron",
    ],
    "CD": ["Valérie Pécresse", "Nicolas Dupont-Aignan"],
    "DCD": [
        "Valérie Pécresse",
        "Nicolas Dupont-Aignan",
        "Marine Le Pen",
        "Éric Zemmour",
    ],
    "D": ["Marine Le Pen", "Éric Zemmour"],
}

trad = {
    "TD": "à droite (tout les partis)",
    "TG": "à gauche (tout les partis)",
    "GCG": "à gauche et au centre-gauche",
    "DCD": "à droite et au centre-droit",
    "G": "à gauche",
    "CD": "pour le centre-droite",
    "C": "pour le centre",
    "D": "à droite",
    "CG": "pour le centre-gauche",
    "CGCCD": 'pour le bloc central (centre-gauche, centre et centre-droite)',
    "tauTD": "à gauche (tout les partis)",
    "tauTG": "à droite (tout les partis)",
    "tauGCG": "à gauche et au centre-gauche",
    "tauDCD": "à droite et au centre-droit",
    "tauG": "à gauche",
    "tauCD": "pour le centre-droite",
    "tauC": "pour le centre",
    "tauD": "à droite",
    "tauCG": "pour le centre-gauche",
    "tauCGCCD": 'pour le bloc central (centre-gauche, centre et centre-droite)',
    "par": "participation"
}


def political_align(blocs):
    if len(blocs) == 6:
        return ["G", "CG", "C", "CD", "D"]
    elif len(blocs) == 4:
        return ["CGCCD", "D", "G"]
    else:
        return ["TG", "TD"]


type_trad = {"pres": "présidentielles", "leg": "leglisatives"}


client_kwargs = "https://" + "minio.lab.sspcloud.fr"


display_config_converter = {
    "type": {
        "pres": "élections présidentielles",
        "leg": "élections législatives",
        "ref": "réferundum",
    },
    "political_division": {
        str(["TG", "TD", "par"]): "division gauche / droite",
        str(["GCG", "DCD", "C", "par"]): "division gauche / droite / centre",
        str(
            ["G", "D", "CG", "CD", "C", "par"]
        ): "division en 5 blocs : gauche / centre gauche / centre / cente droite / droite",
        str(["CGCCD", "D", "G", "par"]): "division extrême gauche / centre-gauche, centre et centre-droite / extrême droite"
    },
}


def convert(category, key):
    if category == "political_division":
        key = str(key)
    return display_config_converter[category].get(key)


def reverse_convert(category, value):
    for k, v in display_config_converter[category].items():
        if v == value:
            if category == "political_division":
                import ast

                return ast.literal_eval(k)
            return k
    return None
