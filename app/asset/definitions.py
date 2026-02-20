DATA = "data"

COMMUNES_MAP_PATH = "s3://arthurmanceau/election_modeling_uhcp/data/raw/geo_data/communes-version-simplifiee.geojson"
RESULT_FULL_PATH = "results_full"


colors_dict = {
    "voteG": "#bb1840",
    "voteCG": "#ffc0c0",
    "voteC": "#FED700",
    "voteCD": "#0066cc",
    "voteD": "#0D378A",
    "voteTG": "#ffc0c0",
    "voteTD": "#0066cc",
    "voteGCG": "#ffc0c0",
    "voteDCD": "#0066cc",
}


def get_colors(blocs, colors_dict):
    """Returns a list in the same order as bloc, with the associated elements"""
    return [colors_dict.get(bloc, None) for bloc in blocs]


trad = {
    "voteTD": "à gauche (tout les partis)",
    "voteTG": "à droite (tout les partis)",
    "voteGCG": "à gauche et au centre-gauche",
    "voteDCD": "à droite et au centre-droit",
    "voteG": "à gauche",
    "voteCD": "pour le centre-droite",
    "voteC": "pour le centre",
    "voteD": "à droite",
    "voteCG": "pour le centre-gauche",
    "par": "participation",
}


def political_align(blocs):
    if len(blocs) == 6:
        return ["voteG", "voteCG", "voteC", "voteCD", "voteD"]
    elif len(blocs) == 4:
        return ["voteGCG", "voteC", "voteDCD"]
    else:
        return ["voteTG", "voteTD"]


type_trad = {"pres": "présidentielles", "leg": "leglisatives"}


client_kwargs = "https://" + "minio.lab.sspcloud.fr"


display_config_converter = {
    "type": {
        "pres": "élections présidentielles",
        "leg": "élections législatives",
        "ref": "réferundum",
    },
    "political_division": {
        str(["voteTG", "voteTD", "par"]): "division gauche / droite",
        str(
            ["voteGCG", "voteDCD", "voteC", "par"]
        ): "division gauche / droite / centre",
        str(
            ["voteG", "voteD", "voteCG", "voteCD", "voteC", "par"]
        ): "division en 5 blocs : gauche / centre gauche / centre / cente droite / droite",
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
