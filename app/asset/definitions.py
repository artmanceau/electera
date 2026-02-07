DATA = "data"

COMMUNES_MAP_PATH = "s3://arthurmanceau/election_modeling_uhcp/data/raw/geo_data/communes-version-simplifiee.geojson"
RESULT_FULL_PATH = "results_full"

colors = ["#bb1840", "#ffc0c0", "#FED700", "#0066cc", "#0D378A"]
blocs = ["G", "CG", "C", "CD", "D"]
trends = ["par"] + [f"vote{b}" for b in blocs]
trad = {
    "G": "à gauche",
    "CD": "pour le centre-droite",
    "C": "pour le centre",
    "D": "à droite",
    "CG": "pour le centre-gauche",
}
type_trad = {"pres": "présidentielles", "leg": "leglisatives"}


client_kwargs = 'https://'+'minio.lab.sspcloud.fr'


display_config_converter = {
    'type': {
        'pres': 'élections présidentielles',
        'leg': 'élections législatives',
        'ref': 'réferundum'
    },
    'political_division': {
        str(["voteTG", "voteTD", "par"]): 'division gauche / droite',
        str(["voteGCG", "voteDCD", 'voteC', "par"]): 'division gauche / droite / centre',
        str([["voteG", "voteD", "voteCG", "voteCD", 'voteC', "par"]]): "division en 5 blocs : gauche / centre gauche / centre / cente droite / droite"
    }
}


def convert(category, key):
    if category == 'political_division':
        key = str(key)
    return display_config_converter[category].get(key)


def reverse_convert(category, value):
    for k, v in display_config_converter[category].items():
        if v == value:
            if category == 'political_division':
                import ast
                return ast.literal_eval(k)
            return k
    return None