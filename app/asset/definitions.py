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