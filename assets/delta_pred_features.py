BASE_FEATURES = [
    # inscrits
    "inscrits",

    # geo
    "lat",
    "long",
    "distanceparis",
    "distancelyon",
    "distancemarseille",
    "dep_num",

    # vbbm
    "F_raw_popcommunesvbbm/vbbm",
    "F_raw_popcommunesvbbm/vbbmpauvresriches",

    "F_rank_pibdepartements/pib",

    "F_rank_capitalimmobiliercommunes/capitalratio",
    "F_rank_etrangerscommunes/petranger",
    "F_raw_revcommunes/revratio",
    "F_rank_popdepartements/popcom5u",
    "F_rank_popdepartements/popcoma",

    # agesexcommunes
    "F_rank_agesexcommunes/age",
    "F_rank_agesexcommunes/agef",
    "F_rank_agesexcommunes/ageh",
    
    "F_rank_agesexcommunes/prop014",
    "F_rank_agesexcommunes/prop1539",
    "F_rank_agesexcommunes/prop4059",
    "F_rank_agesexcommunes/prop60p",

    "F_rank_agesexcommunes/propf",
    "F_rank_agesexcommunes/propf014",
    "F_rank_agesexcommunes/propf1539",
    "F_rank_agesexcommunes/propf4059",
    "F_rank_agesexcommunes/propf60p",

    # cspcommunes
    "F_rank_cspcommunes/pagri",
    "F_rank_cspcommunes/paica",
    "F_rank_cspcommunes/paind",
    "F_rank_cspcommunes/pcadr",
    "F_rank_cspcommunes/pcapi",
    "F_rank_cspcommunes/pempl",
    "F_rank_cspcommunes/pindp",
    "F_rank_cspcommunes/pint",
    "F_rank_cspcommunes/pouem",
    "F_rank_cspcommunes/pouvr",
    "F_rank_cspcommunes/ppint",
    "F_rank_diplomescommunes/pbac",
    "F_rank_diplomescommunes/psup",
    "F_rank_proprietairescommunes/ppropri",
    "F_raw_pibcommunes/pibratio",
    "F_raw_revcommunes/revratiofoy",
    "F_rank_rsacommunes/prsa",

    "F_raw_capitalimmobiliercommunes/prixm2ratio",
    "F_rank_capitalimmobiliercommunes/propappartement",
    "F_rank_naticommunes/pimmigre",
    "F_rank_naticommunes/petranger",
    "F_rank_naticommunes/pnatur",

    "F_raw_basesfiscalescommunes/baseimpotslocauxratio",
    "F_raw_basesfiscalescommunes/recetteimpotslocauxratio",
    "F_raw_basesfiscalescommunes/recetteratio",
    "F_raw_basesfiscalescommunes/tauximpotslocauxratio",
]


def make_features(metric="pct_change"):
    return [f.format(metric=metric) if "{metric}" in f else f for f in BASE_FEATURES]
