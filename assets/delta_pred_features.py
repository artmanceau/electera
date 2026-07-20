BASE_FEATURES = [
    # inscrits
    "inscrits",

    # geo
    "dep_num",

    # vbbm
    "F_raw_popcommunesvbbm/vbbm",
    "F_raw_popcommunesvbbm/vbbmpauvresriches",

    "F_raw_pibdepartements/pib",

    "F_raw_capitalimmobiliercommunes/capitalratio",

    "F_rank_etrangerscommunes/petranger",
    "F_pct_change_etrangerscommunes/petranger",

    "F_raw_revcommunes/revratio",
    "F_pct_change_revcommunes/revratio",

    "F_raw_popdepartements/popcom5u",
    "F_raw_popdepartements/popcoma",

    # agesexcommunes
    "F_rank_agesexcommunes/age",
    "F_pct_change_agesexcommunes/age",

    "F_raw_agesexcommunes/pop",
    "F_pct_change_agesexcommunes/pop",
    
    "F_rank_agesexcommunes/prop014",
    "F_rank_agesexcommunes/prop1539",
    "F_rank_agesexcommunes/prop4059",
    "F_rank_agesexcommunes/prop60p",

    "F_pct_change_agesexcommunes/prop014",
    "F_pct_change_agesexcommunes/prop1539",
    "F_pct_change_agesexcommunes/prop4059",
    "F_pct_change_agesexcommunes/prop60p",

    "F_rank_agesexcommunes/propf",
    "F_pct_change_agesexcommunes/propf",
   

    # cspcommunes
    "F_rank_cspcommunes/pagri",
    "F_rank_cspcommunes/paica",
    "F_rank_cspcommunes/paind",
    "F_rank_cspcommunes/pcadr",
    "F_rank_cspcommunes/pcapi",
    "F_rank_cspcommunes/pempl",
    "F_rank_cspcommunes/pindp",
    "F_rank_cspcommunes/pouem",
    "F_rank_cspcommunes/pouvr",
    "F_rank_cspcommunes/ppint",

    "F_pct_change_cspcommunes/pagri",
    "F_pct_change_cspcommunes/paica",
    "F_pct_change_cspcommunes/paind",
    "F_pct_change_cspcommunes/pcadr",
    "F_pct_change_cspcommunes/pcapi",
    "F_pct_change_cspcommunes/pempl",
    "F_pct_change_cspcommunes/pindp",
    "F_pct_change_cspcommunes/pouem",
    "F_pct_change_cspcommunes/pouvr",
    "F_pct_change_cspcommunes/ppint",

    "F_rank_diplomescommunes/pbac",
    "F_rank_diplomescommunes/psup",

    "F_raw_proprietairescommunes/ppropri",
    "F_pct_change_proprietairescommunes/ppropri",

    "F_raw_pibcommunes/pibratio",
    "F_raw_revcommunes/revratiofoy",
    "F_rank_rsacommunes/prsa",

    "F_raw_capitalimmobiliercommunes/prixm2ratio",
    "F_pct_change_capitalimmobiliercommunes/prixm2ratio",

    "F_rank_capitalimmobiliercommunes/propappartement",

    "F_rank_naticommunes/pimmigre",
    "F_rank_naticommunes/pnatur",
]


def make_features(metric="pct_change"):
    return [f.format(metric=metric) if "{metric}" in f else f for f in BASE_FEATURES]
