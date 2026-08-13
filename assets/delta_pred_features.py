BASE_FEATURES = [
    # inscrits
    "inscrits",
    # geo
    "dep_num",
    "lat",
    "long",
    "distanceparis",
    # vbbm
    "F_raw_popcommunesvbbm/vbbmpauvresriches",
    # PIB
    "F_raw_pibdepartements/pib",
    "F_raw_pibcommunes/pibratio",
    "F_pct_change_pibcommunes/pibratio",
    # Revenu
    "F_raw_revcommunes/revratio",
    "F_pct_change_revcommunes/revratio",
    # Aire urbaine
    "F_raw_popdepartements/popcom5u",
    "F_raw_popdepartements/popcoma",
    # agesexcommunes
    "F_raw_agesexcommunes/age",
    "F_pct_change_agesexcommunes/age",
    "F_raw_agesexcommunes/pop",
    "F_pct_change_agesexcommunes/pop",
    "F_raw_agesexcommunes/prop014",
    "F_raw_agesexcommunes/prop1539",
    "F_raw_agesexcommunes/prop4059",
    "F_raw_agesexcommunes/prop60p",
    "F_pct_change_agesexcommunes/prop014",
    "F_pct_change_agesexcommunes/prop1539",
    "F_pct_change_agesexcommunes/prop4059",
    "F_pct_change_agesexcommunes/prop60p",
    "F_raw_agesexcommunes/propf",
    "F_pct_change_agesexcommunes/propf",
    # cspcommunes
    "F_raw_cspcommunes/pagri",
    "F_raw_cspcommunes/pindp",
    "F_raw_cspcommunes/pcadr",
    "F_raw_cspcommunes/ppint",
    "F_raw_cspcommunes/pempl",
    "F_raw_cspcommunes/pouvr",
    "F_raw_cspcommunes/pchom",
    "F_raw_cspcommunes/pouem",
    "F_raw_cspcommunes/pcapi",
    "F_raw_cspcommunes/paind",
    "F_pct_change_cspcommunes/pagri",
    "F_pct_change_cspcommunes/pindp",
    "F_pct_change_cspcommunes/pcadr",
    "F_pct_change_cspcommunes/ppint",
    "F_pct_change_cspcommunes/pempl",
    "F_pct_change_cspcommunes/pouvr",
    "F_pct_change_cspcommunes/pchom",
    "F_pct_change_cspcommunes/pouem",
    "F_pct_change_cspcommunes/pcapi",
    "F_pct_change_cspcommunes/paind",
    # Diplomes
    "F_raw_diplomescommunes/pbac",
    "F_raw_diplomescommunes/psup",
    # Propriétaires
    "F_raw_proprietairescommunes/ppropri",
    "F_pct_change_proprietairescommunes/ppropri",
    # RSA
    "F_raw_rsacommunes/prsa",
    # Immobilier
    "F_raw_capitalimmobiliercommunes/prixm2ratio",
    "F_pct_change_capitalimmobiliercommunes/prixm2ratio",
    "F_raw_capitalimmobiliercommunes/propappartement",
    "F_raw_capitalimmobiliercommunes/propappartement",
    "F_raw_capitalimmobiliercommunes/capitalratio",
    "F_pct_change_capitalimmobiliercommunes/propappartement",
    "F_pct_change_capitalimmobiliercommunes/capitalratio",
    # Immigration
    "F_raw_naticommunes/pimmigre",
    "F_pct_change_naticommunes/pimmigre",
    "F_raw_naticommunes/pnatur",
    "F_pct_change_naticommunes/pnatur",
    "F_raw_etrangerscommunes/petranger",
    "F_pct_change_etrangerscommunes/petranger",
]


def make_features(metric="pct_change"):
    return [f.format(metric=metric) if "{metric}" in f else f for f in BASE_FEATURES]
