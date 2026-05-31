FEATURE_DICT = {
    "popcommunes/pop": {
        "annees_disponibles": list(range(1790, 2023)),
        "description": "Population de la commune (territoire actuel, codes communes 2022).",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": (
            "Séries homogénéisées pour les communes actives en 2022. "
            "Sources INSEE (1876-2022) complétées par SGF (1801-1876). "
            "Les années intercensitaires ont été interpolées linéairement. "
            "Les années 1790-1800 sont reconstruites à partir de 1801 et de dénombrements partiels. "
            "Les populations 2020-2022 sont extrapolées à partir des tendances récentes."
        ),
        "source": "INSEE + SGF"
    },

    "popcommunes/popagglo": {
        "annees_disponibles": list(range(1790, 2023)),
        "description": "Population de l'agglomération de rattachement (unité urbaine 2022).",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "agglomération",
        "notes": (
            "Calculée à partir des communes rattachées aux unités urbaines définies en 2022. "
            "Permet des comparaisons homogènes sur longue période."
        ),
        "source": "INSEE + calculs auteurs"
    },

    "popcommunes/percommu": {
        "annees_disponibles": list(range(1790, 2023)),
        "description": "Percentile de la répartition de la population selon la taille de commune.",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": (
            "Défini comme la part cumulée de population vivant dans des communes "
            "de taille inférieure ou égale à la commune considérée."
        ),
        "source": "Calculé à partir des populations communales"
    },

    "popcommunes/peragglo": {
        "annees_disponibles": list(range(1790, 2023)),
        "description": "Percentile de la répartition de la population selon la taille d'agglomération.",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "agglomération",
        "notes": (
            "Défini comme la part cumulée de population vivant dans des agglomérations "
            "de taille inférieure ou égale à l'agglomération considérée."
        ),
        "source": "Calculé à partir des populations d'agglomérations"
    },

    "popdepartements/pop": {
        "annees_disponibles": list(range(1780, 2023)),
        "description": "Population des départements au cours du temps.",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": (
            "Série construite par agrégation des populations communales "
            "sur les frontières départementales actuelles."
        ),
        "source": "popdepartements.csv"
    },

    "popdepartements/popcom2u": {
        "annees_disponibles": list(range(1780, 2023)),
        "description": "Part de la population urbaine (communes ≥ 2000 habitants).",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": (
            "Proportion de la population vivant dans des communes de 2000 habitants ou plus, "
            "calculée à partir des données communales agrégées."
        ),
        "source": "popdepartements.csv"
    },

    "popdepartements/popcom5u": {
        "annees_disponibles": list(range(1780, 2023)),
        "description": "Part de la population urbaine (communes ≥ 5000 habitants).",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": (
            "Proportion de la population vivant dans des communes de 5000 habitants ou plus."
        ),
        "source": "popdepartements.csv"
    },

    "popdepartements/popcom10u": {
        "annees_disponibles": list(range(1780, 2023)),
        "description": "Part de la population urbaine (communes ≥ 10000 habitants).",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": (
            "Proportion de la population vivant dans des communes de 10000 habitants ou plus."
        ),
        "source": "popdepartements.csv"
    },

    "popdepartements/popcoma": {
        "annees_disponibles": list(range(1780, 2023)),
        "description": "Population moyenne des communes (pondérée) dans chaque département.",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": (
            "Moyenne pondérée par la population des communes du département, "
            "reflétant la structure d'urbanisation interne."
        ),
        "source": "popdepartements.csv"
    },

    "popcommuneselecteurs/electeurs": {
        "annees_disponibles": list(range(1848, 2023)),
        "description": "Population électorale (ayant le droit de vote) au niveau communal.",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": (
            "Estimations basées sur les composantes démographiques. "
            "Définition variable selon les périodes : "
            "1848–1943 hommes français 21+, 1944–1973 tous Français 21+, "
            "depuis 1974 tous Français 18+. Données estimées avec incertitude."
        ),
        "source": "popcommuneselecteurs.csv"
    },

    "popcommunesvbbm/vbbm": {
        "annees_disponibles": list(range(1790, 2023)),
        "description": "Typologie des communes : villages, bourgs, banlieues, métropoles.",
        "type": "categorical",
        "unit": "classe (1-4)",
        "aggregation_level": "commune",
        "notes": (
            "Classification basée sur la taille et la structure urbaine des communes "
            "(villages <2000, bourgs 2000–100k, banlieues et métropoles)."
        ),
        "source": "popcommunesvbbm.csv"
    },

    "popcommunesvbbm/vbbmpauvresriches": {
        "annees_disponibles": list(range(1790, 2023)),
        "description": "Typologie socio-économique des communes (pauvres à riches).",
        "type": "categorical",
        "unit": "classe (1-8)",
        "aggregation_level": "commune",
        "notes": (
            "Classification croisée par type de commune et niveau de revenu "
            "(villages, bourgs, banlieues, métropoles, chacun divisé en deux groupes)."
        ),
        "source": "popcommunesvbbm.csv"
    },

    "popcommunesvbbm/vbbmpauvresrichescap": {
        "annees_disponibles": list(range(1790, 2023)),
        "description": "Typologie socio-économique basée sur le capital immobilier par habitant.",
        "type": "categorical",
        "unit": "classe (1-8)",
        "aggregation_level": "commune",
        "notes": (
            "Même structure que vbbmpauvresriches mais fondée sur le capital immobilier "
            "par habitant plutôt que le revenu."
        ),
        "source": "popcommunesvbbm.csv"
    },
    "agesexcommunes/pop": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population totale par commune",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Série issue des recensements. Panel de communes présentes au moins une fois entre 1960 et 2022 ; le nombre de communes varie dans le temps en raison des fusions et disparitions.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/popf": {
        "annees_disponibles": list(range(1962, 2023)),
        "description": "Population féminine par commune",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Disponible de 1962 à 2022 selon la note du fichier ; construite à partir des recensements par sexe.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/poph": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population masculine par commune",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Série issue des recensements par sexe.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/propf": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Proportion de femmes dans la population",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Part des femmes dans la population totale de la commune.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/popf014": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population féminine âgée de 0 à 14 ans",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Population par âge et sexe issue des recensements.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/popf1539": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population féminine âgée de 15 à 39 ans",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Population par âge et sexe issue des recensements.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/popf4059": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population féminine âgée de 40 à 59 ans",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Population par âge et sexe issue des recensements.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/popf60p": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population féminine de 60 ans et plus",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Population par âge et sexe issue des recensements.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/popm014": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population masculine âgée de 0 à 14 ans",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Population par âge et sexe issue des recensements.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/popm1539": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population masculine âgée de 15 à 39 ans",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Population par âge et sexe issue des recensements.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/popm4059": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population masculine âgée de 40 à 59 ans",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Population par âge et sexe issue des recensements.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/popm60p": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population masculine de 60 ans et plus",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Population par âge et sexe issue des recensements.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/propf014": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Proportion de femmes parmi les 0–14 ans",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Part des filles dans la population 0–14 ans.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/propf1539": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Proportion de femmes parmi les 15–39 ans",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Part des femmes dans la classe d’âge 15–39 ans.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/propf4059": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Proportion de femmes parmi les 40–59 ans",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Part des femmes dans la classe d’âge 40–59 ans.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/propf60p": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Proportion de femmes parmi les 60 ans et plus",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Part des femmes parmi les 60 ans et plus.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/agef": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Âge moyen des femmes",
        "type": "duration",
        "unit": "années",
        "aggregation_level": "commune",
        "notes": "Âge moyen de la population féminine.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/ageh": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Âge moyen des hommes",
        "type": "duration",
        "unit": "années",
        "aggregation_level": "commune",
        "notes": "Âge moyen de la population masculine.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/age": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Âge moyen de la population",
        "type": "duration",
        "unit": "années",
        "aggregation_level": "commune",
        "notes": "Âge moyen global de la commune.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/prop014": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des 0–14 ans dans la population",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Indicateur de structure par âge.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/prop1539": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des 15–39 ans dans la population",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Indicateur de structure par âge.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/prop4059": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des 40–59 ans dans la population",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Indicateur de structure par âge.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/prop60p": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des 60 ans et plus dans la population",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Indicateur de vieillissement.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/perpropf": {
        "annees_disponibles": list(range(1790, 2023)),
        "description": "Percentile de la part des femmes dans la population",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Position relative des communes selon la proportion de femmes.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/perage": {
        "annees_disponibles": list(range(1790, 2023)),
        "description": "Percentile de l’âge moyen",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Position relative des communes selon l’âge moyen.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/perprop014": {
        "annees_disponibles": list(range(1790, 2023)),
        "description": "Percentile de la part des 0–14 ans",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Position relative des communes selon la part des jeunes.",
        "source": "agesexcommunes.dta"
    },

    "agesexcommunes/perprop60p": {
        "annees_disponibles": list(range(1790, 2023)),
        "description": "Percentile de la part des 60 ans et plus",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Position relative des communes selon le vieillissement.",
        "source": "agesexcommunes.dta"
    },
    "agesexdepartements/pop": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Population totale par département",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Série construite à partir des recensements départementaux (1851–1954) et des données communales agrégées (1960–2022) avec interpolation pour les années intercensitaires avant 1960.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/popf": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Population féminine par département",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Série homogénéisée sur 1851–2022 à partir des recensements et des agrégations communales.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/poph": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Population masculine par département",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Série homogénéisée sur 1851–2022 à partir des recensements et des agrégations communales.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/propf": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Proportion de femmes dans la population",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Part des femmes dans la population totale du département.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/popf014": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Population féminine âgée de 0 à 14 ans",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Série issue des recensements et agrégations communales (post-1960) et tabulations départementales (pré-1960).",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/popf1539": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Population féminine âgée de 15 à 39 ans",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Série homogénéisée sur 1851–2022.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/popf4059": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Population féminine âgée de 40 à 59 ans",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Série homogénéisée sur 1851–2022.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/popf60p": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Population féminine âgée de 60 ans et plus",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Série homogénéisée sur 1851–2022.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/popm014": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Population masculine âgée de 0 à 14 ans",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Série issue des recensements et agrégations communales (post-1960) et tabulations départementales (pré-1960).",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/popm1559": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Population masculine âgée de 15 à 59 ans",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Classe d’âge masculine intermédiaire (telle que définie dans le fichier source).",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/popm60p": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Population masculine âgée de 60 ans et plus",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Série homogénéisée sur 1851–2022.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/propf014": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Proportion de femmes parmi les 0–14 ans",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Part des filles dans la population 0–14 ans.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/propf1539": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Proportion de femmes parmi les 15–39 ans",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Part des femmes dans la classe d’âge 15–39 ans.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/propf4059": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Proportion de femmes parmi les 40–59 ans",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Part des femmes dans la classe d’âge 40–59 ans.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/propf60p": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Proportion de femmes parmi les 60 ans et plus",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Part des femmes parmi les 60 ans et plus.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/prop014": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Part des 0–14 ans dans la population",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Structure par âge du département.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/prop1539": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Part des 15–39 ans dans la population",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Structure par âge du département.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/prop4059": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Part des 40–59 ans dans la population",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Structure par âge du département.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/prop60p": {
        "annees_disponibles": list(range(1851, 2023)),
        "description": "Part des 60 ans et plus dans la population",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Indicateur de vieillissement démographique.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/agef": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Âge moyen des femmes",
        "type": "duration",
        "unit": "années",
        "aggregation_level": "département",
        "notes": "Disponible uniquement pour 1960–2022 selon la documentation.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/ageh": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Âge moyen des hommes",
        "type": "duration",
        "unit": "années",
        "aggregation_level": "département",
        "notes": "Disponible uniquement pour 1960–2022 selon la documentation.",
        "source": "agesexdepartements.csv"
    },

    "agesexdepartements/age": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Âge moyen de la population",
        "type": "duration",
        "unit": "années",
        "aggregation_level": "département",
        "notes": "Disponible uniquement pour 1960–2022 selon la documentation.",
        "source": "agesexdepartements.csv"
    },
    "diplomesdepartements/conjsignf": {
        "annees_disponibles": list(range(1686, 1906)),
        "description": "Nombre de femmes signant leur acte de mariage",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Série issue des registres de mariages (signature des conjointes). Années observées discontinues (1686–1905) avec interpolation linéaire entre vagues de collecte.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/conjnosif": {
        "annees_disponibles": list(range(1686, 1906)),
        "description": "Nombre de femmes ne signant pas leur acte de mariage",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Complément aux signatures de mariage ; données par vagues de recensement historique et interpolation.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/conjsignh": {
        "annees_disponibles": list(range(1686, 1906)),
        "description": "Nombre d’hommes signant leur acte de mariage",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Basé sur la signature des actes de mariage masculins ; données historiques agrégées et interpolées.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/conjnosih": {
        "annees_disponibles": list(range(1686, 1906)),
        "description": "Nombre d’hommes ne signant pas leur acte de mariage",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Complément aux signatures masculines dans les registres de mariage.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/pconjsignf": {
        "annees_disponibles": list(range(1686, 1906)),
        "description": "Proportion de femmes signant leur acte de mariage",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Taux d’alphabétisation proxy via signature des actes de mariage.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/pconjsignh": {
        "annees_disponibles": list(range(1686, 1906)),
        "description": "Proportion d’hommes signant leur acte de mariage",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Indicateur d’alphabétisation via signature des mariages.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/pconjsign": {
        "annees_disponibles": list(range(1686, 1906)),
        "description": "Proportion totale de signatures aux actes de mariage",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Moyenne hommes/femmes des taux de signature des actes de mariage.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/conslirH": {
        "annees_disponibles": list(range(1827, 1907)),
        "description": "Nombre de conscrits sachant lire",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Données issues des conscriptions militaires (1827–1906), interpolées entre vagues de collecte.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/consnolH": {
        "annees_disponibles": list(range(1827, 1907)),
        "description": "Nombre de conscrits ne sachant pas lire",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Inclut cas inconnus dans certaines périodes historiques.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/pconslirH": {
        "annees_disponibles": list(range(1827, 1907)),
        "description": "Proportion de conscrits sachant lire",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Indicateur d’alphabétisation masculine issu des conscrits militaires.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/alphaf": {
        "annees_disponibles": list(range(1866, 1947)),
        "description": "Nombre de femmes alphabétisées (20 ans et plus)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Recensements 1866–1946 ; femmes sachant lire et écrire.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/alphah": {
        "annees_disponibles": list(range(1866, 1947)),
        "description": "Nombre d’hommes alphabétisés (20 ans et plus)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Recensements 1866–1946.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/nonalf": {
        "annees_disponibles": list(range(1866, 1947)),
        "description": "Nombre de femmes non alphabétisées (20 ans et plus)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Inclut personnes ne sachant ni lire ni écrire ou seulement lire.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/nonalh": {
        "annees_disponibles": list(range(1866, 1947)),
        "description": "Nombre d’hommes non alphabétisés (20 ans et plus)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Recensements 1866–1946.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/palphaf": {
        "annees_disponibles": list(range(1866, 1947)),
        "description": "Proportion de femmes alphabétisées (20 ans et plus)",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Taux d’alphabétisation féminin.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/palphah": {
        "annees_disponibles": list(range(1866, 1947)),
        "description": "Proportion d’hommes alphabétisés (20 ans et plus)",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Taux d’alphabétisation masculin.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/palpha": {
        "annees_disponibles": list(range(1866, 1947)),
        "description": "Proportion de la population alphabétisée (20 ans et plus)",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Indicateur global d’alphabétisation.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/nodiph": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Hommes 25+ sans diplôme ou diplôme faible (≤ CAP/BEP)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Basé sur recensements modernes (1960–2022).",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/nodipf": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Femmes 25+ sans diplôme ou diplôme faible (≤ CAP/BEP)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Recensements modernes.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/nodip": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population 25+ sans diplôme ou diplôme faible",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Agrégation hommes/femmes.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/bach": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Nombre de bacheliers (hommes et femmes, 25+)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Diplôme le plus élevé = baccalauréat général ou technologique.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/suph": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Hommes diplômés du supérieur (25+)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Recensements modernes.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/supf": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Femmes diplômées du supérieur (25+)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Recensements modernes.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/sup": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population diplômée du supérieur (25+)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Agrégation hommes/femmes.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/bac": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Nombre de bacheliers (25+)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Inclut baccalauréat général et technologique.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/bacf": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Femmes bachelières (25+)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Recensements modernes.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/bach1960": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Hommes bacheliers (25+)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "département",
        "notes": "Recensements modernes.",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/pbac": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Proportion de bacheliers (et plus)",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Calculée comme (bac + sup) / (nodip + bac + sup).",
        "source": "diplomesdepartements.csv"
    },

    "diplomesdepartements/psup": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Proportion de diplômés du supérieur",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "département",
        "notes": "Calculée comme sup / (nodip + bac + sup).",
        "source": "diplomesdepartements.csv"
    },
    "alphabetisationcommunes/conjsign": {
        "annees_disponibles": list(range(1686, 1906)),
        "description": "Nombre de conjoints signant leur acte de mariage",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Inclut hommes et femmes mariés dans l’année. Séries construites à partir des registres de mariage (1686–1905) avec interpolation entre vagues de collecte.",
        "source": "alphabetisationcommunes.csv"
    },

    "alphabetisationcommunes/conjnosH": {
        "annees_disponibles": list(range(1686, 1906)),
        "description": "Nombre de conjoints ne signant pas leur acte de mariage",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Complément aux données de signature des actes de mariage.",
        "source": "alphabetisationcommunes.csv"
    },

    "alphabetisationcommunes/pconjsign": {
        "annees_disponibles": list(range(1686, 1906)),
        "description": "Proportion de conjoints signant leur acte de mariage",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Indicateur proxy d’alphabétisation basé sur la capacité à signer les actes de mariage.",
        "source": "alphabetisationcommunes.csv"
    },

    "alphabetisationcommunes/perconjsign": {
        "annees_disponibles": list(range(1686, 1906)),
        "description": "Percentile de la proportion de conjoints signant leur acte de mariage",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Percentile de position dans la distribution intercommunale, pondéré par population.",
        "source": "alphabetisationcommunes.csv"
    },

    "alphabetisationcommunes/alpha": {
        "annees_disponibles": list(range(1866, 1947)),
        "description": "Population alphabétisée (20 ans et plus)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Personnes sachant lire et écrire selon les recensements 1866–1946.",
        "source": "alphabetisationcommunes.csv"
    },

    "alphabetisationcommunes/nonal": {
        "annees_disponibles": list(range(1866, 1947)),
        "description": "Population non alphabétisée (20 ans et plus)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Inclut personnes ne sachant ni lire ni écrire ou seulement lire.",
        "source": "alphabetisationcommunes.csv"
    },

    "alphabetisationcommunes/palpha": {
        "annees_disponibles": list(range(1866, 1947)),
        "description": "Proportion de la population alphabétisée (20 ans et plus)",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Indicateur d’alphabétisation basé sur recensements.",
        "source": "alphabetisationcommunes.csv"
    },

    "alphabetisationcommunes/peralpha": {
        "annees_disponibles": list(range(1866, 1947)),
        "description": "Percentile de la proportion de population alphabétisée",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Percentile pondéré par population 20+ ; calculs harmonisés avec données départementales et estimations intra-départementales.",
        "source": "alphabetisationcommunes.csv"
    },
    "cspcommunes/agri": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Nombre d’agriculteurs (25–54 ans)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Actifs occupés ou chômeurs. Séries issues des recensements avec interpolation des années intercensitaires et corrections de comparabilité (notamment 1962).",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/indp": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Nombre d’indépendants (artisans, commerçants, chefs d’entreprise)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Population active 25–54 ans (actifs occupés ou chômeurs).",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/cadr": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Nombre de cadres et professions intellectuelles supérieures",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Actifs 25–54 ans (occupés ou chômeurs). Nomenclature harmonisée des recensements.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/pint": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Nombre de professions intermédiaires",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Actifs 25–54 ans.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/empl": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Nombre d’employés",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Actifs 25–54 ans.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/ouvr": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Nombre d’ouvriers",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Actifs 25–54 ans.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/pact": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Population active totale (25–54 ans)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Somme des catégories CSP (agri + indp + cadr + pint + empl + ouvr).",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/chom": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Nombre de chômeurs (25–54 ans)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Série reconstruite pour 1962 et harmonisée sur l’ensemble de la période.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/aind": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Agriculteurs et indépendants (25–54 ans)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Somme agri + indp.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/aica": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Agriculteurs, indépendants et cadres (25–54 ans)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Somme agri + indp + cadr.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/ouem": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Ouvriers et employés (25–54 ans)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Somme empl + ouvr.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/capi": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Cadres et professions intermédiaires (25–54 ans)",
        "type": "count",
        "unit": "personnes",
        "aggregation_level": "commune",
        "notes": "Somme cadr + pint.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/pagri": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des agriculteurs parmi les actifs",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "pagri = agri / pact.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/pindp": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des indépendants parmi les actifs",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "pindp = indp / pact.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/pcadr": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des cadres parmi les actifs",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "pcadr = cadr / pact.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/ppint": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des professions intermédiaires",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "ppint = pint / pact.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/pempl": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des employés",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "pempl = empl / pact.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/pouvr": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des ouvriers",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "pouvr = ouvr / pact.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/pchom": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des chômeurs parmi les actifs",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "pchom = chom / pact.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/paind": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des agriculteurs et indépendants",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "paind = aind / pact.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/paica": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des agriculteurs, indépendants et cadres",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "paica = aica / pact.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/pouem": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des ouvriers et employés",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "pouem = ouem / pact.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/pcapi": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Part des cadres et professions intermédiaires",
        "type": "proportion",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "pcapi = capi / pact.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/peragri": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Percentile de la part des agriculteurs",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Distribution intercommunale des parts de CSP.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/perindp": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Percentile de la part des indépendants",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Distribution intercommunale.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/percadr": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Percentile de la part des cadres",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Distribution intercommunale.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/perpint": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Percentile de la part des professions intermédiaires",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Distribution intercommunale.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/perempl": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Percentile de la part des employés",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Distribution intercommunale.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/perouvr": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Percentile de la part des ouvriers",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Distribution intercommunale.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/perchom": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Percentile de la part des chômeurs",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Distribution intercommunale.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/peraind": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Percentile des agriculteurs et indépendants",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Distribution intercommunale.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/peraica": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Percentile des agriculteurs, indépendants et cadres",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Distribution intercommunale.",
        "source": "cspcommunes.csv"
    },

    "cspcommunes/perouem": {
        "annees_disponibles": list(range(1960, 2023)),
        "description": "Percentile des ouvriers et employés",
        "type": "percentile",
        "unit": "[0,1]",
        "aggregation_level": "commune",
        "notes": "Distribution intercommunale.",
        "source": "cspcommunes.csv"
    },
}
