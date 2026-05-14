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
}


def get_colors(blocs, colors_dict):
    """Returns a list in the same order as bloc, with the associated elements"""
    return [colors_dict.get(bloc, None) for bloc in blocs]


trad = {
    "TD": "à gauche (tout les partis)",
    "TG": "à droite (tout les partis)",
    "GCG": "à gauche et au centre-gauche",
    "DCD": "à droite et au centre-droit",
    "G": "à gauche",
    "CD": "pour le centre-droite",
    "C": "pour le centre",
    "D": "à droite",
    "CG": "pour le centre-gauche",
    "par": "participation",
}


def political_align(blocs):
    if len(blocs) == 6:
        return ["G", "CG", "C", "CD", "D"]
    elif len(blocs) == 4:
        return ["GCG", "C", "DCD"]
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
        str(
            ["GCG", "DCD", "C", "par"]
        ): "division gauche / droite / centre",
        str(
            ["G", "D", "CG", "CD", "C", "par"]
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


FEATURES_DICT = {
    # ===== H1. POPULATION (1780-2022) =====
    # popcommunes files
    "popcommunes/pop": "Population de la commune",
    "popcommunes/popagglo": "Population de l'agglomération de rattachement de la commune",
    "popcommunes/percommu": "Percentile de la répartition de la population en fonction de la taille de commune",
    "popcommunes/peragglo": "Percentile de la répartition de la population en fonction de la taille d'agglomération",
    # popdepartements files
    "popdepartements/pop": "Population du département",
    "popdepartements/popcom2u": "Proportion de population urbaine (communes 2000h+) dans le département",
    "popdepartements/popcom5u": "Proportion de population urbaine (communes 5000h+) dans le département",
    "popdepartements/popcom10u": "Proportion de population urbaine (communes 10000h+) dans le département",
    "popdepartements/popcoma": "Population moyenne des communes du département (moyenne pondérée par la population des communes)",
    # popcommuneselecteurs files
    "popcommuneselecteurs/electeurs": "Population disposant du droit de vote (électeurs potentiels)",
    # popcommunesvbbm files
    "popcommunesvbbm/vbbm": "Indicateur villages, bourgs, banlieues, métropoles",
    "popcommunesvbbm/vbbmpauvresriches": "Indicateur villages pauvres-riches, bourgs pauvres-riches, banlieues pauvres-riches, métropoles pauvres-riches (revenu par habitant)",
    "popcommunesvbbm/vbbmpauvresrichescap": "Indicateur villages pauvres-riches, bourgs pauvres-riches, banlieues pauvres-riches, métropoles pauvres-riches (capital immobilier par habitant)",
    # ===== H2. AGE, SEXE ET STRUCTURE DES MENAGES (1851-2022) =====
    # agesexcommunes files
    "agesexcommunes/pop": "Population totale de la commune",
    "agesexcommunes/popf": "Population féminine de la commune",
    "agesexcommunes/poph": "Population masculine de la commune",
    "agesexcommunes/propf": "Proportion de femmes dans la commune",
    "agesexcommunes/popf014": "Population féminine 0-14 ans",
    "agesexcommunes/popf1539": "Population féminine 15-39 ans",
    "agesexcommunes/popf4059": "Population féminine 40-59 ans",
    "agesexcommunes/popf60p": "Population féminine 60 ans et +",
    "agesexcommunes/popm014": "Population masculine 0-14 ans",
    "agesexcommunes/popm1539": "Population masculine 15-39 ans",
    "agesexcommunes/popm4059": "Population masculine 40-59 ans",
    "agesexcommunes/popm60p": "Population masculine 60 ans et +",
    "agesexcommunes/propf014": "Proportion de femmes parmi les 0-14 ans",
    "agesexcommunes/propf1539": "Proportion de femmes parmi les 15-39 ans",
    "agesexcommunes/propf4059": "Proportion de femmes parmi les 40-59 ans",
    "agesexcommunes/propf60p": "Proportion de femmes parmi les 60 ans et +",
    "agesexcommunes/agef": "Âge moyen des femmes de la commune",
    "agesexcommunes/ageh": "Âge moyen des hommes de la commune",
    "agesexcommunes/age": "Âge moyen de la commune",
    "agesexcommunes/prop014": "Proportion de 0-14 ans dans la commune",
    "agesexcommunes/prop1539": "Proportion de 15-39 ans dans la commune",
    "agesexcommunes/prop4059": "Proportion de 40-59 ans dans la commune",
    "agesexcommunes/prop60p": "Proportion de 60 ans et + dans la commune",
    "agesexcommunes/perpropf": "Percentile de la répartition de la population en fonction de la proportion de femmes dans la commune",
    "agesexcommunes/perage": "Percentile de la répartition de la population en fonction de l'âge moyen dans la commune",
    "agesexcommunes/perprop014": "Percentile de la répartition de la population en fonction de la proportion de 0-14 ans dans la commune",
    "agesexcommunes/perprop60p": "Percentile de la répartition de la population en fonction de la proportion de 60 ans et plus dans la commune",
    # agesexdepartements files (same variables at department level)
    "agesexdepartements/pop": "Population totale du département",
    "agesexdepartements/popf": "Population féminine du département",
    "agesexdepartements/poph": "Population masculine du département",
    "agesexdepartements/propf": "Proportion de femmes dans le département",
    "agesexdepartements/popf014": "Population féminine 0-14 ans",
    "agesexdepartements/popf1539": "Population féminine 15-39 ans",
    "agesexdepartements/popf4059": "Population féminine 40-59 ans",
    "agesexdepartements/popf60p": "Population féminine 60 ans et +",
    "agesexdepartements/popm014": "Population masculine 0-14 ans",
    "agesexdepartements/popm1559": "Population masculine 15-59 ans",
    "agesexdepartements/popm60p": "Population masculine 60 ans et +",
    "agesexdepartements/propf014": "Proportion de femmes parmi les 0-14 ans",
    "agesexdepartements/propf1539": "Proportion de femmes parmi les 15-39 ans",
    "agesexdepartements/propf4059": "Proportion de femmes parmi les 40-59 ans",
    "agesexdepartements/propf60p": "Proportion de femmes parmi les 60 ans et +",
    "agesexdepartements/prop014": "Proportion de 0-14 ans dans le département",
    "agesexdepartements/prop1539": "Proportion de 15-39 ans dans le département",
    "agesexdepartements/prop4059": "Proportion de 40-59 ans dans le département",
    "agesexdepartements/prop60p": "Proportion de 60 ans et + dans le département",
    "agesexdepartements/agef": "Âge moyen des femmes du département",
    "agesexdepartements/ageh": "Âge moyen des hommes du département",
    "agesexdepartements/age": "Âge moyen du département",
    # menagescommunes files
    "menagescommunes/nmen": "Nombre total de ménages de la commune",
    "menagescommunes/nmencomp": "Nombre de ménages complexes de la commune",
    "menagescommunes/pmencomp": "Proportion de ménages complexes dans la commune",
    "menagescommunes/permencomp": "Percentile dans la répartition de la population en fonction de la proportion de ménages complexes dans la commune",
    # menagesdepartements files
    "menagesdepartements/nmen": "Nombre total de ménages du département",
    "menagesdepartements/nmencomp": "Nombre de ménages complexes du département",
    "menagesdepartements/pmencomp": "Proportion de ménages complexes du département",
    # ===== H3. FORMATION, DIPLOMES ET RELIGION (1680-2022) =====
    # diplomesdepartements files
    "diplomesdepartements/conjsignf": "Nombre de conjointes (femmes mariées dans l'année) signant leur acte de mariage dans le département",
    "diplomesdepartements/conjnosif": "Nombre de conjointes ne signant pas leur acte de mariage dans le département",
    "diplomesdepartements/conjsignh": "Nombre de conjoints (hommes mariés dans l'année) signant leur acte de mariage dans le département",
    "diplomesdepartements/conjnosih": "Nombre de conjoints ne signant pas leur acte de mariage dans le département",
    "diplomesdepartements/pconjsignf": "Proportion de conjointes signant leur acte de mariage dans le département",
    "diplomesdepartements/pconjsignh": "Proportion de conjoints signant leur acte de mariage dans le département",
    "diplomesdepartements/pconjsign": "Proportion de conjointes et conjoints signant leur acte de mariage dans le département",
    "diplomesdepartements/conslirH": "Nombre de conscrits militaires sachant lire dans le département",
    "diplomesdepartements/consnolH": "Nombre de conscrits militaires ne sachant pas lire dans le département",
    "diplomesdepartements/pconslirH": "Proportion de conscrits militaires sachant lire dans le département",
    "diplomesdepartements/alphaf": "Nombre de femmes alphabétisées (sachant lire et écrire) âgées de 20 ans+ résidentes dans le département",
    "diplomesdepartements/nonalf": "Nombre de femmes non alphabétisées âgées de 20 ans+",
    "diplomesdepartements/alphah": "Nombre d'hommes alphabétisés (sachant lire et écrire) âgés de 20 ans+",
    "diplomesdepartements/nonalh": "Nombre d'hommes non alphabétisés âgés de 20 ans+",
    "diplomesdepartements/palphaf": "Proportion de femmes 20+ alphabétisées",
    "diplomesdepartements/palphah": "Proportion d'hommes 20+ alphabétisés",
    "diplomesdepartements/palpha": "Proportion de la population 20+ alphabétisée",
    "diplomesdepartements/nodiph": "Nombre total d'hommes âgés de 25 ans et + sans diplôme ou dont le diplôme le plus élevé est le BEPC, le brevet des collèges, le BEP ou le CAP",
    "diplomesdepartements/bach": "Nombre total d'hommes 25+ dont le diplôme le plus élevé est le baccalauréat général ou technologique",
    "diplomesdepartements/suph": "Nombre total d'hommes 25+ diplômés du supérieur",
    "diplomesdepartements/nodipf": "Nombre total de femmes 25+ sans diplôme ou dont le diplôme le plus élevé est le BEPC, le brevet des collèges, le BEP ou le CAP",
    "diplomesdepartements/bacf": "Nombre total de femmes 25+ dont le diplôme le plus élevé est le baccalauréat général ou technologique",
    "diplomesdepartements/supf": "Nombre total de personnes 25+ diplômées du supérieur",
    "diplomesdepartements/nodip": "Nombre total de personnes 25+ sans diplôme ou dont le diplôme le plus élevé est le BEPC, le brevet des collèges, le BEP ou le CAP",
    "diplomesdepartements/bac": "Nombre total de personnes 25+ dont le diplôme le plus élevé est le baccalauréat général ou technologique",
    "diplomesdepartements/sup": "Nombre total de personnes 25+ diplômées du supérieur",
    "diplomesdepartements/pbac": "Proportion de bacheliers (et diplômés du supérieur) dans le département",
    "diplomesdepartements/psup": "Proportion diplômés du supérieur dans le département",
    # diplomescommunes files (same variables at commune level)
    "diplomescommunes/nodiph": "Nombre total d'hommes âgés de 25 ans et + résidents de la commune sans diplôme ou dont le diplôme le plus élevé est le BEPC, le brevet des collèges, le BEP ou le CAP",
    "diplomescommunes/bach": "Nombre total d'hommes 25+ dont le diplôme le plus élevé est le baccalauréat général ou technologique",
    "diplomescommunes/suph": "Nombre total d'hommes 25+ diplômés du supérieur",
    "diplomescommunes/nodipf": "Nombre total de femmes 25+ sans diplôme ou dont le diplôme le plus élevé est le BEPC, le brevet des collèges, le BEP ou le CAP",
    "diplomescommunes/bacf": "Nombre total de femmes 25+ dont le diplôme le plus élevé est le baccalauréat général ou technologique",
    "diplomescommunes/supf": "Nombre total de personnes 25+ diplômées du supérieur",
    "diplomescommunes/nodip": "Nombre total de personnes 25+ sans diplôme ou dont le diplôme le plus élevé est le BEPC, le brevet des collèges, le BEP ou le CAP",
    "diplomescommunes/bac": "Nombre total de personnes 25+ dont le diplôme le plus élevé est le baccalauréat général ou technologique",
    "diplomescommunes/sup": "Nombre total de personnes 25+ diplômées du supérieur",
    "diplomescommunes/pbac": "Proportion de bacheliers (et diplômés du supérieur) dans la commune",
    "diplomescommunes/psup": "Proportion diplômés du supérieur dans la commune",
    "diplomescommunes/perbac": "Percentile de la distribution de la proportion de bacheliers entre communes",
    "diplomescommunes/persup": "Percentile de la distribution de la proportion de diplômés du supérieur entre communes",
    # alphabetisationcommunes files
    "alphabetisationcommunes/conjsign": "Nombre de conjoints (hommes et femmes mariés dans l'année) signant leur acte de mariage",
    "alphabetisationcommunes/conjnosH": "Nombre de conjoints ne signant pas leur acte de mariage",
    "alphabetisationcommunes/pconjsign": "Proportion de conjoints signant leur acte de mariage",
    "alphabetisationcommunes/perconjsign": "Percentile de la répartition de la proportion de conjoints signant leur acte de mariage",
    "alphabetisationcommunes/alpha": "Nombre de personnes alphabétisées (sachant lire et écrire) âgées de 20 ans+",
    "alphabetisationcommunes/nonal": "Nombre de personnes non alphabétisées âgés de 20 ans+",
    "alphabetisationcommunes/palpha": "Proportion de personnes 20+ alphabétisées",
    "alphabetisationcommunes/peralpha": "Percentile de la distribution de la proportion de personnes alphabétisées entre communes",
    # publicprivecommunes files
    "publicprivecommunes/perprive": "Percentile dans la distribution de la part du privé dans les écoles primaires au niveau du canton",
    "publicprivecommunes/perpriveseco": "Percentile dans la distribution de la part du privé dans les établissements secondaires au niveau du canton",
    "publicprivecommunes/perprive_comm": "Percentile dans la distribution de la part du privé dans les écoles primaires au niveau de la commune",
    "publicprivecommunes/perpriveseco_comm": "Percentile dans la distribution de la part du privé dans les établissements secondaires au niveau de la commune",
    "publicprivecommunes/prive_total": "Proportion d'élèves scolarisés dans le privé au niveau de la commune (tous niveaux confondus)",
    "publicprivecommunes/prive_prim": "Proportion d'élèves scolarisés dans le privé au niveau de la commune (primaire)",
    "publicprivecommunes/prive_seco": "Proportion d'élèves scolarisés dans le privé au niveau de la commune (secondaire)",
    "publicprivecommunes/ntotal_pu": "Nombre d'élèves scolarisés dans le public au niveau de la commune (primaire + secondaire)",
    "publicprivecommunes/ntotal_pr": "Nombre d'élèves scolarisés dans le privé au niveau de la commune (primaire + secondaire)",
    "publicprivecommunes/ntotal": "Nombre total d'élèves scolarisés au niveau de la commune (public + privé, primaire + secondaire)",
    "publicprivecommunes/nprim_pu": "Nombre d'élèves scolarisés dans le public au niveau de la commune (primaire)",
    "publicprivecommunes/nprim_pr": "Nombre d'élèves scolarisés dans le privé au niveau de la commune (primaire)",
    "publicprivecommunes/nprim": "Nombre total d'élèves scolarisés au niveau de la commune (public + privé, primaire)",
    "publicprivecommunes/nseco_pu": "Nombre d'élèves scolarisés dans le public au niveau de la commune (secondaire)",
    "publicprivecommunes/nseco_pr": "Nombre d'élèves scolarisés dans le privé au niveau de la commune (secondaire)",
    "publicprivecommunes/nseco": "Nombre total d'élèves scolarisés au niveau de la commune (public + privé, secondaire)",
    # religiositedepartements files
    "religiositedepartements/ndepclerge": "Nombre total des prêtres soumis au serment de 1791",
    "religiositedepartements/ndepserment": "Nombre total des prêtres prêtant le serment en 1791",
    "religiositedepartements/ndeprefract": "Nombre total des prêtres refusant le serment en 1791",
    "religiositedepartements/prefract": "Proportion de prêtres réfractaires en 1791",
    "religiositedepartements/pserment": "Proportion de prêtres sermentaires en 1791",
    "religiositedepartements/pclerge": "Effectifs totaux du clergé (prêtres, religieux, religieuses) exprimés en proportion de la population totale (1856)",
    "religiositedepartements/pmessalisants": "Effectifs totaux de personnes allant à la messe du dimanche en 1950 en proportion de la population totale",
    # religiositecommunes files
    "religiositecommunes/perrefract": "Percentile de la commune dans la distribution de la proportion de prêtres réfractaires au niveau du district",
    "religiositecommunes/perpriv": "Percentile dans la distribution 1894 de la part du privé (y compris congréganiste public) dans les écoles primaires au niveau du canton",
    "religiositecommunes/perprivf": "Percentile dans la distribution 1894 de la part du privé (y compris congréganiste public) dans les écoles primaires au niveau du canton (filles)",
    "religiositecommunes/perprive": "Percentile de la commune dans la distribution 2021 de la part du privé dans les écoles primaires au niveau du canton",
    "religiositecommunes/perprives": "Percentile dans la distribution 2021 de la part du privé dans les établissements secondaires au niveau du canton",
    "religiositecommunes/prefract": "Proportion de prêtres réfractaires en 1791",
    "religiositecommunes/nclerge": "Nombre total des prêtres soumis au serment de 1791",
    "religiositecommunes/nrefract": "Nombre total des prêtres refusant le serment en 1791",
    "religiositecommunes/privecanton_prim": "Proportion d'élèves scolarisés dans le privé (y compris congréganiste public) (primaire) au niveau du canton (1894)",
    "religiositecommunes/privecanton_prim_f": "Proportion d'élèves scolarisés dans le privé (y compris congréganiste public) (primaire) au niveau du canton (filles) (1894)",
    # ===== H4. PROFESSIONS, EMPLOIS ET SECTEURS D'ACTIVITE (1851-2022) =====
    # cspcommunes files
    "cspcommunes/agri": "Nombre d'agriculteurs (actifs occupés ou chômeurs) parmi les 25-54 ans",
    "cspcommunes/indp": "Nombre d'indépendants (artisans, commerçants, chefs d'entreprises) (actifs occupés ou chômeurs) 25-54 ans",
    "cspcommunes/cadr": "Nombre de cadres et professions intellectuelles supérieures (actifs occupés ou chômeurs) 25-54 ans",
    "cspcommunes/pint": "Nombre de professions intermédiaires (actifs occupés ou chômeurs) 25-54 ans",
    "cspcommunes/empl": "Nombre d'employés (actifs occupés ou chômeurs) 25-54 ans",
    "cspcommunes/ouvr": "Nombre d'ouvriers (actifs occupés ou chômeurs) 25-54 ans",
    "cspcommunes/pact": "Population active totale (actifs occupés ou chômeurs) 25-54 ans",
    "cspcommunes/chom": "Nombre total de chômeurs (toutes CSP confondues) 25-54 ans",
    "cspcommunes/aind": "Nombre d'agriculteurs et indépendants (actifs occupés ou chômeurs) 25-54 ans",
    "cspcommunes/aica": "Nombre d'agriculteurs, indépendants et cadres (actifs occupés ou chômeurs) 25-54 ans",
    "cspcommunes/ouem": "Nombre d'ouvriers et employés (actifs occupés ou chômeurs) 25-54 ans",
    "cspcommunes/capi": "Nombre de cadres et professions intermédiaires (actifs occupés ou chômeurs) 25-54 ans",
    "cspcommunes/pagri": "Proportion d'agriculteurs parmi les actifs 25-54 ans",
    "cspcommunes/pindp": "Proportion d'indépendants parmi les actifs 25-54 ans",
    "cspcommunes/pcadr": "Proportion de cadres et professions intellectuelles supérieures parmi les actifs 25-54 ans",
    "cspcommunes/ppint": "Proportion de professions intermédiaires parmi les actifs 25-54 ans",
    "cspcommunes/pempl": "Proportions d'employés parmi les actifs 25-54 ans",
    "cspcommunes/pouvr": "Proportion d'ouvriers parmi les actifs 25-54 ans",
    "cspcommunes/pchom": "Proportion de chômeurs parmi les actifs 25-54 ans",
    "cspcommunes/paind": "Proportion d'agriculteurs et indépendants parmi les actifs 25-54 ans",
    "cspcommunes/paica": "Proportion d'agriculteurs, indépendants et cadres parmi les actifs 25-54 ans",
    "cspcommunes/pouem": "Proportion d'ouvriers et employés parmi les actifs 25-54 ans",
    "cspcommunes/pcapi": "Proportion de cadres et professions intermédiaires parmi les actifs 25-54 ans",
    "cspcommunes/peragri": "Percentile de la distribution de la proportion d'agriculteurs",
    "cspcommunes/perindp": "Percentile de la distribution de la proportion d'indépendants",
    "cspcommunes/percadr": "Percentile de la distribution de la proportion de cadres et professions intellectuelles supérieures",
    "cspcommunes/perpint": "Percentile de la distribution de la proportion de professions intermédiaires",
    "cspcommunes/perempl": "Percentile de la distribution de la proportion d'employés",
    "cspcommunes/perouvr": "Percentile de la distribution de la proportion d'ouvriers",
    "cspcommunes/perchom": "Percentile de la distribution de la proportion de chômeurs",
    "cspcommunes/peraind": "Percentile de la distribution de la proportion d'agriculteurs et indépendants",
    "cspcommunes/peraica": "Percentile de la distribution de la proportion d'agriculteurs, indépendants et cadres",
    "cspcommunes/perouem": "Percentile de la distribution de la proportion d'ouvriers et employés",
    # emploicommunes files
    "emploicommunes/emp": "Emploi total au lieu de travail",
    "emploicommunes/empexp": "Emploi exportateur au lieu de travail (activités potentiellement exportatrices de biens et services : agriculture, industrie, commerce de gros, conseil aux entreprises)",
    "emploicommunes/empres": "Emploi exportateur au lieu de travail (toutes les autres activités)",
    # empfoncommunes files
    "empfoncommunes/emp": "Emploi total au lieu de travail",
    "empfoncommunes/adm": "Administration publique, police, justice, etc.",
    "empfoncommunes/agr": "Agriculture, pêche",
    "empfoncommunes/btp": "Bâtiment, travaux publics",
    "empfoncommunes/cii": "Commerce inter-entreprises",
    "empfoncommunes/cri": "Conception-recherche industrielle",
    "empfoncommunes/loi": "Loisirs, culture, sport",
    "empfoncommunes/dis": "Distribution, commerce de détail",
    "empfoncommunes/edu": "Éducation, formation",
    "empfoncommunes/rep": "Réparation, entretien",
    "empfoncommunes/fab": "Fabrication de biens matériels et d'énergie",
    "empfoncommunes/bag": "Banque, assurances, gestion",
    "empfoncommunes/tra": "Transports, logistique",
    "empfoncommunes/con": "Conseil, analyse, prestations intellectuelles",
    "empfoncommunes/san": "Santé, action sociale",
    "empfoncommunes/res": "Restaurants, cafés, services de proximité",
    "empfoncommunes/padm": "Proportion d'emplois dans l'administration publique",
    "empfoncommunes/pagr": "Proportion d'emplois dans l'agriculture",
    "empfoncommunes/pbtp": "Proportion d'emplois dans le bâtiment",
    "empfoncommunes/pcii": "Proportion d'emplois dans le commerce inter-entreprises",
    "empfoncommunes/pcri": "Proportion d'emplois dans la conception-recherche industrielle",
    "empfoncommunes/ploi": "Proportion d'emplois dans les loisirs",
    "empfoncommunes/pdis": "Proportion d'emplois dans la distribution",
    "empfoncommunes/pedu": "Proportion d'emplois dans l'éducation",
    "empfoncommunes/prep": "Proportion d'emplois dans la réparation",
    "empfoncommunes/pfab": "Proportion d'emplois dans la fabrication",
    "empfoncommunes/pbag": "Proportion d'emplois dans la banque",
    "empfoncommunes/ptra": "Proportion d'emplois dans les transports",
    "empfoncommunes/pcon": "Proportion d'emplois dans le conseil",
    "empfoncommunes/psan": "Proportion d'emplois dans la santé",
    "empfoncommunes/pres": "Proportion d'emplois dans les restaurants",
    # ===== H5. NATIONALITES ET ORIGINES ETRANGERES (1851-2022) =====
    # naticommunes files
    "naticommunes/francais": "Nombre de personnes de nationalité française",
    "naticommunes/etranger": "Nombre de personnes de nationalité étrangère",
    "naticommunes/petranger": "Proportion de personnes de nationalité étrangère",
    "naticommunes/peretr": "Percentile de proportion de personnes de nationalité étrangère",
    "naticommunes/etranalg": "Nombre de personnes classées comme musulmans algériens (1962) ou de nationalité algérienne (1968, 1975, 1982, 1999)",
    "naticommunes/frarapat": "Nombre de personnes de nationalité française rapatriées d'Algérie (pieds-noirs)",
    "naticommunes/pfrarapat": "Proportion de rapatriés dans la population totale (1968)",
    "naticommunes/perrapat": "Percentile dans la distribution de la proportion de rapatriés dans la population totale (1968)",
    "naticommunes/frannaiss": "Nombre de personnes de nationalité française à la naissance",
    "naticommunes/frannatur": "Nombre de personnes de nationalité française par naturalisation",
    "naticommunes/etraeuro": "Nombre de personnes de nationalité étrangère européenne",
    "naticommunes/etraxeur": "Nombre de personnes de nationalité étrangère extra-européenne",
    "naticommunes/immigre": "Nombre d'immigrés (personnes nées étrangères à l'étranger)",
    "naticommunes/perimmigre": "Percentile dans la distribution de la proportion d'immigrés dans la population totale",
    "naticommunes/immnatur": "Nombre d'immigrés naturalisés (personnes nées étrangères à l'étranger, présentement de nationalité française)",
    "naticommunes/pimmnatur": "Proportion d'immigrés naturalisés dans la population totale",
    "naticommunes/natur": "Nombre de personnes naturalisées (personnes ayant acquis la nationalité française au cours de leur vie)",
    "naticommunes/pnatur": "Proportion de personnes naturalisées dans la population totale",
    "naticommunes/pnaturfra": "Proportion de personnes naturalisées dans la population de nationalité française",
    # natidepartements files (same variables at department level)
    "natidepartements/francais": "Nombre de personnes de nationalité française",
    "natidepartements/etranger": "Nombre de personnes de nationalité étrangère",
    "natidepartements/petranger": "Proportion de personnes de nationalité étrangère",
    # etrangerscommunes files
    "etrangerscommunes/francais": "Nombre de personnes de nationalité française",
    "etrangerscommunes/etranger": "Nombre de personnes de nationalité étrangère",
    "etrangerscommunes/petranger": "Proportion de personnes de nationalité étrangère",
    "etrangerscommunes/peretranger": "Percentile de proportion de personnes de nationalité étrangère",
    # ===== H6. PRODUCTIONS ET REVENUS (1860-2022) =====
    # pibdepartements files
    "pibdepartements/pibhab": "PIB départemental par habitant (exprimé en % du PIB par habitant moyen au niveau national)",
    "pibdepartements/pib": "PIB départemental (exprimé en % du PIB total au niveau national)",
    "pibdepartements/pop": "Population départementale",
    "pibdepartements/emploH": "Emploi départemental",
    "pibdepartements/empagr": "Emploi dans l'agriculture",
    "pibdepartements/empind": "Emploi dans l'industrie",
    "pibdepartements/empser": "Emploi dans les services",
    # revdepartements files
    "revdepartements/revratio": "Revenu moyen par habitant du département (exprimé en ratio du revenu moyen par habitant de France métropolitaine)",
    "revdepartements/revtot": "Revenu total de l'ensemble des foyers fiscaux (imposables et non imposables) du département (en fraction du total national)",
    "revdepartements/pop": "Nombre total d'habitants du département",
    "revdepartements/revratioadu": "Revenu moyen par adulte du département (exprimé en ratio du revenu moyen par adulte de France métropolitaine)",
    "revdepartements/revratiofoy": "Revenu moyen par foyer du département (exprimé en ratio du revenu moyen par foyer de France métropolitaine)",
    "revdepartements/revmoy": "Revenu moyen par habitant du département (exprimé en euros de 2022)",
    "revdepartements/revmoyadu": "Revenu moyen par adulte du département (exprimé en euros de 2022)",
    "revdepartements/revmoyfoy": "Revenu moyen par foyer du département (exprimé en euros de 2022)",
    "revdepartements/nfoyer": "Nombre total de foyers fiscaux (imposables et non imposables) du département",
    "revdepartements/nadult": "Nombre d'adultes (âgés de 20 ans et +) du département",
    # revcommunes files
    "revcommunes/revratio": "Revenu moyen par habitant de la commune (exprimé en ratio du revenu moyen par habitant de France métropolitaine)",
    "revcommunes/revtot": "Revenu total de la commune en fraction du revenu national total",
    "revcommunes/perrev": "Percentile de la distribution du revenu moyen par habitant entre communes",
    "revcommunes/revratioagglo": "Revenu moyen par habitant de l'agglomération (exprimé en ratio du revenu moyen par habitant de France métropolitaine)",
    "revcommunes/revtotagglo": "Revenu total de l'agglomération en fraction du revenu national total",
    "revcommunes/perrevagglo": "Percentile de la distribution du revenu moyen par habitant entre agglos",
    "revcommunes/pop": "Nombre total d'habitants de la commune",
    "revcommunes/nfoyer": "Nombre total de foyers fiscaux (imposables et non imposables) de la commune",
    "revcommunes/revmoyfoy": "Revenu moyen par foyer de la commune (exprimé en euros de 2022)",
    "revcommunes/revratiofoy": "Revenu moyen par foyer de la commune (exprimé en ratio du revenu moyen par foyer de France métropolitaine)",
    "revcommunes/nadult": "Nombre d'adultes (âgés de 20 ans et +) de la commune",
    "revcommunes/revmoyadu": "Revenu moyen par adulte de la commune (exprimé en euros de 2022)",
    "revcommunes/revratioadu": "Revenu moyen par adulte de la commune (exprimé en ratio du revenu moyen par adulte de France métropolitaine)",
    "revcommunes/revmoy": "Revenu moyen par habitant de la commune (exprimé en euros de 2022)",
    "revcommunes/perrevfoy": "Percentile de la distribution du revenu moyen par foyer entre communes",
    "revcommunes/perrevadu": "Percentile de la distribution du revenu moyen par adulte entre communes",
    # pibcommunes files
    "pibcommunes/pibratio": "PIB communal par habitant (exprimé en % du PIB par habitant moyen au niveau national)",
    "pibcommunes/pibtot": "PIB communal (exprimé en % du PIB total au niveau national)",
    "pibcommunes/perpibratio": "Percentile de la répartition du PIB communal par habitant",
    # ===== H7. CAPITAL IMMOBILIER, BASES FISCALES ET TERRES AGRICOLES (1790-2022) =====
    # capitalimmobilierdepartements files
    "capitalimmobilierdepartements/capitalratio": "Capital immobilier (valeur des logements) par habitant en ratio de la moyenne nationale",
    "capitalimmobilierdepartements/capitalimmo": "Capital immobilier total en fraction du total national",
    "capitalimmobilierdepartements/prixbien": "Prix moyen des logements en euros courants",
    "capitalimmobilierdepartements/prixm2ratio": "Prix moyen par m2 de surface réelle bâtie en ratio de la moyenne nationale",
    "capitalimmobilierdepartements/prixm2": "Prix moyen par m2 en euros courants",
    "capitalimmobilierdepartements/surfacH": "Surface moyenne des logements en m2",
    "capitalimmobilierdepartements/surfaceterrain": "Surface du terrain en m2",
    "capitalimmobilierdepartements/propappartement": "Proportion d'appartements",
    # capitalimmobiliercommunes files
    "capitalimmobiliercommunes/capitalratio": "Capital immobilier (valeur des logements) par habitant en ratio de la moyenne nationale",
    "capitalimmobiliercommunes/capitalimmo": "Capital immobilier total (valeur totale des logements) en fraction du total national",
    "capitalimmobiliercommunes/percap": "Percentile de la distribution du capital immobilier par habitant entre communes",
    "capitalimmobiliercommunes/capitalratioagglo": "Capital immobilier (valeur des logements) par habitant de l'agglo en ratio de la moyenne nationale",
    "capitalimmobiliercommunes/capitalimmoagglo": "Capital immobilier total (valeur totale des logements) de l'agglo en fraction du total national",
    "capitalimmobiliercommunes/percapagglo": "Percentile de la distribution du capital immobilier par habitant entre agglos",
    "capitalimmobiliercommunes/prixbien": "Prix moyen des logements en euros courants",
    "capitalimmobiliercommunes/prixm2ratio": "Prix moyen par m2 de surface réelle bâtie en ratio de la moyenne nationale",
    "capitalimmobiliercommunes/prixm2": "Prix moyen par m2 en euros courants",
    "capitalimmobiliercommunes/surfacH": "Surface moyenne des logements en m2",
    "capitalimmobiliercommunes/surfaceterrain": "Surface du terrain en m2",
    "capitalimmobiliercommunes/propappartement": "Proportion d'appartements",
    # basesfiscalesdepartements files
    "basesfiscalesdepartements/basehabitationratio": "Base habitation par habitant (bases de la contribution personnelle-mobilière/taxe d'habitation divisées par la population) exprimée en ratio de la base habitation par habitant au niveau national",
    "basesfiscalesdepartements/basefonciereratio": "Base foncière par habitant (bases de la contribution foncière/taxe foncière divisées par la population) exprimée en ratio de la base foncière par habitant au niveau national",
    "basesfiscalesdepartements/basehabitation": "Base habitation totale (bases de la contribution personnelle-mobilière/taxe d'habitation) exprimée en fraction de la base habitation totale au niveau national",
    "basesfiscalesdepartements/basefoncierH": "Base foncière totale (bases de la contribution foncière/taxe foncière) exprimée en fraction de la base foncière totale au niveau national",
    "basesfiscalesdepartements/baseimpotslocauxratio": "Base impôts locaux par habitant (moyenne pondérée des bases de la contribution personnelle-mobilière/taxe d'habitation et de la contribution foncière/taxe foncière, divisée par la population) exprimée en ratio de la base par habitant au niveau national",
    "basesfiscalesdepartements/baseimpotslocaux": "Base impôts locaux (moyenne pondérée des bases de la contribution personnelle-mobilière/taxe d'habitation et de la contribution foncière/taxe foncière) exprimée en fraction de la base au niveau national",
    "basesfiscalesdepartements/recettehabitationpratio": "Recette habitation principale (recettes de la contribution personnelle-mobilière en principal) par habitant exprimée en ratio de la recette personnelle-mobilière principale par habitant au niveau national",
    "basesfiscalesdepartements/recettehabitationp": "Recette habitation principale exprimée en fraction de la recette personnelle-mobilière principale au niveau national",
    "basesfiscalesdepartements/tauxhabitationp": "Taux habitation principal (recette habitation principale divisée par base habitation) exprimé en ratio du taux national",
    "basesfiscalesdepartements/recettefoncierepratio": "Recette foncière principale (recettes de la contribution foncière en principal) par habitant exprimée en ratio de la recette foncière principale par habitant au niveau national",
    "basesfiscalesdepartements/recettefoncierep": "Recette foncière principale exprimée en fraction de la recette foncière principale au niveau national",
    "basesfiscalesdepartements/tauxfoncierp": "Taux foncier principal (recette foncière principale divisée par base foncière) exprimé en ratio du taux national",
    "basesfiscalesdepartements/recettehabitationratio": "Recette habitation communale (recettes de la taxe d'habitation au niveau communal) par habitant exprimée en ratio de la recette habitation communale par habitant au niveau national",
    "basesfiscalesdepartements/recettefonciereratio": "Recette foncière communale (recettes de la taxe foncière au niveau communal) par habitant exprimée en ratio de la recette foncière communale par habitant au niveau national",
    "basesfiscalesdepartements/recettehabitation": "Recette habitation communale (recettes de la taxe d'habitation au niveau communal) exprimée en fraction de la recette habitation communale au niveau national",
    "basesfiscalesdepartements/recettefoncierH": "Recette foncière communale (recettes de la taxe foncière au niveau communal) exprimée en ratio de la recette foncière communale au niveau national",
    "basesfiscalesdepartements/tauxhabitationratio": "Taux habitation communal (recette habitation communale divisée par base habitation communale) exprimé en ratio du taux national",
    "basesfiscalesdepartements/tauxfoncierratio": "Taux foncier communal (recette foncière communale divisée par base foncière communale) exprimé en ratio du taux national",
    "basesfiscalesdepartements/recetteimpotslocauxratio": "Recette impôts locaux communaux (taxe d'habitation + taxe foncière) par habitant exprimée en ratio de la recette correspondante par habitant au niveau national",
    "basesfiscalesdepartements/recetteimpotslocaux": "Recette impôts locaux communaux (taxe d'habitation + taxe foncière) exprimée en fraction de la recette correspondante au niveau national",
    "basesfiscalesdepartements/tauximpotslocauxratio": "Taux des impôts locaux communaux (recette foncière + habitation divisée par base foncière + habitation) exprimé en ratio du taux correspondant national",
    "basesfiscalesdepartements/recetteratio": "Recette communale totale (toutes ressources confondues) par habitant exprimée en ratio de la recette par habitant au niveau national",
    "basesfiscalesdepartements/recette": "Recette communale totale (toutes ressources confondues) exprimée en fraction de la recette totale au niveau national",
    # recettescommunes files
    "recettescommunes/baseimpotslocauxratio": "Base des impôts locaux directs par habitant (exprimée en ratio de la moyenne au niveau national)",
    "recettescommunes/recetteimpotslocauxratio": "Recette des impôts locaux directs par habitant (exprimée en ratio de moyenne au niveau national)",
    "recettescommunes/tauximpotslocauxratio": "Taux communal effectif des impôts locaux (exprimé en ratio de la moyenne nationale)",
    "recettescommunes/recetteratio": "Recette communale totale (toutes recettes budgétaires confondues) par habitant (exprimée en ratio de la moyenne nationale)",
    "recettescommunes/baseimpotslocauxtot": "Base des impôts locaux directs en francs courants ou en euros courants",
    "recettescommunes/recetteimpotslocauxtot": "Recette des impôts locaux directs en francs courants ou en euros courants",
    "recettescommunes/tauximpotslocaux": "Taux communal effectif des impôts locaux",
    "recettescommunes/recettetot": "Recette communale totale (toutes recettes budgétaires confondues) en francs courants ou en euros courants",
    "recettescommunes/baseimpotslocaux": "Base des impôts locaux directs par habitant en francs courants ou en euros courants",
    "recettescommunes/recetteimpotslocaux": "Recette des impôts locaux directs par habitant en francs courants ou en euros courants",
    "recettescommunes/recette": "Recette communale totale (toutes recettes budgétaires confondues) par habitant en francs courants ou en euros courants",
    "recettescommunes/basehabitationratio": "Base habitation par habitant exprimée en ratio de la base habitation moyenne au niveau national",
    "recettescommunes/basehabitationtot": "Base habitation totale en euros courants (bases nettes de la taxe d'habitation)",
    "recettescommunes/basehabitation": "Base habitation par habitant en euros courants",
    "recettescommunes/basefonciereratio": "Base foncière par habitant exprimée en ratio de la base foncière moyenne au niveau national",
    "recettescommunes/basefoncieretot": "Base foncière totale en euros courants (bases nettes de la taxe foncière)",
    "recettescommunes/basefoncierH": "Base foncière par habitant en euros courants",
    "recettescommunes/recettehabitationratio": "Recette habitation par habitant exprimée en ratio de la base habitation moyenne au niveau national",
    "recettescommunes/recettehabitationtot": "Recette habitation totale en euros courants (bases nettes de la taxe d'habitation)",
    "recettescommunes/recettehabitation": "Recette habitation par habitant en euros courants",
    "recettescommunes/recettefonciereratio": "Recette foncière par habitant exprimée en ratio de la base foncière moyenne au niveau national",
    "recettescommunes/recettefoncieretot": "Recette foncière totale en euros courants (bases nettes de la taxe foncière)",
    "recettescommunes/recettefoncierH": "Recette foncière par habitant en euros courants",
    "recettescommunes/tauxhabitationratio": "Taux communal effectif de la taxe d'habitation (en ratio de la moyenne nationale)",
    "recettescommunes/tauxhabitation": "Taux communal effectif de la taxe d'habitation (recettes divisées par base)",
    "recettescommunes/tauxfoncierratio": "Taux communal effectif de la taxe foncière (en ratio de la moyenne nationale)",
    "recettescommunes/tauxfoncier": "Taux communal effectif de la taxe foncière (recettes divisées par base)",
    # proprietairescommunes files
    "proprietairescommunes/ppropri": "Proportion de ménages propriétaires de leur logement dans la commune",
    "proprietairescommunes/nlogement": "Nombre total de logements dans la commune (propriétaires, locataires ou logés gratuitement) (résidences principales)",
    "proprietairescommunes/npropri": "Nombre de ménages propriétaires dans la commune",
    "proprietairescommunes/perpropri": "Percentile dans la répartition de la proportion de propriétaires entre communes",
    # terrescommunes files
    "terrescommunes/nexploit": "Nombre d'exploitations agricoles de la commune",
    "terrescommunes/nexploit50p": "Nombre d'exploitations agricoles de plus de 50 hectares",
    "terrescommunes/pexploit50p": "Proportion d'exploitations agricoles de plus de 50 hectares",
    "terrescommunes/surface": "Surface totale exploitées de la commune",
    "terrescommunes/surface50p": "Surface exploitées de plus de 50 hectares de la commune",
    "terrescommunes/psurface50p": "Proportion de surfaces de plus de 50 hectares de la commune",
    "terrescommunes/persur50p": "Percentile de la proportion de surfaces de plus de 50 hectares de la commune",
    # ===== H8. AUTRES DONNEES SOCIOECONOMIQUES LOCALISEES (PERIODE RECENTE) =====
    # rsacommunes files
    "rsacommunes/nrsa": "Nombre d'allocataires du RSA par commune",
    "rsacommunes/prsa": "Proportion d'allocataires du RSA dans la population",
    # crimesdelitscommunes files
    "crimesdelitscommunes/ncrimesdelits": "Nombre total moyen annuel de crimes et délits enregistrés dans la commune",
    "crimesdelitscommunes/nviolences": "Nombre moyen annuel de crimes et délits (violences contre les personnes) enregistrés dans la commune",
    "crimesdelitscommunes/ncambriolages": "Nombre moyen annuel de crimes et délits (cambriolages de logements) enregistrés dans la commune",
    "crimesdelitscommunes/nvolsvoitures": "Nombre moyen annuel de crimes et délits (vols de voitures) enregistrés dans la commune",
    "crimesdelitscommunes/nautresvols": "Nombre moyen annuel de crimes et délits (autres vols) enregistrés dans la commune",
    "crimesdelitscommunes/pcrimesdelits": "Nombre total de crimes et délits par habitant",
    "crimesdelitscommunes/pviolences": "Nombre de crimes et délits (violences contre les personnes) par habitant",
    "crimesdelitscommunes/pcambriolages": "Nombre de crimes et délits (cambriolages de logements) par habitant",
    "crimesdelitscommunes/pvolsvoitures": "Nombre de crimes et délits (vols de voitures) par habitant",
    "crimesdelitscommunes/pautresvols": "Nombre de crimes et délits (autres vols) par habitant",
    "crimesdelitscommunes/percrimesdelits": "Percentile de la répartition du nombre total de crimes et délits par habitant",
    "crimesdelitscommunes/perviolences": "Percentile de la répartition du nombre de crimes et délits (violences contre les personnes) par habitant",
    "crimesdelitscommunes/percambriolages": "Percentile de la répartition du nombre de crimes et délits (cambriolages de logements) par habitant",
    "crimesdelitscommunes/pervolsvoitures": "Percentile de la répartition du nombre de crimes et délits (vols de voitures) par habitant",
    "crimesdelitscommunes/perautresvols": "Percentile de la répartition du nombre de crimes et délits (autres vols) par habitant",
    # isfcommunes files
    "isfcommunes/nisf": "Nombre de contribuables ISF par commune",
    "isfcommunes/pisf": "Proportion de contribuables ISF dans la population",
}
