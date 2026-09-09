# Modélisation des comportements électoraux

Ce projet de recherche propose une double approche pour comprendre et modéliser les comportements électoraux en France.

Les résultats sont disponibles sur le site : https://electera.streamlit.app/

## Installation

```bash
git clone https://github.com/artmanceau/electera.git
```
```bash
cd electera
```

Il faut installer uv (qui gère les dépendances)
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

## Machine Learning [1]

### Setup & Environment

```bash
uv sync # Create
source .venv/bin/activate # Activate
uv pip install -e .
```

Le projet peut être reproduit end-to-end en executant les pipelines suivantes:

### Pipelines

#### 1. Download data
```bash
# Download data from  https://www.unehistoireduconflitpolitique.fr/telecharger.html
uv run python -m electera.pipeline.download_data
```

#### 2. Process data
```bash
# Run data processing
uv run python -m electera.pipeline.data_processing_pl
```

#### 3. Train models
```bash
# To train models on only one election. Use MLFlows to observe models.
uv run python -m electera.pipeline.train_models
```


#### 4.  Election backtester
```bash
# Perform a back-test over all the elections (using previous elections to predict the next one). Use MLFlows to observe models.
# Back-testing can be performed in argo workflows.
uv run python -m electera.pipeline.election_backtester
```


#### 5.Generate explanations
```bash
# Run explainability on the trained models.
uv run python -m electera.pipeline.explain_model
```

## Modèle mathématique [2]

Le dossier model/ contient les fichiers relatifs à la modélisation mathématique des comportements electoraux.

Il consistent en une implémentation du modèle suivant : [Christian Borghesi et Jean-Philippe Bouchaud. ≪ Spatial correlations in vote statistics:
a diffusive field model for decision-making ≫]. Puis :
- La ré-estimation des paramètre pour un plus grand corpus d'elections
- L'introduction d'une méthode de perturbation permettant de quantifier l'impact d'une variable socio-économique sur ce modèle.

Notebooks :
- Compute_correlations_COLLAB : notebook utilisé pour calculer les corrélations.
- Election_mathematical_model : le notebook avec le modèle mathématique du vote, l'estimation des paramètre et la perturbation autour d'un critère socio-démographique.
- Election_statistical_analysis : le notebook qui reproduit les résultat de l'article sur les propriété statistiques des élections
- Spatial_correlations : les corrélations spatiales.
- Linear_regression : exploration des corrélations avec les données socio-économiques, essaie d'une régression linéaire sur le taux de participation.

## Références

[1] Cagé J., Piketty T. (2023) : Une histoire du conflit politique. Élections et inégalités sociales en France, 1789-2022.


[2] Borghesi, C., Bouchaud, JP. (2010) : Spatial correlations in vote statistics: a diffusive field model for decision-making.

## Auteur

Arthur Manceau

Damien Challet
