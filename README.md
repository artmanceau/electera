# Modélisation des comportements électoraux

Ce projet de recherche propose une double approche pour comprendre et modéliser les comportements électoraux en France.

## Installation

Le projet est pour l'instant en mode "privé", pour les collaborateurs ajoutés :

```bash
git clone https://<your-username>@github.com/artmanceau/electera.git
```
```bash
cd electera
```

### Machine Learning [1]

## Setup & Environment

# 1. Navigate to project directory
```bash
cd electera/
```

# 2. Virtual environment
```bash
python -m venv .venv # Create
source .venv/bin/activate # Activate
```

# 3. Install requirements
```bash
pip install -r requirements.txt
```

Several pipeline are available, to run the project end-to-end:

## Pipelines

# 1. Download data
```bash
python src/pipeline/download_data.py
```
Will fetch data from https://www.unehistoireduconflitpolitique.fr/telecharger.html and store it to a specified location (local or S3)

# 2. Process data
```bash
python src/pipeline/data_processing.py
```
Will run data processing and store it to a specified location (local or S3)

# 3. Train models
```bash
python src/pipeline/train_models.py
```
Several models are implemented. Models tracking can be done with MLFlow.

# 4. Generate explanations
```bash
python src/pipeline/election_backtester.py
```
Perform a back-testing with the selected model. Training on previous (and previous previous) election (of the same type) to predict the next one.
Results are stored in a specified location (local or S3)

# 5. Election backtester
```bash
python src/pipeline/explain_model.py
```
Contains several explanability features for the model trained during the back-testing

# 6. polling data
```bash
python src/pipeline/poll_data_extract.py
```
Extract polling data for the elections (from Wikipedia)


## Pipelines
Application pour visualiser les performances du modèle
```bash
streamlit run app/app.py
```

### Modèle mathématique [2]

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
