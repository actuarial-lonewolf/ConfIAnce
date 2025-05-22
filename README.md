# Hackathon Desjardins 2025

Ce dépôt de code sert d'espace qui accueillera votre solution.

1. Les données sont sous le dossier [data](./data/)
1. Voir le [notebook de départ](./notebooks/QuickstartNotebook.ipynb)
1. Les variables qui vont dans votre fichier [`.env`](.env) (à créer!) vous ont été transmises par courriel.
1. Vous n'êtes pas contraint à cette structure de code / ces outils-ci, mais assurez-vous que le Comité d'Évaluation Technique puisse corriger votre solution (En cas de doute, communiquez avec les mentors techniques / le comité organisateur)

**IMPORTANT: Même si vous avez configuré votre poste lors de la formation affaires, vous devez le refaire maintenant. Il s'agit des mêmes instructions.** Vous devriez déjà avoir Python 3.12 et git d'installé, donc commencez par le `git clone` de votre dépôt (celui-ci!). 

## Instructions d'installation

1. Vous assurez d'avoir Python 3.12
1. Créer un environnement virtuel
1. Installer les dépendances
1. Écrire les variables d'environnement dans le fichier `.env` (utiliser les **dernières valeurs reçues aujourd'hui**)
1. Rouler le [notebook](./notebooks/QuickstartNotebook.ipynb)
    1. Sélectionner le bon kernel dans Jupyter
    1. Rouler les instructions


```shell
# 1) Assurez vous que la commande suivante retourne python 3.12 ou similaire
python --version

# 2) Installation de l'environnement virtuel
python -m venv .venv --prompt dsj-hackathon25

# Activez votre environnement virtuel
.venv\Scripts\Activate ## Windows
# ou ...
source .venv/bin/activate ## Linux

# 3) Installation des dépendances
pip install -r requirements.in -c requirements.txt


# 4) Récupérer les variables d'environnement (voir votre boite de courriel personnelle) et les mettre dans un .env
```

## Information complémentaire

1. Si vous éprouvez des difficultés et que vous avez besoin de l'aide d'un mentor technique, assurez-vous que votre code soit "committé" et "pushed". Ainsi, le mentor technique pour tester votre code de son côté pour vous aider.
