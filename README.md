# Moteur de recherche dans la littérature scientifique

> **MOD 7.2 : Introduction à la science des données** (BE séances 4, 5 et 6)  
> **Groupe**: Raphaël Davin, Maxence Dordenart, Mathis Frinzi

---

## Présentation du sujet

Ce projet consiste à concevoir un **moteur de recherche sémantique** appliqué à un corpus de plus de 25 000 publications scientifiques.

Étant donnée une publication requête $q$, l'algorithme doit attribuer un score d'appariement $S(q, c)$ à un ensemble d'articles candidats $C$ composé de :

- **5 articles cités** par la publication requête (exemples positifs $C^+$) ;
- **~25 articles aléatoires** issus du corpus (exemples négatifs $C^-$).

L'objectif est de faire remonter préférentiellement les 5 citations pertinentes en tête de classement, en explorant progressivement plusieurs familles de représentations (**approches creuses**, **représentations denses** et **information de structure** via le graphe de citations).

---

## Prérequis et installation

Ce projet utilise [**uv**](https://docs.astral.sh/uv/) comme gestionnaire d'environnement et de dépendances Python (Python `3.13` spécifié dans `.python-version`).

### 2. Cloner le dépôt et installer les dépendances

```bash
git clone https://github.com/moons-of-iego/search-engine.git

# Crée automatiquement l'environnement .venv et installe les dépendances verrouillées
uv sync
```

### 3. Préparer les données

Assurez-vous que les fichiers de données (`corpus.jsonl`, `queries.jsonl`, `valid.tsv`) sont extraits dans le dossier `dataset/` à la racine du projet :

```bash
unzip dataset.zip -d dataset
```

---

## Utilisation

Les scripts et modules peuvent être exécutés directement dans l'environnement géré par `uv` (sans activation manuelle du `.venv`) :

```bash
# Lancer le point d'entrée principal
uv run main.py

# Exécuter un module spécifique de src/
uv run -m src.analyser
uv run -m src.encoder
uv run -m src.comparator
uv run -m src.engine
```

> Au premier lancement, `src/loader.py` sérialise automatiquement les fichiers bruts en format binaire (`.pkl`) dans `.cache/` afin d'accélérer les exécutions suivantes.

---
