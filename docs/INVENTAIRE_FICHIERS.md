# Inventaire des fichiers du dépôt

Ce document indique **à quoi sert chaque élément principal**, s’il est **indispensable au projet**, **optionnel**, **généré localement** ou **ignoré par Git** (selon `.gitignore`). Les chemins sont relatifs à la racine du dépôt.

---

## Légende

| Élément | Signification |
|--------|----------------|
| **Essentiel** | Code ou config nécessaire pour installer, lancer ou comprendre le projet. |
| **Utile** | Script ou doc utile selon le cas (benchmark, données externes, Docker). |
| **Optionnel** | Fonctionnalité secondaire ; le projet tourne sans. |
| **Stub** | Fichier minimal (ex. redirection) pour ne pas casser d’anciens liens. |
| **Généré** | Créé par un outil (`pip install -e`, tests, benchmark) ; ne pas éditer à la main. |
| **Local / data** | Dossier présent sur ta machine mais **exclu du dépôt** (données lourdes). |
| **Sur Git** | Ce qui est typiquement versionné ; l’inverse suit `.gitignore`. |

---

## Racine

| Fichier / dossier | Rôle | Utile ? | Sur Git |
|-------------------|------|---------|---------|
| `README.md` | Point d’entrée : objectif, installation, commandes. | Essentiel | Oui |
| `pyproject.toml` | Package `sdwpf`, dépendances (source unique), extras `dev` / `experiments` (MLflow). | Essentiel | Oui |
| `Dockerfile` | Image pour exécuter le projet dans un conteneur. | Utile (CI / prod light) | Oui |
| `.gitignore` | Exclut données, caches, rapports, `mlruns`, etc. | Essentiel | Oui |
| `.env` | Secrets / chemins locaux (non fourni). | Optionnel ; **ne pas commit** | Non |

---

## `src/sdwpf/` — package Python

| Fichier | Rôle | Utile ? |
|---------|------|---------|
| `__init__.py` | Exports publics (`train_and_evaluate`, `SdwpfRunResult`, etc.). | Essentiel |
| `constants.py` | Constantes (pas de temps, colonnes, etc.). | Essentiel |
| `paths.py` | Résolution des chemins données / artefacts. | Essentiel |
| `data.py` | Chargement / assemblage des CSV SDWPF. | Essentiel |
| `features.py` | Features calendrier, lags, splits temporels, **`walk_forward_indices`**, horizon. | Essentiel |
| `pipeline.py` | `build_modeling_frame`, `evaluate_on_indices`, `train_and_evaluate` ; baselines ; MLflow optionnel. | Essentiel |

**Dossier `src/sdwpf.egg-info/`** : métadonnées créées par `pip install -e .`. **Généré** — ignoré par Git.

---

## `scripts/` — exécutables

| Fichier | Rôle | Utile ? |
|---------|------|---------|
| `sdwpf_explore.py` | Run rapide + stats ; option `--mlflow`. | Essentiel pour expérimentation |
| `sdwpf_benchmark.py` | Comparaisons (ex. full vs météo) ; écrit CSV/MD dans `reports/`. | Utile pour portfolio / tableaux |
| `sdwpf_visualize.py` | Graphiques (importance features, etc.). | Utile pour figures |
| `fetch_open_meteo_wind.py` | Téléchargement météo type Open-Meteo. | Utile si tu veux ces sources |
| `download_wind_toolkit_nlr.py` | Téléchargement NREL Wind Toolkit (données USA). | Utile pour parcours USA / WTK |
| `clean_artifacts.py` | Efface PNG obsolètes `*_h72_*`, `mlruns/` ; `--reports` pour CSV/MD sous `reports/`. | Utile pour rangement local |
| `sdwpf_walkforward.py` | Évaluation multi-plis (walk-forward) avec résumé statistique des MAE. | Utile pour robustesse / portfolio |

Aucun de ces scripts n’est « inutile » au sens strict : ils correspondent à des **tâches différentes**. Si tu ne touches qu’à la Chine SDWPF, les deux derniers sont **optionnels** pour toi.

---

## `tests/`

| Fichier | Rôle | Utile ? |
|---------|------|---------|
| `test_sdwpf_features.py` | Tests unitaires (splits, horizons, etc.). | Essentiel pour qualité / CI |
| `test_sdwpf_data.py` | Tests nettoyage SCADA / ERA5 et fusion. | Essentiel pour qualité / CI |
| `test_sdwpf_pipeline.py` | Tests smoke `train_and_evaluate` (données synthétiques). | Essentiel pour qualité / CI |

---

## `docs/`

| Fichier | Rôle | Utile ? |
|---------|------|---------|
| `GUIDE.md` | Parcours détaillé, flags, architecture. | Essentiel en complément du README |
| `DOMAINE_ET_PRATIQUES.md` | Domaine métier, sous-domaines, checklist de bonnes pratiques. | Essentiel pour cadrer le projet |
| `PLAN.md` | Roadmap, critères de succès, phases, todo. | Utile pour piloter la suite |
| `INVENTAIRE_FICHIERS.md` | Ce fichier : rôle et utilité des chemins. | Doc de maintenance |

---

## `reports/`

| Contenu | Rôle | Sur Git (d’après `.gitignore`) |
|---------|------|--------------------------------|
| `*.csv`, `*.md` hors exceptions | Sorties **sdwpf_benchmark** (résultats datés). | Non (ignorés) |
| `figures/*.png` | Figures exportées (souvent pour README / portfolio). | Oui (PNG explicitement ré-inclus) |
| `figures/.gitkeep` | Garde le dossier vide sous Git. | Oui |

**Verdict** : le dossier sert à **stocker des résultats** ; ce n’est pas du code source. Les anciens fichiers `*_h72_*` peuvent être **obsolètes** si tu imposes un horizon ≥ 144 pas ; tu peux les supprimer localement sans impact sur le code.

---

## `data/`

Données brutes SDWPF, ERA5, etc. **Essentiel en local** pour lancer le pipeline, mais **`data/china/sdwpf/`** et **`data/usa/nlr_wtk/`** sont **ignorés par Git** (volume). Sans ces dossiers, les scripts échouent au chargement jusqu’à ce que tu télécharges ou copies les fichiers.

---

## `mlruns/`

Artefacts **MLflow** (runs locaux). **Optionnel** : uniquement si tu lances `sdwpf_explore.py` avec `--mlflow`. Dossier **ignoré par Git** — tu peux le supprimer pour libérer de l’espace ; MLflow le recrée.

---

## Caches et outils

| Élément | Rôle | Sur Git |
|---------|------|---------|
| `.pytest_cache/` | Cache pytest | Non |
| `__pycache__/`, `*.pyc` | Bytecode Python | Non |
| `.venv/` / `venv/` | Environnement virtuel | Non |
| `dist/`, `build/` | Builds package | Non |

---

## Synthèse « qu’est-ce que je peux supprimer ? »

- **Sans rien casser dans le repo** : supprimer `mlruns/`, `.pytest_cache/`, `src/sdwpf.egg-info/`, anciens `reports/*.csv|md` devenus inutiles, et figures obsolètes.
- **Ne pas supprimer** : `src/sdwpf/`, `scripts/` (selon ton usage), `tests/`, `README.md`, `docs/GUIDE.md`, `pyproject.toml`, `.gitignore`.
- **Données** : ne pas commit ; sur ta machine, ne les efface que si tu peux les retélécharger.

Si tu ajoutes un nouveau fichier, mets à jour cette liste en une courte ligne dans la section qui convient.
