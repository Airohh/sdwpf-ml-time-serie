# Forecasting wind power production under realistic constraints

*Prévision de production éolienne — pipeline reproductible (séries temporelles, multi-sources).*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest)](https://docs.pytest.org/)

**En une phrase** : anticiper la production pour mieux **planifier** et **équilibrer** le réseau ; ce dépôt **borne** ce qu’on peut attendre de la **météo seule** (réanalyse + calendrier) à horizon type **J+1**, avec un **découpage temporel strict**, des **baselines explicites** et une **lecture honnête** des métriques — y compris quand le modèle ne « bat » pas une référence simple.

**Données & périmètre technique** : **France** (Open-Meteo, ODRE), **USA** (Wind Toolkit NLR), **Chine** (SDWPF SCADA + ERA5). Pipeline **XGBoost**, horizons **≥ 1 jour** (144 pas de 10 min), **train / val / test** temporel, baselines **naive** et **persistance** (quand les features le permettent), **MLflow** optionnel (`experiments`), **Docker**.

**Pour aller plus loin** : **[`docs/GUIDE.md`](docs/GUIDE.md)** (parcours, glossaire, CLI, pièges). Publication Git : **[`docs/PUBLICATION_GITHUB.md`](docs/PUBLICATION_GITHUB.md)** (badges CI, remote, push).

---

## Sommaire

- [Problème métier](#problème-métier)
- [Ce que ce dépôt livre](#ce-que-ce-dépôt-livre)
- [Résultats & lecture « portfolio »](#résultats--lecture-portfolio)
- [Lessons learned](#lessons-learned)
- [Next steps](#next-steps)
- [Aperçu & flux](#aperçu--flux)
- [Aperçu visuel (optionnel)](#aperçu-visuel-optionnel)
- [Structure du dépôt](#structure-du-dépôt)
- [Démarrage rapide](#démarrage-rapide)
- [Documentation](#documentation-complète)
- [Évaluation](#évaluation-rappel)
- [Secrets (NLR)](#secrets-nlr)

---

## Problème métier

Anticiper la production d’un parc éolien sert l’**équilibrage électrique**, le **trading** et les **décisions opérationnelles** (maintenance, contrats). La question n’est pas seulement « quel algorithme tourne ? » mais : **avec quelles informations, à quel horizon, et sous quelles contraintes réalistes la prévision reste-t-elle utile ?**

Ce projet **pose** ce trade-off de façon explicite : en mode **`--meteo-mode`** (SDWPF), les *features* sont essentiellement **calendrier (sin/cos)** et **ERA5** — pas la puissance récente ni le vent SCADA — pour **mesurer** si, dans ce cadre, un gradient boosting apporte encore quelque chose par rapport à une **baseline simple** (moyenne de la cible sur le train), **sans fuite temporelle**.

👉 Le positionnement est donc **produit / risque** autant que **ML** : on documente **la valeur ajoutée informationnelle** de la météo réanalysée à la résolution du jeu, pas un benchmark « classement Kaggle ».

---

## Ce que ce dépôt livre

- Un **fil rouge** éolien **vérifiable** : chargement, features, **split temporel**, métriques (MAE, RMSE, nMAE, skill, biais), figures **01–06** et bench multi-horizon / walk-forward.
- Des **branches données** (France / USA / Chine) qui montrent aussi **les limites** des sources (ex. production ODRE **nationale** vs météo **ponctuelle** non co-localisée — détail dans [`docs/GUIDE.md`](docs/GUIDE.md)).
- Des **tests** (`pytest`) et une **CI** GitHub Actions (Python 3.10 / 3.12).

---

## Résultats & lecture « portfolio »

**Constat principal (SDWPF, mode météo seule, horizon type J+1)** : XGBoost peut produire des courbes **plausibles** sur le jeu test, mais sur un **grand parc**, le **skill** vs la baseline naive est souvent **proche de zéro ou légèrement négatif** en MAE moyenne.

👉 **Ce n’est pas un échec de présentation** : c’est une **conclusion exploitable** — elle indique que, **dans ces conditions** (information limitée à réanalyse + calendrier, sans dynamique locale de la machine), **une référence simple reste compétitive**. Les figures **`05_*`** (synthèse parc) et **`06_*`** (détail par turbine : MAE, nMAE, skill, biais) servent à **montrer ce constat chiffré**, pas à embellir artificiellement le gain.

Pour un recruteur ou un pair : *« j’ai isolé la question métier, appliqué une évaluation honnête, et rendu compte des limites de l’information disponible. »*

---

## Lessons learned

- À **résolution 10 min** et **sans Patv récent / vent SCADA** dans les features, la **météo réanalysée seule** **ne suffit souvent pas** à battre une baseline naive sur le **MAE** agrégé — ce qui **cadre** l’attente métier sur ce jeu et cet horizon.
- La **collocation** des signaux (où est mesurée la prod vs où est prise la météo) **conditionne** l’interprétation : certains couples France ne sont **pas** « turbine + météo au même site ».
- Un **skill faible ou négatif** **documenté** avec un protocole propre vaut mieux qu’un **gain** obtenu par **fuite** ou par comparaison **biaisée** (ex. persistance affichée seulement quand `patv_now` est présent).

---

## Next steps

- Réinjecter **vent SCADA** et/ou **puissance récente** lorsque la politique métier / contrat de données le permet (là où l’impact opérationnel est souvent le plus fort).
- **Multi-horizon** systématisé et **incertitude** (intervalles, quantiles, ensembles) pour l’usage réseau / trading.
- Poursuivre les expérimentations **France / USA** en gardant la **checklist anti-fuite** et une lecture **honnête** des métriques.

---

## Aperçu & flux

```mermaid
flowchart LR
  subgraph sources["Sources"]
    SDWPF[SDWPF SCADA + ERA5]
    FR[France ODRE + Open-Meteo]
    US[USA Wind Toolkit]
  end
  subgraph core["Pipeline"]
    FEAT[Features + découpage temporel]
    XGB[XGBoost]
    MET[MAE / RMSE / nMAE / skill]
  end
  subgraph out["Sorties"]
    PNG[Figures 01–06]
    MLF[MLflow optionnel]
  end
  SDWPF --> FEAT
  FR --> FEAT
  US --> FEAT
  FEAT --> XGB
  XGB --> MET
  MET --> PNG
  XGB --> MLF
```

---

## Aperçu visuel (optionnel)

Les PNG générés par `scripts/sdwpf_visualize.py` sont dans `reports/figures/` (tu peux en **committer** quelques-uns : voir `.gitignore`). Pour afficher des images **dans ce README**, copie-les dans `docs/assets/` (voir [`docs/assets/README.md`](docs/assets/README.md)) puis décommente ou adapte :

<!--
Prévision sur le jeu test (exemple) :
![Prévision vs vérité terrain](docs/assets/preview_forecast.png)

Tableau des métriques (exemple) :
![Métriques par turbine](docs/assets/preview_metrics.png)
-->

*Sans fichiers dans `docs/assets/`, cette section reste vide : le projet reste utilisable.*

---

## Structure du dépôt

```
├── README.md                 # Vous êtes ici
├── LICENSE                   # MIT
├── CONTRIBUTING.md           # Guide contributeur minimal
├── docs/
│   ├── GUIDE.md              # Parcours détaillé, glossaire, commandes, pièges
│   ├── PUBLICATION_GITHUB.md # git init, push, badges CI
│   ├── assets/               # Captures optionnelles pour le README
│   ├── DOMAINE_ET_PRATIQUES.md
│   ├── PLAN.md
│   └── INVENTAIRE_FICHIERS.md
├── .github/workflows/ci.yml  # Tests pytest (GitHub Actions)
├── src/sdwpf/                # Package Python (chargement, features, pipeline)
├── scripts/                  # Exécutables (voir ci-dessous)
├── tests/                    # pytest
├── data/                     # Données locales (souvent hors Git)
├── reports/                  # Benchmarks + reports/figures/*.png (selon .gitignore)
├── pyproject.toml
└── Dockerfile
```

**`scripts/`** :

| Script | Usage |
|--------|--------|
| `fetch_open_meteo_wind.py` | Télécharge la météo horaire Open-Meteo (France) |
| `download_wind_toolkit_nlr.py` | Télécharge Wind Toolkit (NLR / `.env`) |
| `sdwpf_explore.py` | Une turbine : métriques + importances |
| `sdwpf_benchmark.py` | Plusieurs horizons, CSV + Markdown |
| `sdwpf_visualize.py` | Figures PNG (série, nuage, test, importances, **KPI** `05_kpi_*`, **tableau métriques** `06_tableau_metriques_*`) |
| `clean_artifacts.py` | Supprime figures `*_h72_*`, dossier `mlruns/` ; option `--reports` |
| `sdwpf_walkforward.py` | Plusieurs plis test en fin de série (`--n-splits`, `--test-size`) ; moyenne / écart-type des MAE |

---

## Démarrage rapide

Depuis la **racine** du dépôt :

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,experiments]"
```

Sans la dépendance **MLflow** (`pip install -e ".[dev]"` uniquement), les scripts tournent normalement ; `--mlflow` invite à installer l’extra `experiments`.

Données SDWPF lourdes : placer sous `data/china/sdwpf/` (détail dans [`docs/GUIDE.md`](docs/GUIDE.md)).

### Exemples SDWPF (même répertoire racine)

```bash
python scripts/sdwpf_explore.py --meteo-mode --horizon-hours 24
python scripts/sdwpf_explore.py --meteo-mode --train-frac 0.55 --val-frac 0.15
python scripts/sdwpf_benchmark.py --meteo-mode
python scripts/sdwpf_visualize.py --meteo-mode --horizon-days 1
python scripts/sdwpf_visualize.py --meteo-mode --horizon-days 1 --turb-ids 1,2,3,5,8
python scripts/sdwpf_walkforward.py --meteo-mode --horizon-days 1 --n-splits 3 --test-size 5000
```

### Données France / USA

**Limite France** : la production **ODRE** est un agrégat **national** ; la météo **Open-Meteo** est un **point** fixe. Ce n’est **pas** co-localisé : on ne peut pas l’utiliser comme un couple « turbine + météo au même site » (détail dans [`docs/GUIDE.md`](docs/GUIDE.md)).

**USA** : `download_wind_toolkit_nlr.py` exige un **`.env`** avec `NLR_API_KEY` et `NLR_EMAIL` (voir [Secrets](#secrets-nlr)).

```bash
python scripts/fetch_open_meteo_wind.py
python scripts/download_wind_toolkit_nlr.py
```

### XGBoost et GPU

Les scripts acceptent **`--xgb-device`** : **`auto`** (défaut si rien n’est passé : variable **`SDWPF_XGB_DEVICE`**, sinon auto) = essai **`hist` + `device=cuda`**, puis repli **CPU** ; **`cuda`** / **`cpu`** ; **`cuda:N`** pour l’ordinal GPU ([doc GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html), [installation](https://xgboost.readthedocs.io/en/stable/install.html)).

**Paquets** : **`pip install xgboost`** installe le wheel **complet** (algorithme GPU inclus sur **Windows x86_64** et **Linux**, selon le [guide officiel](https://xgboost.readthedocs.io/en/stable/install.html)). **`pip install xgboost-cpu`** = variante **CPU seulement** (plus légère). **Windows** : **Visual C++ Redistributable** souvent requis. **Conda** : `conda install -c conda-forge py-xgboost=*=cuda*` (GPU) ou `=*=cpu*` (CPU).

**Matériel / CUDA** : pilotes **NVIDIA** à jour ; **CUDA 12** et **compute capability ≥ 5.0** pour le GPU. Vérification : `python -c "import xgboost as xgb; print(xgb.build_info())"`.

### Tests

```bash
pytest -q
```

Les mêmes tests s’exécutent sur **GitHub Actions** (`.github/workflows/ci.yml`) pour Python **3.10** et **3.12**.

### Docker

```bash
docker build -t sdwpf-forecast .
docker run --rm -v "%CD%/data:/app/data" sdwpf-forecast python scripts/sdwpf_explore.py --help
```

---

## Documentation complète

| Fichier | Contenu |
|--------|---------|
| **[`docs/GUIDE.md`](docs/GUIDE.md)** | **Entrée principale** : intention, données, glossaire SCADA / ERA5, CLI, MLflow, `.env` |
| [`docs/PUBLICATION_GITHUB.md`](docs/PUBLICATION_GITHUB.md) | `git init`, remote, push, badge CI |
| [`docs/DOMAINE_ET_PRATIQUES.md`](docs/DOMAINE_ET_PRATIQUES.md) | Domaine, sous-domaines, checklist |
| [`docs/PLAN.md`](docs/PLAN.md) | Vision, roadmap, todo |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribuer / tests |

---

## Évaluation (rappel)

- **Découpage temporel** : pas de mélange aléatoire ; option **`--val-frac`** pour **train | val | test** et early stopping XGBoost ; métriques **finales** sur le **test**.
- **Persistance** (`Patv(t)` → cible `Patv(t+h)`) : affichée **seulement** si `patv_now` est dans les features (sinon comparaison biaisée). **Naive** = moyenne de `y` sur le **train**, évaluée sur le **test**.

---

## Secrets (NLR)

Copier `.env.example` vers `.env` et renseigner `NLR_API_KEY` / `NLR_EMAIL` pour `scripts/download_wind_toolkit_nlr.py`.
