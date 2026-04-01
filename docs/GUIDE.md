# Guide technique — prévision éolienne & SDWPF

Document de synthèse : objectifs, choix, fichiers, faux pas et états actuels. Les étapes « mineures » (conventions shell, chemins, détails d’API) sont incluses volontairement pour pouvoir tout retrouver plus tard ou en entretien.

**Entrée générale du dépôt :** [`README.md`](../README.md) à la racine. **Cadre métier et pratiques :** [`DOMAINE_ET_PRATIQUES.md`](DOMAINE_ET_PRATIQUES.md).

## Arborescence (après réorganisation)

| Dossier / fichier | Rôle |
|-------------------|------|
| `docs/GUIDE.md` | Ce document |
| `src/sdwpf/` | Package Python installable (`pip install -e ".[dev]"`) |
| `scripts/` | Téléchargements données + exploration / benchmark / figures SDWPF |
| `tests/` | Tests `pytest` |
| `data/` | Données locales (souvent ignorées par Git si lourdes) |
| `reports/` | Sorties CSV/MD de benchmark ; `reports/figures/` pour les PNG |
| Racine | `README.md`, `pyproject.toml`, `Dockerfile`, `.env.example` |

---

## 1. Point de départ : intention du projet

- **Contexte visé** : préparation type poste **Data Scientist** sur **séries temporelles** et énergie renouvelable (vent / solaire).
- **Techniques cibles évoquées** : **XGBoost**, forêts aléatoires, rigueur « time series » (splits temporels, pas de fuite de données).
- **Mise en production (piste)** : **Docker**, **AWS SageMaker**, **MLflow**, et à terme une **petite bibliothèque** interne pour standardiser entraînement / évaluation.
- **Données prévues en parallèle (branches)** :
  - **Chine** : jeu **SDWPF** (SCADA haute fréquence, cible **Patv** — puissance active), priorité pour une démo **co-localisée** (météo + turbine).
  - **France** : production **ODRE** + météo **Open-Meteo** (un point).
  - **USA** : **Wind Toolkit** via l’API **NLR** (ex-NREL, ressource vent / météo, pas de puissance éolienne « réelle » dans l’export utilisé).

---

## 2. Socle Python du dépôt

- **`pyproject.toml`** : paquet installable **`sdwpf`** (dossier **`src/sdwpf/`**), extra **`[dev]`** (**pytest**), extra **`[experiments]`** (**MLflow**). Commandes depuis la racine : **`pip install -e ".[dev]"`** ou avec suivi **`pip install -e ".[dev,experiments]"`**.
- **`README.md`** (racine) : entrée courte + liens vers ce guide.
- **Dépendances** : **`numpy`**, **`scikit-learn`**, **`xgboost`**, **`matplotlib`** ; **MLflow** uniquement si l’extra **`experiments`** est installée.
- **Tests** : **`tests/test_sdwpf_features.py`** (`pytest -q`), sans gros CSV.
- **Pratique** : environnement virtuel (`.venv` / `venv`) recommandé ; **`.gitignore`** exclut aussi **`*.egg-info/`**, **`dist/`**.

---

## 3. Configuration secrète : `.env` vs `.env.example`

- **But** : ne jamais commiter de clés API ni d’email personnel en dur dans le code ou sur un dépôt public.
- **`.env.example`** : fichier **modèle** avec des clés **vides** (`NLR_API_KEY=`, `NLR_EMAIL=`) et un rappel de copier vers `.env`. Si des secrets ont été mis par erreur dans l’exemple, ils ont été **retirés** et remplacés par des placeholders — il faut alors **renouveler la clé** côté fournisseur si elle a fuité (chat, commit, etc.).
- **`.env`** : fichier **local** listé dans **`.gitignore`**, c’est là que l’on renseigne la vraie clé et l’email pour NLR.
- **`python-dotenv`** : les scripts qui en ont besoin chargent automatiquement `.env` à la racine du projet quand c’est implémenté (ex. script Wind Toolkit).

---

## 4. Règles Git : quoi versionner ou non

- Fichier **`.gitignore`** à la racine :
  - **`.env`**, caches Python (`__pycache__`, `*.pyc`), environnements virtuels.
  - Dossiers de données **lourdes ou téléchargées** : **`data/usa/nlr_wtk/`**, **`data/china/sdwpf/`** (le dépôt reste léger ; les données vivent en local ou ailleurs).

---

## 5. Branche France — météo Open-Meteo

- **Script** : **`scripts/fetch_open_meteo_wind.py`**
- **API** : archive Open-Meteo, requêtes découpées **par année** pour limiter la taille des réponses et gérer les limites de l’API.
- **Point géographique** : latitude **50.95**, longitude **1.85** (nord France / point unique pour la démo).
- **Variables horaires demandées** : vent à 10 m / 100 m, directions, humidité, pression, nébulosité, rafales, etc. (liste dans le script, variable `HOURLY`).
- **Fenêtre temporelle** : de **2020-01-01** jusqu’à **`2025-11-30`** (borne haute alignée avec la disponibilité / logique du jeu ODRE côté calendrier).
- **Sortie** : dossier **`data/france/open_meteo/`**, fichier principal  
  **`open_meteo_wind_hourly_2020_2025.csv`**, avec éventuellement des fichiers annexes (méta, échantillon JSON, en-têtes) générés lors des fetch.
- **Détail « insignifiant » mais utile** : après réorganisation des dossiers, le script a été **mis à jour** pour écrire explicitement sous `data/france/open_meteo/` (plus sous un ancien chemin plat).

---

## 6. Branche France — production ODRE

- **Données** : fichier CSV national placé sous **`data/france/odre/`**  
  (`courbes-de-production-mensuelles-eolien-solaire-complement-de-remuneration.csv` — nom long typique des jeux open data).
- **Nature** : agrégat **national** (courbes / séries de production), **pas** la production d’une éolienne au même endroit que le point Open-Meteo.
- **Conséquence** : tentative de **fusion temporelle** « production + météo au même endroit » **abandonnée** pour ce couple de jeux : ce n’est pas co-localisé géographiquement.
- **Problème technique rencontré au passage** : en travaillant sur des séries vent/production avec dates horaires, des **doublons d’horodatage** (ex. passage à l’heure d’été, **02:00** répété) ont provoqué des erreurs de merge (`ValueError` / index dupliqué). La leçon est notée plutôt que de forcer un merge incohérent géographiquement.

---

## 7. Branche USA — Wind Toolkit (NLR)

- **Script** : **`scripts/download_wind_toolkit_nlr.py`**
- **Documentation API** : domaine **developer.nlr.gov** (migration / continuité par rapport à l’ancienne documentation NREL).
- **Authentification** : **`NLR_API_KEY`** et **`NLR_EMAIL`** (souvent requis par ce type d’API).
- **Comportement API** (rappel dans le script) : une requête typique = **un point WGS84** et **une année** ; la réponse peut être le **corps CSV** directement.
- **Sortie par défaut** : **`data/usa/nlr_wtk/`** (dossier ignoré par Git car volumineux).
- **Piège rencontré** : l’API renvoie une erreur si les variables ne sont pas dans **`.env`** — avoir rempli **seulement** `.env.example` ne suffit pas ; il faut un vrai **`.env`** (ou exporter les variables dans le shell).
- **Shell Windows** : sur **PowerShell**, enchaîner des commandes avec **`&&`** peut poser problème selon la version ; préférer **`;`** ou des commandes séparées (détail de confort au quotidien).

---

## 8. Branche Chine — SDWPF (jeu principal « co-localisé »)

- **Emplacement local** : **`data/china/sdwpf/`** (également dans `.gitignore` si gros volume).
- **Fichier SCADA principal** : souvent sous la forme  
  **`data/china/sdwpf/sdwpf_245days_v1.csv/sdwpf_245days_v1.csv`** (dossier dont le nom se termine par `.csv`, contenant le vrai fichier — artefact de dézip / arborescence d’origine).
- **Colonnes utiles (aperçu)** : `TurbID`, `Day`, `Tmstamp`, variables vent / température / orientations, et surtout **`Patv`** (puissance active).
- **`Day` / `Tmstamp`** : index **relatif** au défi KDD / release (pas une date civile officielle dans le README court) ; pour la **modélisation** on construit un **datetime monotone** (ancre arbitraire + jour + heure) afin de faire un **split temporel** cohérent.
- **README dataset** : fichier type **`README (1)`** dans le dossier SDWPF — liste les sous-jeux :
  1. `sdwpf_245days_v1` (SCADA challenge),
  2. positions des turbines,
  3. `final_phase_test` (phase finale compétition),
  4. **`sdwpf_weather_v2`** (météo type **ERA5** en plus du SCADA).
- **Pistes suivantes** (pas tout fait dans ce doc technique) : joindre météo **v2**, horizons de prédiction plus longs, modèles sans puissance courante pour coller à un cas opérationnel, MLflow, Docker, etc.

### Glossaire des variables (SCADA, features, ERA5)

*Pas 10 minutes entre les lignes ; un **lag** de *k* = *k* pas en arrière = *k*×10 min.*

**Identifiants et temps**

| Nom | Rôle |
|-----|------|
| **`TurbID`** | Numéro de l’éolienne dans le parc (chaque machine a sa série). |
| **`Day`** | Indice de jour relatif au jeu (1…N), pas une date civile dans le CSV v1. |
| **`Tmstamp`** | Heure du pas (ex. `00:10`, `00:20`) sur la grille 10 min. |
| **`datetime`** | Construit dans le code (ancre **`2020-05-01`**) pour trier, fusionner et couper dans le temps. |

**Mesures SCADA (sur la turbine / au site)**

| Nom | Signification usuelle |
|-----|------------------------|
| **`Patv`** | **Puissance active** produite (variable cible la plus souvent), en unité du fichier. |
| **`Wspd`** | **Vitesse du vent** vue par le SCADA (souvent au moyeu ou proche). |
| **`Wdir`** | **Direction du vent** (angles / conventions du fabricant ou du jeu). |
| **`Etmp`** | **Température « extérieure »** (capteur ou canal SCADA du dossier). |
| **`Itmp`** | **Température « intérieure »** (ex. nacelle / équipement). |
| **`Ndir`**, **`Pab1`–`Pab3`**, **`Prtv`** | Autres signaux (orientation / angles de pale, etc.) — utiles surtout en analyse fine ; le pipeline minimal du projet se concentre sur vent + températures + **`Patv`**. |

**Features dérivées pour le modèle (noms dans XGBoost / graphiques)**

| Nom | Signification |
|-----|----------------|
| **`patv_now`** | **`Patv`** à l’instant courant *t* (retirée en **`--meteo-mode`** ou **`--no-patv-now`**). |
| **`patv_lag*k*** | **`Patv`** à *t* − *k* pas (historique de puissance). |
| **`wspd_lag*k*** | **`Wspd`** à *t* − *k* pas (historique de vent mesuré). |
| **`wdir`**, **`etmp`**, **`itmp`** | Valeurs **à l’instant *t***, sans suffixe « lag ». |
| **`hour_sin`**, **`hour_cos`**, **`doy_sin`**, **`doy_cos`** | Encodage cyclique de l’heure et du jour dans l’année (uniquement en **`--meteo-mode`**). |

**Champs météo « ERA5 » (fichier `sdwpf_weather_v2`, réanalyse sur grille)**

Ces colonnes viennent du couple jeu / ERA5 ; elles décrivent l’atmosphère au voisinage du site, pas la mesure locale turbine équivalente à **`Wspd`**.

| Nom | Signification courte |
|-----|------------------------|
| **`T2m`** | Température de l’air à **2 m**. |
| **`Sp`** | Pression au niveau de la surface (surface pressure). |
| **`RelH`** | Humidité relative (0–1 ou % selon normalisation du fichier). |
| **`Wspd_w`** | Vitesse du vent issue de la **réanalyse** (souvent proche du site mais pas identique au SCADA). |
| **`Wdir_w`** | Direction du vent côté réanalyse. |
| **`Tp`** | Champ **Tp** livré dans ce CSV météo (variable ERA5 / dérivée selon l’extraction Baidu — à traiter comme info thermodynamique / précipitation potentielle selon la doc du fournisseur). |

**Cible (variable prédite)**

| Nom | Rôle |
|-----|------|
| **`y_target`** | **`Patv`** à l’instant *t* + *h* pas (horizon *h* : ex. **144** pas = 1 j à pas 10 min). |

Les libellés **français** sur le graphique d’importances (`sdwpf_visualize.py`) reprennent ce glossaire de façon raccourcie.

---

## 9. Package + scripts SDWPF

### Bibliothèque **`src/sdwpf/`**

Code réutilisable (ajout de **`src/`** au `PYTHONPATH` par les scripts ; le **`Dockerfile`** copie **`src/`**).

- **`constants.py`** : ancre **`2020-05-01`**, colonnes ERA5, **`STEPS_PER_HOUR`** (6), **`STEPS_PER_DAY`** (144).
- **`paths.py`** : racine du dépôt, chemins par défaut SCADA / météo.
- **`data.py`** : chargement chunké turbine, ERA5, merge ; **nettoyage** : `sanitize_scada_for_forecasting` retire `inf`, puissance ou vent **négatifs**, `Wdir` hors **`[0, 360]`** si renseignée ; ERA5 : exclut `Wspd_w` **négatif** ; fusion : `inf` traités comme manquants puis lignes ERA5 incomplètes exclues.
- **`features.py`** : lags, cible **`y_target`**, splits temporels **`temporal_split`** / **`temporal_split_train_val_test`**, **`walk_forward_indices`** (walk-forward expansif : listes d’indices train/test contigus), **`resolve_horizon_steps`**.
- **`pipeline.py`** : **`train_and_evaluate`** → **`SdwpfRunResult`** (MAE/RMSE **naive**, **persistance** si `patv_now` ∈ features, XGBoost val/test) ; **`load_frame_for_run`** ; **`maybe_log_mlflow`**.

### Script **`scripts/sdwpf_explore.py`**
- **Rôle** :
  - **Charger une seule turbine** (`TurbID`, défaut **1**) en **parcourant le CSV par chunks** (fichier très grand, millions de lignes au total pour toutes les machines).
  - Construire **`datetime`** à partir de **`Day`** + **`Tmstamp`** avec **ancre calendaire `2020-05-01`** (alignée sur **`sdwpf_weather_v2`**, où `Tmstamp` est une date/heure absolue commençant le **2020-05-01**). Trier, **dédoublonner**, convertir les colonnes numériques, **retirer les lignes** sans **`Patv`** ou **`Wspd`**.
  - Construire des **features** : lags de **`Patv`** et **`Wspd`** (ex. 6 pas = 1 h à pas 10 min), + **`wdir`**, **`etmp`**, **`itmp`**.
  - **Horizon minimal** : **`train_and_evaluate`** impose **`horizon_steps` ≥ **144** (1 jour à pas 10 min). Si aucun `--horizon` / `--horizon-hours` / `--horizon-days` n’est passé aux scripts, l’horizon par défaut est **144 pas**. **`--horizon`** (pas), ou **`--horizon-hours`** / **`--horizon-days`** (1 h = 6 pas, 1 j = 144 pas). **`--meteo-mode`** : raccourci **`--era5 --no-patv-now --no-patv-lags`**, et le modèle ne voit plus que **calendrier (sin/cos heure + jour d’année) + colonnes ERA5** — **aucun** signal SCADA (**`Wspd`**, **`Wdir`**, **`Etmp`**, **`Itmp`**, ni leurs lags). Pour un scénario intermédiaire « vent / capteurs site + ERA5 mais sans **`Patv`** dans les X », utiliser **`--era5 --no-patv-now --no-patv-lags`** **sans** **`--meteo-mode`** (lags **`Wspd`**, **`wdir`**, **`etmp`**, **`itmp`**, ERA5).
  - **XGBoost** sur la même cible **`y_target` = `Patv(t+h)`** avec split **temporel**. Par défaut : **train** puis **test** (ex. 70 % / 30 % via `--train-frac`). Option **`--val-frac`** : découpage **train | validation | test** dans l’ordre chronologique (**train_frac + val_frac inférieurs à 1** en cumul) ; la validation sert à l’**early stopping** XGBoost (et les métriques **val** sont affichées), le **test** reste la seule partie jamais vue à l’entraînement pour le **score final**. **Baseline naive** : moyenne des **`y_target`** sur le **train**, évaluée sur le **test**. **Baseline persistance** (prédire **`Patv(t)`** pour la cible **`Patv(t+h)`**) : affichée **uniquement** lorsque **`patv_now`** fait partie des features — sinon la comparaison serait biaisée (ex. **`--meteo-mode`** sans `Patv`). Le benchmark rapporte aussi **`gain_vs_persistence_mae_pct`** quand la persistance est définie. **Walk-forward** : **`walk_forward_indices`** + script **`scripts/sdwpf_walkforward.py`** (résume **moyenne / écart-type** des MAE test sur plusieurs plis). En interne : **`build_modeling_frame`**, **`evaluate_on_indices`** (indices train/val/test explicites) depuis **`sdwpf.pipeline`**.
- **Détail méthodo** : avec **`patv_now`** (= `Patv(t)`), le modèle connaît l’instantané de la production. **`--no-patv-now`** retire `patv_now`. **`--no-patv-lags`** retire tous les **`patv_lag*`** — sauf **`--meteo-mode`**, où il ne reste **ni** historique **ni** mesures SCADA vent/temp, seulement **temps + ERA5**.
- **Benchmark multi-horizons** : les pas par défaut sont **144, 288, 432, 864** (1 j à 6 j) ; tout pas strictement inférieur à **144** est refusé.
- **Fusion ERA5 (phase 2)** : avec le flag **`--era5`**, le script charge le CSV **`wtb2005_2104_full_new.csv`** (sous `sdwpf_weather_v2/sdwpf_weather/`, sinon chemin via **`--weather-csv`**), filtre la même **`TurbID`**, et fait une **jointure interne** sur **`datetime`** avec les colonnes reanalysis **`T2m`**, **`Sp`**, **`RelH`**, **`Wspd_w`**, **`Wdir_w`**, **`Tp`**. Ces colonnes sont ajoutées aux features XGBoost. Sur un pas de 10 min avec **`patv_now`**, l’apport peut être **faible voire nuisible** (colinéarité, bruit) ; l’intérêt pédagogique est surtout le **pipeline aligné** et les importances (**`Wspd_w`**, **`Wdir_w`**, etc.).

**Commande typique** (à lancer depuis la racine du projet) :

```text
python scripts/sdwpf_explore.py
```

Avec reanalysis :

```text
python scripts/sdwpf_explore.py --era5
```

Options utiles : `--turb-id`, `--csv`, `--train-frac`, **`--val-frac`**, **`--early-stopping-rounds`**, `--chunksize`, `--era5`, `--weather-csv`, **`--horizon`**, **`--horizon-hours`**, **`--horizon-days`**, **`--meteo-mode`**, **`--no-patv-now`**, **`--no-patv-lags`**.

Sans **`Patv`** dans les features en **`--meteo-mode`** (temps calendaire + ERA5 seulement) :

```text
python scripts/sdwpf_explore.py --meteo-mode
```

Exemple **24 h** en mode météo :

```text
python scripts/sdwpf_explore.py --meteo-mode --horizon-hours 24
```

### Benchmark multi-horizons **`scripts/sdwpf_benchmark.py`**

Enchaîne plusieurs horizons (défaut : **1 j → 6 j** en pas : **`144,288,432,864`** ; minimum **144**), écrit **`reports/sdwpf_benchmark_*.csv`** et **`.md`** (dossier **`reports/`** ; fichiers générés souvent ignorés par Git).

```text
python scripts/sdwpf_benchmark.py --meteo-mode
```

### Visuels (PNG)

- Script **`scripts/sdwpf_visualize.py`** : enchaîne chargement, entraînement (avec **`return_test_predictions`** côté package) et export **matplotlib** (backend **Agg**) dans **`reports/figures/`** :
  - série **Patv** + **Wspd** (extrait fin de série) ;
  - nuage **Wspd vs Patv** ;
  - sur le **jeu test** : cible **Patv(t+h)** vs **XGBoost** ;
  - barres des **importances** (top 10) ;
  - figure **`05_kpi_performance_*.png`** : tableau des **MAE / RMSE** (lignes de base, XGB val/test), **gains en %**, barres **MAE** comparatives et encadré de lecture rapide.
- Options : **`--horizon-hours`**, **`--horizon-days`**, **`--meteo-mode`**, **`--meteo-max-lag`** (retards 1..k sur chaque colonne **ERA5** ; défaut **12** ; **0** = pas de lags mais **Wspd_w³** et **sin/cos** direction), **`--xgb-device`** (`auto`, `cpu`, `cuda`, **`cuda:N`** ; [GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html), [installation](https://xgboost.readthedocs.io/en/stable/install.html) : wheel **`xgboost`** = GPU possible sur Win/Linux, **`xgboost-cpu`** = CPU seul), **`--series-points`**, **`--test-plot-points`**, **`--out-dir`**, **`--no-kpi-dashboard`** (désactive la figure 05). L’API **`train_and_evaluate`** accepte aussi des horizons **&lt; 1 jour** en pas (ex. **36** = 6 h). Variable d’environnement **`SDWPF_XGB_DEVICE`** si le flag est omis.

```text
python scripts/sdwpf_visualize.py --meteo-mode --horizon-days 1
python scripts/sdwpf_visualize.py --meteo-mode --horizon-days 1 --turb-ids 1,2,3,5,8
```

Plusieurs **TurbID** : **moyenne** (et écart-type) des séries par instant, **moyenne** des prédictions sur le test par **datetime** (longueurs de test différentes gérées), **importances** XGBoost **moyennées** sur les turbines ; tableau **MAE** par turbine en fin de run.

Les PNG ne s’ouvrent pas dans une fenêtre : **matplotlib Agg** écrit les fichiers seulement. Le chemin complet est affiché en fin de script ; **`--open-folder`** (Windows) ouvre **`reports/figures/`** dans l’Explorateur. Les **`.png`** y sont **visibles dans l’IDE** (exception `.gitignore`).

### MLflow (optionnel)

- Installer l’extra : **`pip install -e ".[dev,experiments]"`** (ou **`[experiments]`** seul sur un environnement déjà prêt).
- Flags **`--mlflow`**, **`--mlflow-experiment`** (défaut : `sdwpf`), **`--mlflow-run-name`**.
- URI de tracking : variable d’environnement **`MLFLOW_TRACKING_URI`** si défini ; sinon stockage local **`mlruns/`** à la racine du projet (URL fichier `file:///…`), **ignoré par Git**.
- Chaque run enregistre des **paramètres** (turbine, horizon, `era5`, flags Patv, tailles) et, si XGBoost est disponible, les **métriques** test `xgboost_test_mae`, `xgboost_test_rmse`. Artefacts : `feature_importance_top.json`, `feature_columns.json`.

```text
python scripts/sdwpf_explore.py --mlflow --mlflow-run-name baseline_h1
```

### Docker

- Fichier **`Dockerfile`** à la racine : image **Python 3.11 slim**, installation **`pip install .`** depuis **`pyproject.toml`**, copie **`src/`** et **`scripts/`**. Données montées en volume (**`-v …:/app/data`**). Commande par défaut : **`--help`**. Pour **MLflow** dans l’image, ajouter **`experiments`** dans la ligne `pip install` du Dockerfile.

---

## 10. Organisation des dossiers « data » (état de réflexe)

- **`data/france/open_meteo/`** — sortie Open-Meteo.
- **`data/france/odre/`** — CSV production nationale ODRE.
- **`data/usa/nlr_wtk/`** — exports Wind Toolkit NLR.
- **`data/china/sdwpf/`** — arborescence SDWPF complète (SCADA, weather v2, final phase test, README, etc.).

Les scripts ont été **alignés** sur cette arborescence pour éviter des chemins obsolètes.

---

## 11. Où vivent les fichiers

- **Racine** : configuration (`pyproject.toml`, `Dockerfile`), **`.env`** / **`.env.example`**, **`.gitignore`**, **`README.md`**.
- **`scripts/`** : **`fetch_open_meteo_wind.py`**, **`download_wind_toolkit_nlr.py`**, **`sdwpf_explore.py`**, **`sdwpf_benchmark.py`**, **`sdwpf_visualize.py`**, **`sdwpf_walkforward.py`**, **`clean_artifacts.py`**.
- **`src/sdwpf/`** : bibliothèque SDWPF (chargement, features, entraînement).
- **`docs/`** : ce guide, **`DOMAINE_ET_PRATIQUES.md`**, **`INVENTAIRE_FICHIERS.md`**.
- **`reports/`** : benchmarks ; **`reports/figures/`** pour les PNG.

---

## 12. Sécurité (rappel)

- Ne pas commiter **`.env`**, ne pas coller de **clé** dans le chat ou dans `.env.example`.
- Si une clé a été exposée : la **révoquer / régénérer** sur le portail **developer.nlr.gov** et mettre à jour **uniquement** le `.env` local.

---

## 13. Suite logique du projet (à faire / à prolonger)

- **Fait** : package **`src/sdwpf`** ; **`--horizon-hours` / `--horizon-days` / `--meteo-mode`** ; **`sdwpf_benchmark.py`** ; MLflow ; Docker.
- **À creuser** : prévisions météo **à l’échéance** (lead time) pour horizons **très longs** ; autres modèles / quantiles ; **SageMaker** si cible AWS.

---

*Dernière mise à jour : réorganisation `docs/` + `scripts/`, train/val/test, benchmark multi-horizons ≥ 1 j.*
