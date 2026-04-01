# Contexte projet — à coller dans ChatGPT

> **Usage :** copie-colle **tout ce fichier** dans une nouvelle conversation ChatGPT pour qu’il comprenne le repo, ce que j’ai fait et où j’en suis.

---

## Projet

- **Nom / thème :** prévision de série temporelle sur données **éoliennes SDWPF** (SCADA + ERA5), cible **Patv** (puissance active) avec horizon **J+1** (144 pas de **10 minutes**).
- **Racine du projet (Windows) :** `C:\Users\Utilisateur\Desktop\Data Scientist - ML  Time Series` (attention : parfois noté avec **deux espaces** après `ML` selon le dossier réel).
- **Stack :** Python, **pandas**, **numpy**, **matplotlib** (backend Agg), **XGBoost** pour la régression.

## Structure utile

- `src/sdwpf/` : chargement CSV, fusion SCADA/ERA5, **features**, `train_and_evaluate`, découpages temporels train/val/test.
- `scripts/sdwpf_visualize.py` : charge les données, entraîne **un XGBoost par TurbID**, exporte des PNG dans `reports/figures/` :
  - `01_*` série Patv + vent moyennée(s),
  - `02_*` scatter Wspd vs Patv,
  - `03_*` vérité vs prédiction sur **jeu test**,
  - `04_*` importances features,
  - `05_*` tableau **KPI** (MAE, RMSE, gains vs baselines) + barres MAE.
- `scripts/reproduce_meteo_figures.py` : rejoue mes scénarios (presets `single`, `multi5`, `multi20`, `multi100`, `all`) avec `--meteo-mode` et `--horizon-days 1`.
- **GPU XGBoost :** `--xgb-device auto` (défaut env `SDWPF_XGB_DEVICE`), ou `cuda` / `cpu`. Doc install : [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html). Les sorties console « device mismatch » sklearn+NumPy sont **filtrées** dans `evaluate_on_indices` quand on n’est pas en `cpu` (bruit uniquement).

## Ce que j’ai fait concrètement

1. Figures en **mode météo** (`--meteo-mode` : features type calendrier + météo ERA5, **sans** `patv_now` ni lags Patv — le code met les lags à 0 dans ce mode).
2. **Horizon :** `--horizon-days 1` → **144 pas** (~24 h).
3. **Turbines :**
   - une turbine (ex. **n°1**) ;
   - groupe **1,2,3,5,8** (moyennes / agrégation des graphiques) ;
   - **20 turbines** 1–20 (bon compromis rapidité / diversité) ;
   - **100 turbines** TurbID **1 à 100** (long : 100 entraînements).
4. Découpage par défaut du script : **train 70 %** / **test 30 %** (pas de `--val-frac` dans mes runs principaux).

## Commandes équivalentes

```text
# Une turbine (GPU si dispo)
python scripts/sdwpf_visualize.py --meteo-mode --horizon-days 1 --turb-ids 1 --xgb-device cuda

# Cinq turbines
python scripts/sdwpf_visualize.py --meteo-mode --horizon-days 1 --turb-ids 1,2,3,5,8 --xgb-device cuda

# Vingt turbines (PowerShell)
python scripts/sdwpf_visualize.py --meteo-mode --horizon-days 1 --turb-ids $((1..20) -join ',') --xgb-device cuda

# Cent turbines (PowerShell : générer la liste)
python scripts/sdwpf_visualize.py --meteo-mode --horizon-days 1 --turb-ids $((1..100) -join ',') --xgb-device cuda

# Reproduction (preset all = single → multi5 → multi20 → multi100)
python scripts/reproduce_meteo_figures.py --preset multi20
python scripts/reproduce_meteo_figures.py --preset multi20 -- --xgb-device cuda
python scripts/reproduce_meteo_figures.py --preset multi100
```

**Terminal :** ne pas coller des **lignes de journal** (warnings, « Turbine 1… ») dans PowerShell : ce ne sont pas des commandes.

## Résultats chiffrés (100 turbines, même protocole)

Agrégat **moyenne des MAE par turbine** sur le **jeu test** (à peu près) :

- **MAE XGBoost (test)** ≈ **353.5**
- **RMSE XGBoost (test)** ≈ **459.4**
- **MAE baseline « moyenne du train sur toute la cible » (test)** ≈ **346.4**
- Donc en moyenne XGB est **légèrement moins bon** que cette naive (**~ −2 %** de « gain »).
- **Baseline persistance** : pas calculée en mode météo-only (pas de colonne `patv_now` dans les features).
- **~11 805 lignes** de modélisation par turbine en moyenne, **10 features**.

Les graphiques **03** montrent la **moyenne par instant** des vérités / prédictions sur le test — ce n’est pas la même chose que la moyenne des MAE par turbine.

## Fichiers PNG générés (exemples de noms)

- `01_series_multi_n20_1-20_h144_meteo.png`
- `03_test_forecast_multi_n100_1-100_h144_meteo.png`
- `05_kpi_performance_multi_n100_1-100_h144_meteo.png`

`h144` = 144 pas ; `meteo` = mode météo.

## Question ouverte pour toi (ChatGPT)

Comment **améliorer** le modèle dans ce cadre (sans forcément tout recoder) : réintroduire **Patv / lags** si le use case le permet, **lags météo**, **validation + early stopping**, horizons plus courts, **walk-forward**, modèle **multi-turbine** partagé, etc.

---

*Fin du contexte — date indicative de la conversation : avril 2026.*
