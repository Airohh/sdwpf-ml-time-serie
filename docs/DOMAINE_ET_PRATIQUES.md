# Domaine du projet et bonnes pratiques

Ce dépôt s’inscrit dans un domaine applicatif précis (prévision de **puissance éolienne**) et en adopte les exigences méthodologiques. Ce document fixe le **cadre** et la **checklist** que le code et la doc cherchent à respecter.

---

## Domaine principal

**Prévision opérationnelle ou quasi-opérationnelle de la production éolienne** (régression sur séries temporelles) : à partir d’un historique **SCADA** et éventuellement de **météo / réanalyse** (ex. ERA5), estimer la **puissance active** à un **horizon** donné (ici : pas de 10 min, horizon par défaut **1 jour** pour éviter un problème « trop facile » dominé par l’autocorrélation à très court terme).

---

## Sous-domaines (où le projet s’ancre)

| Sous-domaine | Rôle dans ce dépôt |
|--------------|--------------------|
| **Éolien / EnR** | Cible = `Patv` ; contraintes de pas de temps et d’horizon typiques du dispatch. |
| **Séries temporelles en apprentissage supervisé** | Fenêtres, lags, alignement `t` → `t+h` ; **aucun mélange aléatoire** des lignes temporelles pour l’évaluation. |
| **Validation temporelle** | Découpages **train | validation | test** dans l’ordre chronologique ; validation pour l’**early stopping** ; **test** pour le score final rapporté. |
| **Fuites de données (leakage)** | Mise en garde explicite sur les baselines qui utiliseraient une information non disponible dans les features (ex. persistance quand `Patv(t)` n’est pas dans le modèle). Purge des cibles qui chevauchent le bloc suivant. Mode **`--meteo-mode`** : uniquement calendrier de l’échéance + ERA5 ; **`--era5-at-target`** est une borne haute assumée, pas un modèle déployable. |
| **Features multimodales** | SCADA (vent, températures, puissance) + **réanalyse** ERA5 co-localisée dans le jeu SDWPF ; branches **France / USA** pour illustrer **sources** et **limites** (non co-localisé, autre type de donnée). |
| **Interprétabilité légère** | Importances de features (XGBoost), scripts de visualisation. |
| **Reproductibilité & traçabilité** | `pyproject.toml` comme source des dépendances, `requirements-lock.txt` pour les versions validées ; graine du modèle centralisée ; XGBoost sur CPU par défaut ; **MLflow** en dépendance **optionnelle** (extra `experiments`). |

---

## Bonnes pratiques attendues dans ce domaine

Les points suivants guident les choix du dépôt (et ce qu’il reste à renforcer plus tard).

1. **Temporalité** : tout découpage d’apprentissage et d’évaluation respecte l’ordre du temps ; pas de validation strictement « au hasard » sur la série.
2. **Baselines alignées sur l’information** : comparer le modèle à une baseline qui n’utilise **que** ce que le modèle peut voir : moyenne de la cible sur le **train**, **climatologie** par heure de l’échéance, **courbe de puissance** sur le vent ERA5 à l’échéance quand le modèle y a accès ; persistance seulement si `Patv(t)` est une feature.
3. **Horizons explicites** : horizon par défaut de **144** pas pour coller à une prévision « jour et au-delà », pas seulement quelques pas devant. Les variables calendaires décrivent l’instant prévu *t* + *h*. Lags et cible se calculent **sur le temps** (`t ± k × 10 min`), pas par ligne : une ligne manquante ne doit pas décaler la cible (voir `add_features`).
4. **Séparation val / test** : early stopping sur la validation ; **ne pas** optimiser sur le test. Purge : une ligne du train ou de la validation dont la cible *t* + *h* tombe dans le bloc suivant est retirée.
5. **Reproductibilité** : bornes dans `pyproject.toml`, versions exactes validées dans `requirements-lock.txt` ; `random_state` fixé pour XGBoost (constante dans `sdwpf.constants`) ; CPU par défaut (CPU et GPU ne donnent pas les mêmes arbres).
6. **Données et secrets** : gros fichiers et dossiers `data/` hors Git ; clés API uniquement dans `.env` (voir `.env.example`).
7. **Qualité des mesures** : exclure les valeurs non physiques évidentes sur SCADA (vent négatif, directions hors plage, `inf`), ramener `Patv < 0` (auxiliaires à l’arrêt) à 0 comme dans la KDD Cup 2022, et filtrer ERA5 incohérent (`Wspd_w` négatif) avant modélisation — voir `sanitize_scada_for_forecasting` et `merge_scada_era5`.
8. **Tests ciblés** : logique de split, d’alignement temporel, de purge et de nettoyage couverte par `pytest` sans dépendre des CSV massifs ; chaque script est exécuté de bout en bout sur un faux SDWPF en CI.
9. **Artefacts jetables** : rapports `reports/*.csv|md` et `mlruns/` produits localement, non traités comme source de vérité du dépôt.
10. **Variables qui n’extrapolent pas** : pas de jour de l’année quand les données couvrent moins d’un an (le test tombe sur des jours jamais vus).

---

## Ce que ce projet ne prétend pas couvrir (volontairement)

- Modèles de séquence profonds (LSTM / Transformers) : hors scope pour garder un pipeline **léger et auditable**.
- Prévision météo à l’échéance (nowcasting / NWP downscaling) : les champs ERA5 du jeu sont utilisés **tels quels**. `--era5-at-target` lit la réanalyse à l’échéance comme une prévision parfaite pour mesurer le plafond ; une vraie chaîne opérationnelle utiliserait des **prévisions** météo au pas concerné.

---

## Liens dans le dépôt

- **Guide technique détaillé** : [`GUIDE.md`](GUIDE.md)
- **Entrée courte** : [`README.md`](../README.md)
