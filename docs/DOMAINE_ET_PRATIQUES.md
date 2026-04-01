# Domaine du projet et bonnes pratiques

Ce dépôt s’inscrit dans un domaine applicatif précis (prévision de **puissance éolienne**) et en adopte les exigences méthodologiques. Ce document fixe le **cadre** et la **checklist** que le code et la doc cherchent à respecter.

---

## Domaine principal

**Prévision opérationnelle ou quasi-opérationnelle de la production éolienne** (régression sur séries temporelles) : à partir d’un historique **SCADA** et éventuellement de **météo / réanalyse** (ex. ERA5), estimer la **puissance active** à un **horizon** donné (ici : pas de 10 min, horizon minimal **≥ 1 jour** pour éviter un problème « trop facile » dominé par l’autocorrélation à très court terme).

---

## Sous-domaines (où le projet s’ancre)

| Sous-domaine | Rôle dans ce dépôt |
|--------------|--------------------|
| **Éolien / EnR** | Cible = `Patv` ; contraintes de pas de temps et d’horizon typiques du dispatch. |
| **Séries temporelles en apprentissage supervisé** | Fenêtres, lags, alignement `t` → `t+h` ; **aucun mélange aléatoire** des lignes temporelles pour l’évaluation. |
| **Validation temporelle** | Découpages **train | validation | test** dans l’ordre chronologique ; validation pour l’**early stopping** ; **test** pour le score final rapporté. |
| **Fuites de données (leakage)** | Mise en garde explicite sur les baselines qui utiliseraient une information non disponible dans les features (ex. persistance quand `Patv(t)` n’est pas dans le modèle). Mode **`--meteo-mode`** : uniquement calendrier + ERA5. |
| **Features multimodales** | SCADA (vent, températures, puissance) + **réanalyse** ERA5 co-localisée dans le jeu SDWPF ; branches **France / USA** pour illustrer **sources** et **limites** (non co-localisé, autre type de donnée). |
| **Interprétabilité légère** | Importances de features (XGBoost), scripts de visualisation. |
| **Reproductibilité & traçabilité** | `pyproject.toml` comme source des dépendances ; graine du modèle centralisée ; **MLflow** en dépendance **optionnelle** (extra `experiments`). |

---

## Bonnes pratiques attendues dans ce domaine

Les points suivants guident les choix du dépôt (et ce qu’il reste à renforcer plus tard).

1. **Temporalité** : tout découpage d’apprentissage et d’évaluation respecte l’ordre du temps ; pas de validation strictement « au hasard » sur la série.
2. **Baselines alignées sur l’information** : comparer le modèle à une baseline qui n’utilise **que** ce que le modèle peut voir (ex. moyenne de la cible sur le **train** projetée sur le test, lorsque la persistance s’avère biaisée).
3. **Horizons explicites** : imposer un minimum d’horizon (ici **144** pas) pour coller à une prévision « jour et au-delà », pas seulement quelques pas devant.
4. **Séparation val / test** : early stopping sur la validation ; **ne pas** optimiser sur le test.
5. **Reproductibilité** : versions de bibliothèques dans `pyproject.toml` ; `random_state` fixé pour XGBoost (constante dans `sdwpf.constants`).
6. **Données et secrets** : gros fichiers et dossiers `data/` hors Git ; clés API uniquement dans `.env` (voir `.env.example`).
7. **Qualité des mesures** : exclure les valeurs non physiques évidentes sur SCADA (puissance / vent négatifs, directions hors plage, `inf`) et filtrer ERA5 incohérent (`Wspd_w` négatif) avant modélisation — voir `sanitize_scada_for_forecasting` et `merge_scada_era5`.
8. **Tests ciblés** : logique de split, d’horizon et de nettoyage couverte par `pytest` sans dépendre des CSV massifs.
9. **Artefacts jetables** : rapports `reports/*.csv|md` et `mlruns/` produits localement, non traités comme source de vérité du dépôt.

---

## Ce que ce projet ne prétend pas couvrir (volontairement)

- Modèles de séquence profonds (LSTM / Transformers) : hors scope pour garder un pipeline **léger et auditable**.
- Prévision météo à l’échéance (nowcasting / NWP downscaling) : les champs ERA5 du jeu sont utilisés **tels quels** dans le pipeline actuel ; une vraie chaîne opérationnelle ferait intervenir des **prévisions** météo au pas concerné, pas seulement l’analyse.
- Évaluation « walk-forward » multi-fenêtres : un seul split configurable pour l’instant ; extension possible plus tard.

---

## Liens dans le dépôt

- **Guide technique détaillé** : [`GUIDE.md`](GUIDE.md)
- **Plan, attentes et todo** : [`PLAN.md`](PLAN.md)
- **Inventaire des fichiers** : [`INVENTAIRE_FICHIERS.md`](INVENTAIRE_FICHIERS.md)
- **Entrée courte** : [`README.md`](../README.md)
