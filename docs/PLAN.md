# Plan de travail — prévision éolienne (séries temporelles)

Document vivant : **objectifs**, **résultats attendus**, **phases** et **todo** pour la suite. Mettre à jour les cases au fil de l’avancement.

**Dernière revue dépôt** : **20** tests `pytest`, package importable, `pyproject.toml` (extra `experiments` = MLflow). Figures `*_h72_*` : retirer avec `scripts/clean_artifacts.py` si besoin.

---

## 1. Vision (à quoi sert le projet)

| Élément | Description |
|--------|-------------|
| **Signal métier** | Prévoir la **puissance éolienne** (`Patv`) à partir de l’historique et de la météo / réanalyse. |
| **Message portfolio** | Montrer un pipeline **honnête** : split **temporel**, pas de baseline trompeuse, scénario **météo-only** documenté. |
| **Périmètre technique actuel** | XGBoost tabulaire sur lags + calendrier + ERA5 ; pas de deep learning dans la baseline. |
| **Données** | Foyer principal **SDWPF** (Chine) ; France / USA pour illustrer **sources** et **limites** (co-localisation, ODRE vs point météo). |

---

## 2. Ce qu’on attend comme “succès”

1. **Reproductibilité** : `pip install -e ".[dev]"` ou `".[dev,experiments]"` + données placées comme dans le README → même logique de split et de métriques.
2. **Évaluation lisible** : métriques **test** pour le score principal ; **val** pour early stopping ; **gain %** vs baseline naive documenté dans le benchmark.
3. **Cohérence méthodo** : horizons **≥ 1 jour** (144 pas) pour les comparaisons sérieuses ; figures et exemples README alignés sur cette règle.
4. **Doc à jour** : `README` ↔ `GUIDE` ↔ `DOMAINE_ET_PRATIQUES` sans contradictions ; pas de fichier “fantôme” promis (ex. ancien `PARCOURS_COMPLET`).
5. **Qualité minimum** : `pytest` vert sur la logique features/splits sans dépendre des CSV multi-Go.

---

## 3. Phases proposées

### Phase A — Stabiliser le “noyau” portfolio (court terme)

- Aligner **artefacts visuels** (PNG) sur horizons **≥ 144** ou retirer du dépôt les PNG obsolètes (`h72`, etc.).
- Vérifier une fois **Docker** (`docker build` + `--help` ou run minimal si `data/` présent).
- **GitHub** : dépôt public ou privé, `.gitignore` déjà adapté ; pas de données ni `.env` commités.

### Phase B — Rigueur d’évaluation (moyen terme)

- **Walk-forward** : **`walk_forward_indices`** + **`scripts/sdwpf_walkforward.py`** (agrégation MAE naive / XGB / persistance sur *n* plis).
- **Baseline persistance** : activée **seulement** si `patv_now` ∈ features (`sdwpf_explore`, `sdwpf_benchmark`, MLflow).
- **Tests** : `train_and_evaluate` couvert sur données synthétiques (`test_sdwpf_pipeline.py`).

### Phase C — Extension métier (optionnel)

- Prévisions météo **à l’échéance** (NWP / API) au lieu ou en équilibre avec ERA5 “instantané”.
- Modèles complémentaires (quantiles, autre learner) **sans** casser la comparaison sur le même split.

---

## 4. Todo list (exécutable)

Légende : `[ ]` à faire · `[x]` fait

### Immédiat — cohérence & nettoyage

- [x] Figures `*_h72_*` et dossier **`mlruns/`** : nettoyage via `python scripts/clean_artifacts.py` (déjà fait une fois sur cet environnement).
- [x] **`.env.example`** : uniquement des placeholders (`NLR_*` vides), pas de secrets.
- [ ] **`docker build -t sdwpf-forecast .`** : à lancer quand **Docker Desktop** (ou équivalent) est démarré ; le `Dockerfile` est aligné sur `pip install .`.

### Documentation

- [x] Mise à jour [**`PLAN.md`**](PLAN.md) / [**`INVENTAIRE_FICHIERS.md`**](INVENTAIRE_FICHIERS.md) après ajouts (tests pipeline, README France).
- [ ] Optionnel : **capture** ou lien vers **1–2 figures** `h144` dans le README.

### Code & tests

- [x] Test d’intégration : **`tests/test_sdwpf_pipeline.py`** — `train_and_evaluate` sur données **synthétiques** (mode météo, chemin SCADA complet, `return_test_predictions`).
- [x] (Phase B) **`walk_forward_indices`** ; **`build_modeling_frame`** / **`evaluate_on_indices`** ; **`sdwpf_walkforward.py`** ; **persistance** si `patv_now` ∈ features.

### Données & scripts auxiliaires

- [x] Branche **France** : avertissement **ODRE / Open-Meteo non co-localisés** ajouté au **README** (renvoi au GUIDE).
- [x] Branche **USA** : rappel **`.env` NLR** au README (section Données France / USA).

### Suivi

- [x] Items ci-dessus tenus à jour au fil des passes ; historique en bas de fichier.

---

## 5. Références dans le dépôt

| Fichier | Rôle |
|---------|------|
| [`README.md`](../README.md) | Entrée, commandes, structure |
| [`DOMAINE_ET_PRATIQUES.md`](DOMAINE_ET_PRATIQUES.md) | Domaine, sous-domaines, checklist méthodo |
| [`GUIDE.md`](GUIDE.md) | Détail technique, CLI, glossaire |
| [`INVENTAIRE_FICHIERS.md`](INVENTAIRE_FICHIERS.md) | Rôle de chaque dossier / fichier |

---

*Historique :*

- 2026-03-30 — Création du plan ; recheck : `pytest` OK, import `sdwpf` OK.
- 2026-03-30 — Nettoyage artefacts (`clean_artifacts`) ; nettoyage SCADA/ERA5 dans `data.py` ; `pytest` 14 passed.
- 2026-03-30 — Suite : `test_sdwpf_pipeline.py` (3 tests), README France/USA, `pytest` 17 passed ; Docker build non exécuté (moteur Docker indisponible sur la machine de build).
- 2026-03-30 — Persistance + `walk_forward_indices` ; refactor **`build_modeling_frame`** / **`evaluate_on_indices`** ; **`sdwpf_walkforward.py`** ; `pytest` 20 passed.
