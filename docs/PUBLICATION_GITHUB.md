# Publier sur GitHub

## 1. Créer le dépôt

Sur [GitHub](https://github.com/new), créer un dépôt (ex. `wind-power-forecasting` ou `sdwpf-forecast-pipeline`), **sans** cocher « Add README » si tu pushes un dépôt local déjà prêt.

## 2. Première poussée

Depuis la racine du projet (PowerShell ou bash) :

```bash
git init
git add .
git commit -m "Initial commit: pipeline prévision éolienne SDWPF"
git branch -M main
git remote add origin https://github.com/VOTRE_USER/VOTRE_REPO.git
git push -u origin main
```

Remplace `VOTRE_USER` / `VOTRE_REPO`.

## 3. Badge CI dans le README

Ajoute sous les autres badges du `README.md` (après le premier titre), en remplaçant `VOTRE_USER` et `VOTRE_REPO` :

```markdown
[![CI](https://github.com/VOTRE_USER/VOTRE_REPO/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/VOTRE_USER/VOTRE_REPO/actions)
```

## 4. Données & secrets

- Les dossiers lourds (`data/china/sdwpf/`, etc.) sont dans `.gitignore` : les **données** restent en local ou sur un stockage perso ; le README indique où les placer.
- Ne jamais committer de `.env` (déjà ignoré).

## 5. Aperçus visuels (optionnel)

Voir [`docs/assets/README.md`](assets/README.md) pour ajouter 1–2 PNG dans le README.
