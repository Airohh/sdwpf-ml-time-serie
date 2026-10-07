# Contribuer

Les retours sont les bienvenus (issues, pull requests).

- **Python** : version minimale indiquée dans `pyproject.toml` (`requires-python`).
- **Tests** : `pytest -q` depuis la racine après `pip install -e ".[dev]"` (versions exactes : `pip install -c requirements-lock.txt -e ".[dev]"`).
- **Lint** : `ruff check .` doit passer (vérifié en CI).
- **Style** : privilégier des changements **ciblés** ; aligner imports et conventions du dossier concerné.
- **Données** : ne pas committer de CSV SDWPF lourds (voir `.gitignore`).
