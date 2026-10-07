# Rejouer l'environnement Python (données SDWPF à monter en volume).
# Build : docker build -t sdwpf-forecast .   (Docker Desktop / moteur actif)
# Exemple (bash ou PowerShell) :
#   docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/reports:/app/reports" \
#     sdwpf-forecast python scripts/sdwpf_explore.py --meteo-mode
#
# Suivi MLflow dans le conteneur : remplacer `.` par `".[experiments]"` dans pip install.

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md requirements-lock.txt ./
COPY src/ src/
# Versions figées (mêmes que la CI locale) via le fichier de contraintes.
RUN pip install --no-cache-dir -c requirements-lock.txt .

COPY scripts/ scripts/

RUN useradd --create-home --uid 1000 app \
    && mkdir -p reports/figures \
    && chown -R app /app
USER app

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

CMD ["python", "scripts/sdwpf_explore.py", "--help"]
