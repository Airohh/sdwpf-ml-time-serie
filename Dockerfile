# Rejouer l’environnement Python (données SDWPF à monter en volume).
# Build : docker build -t sdwpf-forecast . (nécessite Docker Desktop / moteur actif).
# Exemple :
#   docker build -t sdwpf-forecast .
#   docker run --rm -v "%CD%/data:/app/data" sdwpf-forecast python scripts/sdwpf_explore.py --turb-id 1
#
# Suivi MLflow dans le conteneur : reconstruire avec
#   RUN pip install --no-cache-dir ".[experiments]"

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

COPY scripts/ scripts/

ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/sdwpf_explore.py", "--help"]
