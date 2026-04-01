"""
Wind Toolkit Data V2 — téléchargement direct CSV (NLR / ex-NREL).

Doc: https://developer.nlr.gov/docs/wind/wind-toolkit/wtk-download/

- Réponse `.csv` : un seul POINT WGS84 et une seule année par requête ; le corps de la réponse est le CSV.
- WKT : POINT(longitude latitude) — longitude en premier.
- Années classiques WTK CONUS : 2007–2014 uniquement.
- Couverture : États-Unis (continental), pas la France.
- Export par défaut : data/usa/nlr_wtk/

Variables d'environnement (recommandé) :
  NLR_API_KEY   clé https://developer.nlr.gov/signup/
  NLR_EMAIL     email (requis par l'API)

Usage :
  1. copy .env.example .env  puis édite .env (clé + email)
  ou
  set NLR_API_KEY=... & set NLR_EMAIL=...  (PowerShell / cmd)

  python scripts/download_wind_toolkit_nlr.py

Options : voir --help
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

BASE_CSV = "https://developer.nlr.gov/api/wind-toolkit/v2/wind/wtk-download.csv"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_REPO_ROOT / ".env")

# Attributs valides (liste doc NLR). Gardez la liste courte pour alléger les fichiers.
DEFAULT_ATTRIBUTES = (
    "windspeed_100m,winddirection_100m,windspeed_80m,"
    "temperature_100m,relativehumidity_2m,pressure_0m"
)


def download_one_year(
    *,
    api_key: str,
    email: str,
    lon: float,
    lat: float,
    year: int,
    interval: int,
    attributes: str,
    utc: bool,
    leap_day: bool,
    out_path: Path,
    timeout: int = 600,
) -> None:
    wkt = f"POINT({lon} {lat})"
    params = {
        "api_key": api_key,
        "wkt": wkt,
        "names": str(year),
        "email": email,
        "interval": str(interval),
        "attributes": attributes,
        "utc": "true" if utc else "false",
        "leap_day": "true" if leap_day else "false",
    }
    # Optionnel mais utile pour l’opérateur NLR
    params["reason"] = "research-local-wind-power-forecasting"
    params["affiliation"] = "personal"

    r = requests.get(BASE_CSV, params=params, timeout=timeout)
    if r.status_code != 200:
        print(f"HTTP {r.status_code} pour {year}: {r.text[:500]}", file=sys.stderr)
        r.raise_for_status()

    text = r.text
    if text.lstrip().startswith("{") or '"errors"' in text[:200]:
        print(f"Réponse inattendue (JSON?) pour {year}:\n{text[:800]}", file=sys.stderr)
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"OK {year} -> {out_path} ({len(text)} caractères)")


def main() -> None:
    _load_dotenv()

    p = argparse.ArgumentParser(description="Wind Toolkit V2 — téléchargement CSV par année")
    p.add_argument("--lon", type=float, default=-101.0, help="Longitude WGS84 (ex. Texas)")
    p.add_argument("--lat", type=float, default=35.0, help="Latitude WGS84")
    p.add_argument(
        "--years",
        type=str,
        default="2010,2011,2012",
        help="Années séparées par des virgules (2007-2014 pour WTK classique)",
    )
    p.add_argument("--interval", type=int, default=60, choices=[5, 15, 30, 60], help="Minutes")
    p.add_argument("--utc", action="store_true", help="Timestamps UTC (sinon heure locale site sans DST)")
    p.add_argument("--no-leap-day", action="store_true", help="Exclure 29 février")
    p.add_argument("--attributes", type=str, default=DEFAULT_ATTRIBUTES, help="Liste comma-separated NLR")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Dossier de sortie (défaut: data/usa/nlr_wtk/)",
    )
    args = p.parse_args()

    api_key = os.environ.get("NLR_API_KEY", "").strip()
    email = os.environ.get("NLR_EMAIL", "").strip()
    if not api_key or not email:
        print(
            "Définissez NLR_API_KEY et NLR_EMAIL (voir docstring).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = _REPO_ROOT / "data" / "usa" / "nlr_wtk"

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    for y in years:
        if y < 2007 or y > 2014:
            print(
                f"Avertissement: l’année {y} est hors plage 2007–2014 pour le WTK V2 classique.",
                file=sys.stderr,
            )

    for year in years:
        out_file = out_dir / f"wtk_point_{args.lat}_{args.lon}_{year}_interval{args.interval}.csv"
        download_one_year(
            api_key=api_key,
            email=email,
            lon=args.lon,
            lat=args.lat,
            year=year,
            interval=args.interval,
            attributes=args.attributes,
            utc=args.utc,
            leap_day=not args.no_leap_day,
            out_path=out_file,
        )
        # Limite doc : CSV ≤ 1 requête / s
        time.sleep(1.1)

    print("Terminé.")


if __name__ == "__main__":
    main()
