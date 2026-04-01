"""
Download Open-Meteo Historical Weather API (hourly) for one point,
chunked by calendar year, 2020-01-01 through 2025-11-30.
Output: <repo>/data/france/open_meteo/open_meteo_wind_hourly_2020_2025.csv

Lancer depuis la racine du dépôt :
  python scripts/fetch_open_meteo_wind.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = 50.95, 1.85
HOURLY = [
    "temperature_2m",
    "relativehumidity_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "surface_pressure",
    "cloud_cover",
    "wind_gusts_10m",
]

# Align with Energy Production Dataset (ODRE/Kaggle) upper bound
END_DATE_GLOBAL = "2025-11-30"


def year_windows() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for y in range(2020, 2026):
        start = f"{y}-01-01"
        if y < 2025:
            end = f"{y}-12-31"
        else:
            end = END_DATE_GLOBAL
        out.append((start, end))
    return out


def fetch_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY,
        "timezone": "Europe/Paris",
        "wind_speed_unit": "ms",
    }
    r = requests.get(BASE_URL, params=params, timeout=120)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(payload)
    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        raise RuntimeError("No hourly.time in response")
    return pd.DataFrame(hourly)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "data" / "france" / "open_meteo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "open_meteo_wind_hourly_2020_2025.csv"
    meta_path = out_dir / "open_meteo_fetch_meta.txt"

    frames: list[pd.DataFrame] = []
    lines: list[str] = []

    for start, end in year_windows():
        print(f"Fetching {start} .. {end}")
        df = fetch_chunk(start, end)
        df["time"] = pd.to_datetime(df["time"])
        frames.append(df)
        lines.append(f"{start} {end} rows={len(df)}")
        time.sleep(1.0)

    meteo = pd.concat(frames, ignore_index=True)
    meteo = meteo.drop_duplicates(subset=["time"], keep="first").sort_values("time")
    meteo.to_csv(out_csv, index=False)

    meta = [
        f"latitude_request={LAT} longitude_request={LON}",
        f"rows={len(meteo)} time_min={meteo['time'].min()} time_max={meteo['time'].max()}",
        *lines,
    ]
    meta_path.write_text("\n".join(meta) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv} ({len(meteo)} rows)")


if __name__ == "__main__":
    main()
