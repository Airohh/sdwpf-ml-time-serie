from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repo root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent


def default_scada_csv_path() -> Path:
    sd = project_root() / "data" / "china" / "sdwpf"
    nested = sd / "sdwpf_245days_v1.csv" / "sdwpf_245days_v1.csv"
    flat = sd / "sdwpf_245days_v1.csv"
    if nested.is_file():
        return nested
    if flat.is_file():
        return flat
    return nested


def default_weather_csv_path() -> Path:
    return (
        project_root()
        / "data"
        / "china"
        / "sdwpf"
        / "sdwpf_weather_v2"
        / "sdwpf_weather"
        / "wtb2005_2104_full_new.csv"
    )
