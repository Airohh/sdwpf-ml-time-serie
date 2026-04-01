from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sdwpf.constants import ERA5_EXTRA_COLS, SDWPF_V1_ANCHOR


def _tm_to_timedelta(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str), format="%H:%M", errors="coerce")
    return pd.to_timedelta(parsed.dt.hour, unit="h") + pd.to_timedelta(
        parsed.dt.minute, unit="m"
    )


def sanitize_scada_for_forecasting(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie la série SCADA d'une turbine avant features / modèle.

    - Convertit les colonnes numériques usuelles, retire ``inf`` / ``-inf``.
    - Exige ``Patv`` et ``Wspd`` présents ; écarte les valeurs non physiques
      (puissance ou vent négatifs ; ``Wdir`` hors ``[0, 360]`` lorsqu'elle est renseignée).
    """
    if "datetime" not in df.columns or "Patv" not in df.columns or "Wspd" not in df.columns:
        raise ValueError("sanitize_scada_for_forecasting requires datetime, Patv, Wspd columns")
    out = df.sort_values("datetime").reset_index(drop=True)
    for c in ("Patv", "Wspd", "Wdir", "Etmp", "Itmp"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Patv", "Wspd"])
    out = out.loc[(out["Patv"] >= 0) & (out["Wspd"] >= 0)]
    if "Wdir" in out.columns:
        wd = out["Wdir"]
        bad_wdir = wd.notna() & ((wd < 0) | (wd > 360))
        out = out.loc[~bad_wdir]
    return out.reset_index(drop=True)


def load_one_turbine(
    csv_path: Path,
    turb_id: int,
    chunksize: int = 500_000,
    anchor: str = SDWPF_V1_ANCHOR,
) -> pd.DataFrame:
    """Stream the large SCADA CSV and keep rows for a single TurbID."""
    parts: list[pd.DataFrame] = []
    base = pd.Timestamp(anchor)
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        sub = chunk.loc[chunk["TurbID"] == turb_id].copy()
        if sub.empty:
            continue
        day_off = pd.to_timedelta(sub["Day"].astype(int) - 1, unit="D")
        sub["datetime"] = base + day_off + _tm_to_timedelta(sub["Tmstamp"])
        parts.append(sub)
    if not parts:
        raise ValueError(f"No rows for TurbID={turb_id} in {csv_path}")
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    if df["datetime"].duplicated().any():
        df = df.drop_duplicates(subset=["datetime"], keep="first")
    return sanitize_scada_for_forecasting(df)


def load_era5_for_turbine(
    weather_csv: Path,
    turb_id: int,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """ERA5 columns from sdwpf_weather_v2 (one row per 10 min per TurbID)."""
    cols = ["TurbID", "Tmstamp", *ERA5_EXTRA_COLS]
    want = set(cols)
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        weather_csv,
        chunksize=chunksize,
        usecols=lambda c: c in want,
    ):
        sub = chunk.loc[chunk["TurbID"] == turb_id].copy()
        if sub.empty:
            continue
        parts.append(sub)
    if not parts:
        raise ValueError(f"No weather rows for TurbID={turb_id} in {weather_csv}")
    w = pd.concat(parts, ignore_index=True)
    w["datetime"] = pd.to_datetime(w["Tmstamp"], errors="coerce")
    w = w.dropna(subset=["datetime"])
    w = w.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="first")
    for c in ERA5_EXTRA_COLS:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.replace([np.inf, -np.inf], np.nan)
    if "Wspd_w" in w.columns:
        w = w.loc[w["Wspd_w"].isna() | (w["Wspd_w"] >= 0)]
    return w


def merge_scada_era5(scada: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    """Inner join on datetime; keeps SCADA rows with co-located reanalysis."""
    out = scada.merge(
        era5[["datetime", *ERA5_EXTRA_COLS]],
        on="datetime",
        how="inner",
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=list(ERA5_EXTRA_COLS))
