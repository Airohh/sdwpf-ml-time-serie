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
      (vent négatif ; ``Wdir`` hors ``[-180, 180]`` lorsqu'elle est renseignée : c'est
      l'angle entre le vent et la nacelle, pas un cap 0–360).
    - ``Patv < 0`` (consommation des auxiliaires à l'arrêt) est ramené à 0, comme dans
      la KDD Cup 2022 : supprimer ces lignes retirerait les périodes de vent faible.
    """
    if "datetime" not in df.columns or "Patv" not in df.columns or "Wspd" not in df.columns:
        raise ValueError("sanitize_scada_for_forecasting requires datetime, Patv, Wspd columns")
    out = df.sort_values("datetime").reset_index(drop=True)
    for c in ("Patv", "Wspd", "Wdir", "Etmp", "Itmp"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Patv", "Wspd"])
    out["Patv"] = out["Patv"].clip(lower=0.0)
    out = out.loc[out["Wspd"] >= 0]
    if "Wdir" in out.columns:
        wd = out["Wdir"]
        bad_wdir = wd.notna() & ((wd < -180) | (wd > 180))
        out = out.loc[~bad_wdir]
    return out.reset_index(drop=True)


def _rows_by_turbine(
    csv_path: Path,
    turb_ids: list[int],
    chunksize: int,
    what: str,
    **read_kw: object,
) -> dict[int, pd.DataFrame]:
    """Une seule lecture par chunks du CSV, lignes regroupées par ``TurbID`` demandé."""
    parts: dict[int, list[pd.DataFrame]] = {int(t): [] for t in turb_ids}
    wanted = set(parts)
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, **read_kw):
        sub = chunk.loc[chunk["TurbID"].isin(wanted)]
        for tid, rows in sub.groupby("TurbID", sort=False):
            parts[int(tid)].append(rows.copy())
    missing = [tid for tid, p in parts.items() if not p]
    if missing:
        ids = ", ".join(map(str, missing))
        raise ValueError(f"No {what} rows for TurbID={ids} in {csv_path}")
    return {tid: pd.concat(p, ignore_index=True) for tid, p in parts.items()}


def load_turbines(
    csv_path: Path,
    turb_ids: list[int],
    chunksize: int = 500_000,
    anchor: str = SDWPF_V1_ANCHOR,
) -> dict[int, pd.DataFrame]:
    """SCADA de plusieurs turbines en **une** passe sur le gros CSV (nettoyées, triées)."""
    base = pd.Timestamp(anchor)
    out: dict[int, pd.DataFrame] = {}
    for tid, df in _rows_by_turbine(csv_path, turb_ids, chunksize, "SCADA").items():
        day_off = pd.to_timedelta(df["Day"].astype(int) - 1, unit="D")
        df["datetime"] = base + day_off + _tm_to_timedelta(df["Tmstamp"])
        df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="first")
        out[tid] = sanitize_scada_for_forecasting(df)
    return out


def load_one_turbine(
    csv_path: Path,
    turb_id: int,
    chunksize: int = 500_000,
    anchor: str = SDWPF_V1_ANCHOR,
) -> pd.DataFrame:
    """Stream the large SCADA CSV and keep rows for a single TurbID."""
    return load_turbines(csv_path, [turb_id], chunksize=chunksize, anchor=anchor)[int(turb_id)]


def _clean_era5(w: pd.DataFrame) -> pd.DataFrame:
    w["datetime"] = pd.to_datetime(w["Tmstamp"], errors="coerce")
    w = w.dropna(subset=["datetime"])
    w = w.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="first")
    for c in ERA5_EXTRA_COLS:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.replace([np.inf, -np.inf], np.nan)
    if "Wspd_w" in w.columns:
        w = w.loc[w["Wspd_w"].isna() | (w["Wspd_w"] >= 0)]
    return w


def load_era5_for_turbines(
    weather_csv: Path,
    turb_ids: list[int],
    chunksize: int = 500_000,
) -> dict[int, pd.DataFrame]:
    """ERA5 (sdwpf_weather_v2) de plusieurs turbines en une passe sur le CSV météo."""
    want = {"TurbID", "Tmstamp", *ERA5_EXTRA_COLS}
    rows = _rows_by_turbine(
        weather_csv, turb_ids, chunksize, "weather", usecols=lambda c: c in want
    )
    return {tid: _clean_era5(w) for tid, w in rows.items()}


def load_era5_for_turbine(
    weather_csv: Path,
    turb_id: int,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """ERA5 columns from sdwpf_weather_v2 (one row per 10 min per TurbID)."""
    return load_era5_for_turbines(weather_csv, [turb_id], chunksize=chunksize)[int(turb_id)]


def merge_scada_era5(scada: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    """Inner join on datetime; keeps SCADA rows with co-located reanalysis."""
    out = scada.merge(
        era5[["datetime", *ERA5_EXTRA_COLS]],
        on="datetime",
        how="inner",
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=list(ERA5_EXTRA_COLS))
