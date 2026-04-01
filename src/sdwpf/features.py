from __future__ import annotations

import numpy as np
import pandas as pd

from sdwpf.constants import (
    ERA5_EXTRA_COLS,
    METEO_WDIR_COS_COL,
    METEO_WDIR_SIN_COL,
    METEO_WSPD_CUBE_COL,
    STEPS_PER_DAY,
)

CALENDAR_FEATURE_COLS = ("hour_sin", "hour_cos", "doy_sin", "doy_cos")


def _add_calendar_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Encode time-of-day and day-of-year as sin/cos (needs ``datetime``)."""
    if "datetime" not in df.columns:
        raise ValueError("calendar features require a 'datetime' column")
    out = df.copy()
    dt = pd.to_datetime(out["datetime"])
    minutes = dt.dt.hour * 60 + dt.dt.minute + dt.dt.second / 60.0
    ang_h = 2 * np.pi * minutes / (24 * 60)
    out["hour_sin"] = np.sin(ang_h)
    out["hour_cos"] = np.cos(ang_h)
    doy = dt.dt.dayofyear.astype(float)
    ang_d = 2 * np.pi * (doy - 1) / 365.25
    out["doy_sin"] = np.sin(ang_d)
    out["doy_cos"] = np.cos(ang_d)
    return out


def _add_meteo_lags_and_physics(
    out: pd.DataFrame, *, meteo_max_lag: int
) -> pd.DataFrame:
    """Lags sur les colonnes ERA5 + Wspd_w³ et encodage cyclique de Wdir_w."""
    if meteo_max_lag < 0:
        raise ValueError("meteo_max_lag must be >= 0")
    era_present = [c for c in ERA5_EXTRA_COLS if c in out.columns]
    for c in era_present:
        s = pd.to_numeric(out[c], errors="coerce")
        for k in range(1, meteo_max_lag + 1):
            out[f"{c}_lag{k}"] = s.shift(k)
    if "Wspd_w" in out.columns:
        w = pd.to_numeric(out["Wspd_w"], errors="coerce").to_numpy(dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(w, 0.0)
        out[METEO_WSPD_CUBE_COL] = np.power(w, 3.0)
    if "Wdir_w" in out.columns:
        wd = pd.to_numeric(out["Wdir_w"], errors="coerce").to_numpy(dtype=np.float64)
        rad = np.deg2rad(wd)
        out[METEO_WDIR_SIN_COL] = np.sin(rad)
        out[METEO_WDIR_COS_COL] = np.cos(rad)
    return out


def add_features(
    df: pd.DataFrame,
    max_lag: int = 6,
    horizon: int = 1,
    *,
    calendar_encoding: bool = False,
    meteo_max_lag: int = 0,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    out = df.copy()
    for k in range(1, max_lag + 1):
        out[f"patv_lag{k}"] = out["Patv"].shift(k)
        out[f"wspd_lag{k}"] = out["Wspd"].shift(k)
    out["patv_now"] = out["Patv"]
    out["wdir"] = out["Wdir"]
    out["etmp"] = out["Etmp"]
    out["itmp"] = out["Itmp"]
    out["y_target"] = out["Patv"].shift(-horizon)
    if calendar_encoding:
        out = _add_calendar_cycles(out)
        out = _add_meteo_lags_and_physics(out, meteo_max_lag=int(meteo_max_lag))
    return out


def temporal_split(
    n: int, train_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    cut = int(n * train_frac)
    idx = np.arange(n)
    return idx[:cut], idx[cut:]


def walk_forward_indices(
    n: int,
    *,
    n_splits: int,
    test_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fenêtres walk-forward (train croissant, tests contigus en fin de série).

    Pour chaque pli ``j`` dans ``0 .. n_splits-1`` (du plus ancien test au plus récent),
    le **train** est ``0 : test_start``, le **test** est ``test_start : test_start + test_size``.

    - ``n`` : nombre de lignes **déjà ordonnées dans le temps**.
    - ``test_size`` : taille **fixe** de chaque bloc test (en nombre d'échantillons).
    """
    if n_splits < 1 or test_size < 1:
        raise ValueError("n_splits and test_size must be >= 1")
    if n < n_splits * test_size + 1:
        raise ValueError(
            f"need n >= n_splits * test_size + 1; got n={n}, "
            f"n_splits={n_splits}, test_size={test_size}"
        )
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(n_splits):
        test_start = n - (n_splits - j) * test_size
        test_end = test_start + test_size
        train_idx = np.arange(0, test_start, dtype=np.int64)
        test_idx = np.arange(test_start, test_end, dtype=np.int64)
        if train_idx.size < 1:
            raise ValueError("walk_forward_indices: empty train fold; increase n or reduce n_splits")
        folds.append((train_idx, test_idx))
    return folds


def temporal_split_train_val_test(
    n: int, train_frac: float, val_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Découpage temporel : train | validation | test (dans cet ordre chronologique)."""
    if train_frac <= 0 or val_frac <= 0:
        raise ValueError("train_frac and val_frac must be positive")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1 (leave room for test)")
    cut_tr = int(n * train_frac)
    cut_va = int(n * (train_frac + val_frac))
    if cut_tr < 1 or cut_va - cut_tr < 1 or n - cut_va < 1:
        raise ValueError("series too short for train / val / test with these fractions")
    idx = np.arange(n)
    return idx[:cut_tr], idx[cut_tr:cut_va], idx[cut_va:]


def temporal_split_by_unique_datetime(
    datetimes: pd.Series,
    train_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Découpe sur les **instants** uniques triés : toutes les lignes d’un même ``datetime``
    partagent le même fold (empilement multi-turbines).
    """
    if train_frac <= 0 or train_frac >= 1:
        raise ValueError("train_frac must be in (0, 1)")
    dt = pd.to_datetime(datetimes, utc=False)
    u = pd.DatetimeIndex(np.sort(dt.unique()))
    n_u = len(u)
    if n_u < 2:
        raise ValueError("need at least 2 unique datetimes for train/test split")
    cut = int(round(n_u * train_frac))
    cut = max(1, min(cut, n_u - 1))
    tr_times = u[:cut]
    te_times = u[cut:]
    tr_idx = np.flatnonzero(dt.isin(tr_times).to_numpy())
    te_idx = np.flatnonzero(dt.isin(te_times).to_numpy())
    if tr_idx.size < 1 or te_idx.size < 1:
        raise ValueError("datetime split produced empty train or test")
    return tr_idx, te_idx


def temporal_split_train_val_test_by_unique_datetime(
    datetimes: pd.Series,
    train_frac: float,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train | validation | test sur la timeline d’instants uniques (multi-turbines)."""
    if train_frac <= 0 or val_frac <= 0:
        raise ValueError("train_frac and val_frac must be positive")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1 (leave room for test)")
    dt = pd.to_datetime(datetimes, utc=False)
    u = pd.DatetimeIndex(np.sort(dt.unique()))
    n_u = len(u)
    cut_tr = int(round(n_u * train_frac))
    cut_va_end = int(round(n_u * (train_frac + val_frac)))
    cut_tr = max(1, cut_tr)
    cut_va_end = max(cut_tr + 1, cut_va_end)
    cut_va_end = min(cut_va_end, n_u - 1)
    if cut_tr < 1 or cut_va_end - cut_tr < 1 or n_u - cut_va_end < 1:
        raise ValueError("series too short for datetime train/val/test split")
    times_tr = u[:cut_tr]
    times_va = u[cut_tr:cut_va_end]
    times_te = u[cut_va_end:]
    tr_idx = np.flatnonzero(dt.isin(times_tr).to_numpy())
    va_idx = np.flatnonzero(dt.isin(times_va).to_numpy())
    te_idx = np.flatnonzero(dt.isin(times_te).to_numpy())
    if tr_idx.size < 1 or va_idx.size < 1 or te_idx.size < 1:
        raise ValueError("datetime split produced empty train, val, or test")
    return tr_idx, va_idx, te_idx


def build_feature_columns(
    feat: pd.DataFrame,
    max_lag: int,
    no_patv_now: bool,
    no_patv_lags: bool,
    *,
    time_meteo_only: bool = False,
    meteo_max_lag: int = 0,
) -> list[str]:
    era5_in_df = [c for c in ERA5_EXTRA_COLS if c in feat.columns]
    if time_meteo_only:
        cal = [c for c in CALENDAR_FEATURE_COLS if c in feat.columns]
        if len(cal) != len(CALENDAR_FEATURE_COLS):
            raise ValueError(
                "time_meteo_only requires calendar columns; "
                "use add_features(..., calendar_encoding=True)"
            )
        if not era5_in_df:
            raise ValueError(
                "time_meteo_only needs ERA5 columns in the frame (merge weather v2)"
            )
        cols: list[str] = list(cal)
        for c in era5_in_df:
            cols.append(c)
            for k in range(1, int(meteo_max_lag) + 1):
                lc = f"{c}_lag{k}"
                if lc in feat.columns:
                    cols.append(lc)
        for name in (METEO_WSPD_CUBE_COL, METEO_WDIR_SIN_COL, METEO_WDIR_COS_COL):
            if name in feat.columns:
                cols.append(name)
        return cols

    patv_lag_cols = (
        [] if no_patv_lags else [f"patv_lag{k}" for k in range(1, max_lag + 1)]
    )
    return (
        ([] if no_patv_now else ["patv_now"])
        + patv_lag_cols
        + [f"wspd_lag{k}" for k in range(1, max_lag + 1)]
        + ["wdir", "etmp", "itmp"]
        + era5_in_df
    )


def resolve_horizon_steps(
    horizon: int | None,
    horizon_hours: float | None,
    horizon_days: float | None,
) -> int:
    """Horizon in 10-min steps. If hours/days set, they define steps (ignores ``horizon``).
    If all are None, defaults to ``STEPS_PER_DAY`` (1 day). If only ``horizon`` is set, uses it.
    """
    if horizon_hours is not None or horizon_days is not None:
        steps = 0
        if horizon_hours is not None:
            steps += int(round(float(horizon_hours) * 6))
        if horizon_days is not None:
            steps += int(round(float(horizon_days) * STEPS_PER_DAY))
        if steps < 1:
            raise ValueError(
                "horizon from --horizon-hours/--horizon-days must be at least 10 minutes"
            )
        return steps
    if horizon is not None:
        if horizon < 1:
            raise ValueError("--horizon must be >= 1")
        return int(horizon)
    return int(STEPS_PER_DAY)
