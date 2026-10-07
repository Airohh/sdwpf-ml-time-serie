from __future__ import annotations

import numpy as np
import pandas as pd

from sdwpf.constants import (
    ERA5_AT_TARGET_SUFFIX,
    ERA5_EXTRA_COLS,
    METEO_WDIR_COS_COL,
    METEO_WDIR_SIN_COL,
    METEO_WSPD_CUBE_COL,
    STEP_MINUTES,
    STEPS_PER_DAY,
)

CALENDAR_FEATURE_COLS = ("hour_sin", "hour_cos", "doy_sin", "doy_cos")
HOUR_FEATURE_COLS = ("hour_sin", "hour_cos")
DAY_OF_YEAR_FEATURE_COLS = ("doy_sin", "doy_cos")

_STEP = pd.Timedelta(minutes=STEP_MINUTES)


def _value_at_offset(s: pd.Series, dt: pd.DatetimeIndex, steps: int) -> np.ndarray:
    """Valeur de ``s`` à l'instant ``t - steps × 10 min`` (``steps < 0`` : futur), NaN si absent.

    Remplace ``Series.shift`` : un décalage par ligne glisse dès qu'une ligne manque
    (nettoyage, jointure ERA5), et la cible ``t+h`` tomberait alors des jours plus loin.
    """
    by_time = pd.Series(s.to_numpy(), index=dt)
    return by_time.reindex(dt - steps * _STEP).to_numpy()


def _add_calendar_cycles(out: pd.DataFrame, when: pd.DatetimeIndex) -> pd.DataFrame:
    """Heure du jour et jour de l'année en sin/cos, pris à l'instant ``when``."""
    minutes = np.asarray(when.hour * 60 + when.minute + when.second / 60.0, dtype=float)
    ang_h = 2 * np.pi * minutes / (24 * 60)
    out["hour_sin"] = np.sin(ang_h)
    out["hour_cos"] = np.cos(ang_h)
    doy = np.asarray(when.dayofyear, dtype=float)
    ang_d = 2 * np.pi * (doy - 1) / 365.25
    out["doy_sin"] = np.sin(ang_d)
    out["doy_cos"] = np.cos(ang_d)
    return out


def _add_wind_physics(out: pd.DataFrame, suffix: str = "") -> None:
    """Wspd_w³ (∝ énergie du vent) et sin/cos de Wdir_w, pour les colonnes ERA5 ``*suffix``."""
    speed, direction = f"Wspd_w{suffix}", f"Wdir_w{suffix}"
    if speed in out.columns:
        w = pd.to_numeric(out[speed], errors="coerce").to_numpy(dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(w, 0.0)
        out[f"{METEO_WSPD_CUBE_COL}{suffix}"] = np.power(w, 3.0)
    if direction in out.columns:
        wd = pd.to_numeric(out[direction], errors="coerce").to_numpy(dtype=np.float64)
        rad = np.deg2rad(wd)
        out[f"{METEO_WDIR_SIN_COL}{suffix}"] = np.sin(rad)
        out[f"{METEO_WDIR_COS_COL}{suffix}"] = np.cos(rad)


def _add_meteo_lags_and_physics(
    out: pd.DataFrame, *, meteo_max_lag: int, dt: pd.DatetimeIndex
) -> pd.DataFrame:
    """Lags sur les colonnes ERA5 + Wspd_w³ et encodage cyclique de Wdir_w."""
    if meteo_max_lag < 0:
        raise ValueError("meteo_max_lag must be >= 0")
    era_present = [c for c in ERA5_EXTRA_COLS if c in out.columns]
    for c in era_present:
        s = pd.to_numeric(out[c], errors="coerce")
        for k in range(1, meteo_max_lag + 1):
            out[f"{c}_lag{k}"] = _value_at_offset(s, dt, k)
    _add_wind_physics(out)
    return out


def _add_era5_at_target(out: pd.DataFrame, *, dt: pd.DatetimeIndex, horizon: int) -> None:
    """ERA5 à l'échéance ``t+h`` : réanalyse utilisée comme une prévision météo parfaite.

    Inaccessible en opérationnel (il faudrait une prévision NWP) : sert de borne haute
    pour mesurer ce que la météo à l'échéance apporterait au modèle.
    """
    for c in ERA5_EXTRA_COLS:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            out[f"{c}{ERA5_AT_TARGET_SUFFIX}"] = _value_at_offset(s, dt, -horizon)
    _add_wind_physics(out, ERA5_AT_TARGET_SUFFIX)


def add_features(
    df: pd.DataFrame,
    max_lag: int = 6,
    horizon: int = 1,
    *,
    calendar_encoding: bool = False,
    meteo_max_lag: int = 0,
    era5_at_target: bool = False,
) -> pd.DataFrame:
    """Lags et cible ``y_target = Patv(t + horizon × 10 min)``, alignés sur ``datetime``.

    Un instant absent de ``df`` donne NaN (lag ou cible), jamais la ligne suivante.
    Le calendrier (``calendar_encoding``) décrit l'instant prévu ``t+h``, connu à l'avance.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if era5_at_target and not calendar_encoding:
        raise ValueError("era5_at_target requires calendar_encoding (meteo mode)")
    if "datetime" not in df.columns:
        raise ValueError("add_features requires a 'datetime' column")
    out = df.copy()
    dt = pd.DatetimeIndex(pd.to_datetime(out["datetime"]))
    if dt.has_duplicates:
        raise ValueError("add_features requires unique datetimes (one turbine per frame)")
    for k in range(1, max_lag + 1):
        out[f"patv_lag{k}"] = _value_at_offset(out["Patv"], dt, k)
        out[f"wspd_lag{k}"] = _value_at_offset(out["Wspd"], dt, k)
    out["patv_now"] = out["Patv"]
    out["wdir"] = out["Wdir"]
    out["etmp"] = out["Etmp"]
    out["itmp"] = out["Itmp"]
    out["y_target"] = _value_at_offset(out["Patv"], dt, -horizon)
    if calendar_encoding:
        out = _add_calendar_cycles(out, dt + horizon * _STEP)
        out = _add_meteo_lags_and_physics(out, meteo_max_lag=int(meteo_max_lag), dt=dt)
        if era5_at_target:
            _add_era5_at_target(out, dt=dt, horizon=horizon)
    return out


def temporal_split(
    n: int, train_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    cut = int(n * train_frac)
    idx = np.arange(n)
    return idx[:cut], idx[cut:]


def purge_target_overlap(
    datetimes: pd.Series,
    idx: np.ndarray,
    next_block_idx: np.ndarray,
    horizon_steps: int,
) -> np.ndarray:
    """Retire de ``idx`` les lignes dont la cible ``t+h`` tombe dans ou après le bloc suivant.

    Sans cette marge, les dernières cibles du train (ou de la validation) appartiennent à
    la période évaluée ensuite : fuite faible mais réelle, croissante avec l'horizon.
    """
    if not len(idx) or not len(next_block_idx):
        return idx
    dt = pd.to_datetime(datetimes).to_numpy()
    block_start = dt[next_block_idx].min()
    target_time = dt[idx] + np.timedelta64(horizon_steps * STEP_MINUTES, "m")
    return idx[target_time < block_start]


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
    include_day_of_year: bool = False,
    era5_at_target: bool = False,
) -> list[str]:
    """Colonnes passées au modèle.

    ``include_day_of_year`` est désactivé par défaut : sur moins d'un an de données
    (SDWPF : 245 jours), les jours du test n'existent pas dans le train ; doy_sin/cos
    deviennent un horodatage déguisé et l'arbre prolonge le niveau de fin de train.
    """
    era5_in_df = [c for c in ERA5_EXTRA_COLS if c in feat.columns]
    if era5_at_target and not time_meteo_only:
        raise ValueError("era5_at_target is only supported with time_meteo_only")
    if time_meteo_only:
        cal_wanted = HOUR_FEATURE_COLS + (
            DAY_OF_YEAR_FEATURE_COLS if include_day_of_year else ()
        )
        cal = [c for c in cal_wanted if c in feat.columns]
        if len(cal) != len(cal_wanted):
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
        physics = (METEO_WSPD_CUBE_COL, METEO_WDIR_SIN_COL, METEO_WDIR_COS_COL)
        cols.extend(name for name in physics if name in feat.columns)
        if era5_at_target:
            at_target = [
                f"{name}{ERA5_AT_TARGET_SUFFIX}" for name in (*ERA5_EXTRA_COLS, *physics)
            ]
            at_target = [c for c in at_target if c in feat.columns]
            if not at_target:
                raise ValueError(
                    "era5_at_target requires add_features(..., era5_at_target=True)"
                )
            cols.extend(at_target)
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
