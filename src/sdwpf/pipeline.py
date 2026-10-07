from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sdwpf.constants import (
    DEFAULT_METEO_MAX_LAG,
    ERA5_AT_TARGET_SUFFIX,
    POOL_TURB_ID_COL,
    STEP_MINUTES,
    XGBOOST_RANDOM_STATE,
)
from sdwpf.data import load_era5_for_turbine, load_one_turbine, merge_scada_era5
from sdwpf.features import (
    add_features,
    build_feature_columns,
    purge_target_overlap,
    temporal_split,
    temporal_split_by_unique_datetime,
    temporal_split_train_val_test,
    temporal_split_train_val_test_by_unique_datetime,
)
from sdwpf.xgb_device import (
    XGBOOST_AVAILABLE,
    device_mismatch_warnings_silenced,
    fit_xgb_regressor,
    predict_on_device,
    resolve_xgb_device_spec,
)

try:
    import mlflow
except ImportError:
    mlflow = None


def _xgboost_gain_importances(reg: Any, n_features: int) -> np.ndarray:
    """Pondération ``gain`` normalisée, sans ``feature_importances_`` sklearn (évite un predict CPU)."""
    imp = np.zeros(n_features, dtype=np.float64)
    scores = reg.get_booster().get_score(importance_type="gain")
    for k, v in scores.items():
        if isinstance(k, str) and k.startswith("f"):
            try:
                idx = int(k[1:])
            except ValueError:
                continue
            if 0 <= idx < n_features:
                imp[idx] = float(v)
    s = float(np.sum(imp))
    if s > 1e-12:
        imp = imp / s
    return imp


def _group_mean_predict(
    keys_train: np.ndarray, y_train: np.ndarray, keys_all: np.ndarray, fallback: float
) -> np.ndarray:
    """Moyenne train de la cible par clé (ex. heure de l'échéance), ``fallback`` si clé inconnue."""
    means = pd.Series(y_train).groupby(keys_train).mean()
    return pd.Series(keys_all).map(means).fillna(fallback).to_numpy(dtype=np.float64)


def _power_curve_predict(
    wind_train: np.ndarray,
    y_train: np.ndarray,
    wind_all: np.ndarray,
    fallback: float,
    bin_width: float = 0.5,
) -> np.ndarray:
    """Courbe de puissance empirique : moyenne train par classe de vent, interpolée."""
    ok = np.isfinite(wind_train)
    if not ok.any():
        return np.full(len(wind_all), fallback, dtype=np.float64)
    curve = pd.Series(y_train[ok]).groupby(np.floor(wind_train[ok] / bin_width)).mean()
    centers = (curve.index.to_numpy(dtype=np.float64) + 0.5) * bin_width
    pred = np.interp(wind_all, centers, curve.to_numpy(dtype=np.float64))
    return np.where(np.isfinite(wind_all), pred, fallback)


@dataclass
class SdwpfRunResult:
    horizon_steps: int
    horizon_minutes: int
    n_rows: int
    n_features: int
    feature_cols: list[str]
    patv_feature_mode: str
    # Toujours calculé : prédit train-mean(y) partout (aucune feature *t* ni *Patv*).
    naive_mean_train_test_mae: float = 0.0
    naive_mean_train_test_rmse: float = 0.0
    # Uniquement si ``patv_now`` est dans les features : prédire Patv(t) pour cible Patv(t+h).
    persistence_test_mae: float | None = None
    persistence_test_rmse: float | None = None
    # Moyenne train de la cible par heure du jour de l'échéance (cycle diurne).
    climatology_test_mae: float | None = None
    climatology_test_rmse: float | None = None
    # Courbe de puissance empirique (train) sur Wspd_w à l'échéance ; seulement avec ERA5 à t+h.
    power_curve_test_mae: float | None = None
    power_curve_test_rmse: float | None = None
    # Lignes train / val retirées car leur cible t+h tombe dans le bloc suivant.
    n_purged: int = 0
    xgboost_val_mae: float | None = None
    xgboost_val_rmse: float | None = None
    xgboost_test_mae: float | None = None
    xgboost_test_rmse: float | None = None
    # Références installée / production (train) pour nMAE éolien (%)
    train_patv_pmax: float = 0.0
    train_patv_mean: float = 0.0
    naive_test_nmae_vs_pmax_pct: float = float("nan")
    naive_test_nmae_vs_mean_patv_pct: float = float("nan")
    xgboost_test_nmae_vs_pmax_pct: float | None = None
    xgboost_test_nmae_vs_mean_patv_pct: float | None = None
    # Skill = 1 − MAE_model / MAE_naive (0 = égal à la naive ; >0 mieux, <0 pire)
    xgboost_skill_vs_naive: float | None = None
    naive_test_bias: float = 0.0
    xgboost_test_bias: float | None = None
    top_importances: list[tuple[str, float]] = field(default_factory=list)
    # Set when train_and_evaluate(..., return_test_predictions=True)
    test_datetime: np.ndarray | None = None
    y_test: np.ndarray | None = None
    xgboost_pred_test: np.ndarray | None = None


def describe_feature_mode(
    *,
    no_patv_now: bool,
    no_patv_lags: bool,
    time_meteo_only: bool,
    meteo_max_lag: int = 0,
    include_day_of_year: bool = False,
    era5_at_target: bool = False,
) -> str:
    if time_meteo_only:
        cal = "calendar of t+h (hour" + (", day of year)" if include_day_of_year else ")")
        era5 = f"ERA5 at t + lags 1..{meteo_max_lag}" if meteo_max_lag > 0 else "ERA5 at t"
        parts = [cal, era5, "Wspd_w³ + Wdir_w sin/cos"]
        if era5_at_target:
            parts.append("ERA5 at t+h (oracle)")
        return " + ".join(parts)
    patv_bits = []
    if not no_patv_now:
        patv_bits.append("patv_now")
    if not no_patv_lags:
        patv_bits.append("patv_lags")
    return "+".join(patv_bits) if patv_bits else "no Patv features"


def build_modeling_frame(
    df: pd.DataFrame,
    *,
    horizon_steps: int,
    max_lag: int = 6,
    no_patv_now: bool = False,
    no_patv_lags: bool = False,
    time_meteo_only: bool = False,
    meteo_max_lag: int = DEFAULT_METEO_MAX_LAG,
    include_day_of_year: bool = False,
    era5_at_target: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Construit la table de modélisation (features + ``y_target``) alignée sur ``df``."""
    if horizon_steps < 1:
        raise ValueError(f"horizon_steps must be >= 1; got {horizon_steps}")
    mlag = int(meteo_max_lag) if time_meteo_only else 0
    if mlag < 0:
        raise ValueError("meteo_max_lag must be >= 0")
    max_lag_eff = 0 if time_meteo_only else max_lag
    trim_start = max(max_lag_eff, mlag)
    feat = add_features(
        df,
        max_lag=max_lag_eff,
        horizon=horizon_steps,
        calendar_encoding=time_meteo_only,
        meteo_max_lag=mlag,
        era5_at_target=era5_at_target,
    )
    # Échauffement des lags, puis lignes sans cible (fin de série ou instant t+h absent).
    feat = feat.iloc[trim_start:]
    feat = feat.dropna(subset=["y_target"]).reset_index(drop=True)
    feature_cols = build_feature_columns(
        feat,
        max_lag_eff,
        no_patv_now,
        no_patv_lags,
        time_meteo_only=time_meteo_only,
        meteo_max_lag=mlag,
        include_day_of_year=include_day_of_year,
        era5_at_target=era5_at_target,
    )
    if not feature_cols:
        raise ValueError(
            "No features: allow patv_now or patv_lags, or include SCADA/ERA5 columns."
        )
    return feat, feature_cols


def evaluate_on_indices(
    feat: pd.DataFrame,
    feature_cols: list[str],
    *,
    horizon_steps: int,
    patv_feature_mode: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    xgb_params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 30,
    return_test_predictions: bool = False,
    xgb_device: str | None = None,
    purge: bool = True,
) -> SdwpfRunResult:
    """Évalue XGBoost + baselines sur des indices **déjà** calculés (table ``feat``).

    ``purge`` retire du train (et de la validation) les lignes dont la cible ``t+h``
    tombe dans le bloc suivant (voir ``purge_target_overlap``).
    """
    n = len(feat)
    for name, idx in (
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ):
        if idx.size and (idx.min() < 0 or idx.max() >= n or len(np.unique(idx)) != len(idx)):
            raise ValueError(f"Invalid or duplicate {name} indices for frame of length {n}")
    if not len(test_idx):
        raise ValueError("test_idx must be non-empty")
    if not len(train_idx):
        raise ValueError("train_idx must be non-empty")

    st_tr = set(map(int, train_idx))
    st_te = set(map(int, test_idx))
    if st_tr & st_te:
        raise ValueError("train_idx and test_idx must be disjoint")
    if len(val_idx):
        st_va = set(map(int, val_idx))
        if st_tr & st_va or st_va & st_te:
            raise ValueError("train, val, and test indices must be disjoint")

    n_before = len(train_idx) + len(val_idx)
    if purge:
        if "datetime" not in feat.columns:
            raise ValueError("purge requires a 'datetime' column (or pass purge=False)")
        dts = feat["datetime"]
        if len(val_idx):
            train_idx = purge_target_overlap(dts, train_idx, val_idx, horizon_steps)
            val_idx = purge_target_overlap(dts, val_idx, test_idx, horizon_steps)
            if not len(val_idx):
                raise ValueError(
                    "validation block is shorter than the horizon: nothing left after purge; "
                    "increase val_frac or the amount of data"
                )
        else:
            train_idx = purge_target_overlap(dts, train_idx, test_idx, horizon_steps)
        if not len(train_idx):
            raise ValueError("train is empty after purging targets that overlap the next block")
    n_purged = n_before - len(train_idx) - len(val_idx)

    X = feat[feature_cols].to_numpy(dtype=np.float64)
    y = feat["y_target"].to_numpy(dtype=np.float64)

    X_tr = X[train_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    X_va = X[val_idx] if len(val_idx) else None
    y_va = y[val_idx] if len(val_idx) else None

    def mae_rmse_on(pred: np.ndarray, idx: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
        mae = mean_absolute_error(y_true, pred[idx])
        rmse = float(np.sqrt(mean_squared_error(y_true, pred[idx])))
        return mae, rmse

    def test_mae_rmse(pred: np.ndarray) -> tuple[float, float]:
        return mae_rmse_on(pred, test_idx, y_te)

    y_train_mean = float(np.mean(y_tr))
    naive_pred = np.full(len(y), y_train_mean, dtype=np.float64)
    naive_mae, naive_rmse = test_mae_rmse(naive_pred)

    persistence_mae = persistence_rmse = None
    if "patv_now" in feature_cols:
        pers_pred = feat["patv_now"].to_numpy(dtype=np.float64)
        persistence_mae, persistence_rmse = test_mae_rmse(pers_pred)

    clim_mae = clim_rmse = None
    if "datetime" in feat.columns:
        target_hour = (
            pd.to_datetime(feat["datetime"]) + pd.Timedelta(minutes=horizon_steps * STEP_MINUTES)
        ).dt.hour.to_numpy()
        clim_pred = _group_mean_predict(target_hour[train_idx], y_tr, target_hour, y_train_mean)
        clim_mae, clim_rmse = test_mae_rmse(clim_pred)

    pc_mae = pc_rmse = None
    pc_col = f"Wspd_w{ERA5_AT_TARGET_SUFFIX}"
    if pc_col in feat.columns:
        wind = feat[pc_col].to_numpy(dtype=np.float64)
        pc_pred = _power_curve_predict(wind[train_idx], y_tr, wind, y_train_mean)
        pc_mae, pc_rmse = test_mae_rmse(pc_pred)

    x_mae = x_rmse = None
    x_val_mae = x_val_rmse = None
    top_imp: list[tuple[str, float]] = []

    use_es = (
        XGBOOST_AVAILABLE
        and X_va is not None
        and len(val_idx) >= 1
        and early_stopping_rounds > 0
    )
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": XGBOOST_RANDOM_STATE,
        "n_jobs": -1,
    }
    if use_es:
        params["early_stopping_rounds"] = int(early_stopping_rounds)
    if xgb_params:
        params.update(xgb_params)

    train_patv_pmax = train_patv_mean = 0.0
    naive_test_nmae_p = float("nan")
    naive_test_nmae_m = float("nan")
    naive_test_bias = 0.0
    if "Patv" in feat.columns:
        patv_tr = feat["Patv"].iloc[train_idx].to_numpy(dtype=np.float64)
        train_patv_pmax = float(np.nanmax(patv_tr))
        train_patv_mean = float(np.nanmean(patv_tr))

    def _pct_mae_over_ref(mae: float, ref: float) -> float:
        if not np.isfinite(mae) or not np.isfinite(ref) or ref <= 1e-12:
            return float("nan")
        return float(100.0 * mae / ref)

    if "Patv" in feat.columns and train_patv_pmax > 1e-12:
        naive_test_nmae_p = _pct_mae_over_ref(naive_mae, train_patv_pmax)
    if "Patv" in feat.columns and train_patv_mean > 1e-12:
        naive_test_nmae_m = _pct_mae_over_ref(naive_mae, train_patv_mean)
    naive_test_bias = float(np.mean(y_te - y_train_mean))

    y_hat = None
    skill_vs_naive: float | None = None
    x_bias = None
    x_nmae_p = x_nmae_m = None
    if XGBOOST_AVAILABLE:
        fit_kw: dict[str, Any] = {}
        if use_es and y_va is not None:
            fit_kw["eval_set"] = [(X_va, y_va)]
            fit_kw["verbose"] = False

        spec = resolve_xgb_device_spec(xgb_device)
        with device_mismatch_warnings_silenced(spec):
            reg, fit_device = fit_xgb_regressor(params, X_tr, y_tr, fit_kw=fit_kw, spec=spec)
            y_hat = predict_on_device(reg, X, fit_device)
        if X_va is not None and len(val_idx) >= 1 and y_va is not None:
            x_val_mae, x_val_rmse = mae_rmse_on(y_hat, val_idx, y_va)
        x_mae, x_rmse = test_mae_rmse(y_hat)
        if naive_mae > 1e-12 and x_mae is not None and np.isfinite(x_mae):
            skill_vs_naive = float(1.0 - float(x_mae) / float(naive_mae))
        x_bias = float(np.mean(y_te - y_hat[test_idx]))
        if train_patv_pmax > 1e-12 and x_mae is not None:
            x_nmae_p = _pct_mae_over_ref(float(x_mae), train_patv_pmax)
        if train_patv_mean > 1e-12 and x_mae is not None:
            x_nmae_m = _pct_mae_over_ref(float(x_mae), train_patv_mean)
        imp = _xgboost_gain_importances(reg, len(feature_cols))
        order = np.argsort(imp)[::-1][: min(12, len(feature_cols))]
        top_imp = [(feature_cols[i], float(imp[i])) for i in order]

    dt_te = y_te_out = x_te = None
    if return_test_predictions and "datetime" in feat.columns:
        dt_te = feat["datetime"].iloc[test_idx].to_numpy(dtype="datetime64[ns]")
        y_te_out = y_te.copy()
        if y_hat is not None:
            x_te = y_hat[test_idx]

    return SdwpfRunResult(
        horizon_steps=horizon_steps,
        horizon_minutes=horizon_steps * 10,
        n_rows=len(feat),
        n_features=len(feature_cols),
        feature_cols=list(feature_cols),
        patv_feature_mode=patv_feature_mode,
        naive_mean_train_test_mae=naive_mae,
        naive_mean_train_test_rmse=naive_rmse,
        persistence_test_mae=persistence_mae,
        persistence_test_rmse=persistence_rmse,
        climatology_test_mae=clim_mae,
        climatology_test_rmse=clim_rmse,
        power_curve_test_mae=pc_mae,
        power_curve_test_rmse=pc_rmse,
        n_purged=n_purged,
        xgboost_val_mae=x_val_mae,
        xgboost_val_rmse=x_val_rmse,
        xgboost_test_mae=x_mae,
        xgboost_test_rmse=x_rmse,
        train_patv_pmax=train_patv_pmax,
        train_patv_mean=train_patv_mean,
        naive_test_nmae_vs_pmax_pct=naive_test_nmae_p,
        naive_test_nmae_vs_mean_patv_pct=naive_test_nmae_m,
        xgboost_test_nmae_vs_pmax_pct=x_nmae_p,
        xgboost_test_nmae_vs_mean_patv_pct=x_nmae_m,
        xgboost_skill_vs_naive=skill_vs_naive,
        naive_test_bias=naive_test_bias,
        xgboost_test_bias=x_bias,
        top_importances=top_imp,
        test_datetime=dt_te,
        y_test=y_te_out,
        xgboost_pred_test=x_te,
    )


def train_and_evaluate(
    df: pd.DataFrame,
    *,
    horizon_steps: int,
    train_frac: float = 0.7,
    val_frac: float | None = None,
    max_lag: int = 6,
    no_patv_now: bool = False,
    no_patv_lags: bool = False,
    time_meteo_only: bool = False,
    meteo_max_lag: int = DEFAULT_METEO_MAX_LAG,
    include_day_of_year: bool = False,
    era5_at_target: bool = False,
    xgb_params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 30,
    return_test_predictions: bool = False,
    xgb_device: str | None = None,
) -> SdwpfRunResult:
    mlag = int(meteo_max_lag) if time_meteo_only else 0
    feat, feature_cols = build_modeling_frame(
        df,
        horizon_steps=horizon_steps,
        max_lag=max_lag,
        no_patv_now=no_patv_now,
        no_patv_lags=no_patv_lags,
        time_meteo_only=time_meteo_only,
        meteo_max_lag=mlag,
        include_day_of_year=include_day_of_year,
        era5_at_target=era5_at_target,
    )
    if val_frac is not None and val_frac > 0:
        train_idx, val_idx, test_idx = temporal_split_train_val_test(
            len(feat), train_frac, val_frac
        )
    else:
        train_idx, test_idx = temporal_split(len(feat), train_frac)
        val_idx = np.array([], dtype=np.int64)

    patv_desc = describe_feature_mode(
        no_patv_now=no_patv_now,
        no_patv_lags=no_patv_lags,
        time_meteo_only=time_meteo_only,
        meteo_max_lag=mlag,
        include_day_of_year=include_day_of_year,
        era5_at_target=era5_at_target,
    )
    return evaluate_on_indices(
        feat,
        feature_cols,
        horizon_steps=horizon_steps,
        patv_feature_mode=patv_desc,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        xgb_params=xgb_params,
        early_stopping_rounds=early_stopping_rounds,
        return_test_predictions=return_test_predictions,
        xgb_device=xgb_device,
    )


def train_and_evaluate_pooled(
    frames: list[pd.DataFrame],
    turb_ids: list[int],
    *,
    horizon_steps: int,
    train_frac: float = 0.7,
    val_frac: float | None = None,
    max_lag: int = 6,
    no_patv_now: bool = False,
    no_patv_lags: bool = False,
    time_meteo_only: bool = False,
    meteo_max_lag: int = DEFAULT_METEO_MAX_LAG,
    include_day_of_year: bool = False,
    era5_at_target: bool = False,
    xgb_params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 30,
    return_test_predictions: bool = False,
    xgb_device: str | None = None,
) -> SdwpfRunResult:
    """Un seul XGBoost sur plusieurs turbines : empile les tables de modélisation et
    ajoute la colonne ``POOL_TURB_ID_COL`` (numérique). Découpage **par instants uniques**
    pour que train/val/test soient alignés dans le temps sur toutes les éoliennes.
    """
    if len(frames) != len(turb_ids):
        raise ValueError("frames and turb_ids must have the same length")
    if not frames:
        raise ValueError("at least one turbine frame is required")
    mlag = int(meteo_max_lag) if time_meteo_only else 0
    parts: list[pd.DataFrame] = []
    base_cols: list[str] | None = None
    for df, tid in zip(frames, turb_ids, strict=True):
        feat, cols = build_modeling_frame(
            df,
            horizon_steps=horizon_steps,
            max_lag=max_lag,
            no_patv_now=no_patv_now,
            no_patv_lags=no_patv_lags,
            time_meteo_only=time_meteo_only,
            meteo_max_lag=mlag,
            include_day_of_year=include_day_of_year,
            era5_at_target=era5_at_target,
        )
        if base_cols is None:
            base_cols = cols
        elif cols != base_cols:
            raise ValueError(
                "feature columns differ between turbines; check data / missing ERA5 columns"
            )
        block = feat.copy()
        block[POOL_TURB_ID_COL] = float(tid)
        parts.append(block)
    assert base_cols is not None
    feat_all = pd.concat(parts, ignore_index=True)
    feat_all = feat_all.sort_values(
        ["datetime", POOL_TURB_ID_COL], kind="mergesort"
    ).reset_index(drop=True)
    feature_cols = [POOL_TURB_ID_COL] + base_cols
    if val_frac is not None and val_frac > 0:
        train_idx, val_idx, test_idx = temporal_split_train_val_test_by_unique_datetime(
            feat_all["datetime"],
            train_frac,
            val_frac,
        )
    else:
        train_idx, test_idx = temporal_split_by_unique_datetime(
            feat_all["datetime"],
            train_frac,
        )
        val_idx = np.array([], dtype=np.int64)
    patv_desc = (
        describe_feature_mode(
            no_patv_now=no_patv_now,
            no_patv_lags=no_patv_lags,
            time_meteo_only=time_meteo_only,
            meteo_max_lag=mlag,
            include_day_of_year=include_day_of_year,
            era5_at_target=era5_at_target,
        )
        + f" | pooled {len(frames)} turbines ({POOL_TURB_ID_COL})"
    )
    return evaluate_on_indices(
        feat_all,
        feature_cols,
        horizon_steps=horizon_steps,
        patv_feature_mode=patv_desc,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        xgb_params=xgb_params,
        early_stopping_rounds=early_stopping_rounds,
        return_test_predictions=return_test_predictions,
        xgb_device=xgb_device,
    )


def load_frame_for_run(
    *,
    csv_path: Path,
    turb_id: int,
    chunksize: int,
    era5: bool,
    weather_csv: Path | None,
    default_weather: Path,
) -> pd.DataFrame:
    df = load_one_turbine(csv_path, turb_id, chunksize=chunksize)
    if era5:
        wpath = weather_csv if weather_csv is not None else default_weather
        era = load_era5_for_turbine(wpath, turb_id, chunksize=chunksize)
        df = merge_scada_era5(df, era)
    return df


def maybe_log_mlflow(
    result: SdwpfRunResult,
    *,
    project_root: Path,
    experiment: str,
    run_name: str,
    extra_params: dict[str, Any],
) -> None:
    if mlflow is None:
        print("mlflow not installed; from repo root: pip install -e \".[dev,experiments]\"")
        return
    runs_dir = (project_root / "mlruns").resolve()
    uri = os.environ.get("MLFLOW_TRACKING_URI", runs_dir.as_uri())
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    run_kw: dict[str, Any] = {}
    if run_name.strip():
        run_kw["run_name"] = run_name.strip()
    with mlflow.start_run(**run_kw):
        mlflow.log_params(
            {
                **extra_params,
                "horizon_steps": result.horizon_steps,
                "horizon_minutes": result.horizon_minutes,
                "n_features": result.n_features,
                "n_rows": result.n_rows,
                "patv_feature_mode": result.patv_feature_mode,
                "n_purged": result.n_purged,
            }
        )
        metrics_base: dict[str, float] = {
            "naive_mean_train_test_mae": result.naive_mean_train_test_mae,
            "naive_mean_train_test_rmse": result.naive_mean_train_test_rmse,
        }
        for k, v in (
            ("persistence_test_mae", result.persistence_test_mae),
            ("persistence_test_rmse", result.persistence_test_rmse),
            ("climatology_test_mae", result.climatology_test_mae),
            ("climatology_test_rmse", result.climatology_test_rmse),
            ("power_curve_test_mae", result.power_curve_test_mae),
            ("power_curve_test_rmse", result.power_curve_test_rmse),
        ):
            if v is not None:
                metrics_base[k] = float(v)
        mlflow.log_metrics(metrics_base)
        if result.xgboost_val_mae is not None:
            vm = {"xgboost_val_mae": result.xgboost_val_mae}
            if result.xgboost_val_rmse is not None:
                vm["xgboost_val_rmse"] = result.xgboost_val_rmse
            mlflow.log_metrics(vm)
        if result.xgboost_test_mae is not None:
            xm = {"xgboost_test_mae": result.xgboost_test_mae}
            if result.xgboost_test_rmse is not None:
                xm["xgboost_test_rmse"] = result.xgboost_test_rmse
            for k, v in (
                ("xgboost_test_nmae_vs_pmax_pct", result.xgboost_test_nmae_vs_pmax_pct),
                ("xgboost_test_nmae_vs_mean_patv_pct", result.xgboost_test_nmae_vs_mean_patv_pct),
                ("xgboost_skill_vs_naive", result.xgboost_skill_vs_naive),
                ("xgboost_test_bias", result.xgboost_test_bias),
            ):
                if v is not None and isinstance(v, (float, int)) and np.isfinite(float(v)):
                    xm[k] = float(v)
            mlflow.log_metrics(xm)
        for k, v in (
            ("naive_test_nmae_vs_pmax_pct", result.naive_test_nmae_vs_pmax_pct),
            ("naive_test_nmae_vs_mean_patv_pct", result.naive_test_nmae_vs_mean_patv_pct),
            ("naive_test_bias", result.naive_test_bias),
        ):
            if isinstance(v, (float, int)) and np.isfinite(float(v)):
                mlflow.log_metric(k, float(v))
        if result.train_patv_pmax > 1e-12:
            mlflow.log_metric("train_patv_pmax", float(result.train_patv_pmax))
        if result.train_patv_mean > 1e-12:
            mlflow.log_metric("train_patv_mean", float(result.train_patv_mean))
        if result.top_importances:
            mlflow.log_dict(
                dict(result.top_importances[:10]),
                "feature_importance_top.json",
            )
        mlflow.log_dict(
            {"columns": result.feature_cols},
            "feature_columns.json",
        )
    print(f"MLflow run logged (experiment={experiment!r}, uri={uri})")
