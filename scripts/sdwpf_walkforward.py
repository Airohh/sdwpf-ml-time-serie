"""
Évaluation walk-forward : plusieurs fenêtres test en fin de série (train croissant).

Pour chaque pli : bloc test de taille fixe ; le train est tout ce qui précède.
Option : découper le train en train / validation (fin du train) pour early stopping.

Usage (racine du dépôt) :
  python scripts/sdwpf_walkforward.py --horizon-days 1 --n-splits 3 --test-size 5000
  python scripts/sdwpf_walkforward.py --meteo-mode --n-splits 2 --test-size 3000 --val-frac-within-train 0.12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

from sdwpf import (
    DEFAULT_METEO_MAX_LAG,
    build_modeling_frame,
    default_scada_csv_path,
    default_weather_csv_path,
    evaluate_on_indices,
    load_frame_for_run,
    resolve_horizon_steps,
    walk_forward_indices,
)
from sdwpf.pipeline import _patv_feature_description, resolve_xgb_device_spec


def main() -> None:
    p = argparse.ArgumentParser(description="SDWPF walk-forward (indices expansifs)")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--turb-id", type=int, default=1)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--horizon-hours", type=float, default=None)
    p.add_argument("--horizon-days", type=float, default=None)
    p.add_argument("--n-splits", type=int, default=3, help="Nombre de blocs test successifs (fin de série)")
    p.add_argument(
        "--test-size",
        type=int,
        default=5_000,
        help="Nombre de lignes (après features) par pli test",
    )
    p.add_argument(
        "--val-frac-within-train",
        type=float,
        default=None,
        help="Part validation = fin du bloc train de chaque pli (ex. 0.12). Sinon pas de val / pas d'early stopping.",
    )
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Utilisé seulement si --val-frac-within-train est défini",
    )
    p.add_argument("--meteo-mode", action="store_true")
    p.add_argument(
        "--meteo-max-lag",
        type=int,
        default=DEFAULT_METEO_MAX_LAG,
        help="Avec --meteo-mode : retards ERA5 1..k (0 = pas de lags ERA5).",
    )
    p.add_argument("--era5", action="store_true")
    p.add_argument("--weather-csv", type=Path, default=None)
    p.add_argument(
        "--xgb-device",
        type=str,
        default=None,
        metavar="SPEC",
        help="auto | cpu | cuda | cuda:N ; si omis → SDWPF_XGB_DEVICE ou auto.",
    )
    args = p.parse_args()
    try:
        xgb_dev = resolve_xgb_device_spec(args.xgb_device)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    if args.n_splits < 1 or args.test_size < 1:
        raise SystemExit("--n-splits and --test-size must be >= 1")
    if args.val_frac_within_train is not None and (
        args.val_frac_within_train <= 0 or args.val_frac_within_train >= 1
    ):
        raise SystemExit("--val-frac-within-train must be in (0, 1) when set")
    if args.meteo_max_lag < 0:
        raise SystemExit("--meteo-max-lag must be >= 0")

    era5 = args.era5 or args.meteo_mode
    no_patv_now = args.meteo_mode
    no_patv_lags = args.meteo_mode

    try:
        h = resolve_horizon_steps(args.horizon, args.horizon_hours, args.horizon_days)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    csv_path = args.csv or default_scada_csv_path()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    if era5:
        wpath = args.weather_csv or default_weather_csv_path()
        if not wpath.is_file():
            raise SystemExit(f"ERA5 CSV not found: {wpath}")

    print(f"Loading TurbID={args.turb_id} (ERA5={era5}, meteo={args.meteo_mode})...")
    df = load_frame_for_run(
        csv_path=csv_path,
        turb_id=args.turb_id,
        chunksize=args.chunksize,
        era5=era5,
        weather_csv=args.weather_csv,
        default_weather=default_weather_csv_path(),
    )
    print(f"Rows raw frame: {len(df)}")

    mmlag = args.meteo_max_lag if args.meteo_mode else 0
    feat, feature_cols = build_modeling_frame(
        df,
        horizon_steps=h,
        max_lag=6,
        no_patv_now=no_patv_now,
        no_patv_lags=no_patv_lags,
        time_meteo_only=args.meteo_mode,
        meteo_max_lag=mmlag,
    )
    n = len(feat)
    patv_desc = _patv_feature_description(
        no_patv_now=no_patv_now,
        no_patv_lags=no_patv_lags,
        time_meteo_only=args.meteo_mode,
        meteo_max_lag=mmlag,
    )
    try:
        folds = walk_forward_indices(n, n_splits=args.n_splits, test_size=args.test_size)
    except ValueError as e:
        raise SystemExit(
            f"{e}. Increase data, lower --n-splits or --test-size, or shorten horizon."
        ) from e

    print(
        f"Modeling rows={n}; horizon_steps={h}; {args.n_splits} folds x test_size={args.test_size}"
    )

    es = int(args.early_stopping_rounds)
    vf = args.val_frac_within_train

    xgb_maes: list[float] = []
    naive_maes: list[float] = []
    persist_maes: list[float] = []

    for k, (train_block, test_idx) in enumerate(folds):
        if vf is not None:
            ntr = len(train_block)
            n_va = max(1, int(round(ntr * vf)))
            if ntr - n_va < 2:
                raise SystemExit(
                    f"Fold {k}: train too short after val split ({ntr} rows). "
                    "Lower --val-frac-within-train or --test-size."
                )
            tr_idx = train_block[:-n_va]
            va_idx = train_block[-n_va:]
            es_eff = es if es > 0 else 0
        else:
            tr_idx, va_idx = train_block, np.array([], dtype=np.int64)
            es_eff = 0

        r = evaluate_on_indices(
            feat,
            feature_cols,
            horizon_steps=h,
            patv_feature_mode=patv_desc,
            train_idx=tr_idx,
            val_idx=va_idx,
            test_idx=test_idx,
            early_stopping_rounds=es_eff,
            xgb_device=xgb_dev,
        )
        xm = r.xgboost_test_mae
        if xm is None:
            raise SystemExit("xgboost not available; pip install xgboost")
        xgb_maes.append(xm)
        naive_maes.append(r.naive_mean_train_test_mae)
        if r.persistence_test_mae is not None:
            persist_maes.append(r.persistence_test_mae)
        ptxt = f"persist_MAE={r.persistence_test_mae:.2f}" if r.persistence_test_mae else "no persist"
        print(
            f"  fold {k + 1}/{args.n_splits}: train={len(tr_idx)} val={len(va_idx)} test={len(test_idx)} | "
            f"naive_MAE={r.naive_mean_train_test_mae:.2f} XGB_test_MAE={xm:.2f} | {ptxt}"
        )

    xa = np.array(xgb_maes, dtype=float)
    na = np.array(naive_maes, dtype=float)
    print("\n--- Aggregates (over folds) ---")
    def _msummary(a: np.ndarray) -> str:
        if len(a) < 2:
            return f"{float(a.mean()):.4f}"
        return f"mean={float(a.mean()):.4f} std={float(a.std(ddof=1)):.4f}"

    print(f"XGBoost test MAE:   {_msummary(xa)}")
    print(f"Naive test MAE:     {_msummary(na)}")
    if persist_maes:
        pa = np.array(persist_maes, dtype=float)
        print(f"Persistence MAE:    {_msummary(pa)}")


if __name__ == "__main__":
    main()
