"""
SDWPF: one turbine, temporal split, XGBoost sur Patv(t+h).

Horizon par défaut : 1 jour (144 pas de 10 min) ; h minimal = 1 pas.

Uses package ``sdwpf`` under ``src/``. Run from repo root:
  python scripts/sdwpf_explore.py
  python scripts/sdwpf_explore.py --meteo-mode --horizon-days 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sdwpf_paths import prepend_src, repo_root_from_here

_ROOT = repo_root_from_here(__file__)
prepend_src(_ROOT)

from sdwpf import (
    DEFAULT_METEO_MAX_LAG,
    default_scada_csv_path,
    default_weather_csv_path,
    load_frame_for_run,
    maybe_log_mlflow,
    project_root,
    train_and_evaluate,
)
from sdwpf.cli_common import (
    add_xgb_device_argument,
    apply_meteo_mode_explore_style,
    resolve_horizon_steps_or_exit,
    resolve_xgb_device_or_exit,
    validate_meteo_max_lag,
    validate_temporal_fractions,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="SDWPF: prévision Patv à l’horizon t+h (XGBoost)"
    )
    p.add_argument("--csv", type=Path, default=None, help="Path to sdwpf_245days_v1.csv")
    p.add_argument("--turb-id", type=int, default=1)
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Part du temps pour l’entraînement (chronologique). Avec --val-frac, "
        "validation puis test suivent sans mélange.",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="Part validation (entre train et test). Ex. 0.15 avec train 0.55 → "
        "~30 %% test. Active early stopping XGBoost.",
    )
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Si --val-frac > 0 : arrêt anticipé XGBoost (0 pour désactiver).",
    )
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument(
        "--era5",
        action="store_true",
        help="Merge ERA5 from sdwpf_weather_v2 (wtb* CSV)",
    )
    p.add_argument("--weather-csv", type=Path, default=None)
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Horizon en pas 10 min (ignoré si --horizon-hours/--horizon-days). "
        "Par défaut 144 pas (1 j) si rien n’est précisé.",
    )
    p.add_argument(
        "--horizon-hours",
        type=float,
        default=None,
        help="Forecast horizon in hours (1 h = 6 steps). Combine with --horizon-days.",
    )
    p.add_argument(
        "--horizon-days",
        type=float,
        default=None,
        help="Forecast horizon in days (1 d = 144 steps). Combine with --horizon-hours.",
    )
    p.add_argument("--no-patv-now", action="store_true")
    p.add_argument("--no-patv-lags", action="store_true")
    p.add_argument(
        "--meteo-mode",
        action="store_true",
        help="Shorthand: --era5 --no-patv-now --no-patv-lags; XGBoost voit seulement "
        "le temps (calendrier sin/cos) + ERA5, pas le vent SCADA ni Patv",
    )
    p.add_argument(
        "--meteo-max-lag",
        type=int,
        default=DEFAULT_METEO_MAX_LAG,
        help="Avec --meteo-mode : retards ERA5 1..k (0 = pas de lags sur ERA5).",
    )
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", type=str, default="sdwpf")
    p.add_argument("--mlflow-run-name", type=str, default="")
    add_xgb_device_argument(p)
    args = p.parse_args()
    xgb_dev = resolve_xgb_device_or_exit(args.xgb_device)

    apply_meteo_mode_explore_style(args)
    validate_meteo_max_lag(args.meteo_max_lag)

    validate_temporal_fractions(args.train_frac, args.val_frac)

    horizon_steps = resolve_horizon_steps_or_exit(
        args.horizon,
        args.horizon_hours,
        args.horizon_days,
    )

    csv_path = args.csv or default_scada_csv_path()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    print(f"Loading TurbID={args.turb_id} from {csv_path} ...")
    if args.era5:
        wpath = args.weather_csv or default_weather_csv_path()
        if not wpath.is_file():
            raise SystemExit(f"ERA5 CSV not found: {wpath}")
        print(f"Merging ERA5 from {wpath} ...")

    df = load_frame_for_run(
        csv_path=csv_path,
        turb_id=args.turb_id,
        chunksize=args.chunksize,
        era5=args.era5,
        weather_csv=args.weather_csv,
        default_weather=default_weather_csv_path(),
    )
    print(f"Rows in frame: {len(df)}")

    try:
        result = train_and_evaluate(
            df,
            horizon_steps=horizon_steps,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            max_lag=6,
            no_patv_now=args.no_patv_now,
            no_patv_lags=args.no_patv_lags,
            time_meteo_only=args.meteo_mode,
            meteo_max_lag=args.meteo_max_lag,
            early_stopping_rounds=args.early_stopping_rounds,
            xgb_device=xgb_dev,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    if args.val_frac is not None:
        te = 100 * (1.0 - args.train_frac - args.val_frac)
        split_lbl = (
            f"train {100 * args.train_frac:.0f}% | val {100 * args.val_frac:.0f}% | "
            f"test {te:.0f}%"
        )
    else:
        split_lbl = f"test last {100 * (1 - args.train_frac):.0f}%"
    print(
        f"Forecast horizon: {horizon_steps} step(s) ({result.horizon_minutes} min, "
        f"~{result.horizon_minutes / 60:.2f} h); rows: {result.n_rows}; "
        f"features: {result.n_features} ({result.patv_feature_mode})"
    )
    print(
        f"Naive mean(train y) ({split_lbl}) MAE={result.naive_mean_train_test_mae:.4f} "
        f"RMSE={result.naive_mean_train_test_rmse:.4f}"
    )
    if result.persistence_test_mae is not None:
        print(
            f"Persistence Patv(t)->Patv(t+h) ({split_lbl}) "
            f"MAE={result.persistence_test_mae:.4f} RMSE={result.persistence_test_rmse:.4f}"
        )
    if result.xgboost_val_mae is not None:
        print(
            f"XGBoost validation ({split_lbl}) MAE={result.xgboost_val_mae:.4f} "
            f"RMSE={result.xgboost_val_rmse:.4f}"
        )
    if result.xgboost_test_mae is not None:
        imp = 100.0 * (
            1.0
            - result.xgboost_test_mae / max(result.naive_mean_train_test_mae, 1e-9)
        )
        extra = ""
        if result.persistence_test_mae is not None and result.persistence_test_mae > 1e-9:
            gp = 100.0 * (1.0 - result.xgboost_test_mae / result.persistence_test_mae)
            extra = f" ; vs persistence MAE ~ {gp:.1f}%"
        print(
            f"XGBoost test ({split_lbl}) MAE={result.xgboost_test_mae:.4f} "
            f"RMSE={result.xgboost_test_rmse:.4f} (gain vs naive MAE ~ {imp:.1f}%){extra}"
        )
        print("Top feature importances:")
        for name, imp in result.top_importances:
            print(f"  {name:12s} {imp:.4f}")
    else:
        print("xgboost not installed; pip install xgboost")

    if args.mlflow:
        maybe_log_mlflow(
            result,
            project_root=project_root(),
            experiment=args.mlflow_experiment,
            run_name=args.mlflow_run_name,
            extra_params={
                "turb_id": args.turb_id,
                "train_frac": args.train_frac,
                "val_frac": args.val_frac,
                "early_stopping_rounds": args.early_stopping_rounds,
                "era5": args.era5,
                "no_patv_now": args.no_patv_now,
                "no_patv_lags": args.no_patv_lags,
                "meteo_mode": args.meteo_mode,
                "chunksize": args.chunksize,
            },
        )


if __name__ == "__main__":
    main()
