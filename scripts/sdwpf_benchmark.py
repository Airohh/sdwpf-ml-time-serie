"""
Multi-horizon benchmark: XGBoost sur SDWPF (horizon en pas 10 min, h ≥ 1 possible).

Writes CSV + Markdown under reports/ for interview / MLflow-style traceability.

Usage (repo root):
  python scripts/sdwpf_benchmark.py
  python scripts/sdwpf_benchmark.py --meteo-mode --turb-id 1
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd

from sdwpf import (
    DEFAULT_METEO_MAX_LAG,
    default_scada_csv_path,
    default_weather_csv_path,
    load_frame_for_run,
    project_root,
    train_and_evaluate,
)
from sdwpf.pipeline import resolve_xgb_device_spec


def _label_hours(steps: int) -> str:
    h = steps * 10 / 60.0
    if h < 24:
        return f"{h:.0f}h" if h == int(h) else f"{h:.1f}h"
    d = h / 24.0
    return f"{d:.1f}d" if d != int(d) else f"{int(d)}d"


def main() -> None:
    p = argparse.ArgumentParser(description="SDWPF multi-horizon benchmark table")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--turb-id", type=int, default=1)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="Bloc validation temporel (entre train et test). Requiert train_frac+val_frac<1.",
    )
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Avec --val-frac, arrêt anticipé XGBoost (0=désactivé).",
    )
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument(
        "--horizons-steps",
        type=str,
        default="144,288,432,864",
        help="Pas 10 min, séparés par des virgules (ex. 36=6h, 144=1j). Défaut : 1 j … 6 j",
    )
    p.add_argument(
        "--meteo-mode",
        action="store_true",
        help="Temps (calendrier) + ERA5 seulement, sans Patv ni mesures SCADA vent/temp; "
        "implique --era5",
    )
    p.add_argument("--era5", action="store_true")
    p.add_argument(
        "--meteo-max-lag",
        type=int,
        default=DEFAULT_METEO_MAX_LAG,
        help="Avec --meteo-mode : retards ERA5 1..k.",
    )
    p.add_argument("--weather-csv", type=Path, default=None)
    p.add_argument(
        "--xgb-device",
        type=str,
        default=None,
        metavar="SPEC",
        help="auto | cpu | cuda | cuda:N ; si omis → SDWPF_XGB_DEVICE ou auto.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: reports/ under repo root",
    )
    args = p.parse_args()
    try:
        xgb_dev = resolve_xgb_device_spec(args.xgb_device)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    if args.val_frac is not None and args.val_frac <= 0:
        raise SystemExit("--val-frac must be positive when set")
    if args.val_frac is not None and args.train_frac + args.val_frac >= 1:
        raise SystemExit("--train-frac + --val-frac must be < 1")

    era5 = args.era5 or args.meteo_mode
    no_patv_now = args.meteo_mode
    no_patv_lags = args.meteo_mode

    csv_path = args.csv or default_scada_csv_path()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    if era5:
        wpath = args.weather_csv or default_weather_csv_path()
        if not wpath.is_file():
            raise SystemExit(f"ERA5 CSV not found: {wpath}")

    horizons = [int(x.strip()) for x in args.horizons_steps.split(",") if x.strip()]
    if any(h < 1 for h in horizons):
        raise SystemExit("All horizon steps must be >= 1")
    if args.meteo_max_lag < 0:
        raise SystemExit("--meteo-max-lag must be >= 0")

    print(f"Loading TurbID={args.turb_id} (ERA5={era5}, meteo_mode={args.meteo_mode})...")
    df = load_frame_for_run(
        csv_path=csv_path,
        turb_id=args.turb_id,
        chunksize=args.chunksize,
        era5=era5,
        weather_csv=args.weather_csv,
        default_weather=default_weather_csv_path(),
    )
    print(f"Rows: {len(df)}")

    rows: list[dict] = []
    for h_steps in horizons:
        r = train_and_evaluate(
            df,
            horizon_steps=h_steps,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            max_lag=6,
            no_patv_now=no_patv_now,
            no_patv_lags=no_patv_lags,
            time_meteo_only=args.meteo_mode,
            meteo_max_lag=args.meteo_max_lag,
            early_stopping_rounds=args.early_stopping_rounds,
            xgb_device=xgb_dev,
        )
        x_mae = r.xgboost_test_mae
        n_mae = r.naive_mean_train_test_mae
        p_mae = r.persistence_test_mae
        gain = (
            (100.0 * (1.0 - x_mae / n_mae)) if x_mae is not None and n_mae > 0 else None
        )
        gain_p = (
            (100.0 * (1.0 - x_mae / p_mae))
            if x_mae is not None and p_mae is not None and p_mae > 0
            else None
        )
        rows.append(
            {
                "horizon_steps": h_steps,
                "horizon_label": _label_hours(h_steps),
                "horizon_minutes": r.horizon_minutes,
                "n_rows": r.n_rows,
                "n_features": r.n_features,
                "naive_mean_mae": n_mae,
                "naive_mean_rmse": r.naive_mean_train_test_rmse,
                "persistence_mae": p_mae,
                "persistence_rmse": r.persistence_test_rmse,
                "xgboost_val_mae": r.xgboost_val_mae,
                "xgboost_val_rmse": r.xgboost_val_rmse,
                "xgboost_mae": x_mae,
                "xgboost_rmse": r.xgboost_test_rmse,
                "gain_vs_naive_mae_pct": gain,
                "gain_vs_persistence_mae_pct": gain_p,
            }
        )
        if r.xgboost_test_mae is not None:
            vtxt = ""
            if r.xgboost_val_mae is not None:
                vtxt = f" val_MAE={r.xgboost_val_mae:.2f}"
            ptxt = ""
            if gain_p is not None:
                ptxt = f" gain_vs_persist={gain_p:.1f}%"
            gtxt = f"{gain:.1f}%" if gain is not None else "—"
            print(
                f"  h={h_steps} ({_label_hours(h_steps)}): "
                f"naive_MAE={n_mae:.2f}{vtxt} test_MAE={r.xgboost_test_mae:.2f} "
                f"(gain naive {gtxt}){ptxt}"
            )
        else:
            print(f"  h={h_steps}: naive_MAE={n_mae:.2f} (no xgb)")

    out_dir = args.out_dir or (project_root() / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "meteo" if args.meteo_mode else "full_feats"
    csv_out = out_dir / f"sdwpf_benchmark_{tag}_{ts}.csv"
    md_out = out_dir / f"sdwpf_benchmark_{tag}_{ts}.md"

    table = pd.DataFrame(rows)
    table.to_csv(csv_out, index=False)

    mode_line = (
        "**Mode:** temps calendaire (sin/cos) + ERA5 seulement — pas de `Patv`, pas de vent "
        "/ températures SCADA dans les features\n\n"
        if args.meteo_mode
        else "**Mode:** toutes les features Patv autorisées par défaut\n\n"
    )
    vf = args.val_frac
    split_note = (
        f"**Découpage:** train {args.train_frac:.0%} | val {vf:.0%} | test {1 - args.train_frac - vf:.0%}  \n\n"
        if vf is not None
        else f"**Découpage:** train {args.train_frac:.0%} | test {1 - args.train_frac:.0%}  \n\n"
    )
    md_lines = [
        "# SDWPF benchmark",
        "",
        f"**TurbID:** {args.turb_id}  ",
        split_note.rstrip(),
        f"**ERA5:** {era5}  ",
        mode_line.rstrip(),
        "",
        "| Horizon | Steps | min | Naive MAE | Pers. MAE | XGB val MAE | XGB test MAE | Gain vs naive % | Gain vs pers. % | test RMSE |",
        "|---------|-------|-----|-----------|-----------|-------------|--------------|-----------------|-----------------|-----------|",
    ]
    for row in rows:
        xmae = row["xgboost_mae"]
        xrmse = row["xgboost_rmse"]
        vmae = row["xgboost_val_mae"]
        nmae = row["naive_mean_mae"]
        pmae = row["persistence_mae"]
        g = row["gain_vs_naive_mae_pct"]
        gp = row["gain_vs_persistence_mae_pct"]
        xstr = f"{xmae:.2f}" if xmae is not None else "—"
        vstr = f"{vmae:.2f}" if vmae is not None else "—"
        rstr = f"{xrmse:.2f}" if xrmse is not None else "—"
        gstr = f"{g:.1f}" if g is not None else "—"
        gpstr = f"{gp:.1f}" if gp is not None else "—"
        pmstr = f"{pmae:.2f}" if pmae is not None else "—"
        md_lines.append(
            f"| {row['horizon_label']} | {row['horizon_steps']} | "
            f"{row['horizon_minutes']} | {nmae:.2f} | {pmstr} | {vstr} | {xstr} | {gstr} | {gpstr} | {rstr} |"
        )
    md_lines += ["", f"CSV: `{csv_out.name}`", ""]
    md_out.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {csv_out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
