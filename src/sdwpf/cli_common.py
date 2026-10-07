"""Arguments CLI et validations partagées par les scripts (sans logique métier)."""

from __future__ import annotations

import argparse
from typing import Any

from sdwpf.features import resolve_horizon_steps
from sdwpf.xgb_device import resolve_xgb_device_spec

XGB_DEVICE_HELP = "cpu | auto | cuda | cuda:N ; si omis → SDWPF_XGB_DEVICE ou cpu."


def add_xgb_device_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str | None = None,
) -> None:
    parser.add_argument(
        "--xgb-device",
        type=str,
        default=None,
        metavar="SPEC",
        help=help_text or XGB_DEVICE_HELP,
    )


def resolve_xgb_device_or_exit(raw: str | None) -> str:
    try:
        return resolve_xgb_device_spec(raw)
    except ValueError as e:
        raise SystemExit(str(e)) from e


def validate_temporal_fractions(train_frac: float, val_frac: float | None) -> None:
    if val_frac is not None and val_frac <= 0:
        raise SystemExit("--val-frac must be positive when set")
    if val_frac is not None and train_frac + val_frac >= 1:
        raise SystemExit("--train-frac + --val-frac must be < 1")


def validate_meteo_max_lag(meteo_max_lag: int) -> None:
    if meteo_max_lag < 0:
        raise SystemExit("--meteo-max-lag must be >= 0")


def resolve_horizon_steps_or_exit(
    horizon: int | None,
    horizon_hours: float | None,
    horizon_days: float | None,
) -> int:
    try:
        return resolve_horizon_steps(horizon, horizon_hours, horizon_days)
    except ValueError as e:
        raise SystemExit(str(e)) from e


def apply_meteo_mode_explore_style(args: Any) -> None:
    """``--meteo-mode`` : active ERA5 et retire Patv / lags Patv des features."""
    if args.meteo_mode:
        args.era5 = True
        args.no_patv_now = True
        args.no_patv_lags = True


def meteo_flags_benchmark_style(args: Any) -> tuple[bool, bool, bool]:
    """Retourne ``(era5, no_patv_now, no_patv_lags)`` comme benchmark / walkforward."""
    era5 = bool(args.era5 or args.meteo_mode)
    no = bool(args.meteo_mode)
    return era5, no, no


def parse_turb_ids(spec: str) -> list[int]:
    """``"1,2,5"`` ou ``"1-20,25"`` → IDs dans l'ordre donné, sans doublon."""
    out: list[int] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        try:
            ids = range(int(lo), int(hi) + 1) if sep else [int(part)]
        except ValueError as e:
            raise SystemExit(f"TurbID invalide : {part!r} (ex. 1,2,5 ou 1-20)") from e
        out.extend(i for i in ids if i not in out)
    if not out:
        raise SystemExit(f"Aucun TurbID dans {spec!r}")
    return out


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Options de features / modèle communes aux scripts SDWPF."""
    parser.add_argument(
        "--day-of-year",
        action="store_true",
        help="Ajoute le jour de l'année (sin/cos). Déconseillé sous un an de données : "
        "les jours du test n'existent pas dans le train.",
    )
    parser.add_argument(
        "--era5-at-target",
        action="store_true",
        help="Avec --meteo-mode : ajoute ERA5 à l'échéance t+h (prévision météo parfaite, "
        "borne haute inaccessible en opérationnel) et la baseline courbe de puissance.",
    )
    parser.add_argument(
        "--xgb-objective",
        type=str,
        default=None,
        metavar="OBJ",
        help="Objectif XGBoost (défaut reg:squarederror ; reg:absoluteerror optimise la MAE).",
    )


def validate_model_arguments(args: Any) -> None:
    if args.era5_at_target and not args.meteo_mode:
        raise SystemExit("--era5-at-target requires --meteo-mode")


def xgb_params_from_args(args: Any) -> dict[str, Any] | None:
    return {"objective": args.xgb_objective} if args.xgb_objective else None


def baseline_summary_lines(r: Any) -> list[str]:
    """Lignes d'affichage des baselines secondaires et de la purge (``SdwpfRunResult``)."""
    lines: list[str] = []
    if r.climatology_test_mae is not None:
        lines.append(
            f"Climatology (train mean by target hour) MAE={r.climatology_test_mae:.4f} "
            f"RMSE={r.climatology_test_rmse:.4f}"
        )
    if r.power_curve_test_mae is not None:
        lines.append(
            f"Power curve on ERA5 wind at t+h MAE={r.power_curve_test_mae:.4f} "
            f"RMSE={r.power_curve_test_rmse:.4f}"
        )
    if r.n_purged:
        lines.append(f"Purged {r.n_purged} train/val rows whose target overlaps the next block")
    return lines
