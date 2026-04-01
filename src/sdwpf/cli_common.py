"""Arguments CLI et validations partagées par les scripts (sans logique métier)."""

from __future__ import annotations

import argparse
from typing import Any

from sdwpf.features import resolve_horizon_steps
from sdwpf.pipeline import resolve_xgb_device_spec

XGB_DEVICE_HELP = "auto | cpu | cuda | cuda:N ; si omis → SDWPF_XGB_DEVICE ou auto."


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
