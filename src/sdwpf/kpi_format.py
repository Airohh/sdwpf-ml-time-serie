"""Formatage des cellules pour tableaux KPI / métriques (figures 05–06)."""

from __future__ import annotations

import numpy as np


def fmt_float(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def fmt_pct_gain(model_mae: float | None, baseline_mae: float | None) -> str:
    if (
        model_mae is None
        or baseline_mae is None
        or not np.isfinite(model_mae)
        or baseline_mae <= 1e-12
    ):
        return "—"
    return f"{100.0 * (1.0 - float(model_mae) / float(baseline_mae)):.1f}"


def extend_float_row_avg(vals: list[float | None], *, n_t: int, nd: int = 3) -> list[str]:
    row = [fmt_float(v, nd) for v in vals]
    if n_t > 1:
        finite = [float(x) for x in vals if x is not None and np.isfinite(x)]
        row.append(fmt_float(float(np.mean(finite)), nd) if finite else "—")
    return row


def extend_skill_row(vals: list[float | None], *, n_t: int) -> list[str]:
    row: list[str] = []
    for v in vals:
        if v is None or not np.isfinite(v):
            row.append("—")
        else:
            row.append(f"{float(v):.4f}")
    if n_t > 1:
        finite = [float(x) for x in vals if x is not None and np.isfinite(x)]
        row.append(f"{float(np.mean(finite)):.4f}" if finite else "—")
    return row


def triple_stat_row(vals: list[float | None], nd: int = 3) -> list[str]:
    """Une ligne tableau compact : moyenne, min, max (chaînes formatées)."""
    finite = [float(x) for x in vals if x is not None and np.isfinite(x)]
    if not finite:
        return ["—", "—", "—"]
    return [
        fmt_float(float(np.mean(finite)), nd),
        fmt_float(float(np.min(finite)), nd),
        fmt_float(float(np.max(finite)), nd),
    ]


def triple_int_row(rows: list[int]) -> list[str]:
    if not rows:
        return ["—", "—", "—"]
    a = np.asarray(rows, dtype=float)
    return [
        str(int(round(float(np.mean(a))))),
        str(int(np.min(a))),
        str(int(np.max(a))),
    ]
