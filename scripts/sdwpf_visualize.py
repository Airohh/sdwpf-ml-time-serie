"""
Figures PNG pour SDWPF : série Patv / vent, scatter, comparaison test, importances,
tableau KPI + barres (05_kpi_performance_*.png), tableau métriques seul (06_tableau_metriques_*.png).

Une ou plusieurs turbines (--turb-ids) : avec plusieurs IDs, on agrège la moyenne
(± écart-type sur les séries et le jeu test) sur l’ensemble des éoliennes.

  python scripts/sdwpf_visualize.py --meteo-mode --horizon-days 1
  python scripts/sdwpf_visualize.py --meteo-mode --horizon-days 1 --turb-ids 1,2,3,5,8
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from sdwpf_paths import prepend_src, repo_root_from_here

_ROOT = repo_root_from_here(__file__)
prepend_src(_ROOT)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import use as mpl_use
from matplotlib.gridspec import GridSpec

mpl_use("Agg")

from sdwpf import (
    default_scada_csv_path,
    default_weather_csv_path,
    load_frame_for_run,
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
from sdwpf.constants import DEFAULT_METEO_MAX_LAG
from sdwpf.kpi_format import (
    extend_float_row_avg,
    extend_skill_row,
    fmt_float,
    fmt_pct_gain,
    triple_int_row,
    triple_stat_row,
)
from sdwpf.pipeline import SdwpfRunResult


def _parse_turb_ids(spec: str, fallback: int) -> list[int]:
    if not spec or not spec.strip():
        return [fallback]
    out = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not out:
        return [fallback]
    return out


def _file_tag(turb_ids: list[int], horizon_steps: int, meteo: bool) -> str:
    if len(turb_ids) == 1:
        t = f"t{turb_ids[0]}"
    elif len(turb_ids) <= 5:
        t = "multi_" + "-".join(str(x) for x in turb_ids)
    else:
        t = f"multi_n{len(turb_ids)}_{turb_ids[0]}-{turb_ids[-1]}"
    t += f"_h{horizon_steps}"
    if meteo:
        t += "_meteo"
    return t


def _feature_label_fr(name: str) -> str:
    """Libellé français pour les noms de variables (importances XGBoost)."""
    exact: dict[str, str] = {
        "hour_sin": "Heure du jour (sin)",
        "hour_cos": "Heure du jour (cos)",
        "doy_sin": "Jour dans l’année (sin)",
        "doy_cos": "Jour dans l’année (cos)",
        "patv_now": "Puissance active Patv (instant t)",
        "wdir": "Direction du vent (SCADA)",
        "etmp": "Température extérieure (SCADA)",
        "itmp": "Température intérieure / nacelle (SCADA)",
        "T2m": "Température à 2 m (ERA5)",
        "Sp": "Pression de surface (ERA5)",
        "RelH": "Humidité relative (ERA5)",
        "Wspd_w": "Vitesse du vent (ERA5)",
        "Wdir_w": "Direction du vent (ERA5)",
        "Tp": "Paramètre Tp (ERA5)",
        "Wspd_w_cube": "Vitesse vent ERA5 au cube (∝ énergie éolienne)",
        "Wdir_w_sin": "Direction vent ERA5 (sin)",
        "Wdir_w_cos": "Direction vent ERA5 (cos)",
    }
    if name in exact:
        return exact[name]
    m = re.match(r"^([A-Za-z0-9_]+)_lag(\d+)$", name)
    if m:
        base, k = m.group(1), int(m.group(2))
        fr_base = exact.get(base, base)
        return f"{fr_base}, retard {k} pas ({k * 10} min)"
    m = re.match(r"^patv_lag(\d+)$", name)
    if m:
        k = int(m.group(1))
        return f"Patv retardée de {k} pas ({k * 10} min)"
    m = re.match(r"^wspd_lag(\d+)$", name)
    if m:
        k = int(m.group(1))
        return f"Vitesse du vent Wspd retardée de {k} pas ({k * 10} min)"
    return name


def _export_performance_metrics_table_figure(
    results: list[SdwpfRunResult],
    turb_ids: list[int],
    horizon_steps: int,
    multi_label: str,
    train_frac: float,
    val_frac: float | None,
    out_path: Path,
    *,
    patv_mode_summary: str,
) -> None:
    """Figure dédiée : tableau visuel des mesures de performance (jeu test).

    Peu de turbines : métriques en lignes, turbines en colonnes.
    Beaucoup de turbines (≥ ``TRANSPOSE_METRICS_IF_N_TURBINES``) : une ligne par
    turbine (lisible), métriques en colonnes + ligne « μ parc ».
    """
    TRANSPOSE_METRICS_IF_N_TURBINES = 18
    n_t = len(results)

    xgb_mae = [r.xgboost_test_mae for r in results]
    naive_mae = [r.naive_mean_train_test_mae for r in results]
    persist_mae = [r.persistence_test_mae for r in results]

    def _avg(vals: list[float | None]) -> float | None:
        finite = [float(x) for x in vals if x is not None and np.isfinite(x)]
        return float(np.mean(finite)) if finite else None

    hz_h = horizon_steps * 10 / 60.0
    split_txt = f"Train {train_frac:.0%}"
    if val_frac is not None and val_frac > 0:
        split_txt += f" · Val {val_frac:.0%} · Test {1 - train_frac - val_frac:.0%}"
    else:
        split_txt += f" · Test {1 - train_frac:.0%}"

    subtitle = (
        f"Mode features : {patv_mode_summary} · "
        "nMAE = 100×MAE/réf. (Pmax train = max Patv sur le train). "
        "Skill > 0 ⇒ XGB meilleur que la baseline."
    )

    if n_t >= TRANSPOSE_METRICS_IF_N_TURBINES:
        col_labels = [
            "ID",
            "MAE XGB",
            "RMSE XGB",
            "MAE naïf",
            "nMAE\n%Pmax",
            "nMAE\n%μPatv",
            "Skill",
            "Gain %",
            "Bias XGB",
        ]
        rows_cell: list[list[str]] = []
        for i in range(n_t):
            r = results[i]
            sk = r.xgboost_skill_vs_naive
            sk_s = f"{float(sk):.3f}" if sk is not None and np.isfinite(sk) else "—"
            rows_cell.append(
                [
                    str(turb_ids[i]),
                    fmt_float(r.xgboost_test_mae, 1),
                    fmt_float(r.xgboost_test_rmse, 1),
                    fmt_float(naive_mae[i], 1),
                    fmt_float(r.xgboost_test_nmae_vs_pmax_pct, 1),
                    fmt_float(r.xgboost_test_nmae_vs_mean_patv_pct, 1),
                    sk_s,
                    fmt_pct_gain(xgb_mae[i], naive_mae[i]),
                    fmt_float(r.xgboost_test_bias, 1),
                ]
            )
        mn = _avg(naive_mae)
        mx = _avg(xgb_mae)
        sks = [
            float(s)
            for s in (r.xgboost_skill_vs_naive for r in results)
            if s is not None and np.isfinite(s)
        ]
        rows_cell.append(
            [
                "μ parc",
                fmt_float(_avg(xgb_mae), 1),
                fmt_float(_avg([r.xgboost_test_rmse for r in results]), 1),
                fmt_float(mn, 1),
                fmt_float(_avg([r.xgboost_test_nmae_vs_pmax_pct for r in results]), 1),
                fmt_float(_avg([r.xgboost_test_nmae_vs_mean_patv_pct for r in results]), 1),
                f"{float(np.mean(sks)):.3f}" if sks else "—",
                fmt_pct_gain(mx, mn) if mn is not None and mx is not None else "—",
                fmt_float(_avg([r.xgboost_test_bias for r in results]), 1),
            ]
        )
        skill_col_idx = 6
        dpi = 200
        n_data_rows = len(rows_cell)
        fig_h = min(48.0, max(10.0, 2.2 + n_data_rows * 0.22))
        fig = plt.figure(figsize=(14, fig_h), facecolor="#f6f7f9")
        fig.suptitle(
            "Mesures de performance — jeu de test (une ligne par turbine)",
            fontsize=14,
            fontweight="bold",
            color="#1a1d21",
            y=0.985,
        )
        fig.text(
            0.5,
            0.965,
            f"{multi_label} · ~{hz_h:.1f} h · {split_txt} · {n_t} turbines",
            ha="center",
            fontsize=11,
            color="#212529",
        )
        fig.text(0.5, 0.948, subtitle, ha="center", fontsize=10, color="#495057")
        ax = fig.add_axes([0.04, 0.02, 0.92, 0.915])
        ax.axis("off")
        tbl = ax.table(
            cellText=rows_cell,
            colLabels=col_labels,
            loc="upper center",
            cellLoc="center",
            colLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.12, 2.0)
        n_tbl_rows = n_data_rows + 1  # + header
        last_body_row = n_tbl_rows - 1
        for (row_i, col_j), cell in tbl.get_celld().items():
            if row_i == 0:
                cell.set_facecolor("#1a365d")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_fontsize(10)
                continue
            is_mean_row = row_i == last_body_row
            if is_mean_row:
                cell.set_facecolor("#fff9e6")
                cell.get_text().set_fontweight("bold")
                cell.set_edgecolor("#b8860b")
                cell.set_linewidth(1.2)
            else:
                stripe = "#ffffff" if row_i % 2 == 1 else "#f1f5f9"
                cell.set_facecolor(stripe)
                cell.set_edgecolor("#94a3b8")
            if col_j == skill_col_idx and not is_mean_row and row_i > 0:
                try:
                    txt = rows_cell[row_i - 1][col_j].replace("—", "").strip()
                    if txt:
                        v = float(txt)
                        if v > 0.001:
                            cell.set_facecolor("#c6f6d5")
                        elif v < -0.001:
                            cell.set_facecolor("#fed7d7")
                except ValueError:
                    pass
            cell.get_text().set_fontsize(10)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    # ——— mode large : peu de colonnes turbines ———
    col_labels = [f"Turbine {tid}" for tid in turb_ids]
    if n_t > 1:
        col_labels.append("Moyenne")

    def _row_from_optional(vals: list[float | None], nd: int = 2) -> list[str]:
        cells = [fmt_float(v, nd) for v in vals]
        if n_t > 1:
            a = _avg(vals)
            cells.append(fmt_float(a, nd) if a is not None else "—")
        return cells

    def _row_gain_vs_naive() -> list[str]:
        cells = [fmt_pct_gain(xgb_mae[i], naive_mae[i]) for i in range(n_t)]
        if n_t > 1:
            mn = _avg(naive_mae)
            mx = _avg(xgb_mae)
            cells.append(fmt_pct_gain(mx, mn) if mn is not None and mx is not None else "—")
        return cells

    def _row_gain_vs_persist() -> list[str]:
        cells = [fmt_pct_gain(xgb_mae[i], persist_mae[i]) for i in range(n_t)]
        if n_t > 1:
            mp = _avg(persist_mae)
            mx = _avg(xgb_mae)
            cells.append(fmt_pct_gain(mx, mp) if mp is not None and mx is not None else "—")
        return cells

    rows_lbl = []
    rows_cell = []

    def add_row(label: str, values: list[str]) -> None:
        rows_lbl.append(label)
        rows_cell.append(values)

    add_row("MAE XGBoost (test)", _row_from_optional(xgb_mae, 2))
    add_row("RMSE XGBoost (test)", _row_from_optional([r.xgboost_test_rmse for r in results], 2))
    add_row("MAE baseline — moy. cible train (test)", _row_from_optional(naive_mae, 2))
    add_row("RMSE baseline — moy. cible train (test)", _row_from_optional([r.naive_mean_train_test_rmse for r in results], 2))

    has_persist = any(x is not None and np.isfinite(x) for x in persist_mae)
    if has_persist:
        add_row("MAE persistance Patv(t) (test)", _row_from_optional(persist_mae, 2))

    add_row("nMAE XGB vs Pmax train (%)", _row_from_optional([r.xgboost_test_nmae_vs_pmax_pct for r in results], 2))
    add_row("nMAE XGB vs moy. Patv train (%)", _row_from_optional([r.xgboost_test_nmae_vs_mean_patv_pct for r in results], 2))
    add_row("Skill vs naive (1 − MAE/MAE)", extend_skill_row([r.xgboost_skill_vs_naive for r in results], n_t=n_t))
    add_row("Gain vs naive = Skill×100 (%)", _row_gain_vs_naive())
    add_row("Bias XGB (moy. y − ŷ, test)", _row_from_optional([r.xgboost_test_bias for r in results], 2))
    add_row("Bias baseline (test)", _row_from_optional([r.naive_test_bias for r in results], 2))

    if has_persist:
        add_row("Gain XGB vs persistance (%)", _row_gain_vs_persist())

    n_rows = len(rows_lbl)
    fig_h = max(8.0, 1.6 + n_rows * 0.55)
    fig = plt.figure(
        figsize=(min(18, 3.0 + 1.5 * len(col_labels)), fig_h),
        facecolor="#f6f7f9",
    )
    fig.suptitle(
        "Mesures de performance — jeu de test",
        fontsize=14,
        fontweight="bold",
        color="#1a1d21",
        y=0.97,
    )
    fig.text(
        0.5,
        0.925,
        f"{multi_label} · Horizon ≈ {hz_h:.2f} h ({horizon_steps} pas × 10 min) · {split_txt}",
        ha="center",
        fontsize=11,
        color="#333333",
    )
    fig.text(0.5, 0.89, subtitle, ha="center", fontsize=10, color="#495057", style="italic")
    ax = fig.add_axes([0.05, 0.04, 0.9, 0.82])
    ax.axis("off")

    tbl = ax.table(
        cellText=rows_cell,
        rowLabels=rows_lbl,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
        colLoc="center",
        rowLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.65)

    for (row_i, col_j), cell in tbl.get_celld().items():
        if row_i == 0 and col_j == -1:
            cell.set_facecolor("#1a365d")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            continue
        if row_i == 0:
            cell.set_facecolor("#1a365d")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            continue
        if col_j == -1:
            cell.set_facecolor("#e2e8f0")
            cell.get_text().set_fontweight("600")
            cell.set_edgecolor("#94a3b8")
            continue
        body_color = "#ffffff" if row_i % 2 == 1 else "#f1f5f9"
        cell.set_facecolor(body_color)
        cell.set_edgecolor("#94a3b8")
        if 1 <= row_i <= len(rows_lbl) and rows_lbl[row_i - 1].startswith("Skill vs naive"):
            try:
                txt = rows_cell[row_i - 1][col_j].replace("—", "").strip()
                if txt:
                    v = float(txt)
                    if v > 0.001:
                        cell.set_facecolor("#c6f6d5")
                    elif v < -0.001:
                        cell.set_facecolor("#fed7d7")
            except ValueError:
                pass

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _export_kpi_performance_figure(
    results: list[SdwpfRunResult],
    turb_ids: list[int],
    horizon_steps: int,
    multi_label: str,
    train_frac: float,
    val_frac: float | None,
    out_path: Path,
    *,
    patv_mode_summary: str,
) -> None:
    """Tableau des KPI + barres MAE (jeu test) : naive / persistance / XGBoost."""
    n_t = len(results)
    col_labels = [f"Turb. {tid}" for tid in turb_ids]
    if n_t > 1:
        col_labels.append("Moyenne")
    n_col = len(col_labels)

    def extend_float_row(vals: list[float | None]) -> list[str]:
        return extend_float_row_avg(vals, n_t=n_t, nd=3)

    xgb_mae_l = [r.xgboost_test_mae for r in results]
    naive_mae_l = [r.naive_mean_train_test_mae for r in results]
    persist_mae_l = [r.persistence_test_mae for r in results]

    hz_h = horizon_steps * 10 / 60.0
    split_txt = f"train {train_frac:.0%}"
    if val_frac is not None and val_frac > 0:
        split_txt += f" | val {val_frac:.0%} | test {1 - train_frac - val_frac:.0%}"
    else:
        split_txt += f" | test {1 - train_frac:.0%}"
    if len(split_txt) > 48:
        split_txt = split_txt[:45] + "…"

    row_labels = [
        "Horizon (pas)",
        "Horizon (~h)",
        "Lignes modélisation",
        "Nb features",
        "Découpage",
        "MAE naive (test)",
        "RMSE naive (test)",
        "MAE persistance (test)",
        "MAE XGB val",
        "MAE XGB test",
        "RMSE XGB test",
        "Pmax Patv train (proxy install.)",
        "Moy. Patv train",
        "nMAE naive vs Pmax (%)",
        "nMAE XGB vs Pmax (%)",
        "nMAE naive vs moy.Patv (%)",
        "nMAE XGB vs moy.Patv (%)",
        "Skill XGB vs naive (1−MAE/MAE)",
        "Gain XGB vs naive = Skill×100 (%)",
        "Bias naive (test)",
        "Bias XGB (test)",
        "Gain XGB vs persist. (%)",
    ]

    cells: list[list[str]] = []
    cells.append([str(horizon_steps)] * n_col)
    cells.append([f"{hz_h:.2f}"] * n_col)
    rn = [str(r.n_rows) for r in results]
    if n_t > 1:
        rn.append(str(int(round(float(np.mean([r.n_rows for r in results]))))))
    cells.append(rn)
    nf = [str(r.n_features) for r in results]
    if n_t > 1:
        nf.append(str(int(round(float(np.mean([r.n_features for r in results]))))))
    cells.append(nf)
    cells.append([split_txt] * n_col)
    cells.append(extend_float_row(naive_mae_l))
    cells.append(extend_float_row([r.naive_mean_train_test_rmse for r in results]))
    cells.append(extend_float_row(persist_mae_l))
    cells.append(extend_float_row([r.xgboost_val_mae for r in results]))
    cells.append(extend_float_row(xgb_mae_l))
    cells.append(extend_float_row([r.xgboost_test_rmse for r in results]))
    cells.append(extend_float_row([r.train_patv_pmax for r in results]))
    cells.append(extend_float_row([r.train_patv_mean for r in results]))
    cells.append(
        extend_float_row_avg(
            [r.naive_test_nmae_vs_pmax_pct for r in results], n_t=n_t, nd=2
        )
    )
    cells.append(
        extend_float_row_avg(
            [r.xgboost_test_nmae_vs_pmax_pct for r in results], n_t=n_t, nd=2
        )
    )
    cells.append(
        extend_float_row_avg(
            [r.naive_test_nmae_vs_mean_patv_pct for r in results], n_t=n_t, nd=2
        )
    )
    cells.append(
        extend_float_row_avg(
            [r.xgboost_test_nmae_vs_mean_patv_pct for r in results], n_t=n_t, nd=2
        )
    )
    cells.append(extend_skill_row([r.xgboost_skill_vs_naive for r in results], n_t=n_t))

    gains_naive = [fmt_pct_gain(xgb_mae_l[i], naive_mae_l[i]) for i in range(n_t)]
    if n_t > 1:
        mn = [float(x) for x in naive_mae_l if x is not None]
        mx = [float(x) for x in xgb_mae_l if x is not None]
        gains_naive.append(fmt_pct_gain(np.mean(mx), np.mean(mn)) if mn and mx else "—")
    cells.append(gains_naive)
    cells.append(extend_float_row([r.naive_test_bias for r in results]))
    cells.append(extend_float_row([r.xgboost_test_bias for r in results]))

    gains_p = [fmt_pct_gain(xgb_mae_l[i], persist_mae_l[i]) for i in range(n_t)]
    if n_t > 1:
        mp_ = [float(x) for x in persist_mae_l if x is not None and np.isfinite(x)]
        mx = [float(x) for x in xgb_mae_l if x is not None]
        gains_p.append(fmt_pct_gain(np.mean(mx), np.mean(mp_)) if mp_ and mx else "—")
    cells.append(gains_p)

    KPI_COMPACT_IF_N_TURBINES = 22
    compact_mode = n_t > KPI_COMPACT_IF_N_TURBINES
    if compact_mode:
        col_labels = ["Moy. parc", "Min", "Max"]

        naive_nums = [float(x) for x in naive_mae_l if x is not None and np.isfinite(x)]
        xgb_nums_f = [float(x) for x in xgb_mae_l if x is not None and np.isfinite(x)]
        mn_avg = float(np.mean(naive_nums)) if naive_nums else None
        mx_avg = float(np.mean(xgb_nums_f)) if xgb_nums_f else None
        gain_mean = fmt_pct_gain(mx_avg, mn_avg) if mn_avg and mx_avg else "—"
        mp_fin = [float(x) for x in persist_mae_l if x is not None and np.isfinite(x)]
        gain_p_mean = (
            fmt_pct_gain(float(np.mean(xgb_nums_f)), float(np.mean(mp_fin)))
            if mp_fin and xgb_nums_f
            else "—"
        )

        cells = [
            [str(horizon_steps)] * 3,
            [f"{hz_h:.2f}"] * 3,
            triple_int_row([r.n_rows for r in results]),
            triple_int_row([r.n_features for r in results]),
            [split_txt] * 3,
            triple_stat_row(naive_mae_l),
            triple_stat_row([r.naive_mean_train_test_rmse for r in results]),
            triple_stat_row(persist_mae_l),
            triple_stat_row([r.xgboost_val_mae for r in results]),
            triple_stat_row(xgb_mae_l),
            triple_stat_row([r.xgboost_test_rmse for r in results]),
            triple_stat_row([r.train_patv_pmax for r in results]),
            triple_stat_row([r.train_patv_mean for r in results]),
            triple_stat_row([r.naive_test_nmae_vs_pmax_pct for r in results], nd=2),
            triple_stat_row([r.xgboost_test_nmae_vs_pmax_pct for r in results], nd=2),
            triple_stat_row([r.naive_test_nmae_vs_mean_patv_pct for r in results], nd=2),
            triple_stat_row([r.xgboost_test_nmae_vs_mean_patv_pct for r in results], nd=2),
            triple_stat_row([r.xgboost_skill_vs_naive for r in results], nd=4),
            [gain_mean, "—", "—"],
            triple_stat_row([r.naive_test_bias for r in results]),
            triple_stat_row([r.xgboost_test_bias for r in results]),
            [gain_p_mean, "—", "—"],
        ]

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.35, 1.0], width_ratios=[1.55, 1.0], hspace=0.38, wspace=0.28)
    ax_tbl = fig.add_subplot(gs[0, :])
    ax_tbl.axis("off")
    t_compact = (
        f"\nRésumé agrégé : moyenne / min / max sur {n_t} turbines — "
        "lignes par machine : figure 06_tableau_metriques."
        if compact_mode
        else ""
    )
    ax_tbl.set_title(
        f"Indicateurs de performance (jeu test) — {multi_label}\n"
        f"Mode features : {patv_mode_summary}\n"
        "nMAE = 100×MAE/réf. (Pmax train = max Patv train ; sinon moy. Patv train). "
        "Skill = 1 − MAE_XGB/MAE_naive (>0 mieux)."
        + t_compact,
        fontsize=10 if not compact_mode else 9,
        pad=12,
    )

    table = ax_tbl.table(
        cellText=cells,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10 if compact_mode else 8)
    table.scale(1.15, 1.45 if compact_mode else 1.25)
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(weight="bold")

    ax_bar = fig.add_subplot(gs[1, 0])
    naive_nums = [float(x) for x in naive_mae_l if x is not None]
    xgb_nums = [float(x) for x in xgb_mae_l if x is not None]
    mean_naive = float(np.mean(naive_nums)) if naive_nums else 0.0
    mean_xgb = float(np.mean(xgb_nums)) if xgb_nums else 0.0
    persist_finite = [float(x) for x in persist_mae_l if x is not None and np.isfinite(x)]
    categories = ["Naive\n(moy. train)", "XGBoost\n(test)"]
    heights = [mean_naive, mean_xgb]
    colors = ["#6c757d", "#0d6efd"]
    if persist_finite:
        categories.insert(1, "Persistance\nPatv(t)")
        heights.insert(1, float(np.mean(persist_finite)))
        colors.insert(1, "#fd7e14")
    y_pos = np.arange(len(categories))
    ax_bar.barh(y_pos, heights, color=colors, height=0.55, edgecolor="white")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(categories, fontsize=9)
    ax_bar.set_xlabel("MAE sur le jeu test (moyenne sur les turbines)", fontsize=9)
    ax_bar.set_title("Comparaison des MAE (test)", fontsize=10)
    ax_bar.invert_yaxis()
    for i, v in enumerate(heights):
        ax_bar.text(v, i, f"  {v:.2f}", va="center", fontsize=9)

    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis("off")
    _skills = [
        float(s)
        for s in (r.xgboost_skill_vs_naive for r in results)
        if s is not None and np.isfinite(s)
    ]
    mean_skill = float(np.mean(_skills)) if _skills else float("nan")
    _biases_x = [
        float(b)
        for b in (r.xgboost_test_bias for r in results)
        if b is not None and np.isfinite(b)
    ]
    mean_bias_x = float(np.mean(_biases_x)) if _biases_x else float("nan")
    lines = [
        "Lecture rapide",
        "— MAE / RMSE : jeu test (même unité que Patv).",
        "— nMAE % : MAE normalisé (standard éolien, comparer entre parcs).",
        "— Skill : 1 − MAE_XGB/MAE_naive ; Gain % = Skill × 100.",
        "— Bias : moyenne (y − ŷ) ; >0 sous-prédit la puissance.",
        "— Persistance : seulement si Patv(t) est une feature.",
        "",
        f"Résumé : XGBoost test MAE moyen = {mean_xgb:.3f}",
        f"vs naive = {mean_naive:.3f} ({fmt_pct_gain(mean_xgb, mean_naive)} % de gain)",
    ]
    if np.isfinite(mean_skill):
        lines.append(f"Skill moyen (si fini) ≈ {mean_skill:.4f}")
    if np.isfinite(mean_bias_x):
        lines.append(f"Bias XGB moyen ≈ {mean_bias_x:.3f}")
    if persist_finite:
        mpv = float(np.mean(persist_finite))
        lines.append(f"vs persistance = {mpv:.3f} ({fmt_pct_gain(mean_xgb, mpv)} % de gain)")
    ax_txt.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax_txt.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="sans-serif",
        linespacing=1.35,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _average_importances(results: list[SdwpfRunResult]) -> list[tuple[str, float]]:
    acc: dict[str, list[float]] = {}
    for r in results:
        for name, val in r.top_importances:
            acc.setdefault(name, []).append(val)
    if not acc:
        return []
    mean_sorted = sorted(
        ((n, float(np.mean(v))) for n, v in acc.items()),
        key=lambda x: -x[1],
    )
    return mean_sorted[:10]


def main() -> None:
    p = argparse.ArgumentParser(
        description="SDWPF — export de graphiques PNG (1 ou plusieurs TurbID, moyenne)"
    )
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--turb-id", type=int, default=1)
    p.add_argument(
        "--turb-ids",
        type=str,
        default="",
        help="Liste séparée par virgules, ex. 1,2,3,5. Si vide, utilise --turb-id.",
    )
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="Bloc validation entre train et test (train+val < 1).",
    )
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Avec --val-frac, early stopping XGBoost.",
    )
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--era5", action="store_true")
    p.add_argument("--weather-csv", type=Path, default=None)
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Pas 10 min ; par défaut 144 (1 j) si aucun --horizon-hours/days",
    )
    p.add_argument("--horizon-hours", type=float, default=None)
    p.add_argument("--horizon-days", type=float, default=None)
    p.add_argument("--no-patv-now", action="store_true")
    p.add_argument("--no-patv-lags", action="store_true")
    p.add_argument("--meteo-mode", action="store_true")
    p.add_argument(
        "--meteo-max-lag",
        type=int,
        default=DEFAULT_METEO_MAX_LAG,
        help=(
            "Avec --meteo-mode : retards 1..k sur chaque colonne ERA5. "
            "0 = pas de lags (conserve Wspd_w³ et sin/cos direction)."
        ),
    )
    p.add_argument(
        "--series-points",
        type=int,
        default=1440,
        help="Nombre de derniers pas affichés sur le graphique de série temporelle",
    )
    p.add_argument(
        "--test-plot-points",
        type=int,
        default=800,
        help="Nombre maximal d’instants affichés sur le graphique de prévision (jeu test)",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--open-folder",
        action="store_true",
        help="Ouvre le dossier des figures dans l’explorateur Windows après export",
    )
    p.add_argument(
        "--no-kpi-dashboard",
        action="store_true",
        help="Ne génère pas les figures 05 et 06 (KPI + tableau métriques)",
    )
    add_xgb_device_argument(
        p,
        help_text=(
            "XGBoost : auto | cpu | cuda | cuda:N (ordinal GPU, voir doc XGBoost GPU). "
            "Si omis : SDWPF_XGB_DEVICE ou auto."
        ),
    )
    args = p.parse_args()
    xgb_dev = resolve_xgb_device_or_exit(args.xgb_device)

    apply_meteo_mode_explore_style(args)
    if args.meteo_mode:
        validate_meteo_max_lag(args.meteo_max_lag)

    validate_temporal_fractions(args.train_frac, args.val_frac)

    turb_ids = _parse_turb_ids(args.turb_ids, args.turb_id)
    horizon_steps = resolve_horizon_steps_or_exit(
        args.horizon, args.horizon_hours, args.horizon_days
    )

    csv_path = args.csv or default_scada_csv_path()
    if not csv_path.is_file():
        raise SystemExit(f"Fichier CSV introuvable : {csv_path}")
    if args.era5:
        wpath = args.weather_csv or default_weather_csv_path()
        if not wpath.is_file():
            raise SystemExit(f"Fichier CSV ERA5 introuvable : {wpath}")

    out_dir = args.out_dir or (project_root() / "reports" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _file_tag(turb_ids, horizon_steps, args.meteo_mode)

    multi_label = (
        f"moyenne sur {len(turb_ids)} turbines (n° {', '.join(map(str, turb_ids[:12]))}"
        + (" …" if len(turb_ids) > 12 else "")
        if len(turb_ids) > 1
        else f"turbine n° {turb_ids[0]}"
    )

    # --- Chargement et agrégation
    print(f"Chargement de {len(turb_ids)} turbine(s)...")
    series_parts: list[pd.DataFrame] = []
    scatter_parts: list[pd.DataFrame] = []
    results: list[SdwpfRunResult] = []

    for tid in turb_ids:
        df = load_frame_for_run(
            csv_path=csv_path,
            turb_id=tid,
            chunksize=args.chunksize,
            era5=args.era5,
            weather_csv=args.weather_csv,
            default_weather=default_weather_csv_path(),
        )
        series_parts.append(df[["datetime", "Patv", "Wspd"]].copy())
        scatter_parts.append(
            df.iloc[:: max(1, len(df) // max(5000 // len(turb_ids), 1))][
                ["Wspd", "Patv"]
            ].copy()
        )
        print(f"  Turbine {tid} : entraînement et évaluation...")
        try:
            results.append(
                train_and_evaluate(
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
                    return_test_predictions=True,
                    xgb_device=xgb_dev,
                )
            )
        except ValueError as e:
            raise SystemExit(str(e)) from e

    combo = pd.concat(series_parts, ignore_index=True)
    agg = (
        combo.groupby("datetime", as_index=False)
        .agg(
            Patv_mean=("Patv", "mean"),
            Patv_std=("Patv", "std"),
            Wspd_mean=("Wspd", "mean"),
            Wspd_std=("Wspd", "std"),
        )
        .sort_values("datetime")
    )
    agg["Patv_std"] = agg["Patv_std"].fillna(0.0)
    agg["Wspd_std"] = agg["Wspd_std"].fillna(0.0)

    n_tail = max(200, min(args.series_points, len(agg)))
    sub = agg.tail(n_tail)

    # --- 1) Série moyenne Patv + Wspd (bande ±1σ)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.fill_between(
        sub["datetime"],
        sub["Patv_mean"] - sub["Patv_std"],
        sub["Patv_mean"] + sub["Patv_std"],
        color="C0",
        alpha=0.2,
        linewidth=0,
    )
    ax1.plot(
        sub["datetime"],
        sub["Patv_mean"],
        color="C0",
        lw=1.0,
        label="Patv — moyenne sur les turbines",
    )
    ax1.set_ylabel("Puissance active Patv (moyenne, unité SCADA)")
    ax1.set_xlabel("Date et heure")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax1.tick_params(axis="x", rotation=30)
    ax2 = ax1.twinx()
    ax2.fill_between(
        sub["datetime"],
        sub["Wspd_mean"] - sub["Wspd_std"],
        sub["Wspd_mean"] + sub["Wspd_std"],
        color="C1",
        alpha=0.15,
        linewidth=0,
    )
    ax2.plot(
        sub["datetime"],
        sub["Wspd_mean"],
        color="C1",
        lw=0.8,
        alpha=0.9,
        label="Wspd — moyenne sur les turbines",
    )
    ax2.set_ylabel("Vitesse du vent Wspd (moyenne)")
    ax1.set_title(
        f"SDWPF — {multi_label}\n"
        f"Série puissance et vent (zone : écart-type entre turbines, ±1 σ)"
    )
    fig.tight_layout()
    p1 = out_dir / f"01_series_{tag}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Enregistré : {p1}")

    # --- 2) Scatter pool
    pool = pd.concat(scatter_parts, ignore_index=True)
    if len(pool) > 8000:
        pool = pool.sample(8000, random_state=42)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pool["Wspd"], pool["Patv"], s=4, alpha=0.2, c="C0")
    ax.set_xlabel("Vitesse du vent (Wspd, SCADA)")
    ax.set_ylabel("Puissance active (Patv)")
    ax.set_title(f"Nuage de points : vent vs puissance — {multi_label}")
    fig.tight_layout()
    p2 = out_dir / f"02_scatter_wspd_patv_{tag}.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Enregistré : {p2}")

    # --- 3) Test : moyenne par instant (plusieurs turbines => longueurs test parfois différentes)
    test_rows: list[pd.DataFrame] = []
    for tid, r in zip(turb_ids, results, strict=False):
        if r.test_datetime is None or r.y_test is None:
            continue
        n = len(r.test_datetime)
        x_col = r.xgboost_pred_test
        if x_col is None:
            x_col = np.full(n, np.nan, dtype=float)
        test_rows.append(
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(r.test_datetime),
                    "y": r.y_test,
                    "x": x_col,
                }
            )
        )
    if not test_rows:
        print(
            "Aucune prédiction sur le jeu test ; graphiques 03 (prévision) et 04 (importances) ignorés."
        )
    else:
        ct = pd.concat(test_rows, ignore_index=True)
        agg_t = (
            ct.groupby("datetime", as_index=False)
            .agg(
                y_mean=("y", "mean"),
                y_std=("y", "std"),
                x_mean=("x", "mean"),
                x_std=("x", "std"),
            )
            .sort_values("datetime")
        )
        for col in ("y_std", "x_std"):
            agg_t[col] = agg_t[col].fillna(0.0)

        xgb_maes = [r.xgboost_test_mae for r in results if r.xgboost_test_mae is not None]
        mae_title = "Jeu de test"
        if xgb_maes:
            mae_title = (
                f"EMA XGBoost (moyenne par turbine) = {float(np.mean(xgb_maes)):.1f}"
            )

        sub_t = agg_t.tail(min(args.test_plot_points, len(agg_t)))
        dt = sub_t["datetime"].to_numpy()
        y_mean = sub_t["y_mean"].to_numpy()
        y_std = sub_t["y_std"].to_numpy()
        x_mean = sub_t["x_mean"].to_numpy()
        x_std = sub_t["x_std"].to_numpy()
        has_x = bool(np.any(np.isfinite(x_mean) & (np.abs(x_mean) > 1e-9)))

        fig, ax = plt.subplots(figsize=(12, 4))
        n = len(dt)
        hz_min = horizon_steps * 10
        ax.plot(
            dt,
            y_mean,
            color="black",
            lw=1.1,
            label=f"Vérité terrain : Patv à t + {horizon_steps} pas ({hz_min} min)",
        )
        ax.fill_between(dt, y_mean - y_std, y_mean + y_std, color="0.5", alpha=0.15, linewidth=0)
        if has_x:
            ax.plot(
                dt,
                x_mean,
                color="C3",
                lw=1.0,
                label="Prévision XGBoost (moyenne par instant)",
            )
            ax.fill_between(
                dt,
                x_mean - x_std,
                x_mean + x_std,
                color="C3",
                alpha=0.12,
                linewidth=0,
            )
        ax.set_title(
            f"Jeu de test (fraction temporelle la plus récente) — {multi_label}\n"
            f"{mae_title} — {n} derniers instants affichés"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel("Puissance active (moyenne sur les turbines)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        p3 = out_dir / f"03_test_forecast_{tag}.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        print(f"Enregistré : {p3}")

        # --- 4) Importances moyennées
        avg_imp = _average_importances(results)
        if avg_imp:
            names, vals = zip(*avg_imp, strict=False)
            labels_fr = [_feature_label_fr(str(n)) for n in names]
            fig, ax = plt.subplots(figsize=(10, 4.5))
            y_pos = np.arange(len(names))
            ax.barh(y_pos, vals, color="steelblue")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_fr, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(
                "Importance moyenne (plus c’est élevé, plus la variable influence la prédiction)"
            )
            ax.set_title(
                f"XGBoost — variables les plus influentes (importance moyennée sur les turbines) — {multi_label}"
            )
            fig.tight_layout()
            p4 = out_dir / f"04_importance_{tag}.png"
            fig.savefig(p4, dpi=150)
            plt.close(fig)
            print(f"Enregistré : {p4}")

    if not args.no_kpi_dashboard:
        p5 = out_dir / f"05_kpi_performance_{tag}.png"
        _export_kpi_performance_figure(
            results,
            turb_ids,
            horizon_steps,
            multi_label,
            args.train_frac,
            args.val_frac,
            p5,
            patv_mode_summary=results[0].patv_feature_mode,
        )
        print(f"Enregistré : {p5}")
        p6 = out_dir / f"06_tableau_metriques_{tag}.png"
        _export_performance_metrics_table_figure(
            results,
            turb_ids,
            horizon_steps,
            multi_label,
            args.train_frac,
            args.val_frac,
            p6,
            patv_mode_summary=results[0].patv_feature_mode,
        )
        print(f"Enregistré : {p6}")

    print("Erreur moyenne absolue (EMA) sur le jeu de test, par turbine (XGBoost) :")
    for tid, r in zip(turb_ids, results, strict=False):
        xm = r.xgboost_test_mae
        print(f"  Turbine {tid} : XGBoost = {xm:.2f}" if xm is not None else f"  Turbine {tid} : —")

    print(f"\nDossier des figures PNG : {out_dir.resolve()}")
    if args.open_folder and sys.platform == "win32":
        os.startfile(out_dir)  # type: ignore[attr-defined]
    elif args.open_folder:
        import subprocess

        subprocess.Popen(["xdg-open", str(out_dir)])  # Linux


if __name__ == "__main__":
    main()
