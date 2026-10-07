"""Smoke tests: every script runs end to end on a small fake SDWPF (same CSV layout)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdwpf.constants import ERA5_EXTRA_COLS

pytest.importorskip("xgboost")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parent.parent
N_DAYS = 12


@pytest.fixture(scope="module")
def fake_sdwpf(tmp_path_factory) -> tuple[Path, Path]:
    """SCADA (TurbID, Day, Tmstamp, …, Patv) + weather v2 for 3 turbines, with idle Patv < 0."""
    rng = np.random.default_rng(0)
    n = N_DAYS * 144
    t = pd.date_range("2020-05-01", periods=n, freq="10min")
    wind = np.clip(7 + np.cumsum(rng.normal(0, 0.25, n)), 0, 20)
    scada, weather = [], []
    for tid in (1, 2, 3):
        v = np.clip(wind + rng.normal(0, 0.5, n), 0, None)
        patv = np.clip(1500 * ((v - 3) / 9) ** 3, 0, 1500) * (v >= 3)
        patv = np.where(v < 3, -0.3, patv + rng.normal(0, 20, n))
        scada.append(
            pd.DataFrame(
                {
                    "TurbID": tid,
                    "Day": np.arange(n) // 144 + 1,
                    "Tmstamp": t.strftime("%H:%M"),
                    "Wspd": v,
                    "Wdir": rng.uniform(-15, 15, n),
                    "Etmp": 20.0,
                    "Itmp": 30.0,
                    "Patv": patv,
                }
            )
        )
        w = pd.DataFrame({"TurbID": tid, "Tmstamp": t.strftime("%Y-%m-%d %H:%M:%S")})
        for c in ERA5_EXTRA_COLS:
            w[c] = rng.normal(1.0, 0.1, n)
        w["Wspd_w"] = np.clip(wind + rng.normal(0, 0.5, n), 0, None)
        weather.append(w)
    d = tmp_path_factory.mktemp("sdwpf")
    scada_csv, weather_csv = d / "scada.csv", d / "weather.csv"
    pd.concat(scada).to_csv(scada_csv, index=False)
    pd.concat(weather).to_csv(weather_csv, index=False)
    return scada_csv, weather_csv


def _run(script: str, *args: object) -> str:
    env = {**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def _data_args(fake_sdwpf: tuple[Path, Path]) -> list[object]:
    scada_csv, weather_csv = fake_sdwpf
    return ["--csv", scada_csv, "--weather-csv", weather_csv, "--chunksize", 5000]


def test_explore_meteo_with_oracle(fake_sdwpf):
    out = _run("sdwpf_explore.py", *_data_args(fake_sdwpf), "--meteo-mode", "--era5-at-target")
    assert "XGBoost test" in out
    assert "Climatology" in out and "Power curve" in out


def test_explore_pooled(fake_sdwpf):
    out = _run("sdwpf_explore.py", *_data_args(fake_sdwpf), "--meteo-mode", "--pooled", "1-2")
    assert "pooled 2 turbines" in out


def test_benchmark_writes_reports(fake_sdwpf, tmp_path):
    _run(
        "sdwpf_benchmark.py",
        *_data_args(fake_sdwpf),
        "--meteo-mode",
        "--horizons-steps",
        "36,144",
        "--out-dir",
        tmp_path,
    )
    (csv,) = tmp_path.glob("*.csv")
    table = pd.read_csv(csv)
    assert list(table["horizon_steps"]) == [36, 144]
    assert table["climatology_mae"].notna().all()
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_walkforward(fake_sdwpf):
    out = _run(
        "sdwpf_walkforward.py",
        *_data_args(fake_sdwpf),
        "--meteo-mode",
        "--n-splits",
        2,
        "--test-size",
        200,
        "--val-frac-within-train",
        0.3,
    )
    assert "fold 2/2" in out and "Aggregates" in out


def test_visualize_exports_all_figures(fake_sdwpf, tmp_path):
    _run(
        "sdwpf_visualize.py",
        *_data_args(fake_sdwpf),
        "--meteo-mode",
        "--turb-ids",
        "1-2",
        "--out-dir",
        tmp_path,
    )
    names = sorted(p.name[:2] for p in tmp_path.glob("*.png"))
    assert names == ["01", "02", "03", "04", "05", "06"]


def test_reproduce_dry_run():
    out = _run("reproduce_meteo_figures.py", "--dry-run", "--preset", "all")
    assert "--val-frac" in out and "1-100" in out
