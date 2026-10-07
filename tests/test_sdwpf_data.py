"""Tests for SCADA / ERA5 cleaning helpers (no large CSV)."""

import numpy as np
import pandas as pd
import pytest

from sdwpf.constants import ERA5_EXTRA_COLS, SDWPF_V1_ANCHOR
from sdwpf.data import (
    load_era5_for_turbine,
    load_era5_for_turbines,
    load_one_turbine,
    load_turbines,
    merge_scada_era5,
    sanitize_scada_for_forecasting,
)


def _tiny_scada() -> pd.DataFrame:
    base = pd.Timestamp(SDWPF_V1_ANCHOR)
    t = pd.date_range(base, periods=6, freq="10min")
    return pd.DataFrame(
        {
            "datetime": t,
            "Patv": [0.0, 10.0, np.inf, -1.0, 5.0, 8.0],
            "Wspd": [3.0, 4.0, 5.0, 6.0, -0.1, 7.0],
            "Wdir": [90.0, 100.0, 180.0, 400.0, 0.0, np.nan],
            "Etmp": [20.0] * 6,
            "Itmp": [25.0] * 6,
        }
    )


def test_sanitize_scada_drops_inf_negative_wspd_and_bad_wdir():
    out = sanitize_scada_for_forecasting(_tiny_scada())
    # Rows 0,1 valides ; 2 inf ; 3 Wdir=400 ; 4 Wspd<0 ; 5 valide (Wdir NaN conservé)
    assert len(out) == 3
    assert not np.isinf(out["Patv"]).any()
    assert (out["Patv"] >= 0).all() and (out["Wspd"] >= 0).all()


def test_sanitize_scada_clips_negative_patv_to_zero():
    df = pd.DataFrame(
        {
            "datetime": pd.date_range(pd.Timestamp(SDWPF_V1_ANCHOR), periods=3, freq="10min"),
            "Patv": [-0.3, 0.0, 250.0],
            "Wspd": [1.5, 2.0, 7.0],
        }
    )
    out = sanitize_scada_for_forecasting(df)
    assert len(out) == 3
    assert out["Patv"].tolist() == [0.0, 0.0, 250.0]


def test_sanitize_scada_wdir_valid_range_is_minus_180_to_180():
    # Wdir = angle vent / nacelle : les valeurs négatives sont normales
    df = pd.DataFrame(
        {
            "datetime": pd.date_range(pd.Timestamp(SDWPF_V1_ANCHOR), periods=5, freq="10min"),
            "Patv": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Wspd": [4.0, 5.0, 6.0, 7.0, 8.0],
            "Wdir": [-180.0, -12.5, 180.0, 270.0, -200.0],
            "Etmp": [20.0] * 5,
            "Itmp": [25.0] * 5,
        }
    )
    out = sanitize_scada_for_forecasting(df)
    assert out["Wdir"].tolist() == [-180.0, -12.5, 180.0]


def test_sanitize_scada_requires_columns():
    with pytest.raises(ValueError):
        sanitize_scada_for_forecasting(pd.DataFrame({"Patv": [1], "Wspd": [2]}))


def test_merge_scada_era5_drops_rows_with_inf_in_era5():
    base = pd.Timestamp(SDWPF_V1_ANCHOR)
    t = pd.date_range(base, periods=2, freq="10min")
    scada = pd.DataFrame(
        {
            "datetime": t,
            "Patv": [1.0, 2.0],
            "Wspd": [5.0, 6.0],
            "Wdir": [0.0, 0.0],
            "Etmp": [20.0, 20.0],
            "Itmp": [25.0, 25.0],
        }
    )
    era = pd.DataFrame({"datetime": t, **{c: [0.1, np.inf] for c in ERA5_EXTRA_COLS}})
    out = merge_scada_era5(scada, era)
    assert len(out) == 1
    assert float(out.iloc[0]["Patv"]) == 1.0
    era_mat = out[list(ERA5_EXTRA_COLS)].to_numpy(dtype=float)
    assert not np.isinf(era_mat).any()


def test_load_era5_filters_negative_wspd_w(tmp_path):
    csv = tmp_path / "w.csv"
    rows = []
    t0 = pd.Timestamp("2020-05-01 00:00:00")
    for i in range(3):
        row = {"TurbID": 1, "Tmstamp": (t0 + pd.Timedelta(minutes=10 * i)).isoformat()}
        for c in ERA5_EXTRA_COLS:
            row[c] = 1.0
        rows.append(row)
    rows[1]["Wspd_w"] = -3.0
    pd.DataFrame(rows).to_csv(csv, index=False)
    df = load_era5_for_turbine(csv, turb_id=1, chunksize=10)
    assert len(df) == 2


def _write_scada_csv(path, turb_ids, n_steps):
    rows = []
    for step in range(n_steps):
        for tid in turb_ids:  # interleaved, like the real file
            rows.append(
                {
                    "TurbID": tid,
                    "Day": step // 144 + 1,
                    "Tmstamp": f"{(step % 144) // 6:02d}:{(step % 6) * 10:02d}",
                    "Wspd": 5.0 + tid,
                    "Wdir": -5.0,
                    "Etmp": 20.0,
                    "Itmp": 25.0,
                    "Patv": float(100 * tid + step),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_turbines_single_pass_matches_one_by_one(tmp_path):
    csv = tmp_path / "scada.csv"
    _write_scada_csv(csv, [1, 2, 3], n_steps=300)
    frames = load_turbines(csv, [3, 1], chunksize=97)
    assert set(frames) == {1, 3}
    for tid in (1, 3):
        one = load_one_turbine(csv, tid, chunksize=97)
        pd.testing.assert_frame_equal(frames[tid], one)
        assert len(one) == 300 and (one["TurbID"] == tid).all()
    assert frames[1]["datetime"].iloc[150] == pd.Timestamp(SDWPF_V1_ANCHOR) + pd.Timedelta(minutes=1500)


def test_load_turbines_reports_missing_ids(tmp_path):
    csv = tmp_path / "scada.csv"
    _write_scada_csv(csv, [1], n_steps=10)
    with pytest.raises(ValueError, match="TurbID=7"):
        load_turbines(csv, [1, 7])


def test_load_era5_for_turbines_groups_by_id(tmp_path):
    csv = tmp_path / "w.csv"
    t0 = pd.Timestamp("2020-05-01 00:00:00")
    rows = []
    for i in range(4):
        for tid in (1, 2):
            row = {"TurbID": tid, "Tmstamp": (t0 + pd.Timedelta(minutes=10 * i)).isoformat()}
            row.update({c: float(tid) for c in ERA5_EXTRA_COLS})
            rows.append(row)
    pd.DataFrame(rows).to_csv(csv, index=False)
    eras = load_era5_for_turbines(csv, [1, 2], chunksize=3)
    assert len(eras[1]) == len(eras[2]) == 4
    assert (eras[2]["Wspd_w"] == 2.0).all()
