"""Tests for SCADA / ERA5 cleaning helpers (no large CSV)."""

import numpy as np
import pandas as pd
import pytest

from sdwpf.constants import ERA5_EXTRA_COLS, SDWPF_V1_ANCHOR
from sdwpf.data import load_era5_for_turbine, merge_scada_era5, sanitize_scada_for_forecasting


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


def test_sanitize_scada_drops_negative_and_inf():
    out = sanitize_scada_for_forecasting(_tiny_scada())
    # Rows 0,1 valides ; 2 inf ; 3 Patv<0 ; 4 Wspd<0 ; 5 valide (Wdir NaN conservé)
    assert len(out) == 3
    assert not np.isinf(out["Patv"]).any()
    assert (out["Patv"] >= 0).all() and (out["Wspd"] >= 0).all()


def test_sanitize_scada_drops_invalid_wdir():
    df = pd.DataFrame(
        {
            "datetime": pd.date_range(pd.Timestamp(SDWPF_V1_ANCHOR), periods=3, freq="10min"),
            "Patv": [1.0, 2.0, 3.0],
            "Wspd": [4.0, 5.0, 6.0],
            "Wdir": [0.0, 500.0, 270.0],
            "Etmp": [20.0, 20.0, 20.0],
            "Itmp": [25.0, 25.0, 25.0],
        }
    )
    out = sanitize_scada_for_forecasting(df)
    assert len(out) == 2
    assert 500.0 not in out["Wdir"].values


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
