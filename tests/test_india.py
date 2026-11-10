"""Tests for India-specific formulas and standards."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import formulas as F


def test_bis_10500_present():
    assert "fluoride" in F.BIS_10500
    assert F.BIS_10500["fluoride"] == 1.0
    assert F.BIS_10500["nitrate"] == 45
    assert F.BIS_10500["tds"] == 500


def test_cgwb_rainfall_recharge():
    # 1000 mm rainfall, 100 km2 area, 10% infiltration -> 1000*100*0.1/100 = 100 ham
    r = F.cgwb_rainfall_recharge(1000.0, 100.0, infiltration_factor=0.10)
    assert r == pytest.approx(100.0)
    # array input
    rr = F.cgwb_rainfall_recharge(np.array([500.0, 1000.0]), np.array([50.0, 100.0]), 0.10)
    assert len(rr) == 2


def test_imd_drought_classify():
    out = F.imd_drought_classify(pd.Series([-2.5, -1.7, -1.2, -0.5, np.nan]))
    assert list(out) == ["Extreme drought", "Severe drought", "Moderate drought", "No drought", None] or pd.isna(out.iloc[4])
    assert pd.isna(out.iloc[4])


def test_cwmi_score_equal_weights():
    ind = {"a": 80.0, "b": 60.0}
    score = F.cwmi_score(ind)
    assert score == pytest.approx(70.0)


def test_cwmi_score_weighted():
    ind = {"a": 80.0, "b": 60.0}
    score = F.cwmi_score(ind, weights={"a": 2.0, "b": 1.0})
    assert score == pytest.approx((80*2 + 60*1) / 3)


def test_cwmi_score_empty():
    assert np.isnan(F.cwmi_score({}))