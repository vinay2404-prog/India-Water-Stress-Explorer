"""Unit tests for the Tier-2 formulas (demand, quality, drought, trend uncertainty)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import formulas as F


def test_effective_rainfall():
    # P <= 250
    assert F.effective_rainfall(100.0) == pytest.approx(100 * (125 - 20) / 125)
    # P > 250
    assert F.effective_rainfall(300.0) == pytest.approx(125 + 0.1 * 300)
    # negative -> nan
    assert np.isnan(F.effective_rainfall(-10.0))


def test_crop_et():
    assert F.crop_et(5.0, 1.2) == pytest.approx(6.0)


def test_irrigation_water_requirement():
    # IWR = max(ETc - Peff, 0)
    assert F.irrigation_water_requirement(10.0, 3.0) == pytest.approx(7.0)
    assert F.irrigation_water_requirement(3.0, 10.0) == pytest.approx(0.0)


def test_wei():
    assert F.wei(40.0, 100.0) == pytest.approx(40.0)
    # zero renewable -> nan
    assert np.isnan(F.wei(10.0, 0.0))


def test_wqi():
    # single param: concentration == standard -> q=100, WQI=100
    assert F.wqi([50.0], [50.0]) == pytest.approx(100.0)
    # two params with equal weights (1/std)
    w = F.wqi([50.0, 100.0], [100.0, 200.0])
    assert w == pytest.approx(50.0)  # both q=50


def test_exceedance():
    assert F.exceedance(10.0, 5.0) == 1
    assert F.exceedance(3.0, 5.0) == 0


def test_thornthwaite_pet():
    pet = F.thornthwaite_pet(np.array([10.0, 20.0, 5.0, -2.0]))
    assert pet[0] > 0
    assert pet[3] == 0.0  # negative temp -> 0


def test_ols_confidence_interval():
    yrs = [2010, 2011, 2012, 2013, 2014]
    dep = [10.0, 11.2, 12.1, 13.0, 14.2]
    slope, lo, hi, p = F.ols_confidence_interval(yrs, dep)
    assert slope > 0
    assert lo < slope < hi
    assert 0 <= p <= 1
    # too few points
    s, *_ = F.ols_confidence_interval([2010, 2011], [1.0, 2.0])
    assert np.isnan(s)


def test_mann_kendall_increasing():
    x = np.arange(10, dtype=float) + np.random.default_rng(1).normal(0, 0.1, 10)
    S, Z, p, trend = F.mann_kendall(x)
    assert trend == "increasing"
    assert p < 0.05


def test_mann_kendall_no_trend():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 20)
    S, Z, p, trend = F.mann_kendall(x)
    assert trend == "no trend"


def test_mann_kendall_short():
    S, Z, p, trend = F.mann_kendall([1.0, 2.0])
    assert np.isnan(S)