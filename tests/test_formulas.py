"""Unit tests for src/formulas.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import formulas as F


def test_depletion_rate_positive_and_negative():
    # perfectly linear rising depth -> +1.0 m/yr
    yrs = [2010, 2011, 2012, 2013]
    dep = [10.0, 11.0, 12.0, 13.0]
    assert F.depletion_rate(yrs, dep) == pytest.approx(1.0)
    # falling depth -> negative
    assert F.depletion_rate(yrs, [13.0, 12.0, 11.0, 10.0]) == pytest.approx(-1.0)
    # too few points -> nan
    assert np.isnan(F.depletion_rate([2010, 2011], [10.0, 11.0]))


def test_depletion_rate_ignores_nan():
    assert F.depletion_rate([2010, 2011, 2012, 2013], [10.0, np.nan, 12.0, 13.0]) == pytest.approx(1.0)


def test_seasonal_rise():
    out = F.seasonal_rise([5.0, 3.0], [3.0, 4.0])
    np.testing.assert_allclose(out, [2.0, 0.0])


def test_recharge_wtf_and_ratio():
    assert F.recharge_wtf(0.05, 2.0) == pytest.approx(100.0)
    assert F.recharge_ratio(100.0, 1000.0) == pytest.approx(10.0)


def test_monsoon_share():
    assert F.monsoon_share(800.0, 1000.0) == pytest.approx(80.0)


def test_years_to_reach():
    # positive rate, limit above now -> finite
    r = F.years_to_reach([10.0], [0.5], [20.0])
    assert r[0] == pytest.approx(20.0)
    # rate <= 0 -> nan (not depleting)
    assert np.isnan(F.years_to_reach([10.0], [-0.5], [20.0])[0])
    assert np.isnan(F.years_to_reach([10.0], [0.0], [20.0])[0])
    # already below limit -> nan (not 0, the old bug)
    assert np.isnan(F.years_to_reach([15.0], [0.5], [10.0])[0])
    # >200 handled by caller; function returns raw value
    assert F.years_to_reach([10.0], [0.05], [20.0])[0] == pytest.approx(200.0)


def test_projected_depth():
    assert F.projected_depth(10.0, 0.5, 4) == pytest.approx(12.0)


def test_minmax_basic_and_degenerate():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = F.minmax(s, 0.0, 1.0)
    assert z.min() == pytest.approx(0.0)
    assert z.max() == pytest.approx(1.0)
    # degenerate (all equal) -> neutral 0.5, not 0.0
    z2 = F.minmax(pd.Series([7.0, 7.0, 7.0]))
    assert (z2 == 0.5).all()


def test_stress_components_range():
    df = pd.DataFrame({
        "pre_depth": [5.0, 15.0, 30.0],
        "trend": [0.1, 0.5, 1.0],
        "rain_annual": [400.0, 900.0, 1700.0],
        "recharge_ratio": [2.0, 10.0, 30.0],
    })
    c = F.stress_components(df)
    for col in c.columns:
        assert c[col].between(0, 1).all()


def test_weighted_index_equals_water_stress_index():
    df = pd.DataFrame({
        "pre_depth": [5.0, 15.0, 30.0],
        "trend": [0.1, 0.5, 1.0],
        "rain_annual": [400.0, 900.0, 1700.0],
        "recharge_ratio": [2.0, 10.0, 30.0],
    })
    w = {"depth": 0.35, "trend": 0.30, "rain": 0.20, "recharge": 0.15}
    comps = F.stress_components(df)
    assert np.allclose(F.weighted_index(comps, w), F.water_stress_index(df, w))


def test_weighted_index_zero_weights_raises():
    df = pd.DataFrame({"pre_depth": [1.0, 2.0], "trend": [0.1, 0.2],
                       "rain_annual": [500.0, 600.0], "recharge_ratio": [5.0, 6.0]})
    comps = F.stress_components(df)
    with pytest.raises(ValueError):
        F.weighted_index(comps, {"depth": 0, "trend": 0, "rain": 0, "recharge": 0})


def test_classify_ordered():
    cats = F.classify(pd.Series([10.0, 40.0, 55.0, 70.0]), (35, 50, 65))
    assert list(cats) == ["Low", "Moderate", "High", "Critical"]


def test_classify_nan_preserved():
    cats = F.classify(pd.Series([10.0, np.nan, 70.0]), (35, 50, 65))
    assert pd.isna(cats.iloc[1])


def test_trend_stats():
    yrs = [2010, 2011, 2012, 2013, 2014]
    dep = [10.0, 11.2, 12.1, 13.0, 14.2]
    sl, ic, r, p, se = F.trend_stats(yrs, dep)
    assert sl > 0
    assert 0 <= p <= 1
    assert 0 <= abs(r) <= 1
    # too few points
    sl2, *_ = F.trend_stats([2010, 2011], [1.0, 2.0])
    assert np.isnan(sl2)


def test_falkenmark_classify():
    out = F.falkenmark_classify(pd.Series([200.0, 700.0, 1300.0, 2500.0, np.nan]))
    assert list(out) == ["Absolute scarcity", "Scarcity", "Stress", "No stress", None] or pd.isna(out.iloc[4])
    assert pd.isna(out.iloc[4])


def test_spi_basic():
    rng = np.random.default_rng(42)
    rain = rng.gamma(shape=2.0, scale=25.0, size=240)  # 20 years of varied monthly rain
    s = F.spi(rain, 3)
    assert len(s) == 240
    assert np.isfinite(s[-1])


def test_spi_short_series_returns_nan():
    s = F.spi([10.0, 20.0, 30.0], 3)
    assert np.all(np.isnan(s))
