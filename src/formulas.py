"""Water-stress formulas used by the app.

All functions are pure and vectorised (work on floats, numpy arrays or pandas Series)
so the same code is used for the whole dataset and for a single district.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- hydrology
def depletion_rate(years, depth):
    """Ordinary least-squares slope of depth-to-water vs. year  [m / year].
        rate = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    Positive  -> water table is getting deeper (depleting).
    Negative  -> water table is rising (recovering)."""
    x = np.asarray(years, float)
    y = np.asarray(depth, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return np.nan
    return (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x ** 2).sum() - x.sum() ** 2)


def seasonal_rise(pre_depth, post_depth):
    """Δh = max(pre-monsoon depth - post-monsoon depth, 0)   [m]
    Depth is measured below ground, so a bigger pre-monsoon depth than post-monsoon
    depth means the water table rose during the monsoon."""
    return np.maximum(np.asarray(pre_depth, float) - np.asarray(post_depth, float), 0)


def recharge_wtf(sy, rise_m):
    """Water-Table-Fluctuation (WTF) recharge per unit area  [mm / year]
        R = Sy × Δh × 1000
    Sy = specific yield (dimensionless), Δh = seasonal water-table rise [m].
    (Gross recharge; draft during the monsoon is not added back.)"""
    return sy * np.asarray(rise_m, float) * 1000.0


def recharge_ratio(recharge_mm, rainfall_mm):
    """Recharge ratio  [% of rainfall]  =  R / P × 100"""
    return np.asarray(recharge_mm, float) / np.asarray(rainfall_mm, float) * 100.0


def monsoon_share(rain_jjas, rain_annual):
    """Monsoon dependence [%]  =  P(Jun–Sep) / P(annual) × 100"""
    return np.asarray(rain_jjas, float) / np.asarray(rain_annual, float) * 100.0


def years_to_reach(depth_now, rate, limit_depth):
    """Years until the pre-monsoon depth reaches `limit_depth` at the current rate
        t = (D_limit − D_now) / rate        (only if rate > 0; NaN otherwise)"""
    depth_now = np.asarray(depth_now, float)
    rate = np.asarray(rate, float)
    limit_depth = np.asarray(limit_depth, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (limit_depth - depth_now) / rate
    # NaN when rate <= 0 (table not deepening), or when the water table is already
    # at/below `limit_depth` (t <= 0): reporting "0 years" would be misleading.
    return np.where((rate > 0) & (t > 0), t, np.nan)


def projected_depth(depth_now, rate, years_ahead):
    """D(t) = D_now + rate × t"""
    return np.asarray(depth_now, float) + np.asarray(rate, float) * years_ahead


# --------------------------------------------------------- composite index
def minmax(s, lo_q=0.05, hi_q=0.95):
    """Robust min–max scaling to 0..1 (values clipped at the 5th / 95th percentile)
        z = (x − P5) / (P95 − P5)"""
    s = pd.Series(s, dtype=float)
    lo, hi = s.quantile(lo_q), s.quantile(hi_q)
    if hi == lo:
        # No usable spread (all values tied): return a neutral midpoint instead of
        # silently biasing every district to "no stress" (0.0) or "max stress" (1.0).
        return pd.Series(0.5, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


COMPONENTS = {
    "depth": "Pre-monsoon depth to water",
    "trend": "Depletion rate",
    "rain": "Rainfall deficit",
    "recharge": "Recharge deficit",
}


def stress_components(df):
    """Four 0..1 stress components (1 = most stressed)."""
    c = pd.DataFrame(index=df.index)
    c["depth"] = minmax(df["pre_depth"])              # deeper  -> more stress
    c["trend"] = minmax(df["trend"])                  # falling -> more stress
    c["rain"] = 1 - minmax(df["rain_annual"])         # drier   -> more stress
    c["recharge"] = 1 - minmax(df["recharge_ratio"])  # weaker recharge -> more stress
    return c


def weighted_index(components, weights):
    """WSI = 100 × Σ(wᵢ·cᵢ) / Σwᵢ      with cᵢ ∈ [0,1]
    Takes a components table so callers that already computed `stress_components`
    avoid computing it twice."""
    c = pd.DataFrame(components, dtype=float)
    w = pd.Series({k: weights[k] for k in c.columns if k in weights}, dtype=float)
    if w.sum() == 0:
        raise ValueError("At least one weight must be above zero.")
    w = w / w.sum()
    return 100 * sum(c[k] * w[k] for k in w.index)


def water_stress_index(df, weights):
    """WSI from a raw district table (convenience wrapper around
    stress_components + weighted_index)."""
    return weighted_index(stress_components(df), weights)


CATEGORY_COLORS = {"Low": "#2a9d8f", "Moderate": "#e9c46a", "High": "#f4a261", "Critical": "#c1121f"}
CATEGORY_ORDER = ["Low", "Moderate", "High", "Critical"]


def classify(score, cuts=(35, 50, 65)):
    """Category from the index:  Low < c1 <= Moderate < c2 <= High < c3 <= Critical."""
    s = pd.Series(score, dtype=float)
    c1, c2, c3 = cuts
    out = pd.Series(np.select([s >= c3, s >= c2, s >= c1], ["Critical", "High", "Moderate"], "Low"), index=s.index)
    return out.where(s.notna())
