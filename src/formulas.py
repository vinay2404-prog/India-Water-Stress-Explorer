"""Water-stress formulas used by the app.

All functions are pure and vectorised (work on floats, numpy arrays or pandas Series)
so the same code is used for the whole dataset and for a single district.
"""
import numpy as np
import pandas as pd
from scipy import stats


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


def trend_stats(years, depth):
    """OLS slope + significance for depth-to-water vs. year.

    Returns (slope, intercept, r, p_value, stderr). NaNs if fewer than 3 valid
    points or zero variance in x. Used by the data pipeline to flag trends
    that are not statistically distinguishable from zero.
    """
    x = np.asarray(years, float)
    y = np.asarray(depth, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    res = stats.linregress(x, y)
    return (float(res.slope), float(res.intercept), float(res.rvalue),
            float(res.pvalue), float(res.stderr))
def spi(monthly_rain, scale=3):
    """Standardised Precipitation Index (SPI) for a monthly rainfall series.

    `monthly_rain` : 1-D array-like of monthly rainfall (chronological, one value
                     per month). Returns an array of SPI values aligned to the
                     input (NaN for the first `scale-1` months and where the fit
                     is degenerate). Fitting is done per calendar month so that
                     seasonality is removed.
    """
    r = np.asarray(monthly_rain, float)
    n = len(r)
    out = np.full(n, np.nan)
    if n < scale + 12:
        return out
    # rolling k-month sums, aligned to the END of the window
    roll = pd.Series(r).rolling(scale, min_periods=scale).sum().to_numpy()
    # fit per calendar month of the *end* month
    months = (np.arange(n) % 12) + 1
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        idx = idx[idx >= scale - 1]
        vals = roll[idx]
        vals = vals[~np.isnan(vals)]
        if len(vals) < 8:
            continue
        # probability of zero
        p0 = float(np.mean(vals == 0))
        pos = vals[vals > 0]
        if len(pos) < 4:
            continue
        try:
            a, loc, sc = stats.gamma.fit(pos, floc=0)
        except Exception:
            continue
        for i in idx:
            x = roll[i]
            if np.isnan(x):
                continue
            if x == 0:
                cdf = p0
            else:
                cdf = p0 + (1 - p0) * stats.gamma.cdf(x, a, loc=0, scale=sc)
            cdf = min(max(cdf, 1e-4), 1 - 1e-4)
            out[i] = float(stats.norm.ppf(cdf))
    return out
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


FALKENMARK_THRESHOLDS = (500, 1000, 1700)  # absolute scarcity / scarcity / stress (m3/person/yr)


def falkenmark_classify(values):
    """Falkenmark water-scarcity class from renewable water per person (m3/yr).

    < 500  -> absolute scarcity
    500-1000 -> scarcity
    1000-1700 -> stress
    >= 1700 -> no stress
    """
    s = pd.Series(values, dtype=float)
    out = pd.Series(np.select([s < 500, s < 1000, s < 1700], ["Absolute scarcity", "Scarcity", "Stress"],
                              "No stress"), index=s.index)
    return out.where(s.notna())
CATEGORY_COLORS = {"Low": "#2a9d8f", "Moderate": "#e9c46a", "High": "#f4a261", "Critical": "#c1121f"}
CATEGORY_ORDER = ["Low", "Moderate", "High", "Critical"]


def classify(score, cuts=(35, 50, 65)):
    """Category from the index:  Low < c1 <= Moderate < c2 <= High < c3 <= Critical."""
    s = pd.Series(score, dtype=float)
    c1, c2, c3 = cuts
    out = pd.Series(np.select([s >= c3, s >= c2, s >= c1], ["Critical", "High", "Moderate"], "Low"), index=s.index)
    return out.where(s.notna())
