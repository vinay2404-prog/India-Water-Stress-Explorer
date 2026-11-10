"""Water-stress formulas used by the app.

All functions are pure and vectorised (work on floats, numpy arrays or pandas Series)
so the same code is used for the whole dataset and for a single district.
"""
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- India-specific standards & methods
# BIS IS 10500:2012 drinking water standards (India) - used for WQI
BIS_10500 = {
    "ph": (6.5, 8.5),        # (min, max) acceptable
    "tds": 500,              # mg/L (desirable)
    "fluoride": 1.0,         # mg/L (desirable) / 1.5 (permissible)
    "nitrate": 45,           # mg/L
    "arsenic": 0.01,         # mg/L
    "iron": 0.3,             # mg/L
    "chloride": 250,         # mg/L
    "sulphate": 200,         # mg/L
    "hardness": 200,         # mg/L (as CaCO3)
    "turbidity": 1,           # NTU
}

# Default WQI standards (BIS IS 10500:2012, India)
BIS_WQI_STANDARDS = {
    "ph": 8.5,
    "tds": 500,
    "fluoride": 1.0,
    "nitrate": 45,
    "arsenic": 0.01,
    "iron": 0.3,
    "chloride": 250,
    "sulphate": 200,
    "hardness": 200,
    "turbidity": 1,
}


def cgwb_rainfall_recharge(rainfall_mm, area_km2, infiltration_factor=0.10, specific_yield=None):
    """CGWB Ground Water Resource Estimation (GEC) rainfall recharge  [ham or MCM].

    GEC method:  R = (P - R_off) * A * f_i
    where f_i is the rainfall infiltration factor (fraction of rainfall that
    recharges groundwater). Typical f_i values (CGWB):
        alluvial ( Indo-Gangetic ): 0.10 - 0.20
        hard rock (peninsular):      0.05 - 0.10
        hilly:                      0.04 - 0.08

    rainfall_mm        : annual rainfall (mm)
    area_km2           : district area (km2)
    infiltration_factor: fraction (default 0.10)
    Returns recharge in ham (hectare-metres) if area in km2 and rainfall in mm:
        R (ham) = P (mm) * A (km2) * f_i / 100
    """
    p = np.asarray(rainfall_mm, float)
    a = np.asarray(area_km2, float)
    r = p * a * float(infiltration_factor) / 100.0  # ham
    if specific_yield is not None:
        # WTF cross-check: R = Sy * dh * A * 10000 (ham)
        pass
    return r


def imd_drought_classify(spi_values):
    """IMD operational drought classification from SPI.

    IMD (India Meteorological Department) uses SPI for drought monitoring:
        SPI >= -1.0          : No drought
        -1.5 <= SPI < -1.0   : Moderate drought
        -2.0 <= SPI < -1.5   : Severe drought
        SPI < -2.0           : Extreme drought
    """
    s = pd.Series(spi_values, dtype=float)
    out = pd.Series(np.select(
        [s < -2.0, s < -1.5, s < -1.0],
        ["Extreme drought", "Severe drought", "Moderate drought"],
        "No drought"), index=s.index)
    return out.where(s.notna())


def cwmi_score(indicators, weights=None):
    """NITI Aayog Composite Water Management Index (CWMI) score.

    CWMI uses 28 indicators across 9 themes (Groundwater, Surface water, etc.)
    with specific weights. This is a simplified version that accepts a dict of
    indicator values and optional weights.

    indicators : dict {indicator_name: value (0-100 scale)}
    weights    : dict {indicator_name: weight} (default: equal weights)
    Returns a single CWMI score (0-100).
    """
    if weights is None:
        weights = {k: 1.0 for k in indicators}
    total_w = sum(weights.get(k, 0) for k in indicators)
    if total_w == 0:
        return np.nan
    score = sum(indicators[k] * weights.get(k, 0) for k in indicators) / total_w
    return float(score)


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


# ---------------------------------------------------------------- demand / water balance
def effective_rainfall(precip_mm):
    """USDA SCS effective rainfall  [mm].

        Peff = P*(125 - 0.2P)/125   if P <= 250 mm
             = 125 + 0.1P            if P >  250 mm
    """
    p = np.asarray(precip_mm, float)
    out = np.where(p <= 250, p * (125 - 0.2 * p) / 125.0, 125.0 + 0.1 * p)
    return np.where(p < 0, np.nan, out)

def thornthwaite_pet(t_mean, lat_deg=20.0):
    """Thornthwaite potential evapotranspiration  [mm/month] (temperature-only).

    Simpler alternative to Penman-Monteith when only temperature is available.
    """
    t = np.asarray(t_mean, float)
    # heat index (only defined for t > 0)
    i = np.zeros_like(t, dtype=float)
    pos = t > 0
    i[pos] = (t[pos] / 5.0) ** 1.514
    I = np.sum(i)  # annual heat index (scalar-ish); handle array by summing over time
    if np.ndim(t) > 0:
        I = float(np.nansum(i))
    if I == 0:
        return np.zeros_like(t)
    a = 6.75e-7 * I ** 3 - 7.71e-5 * I ** 2 + 1.792e-2 * I + 0.49239
    # day-length correction factor (approx, depends on latitude & month)
    # Use a constant factor ~1.0 for simplicity; refine if month is known.
    with np.errstate(invalid="ignore"):
        pet = 1.6 * (10 * t / I) ** a
    pet = np.where(t <= 0, 0.0, pet)
    return pet


def blaney_criddle_pet(t_mean, lat_deg=20.0, doy=180):
    """Blaney-Criddle potential evapotranspiration  [mm/day].

    Used by CWC and irrigation departments in India. Requires only
    temperature and latitude (day-length factor).
    PET = p * (0.46 * T + 8.13)  where p = % of annual daytime hours
    """
    t = np.asarray(t_mean, float)
    p_factor = 0.29 + 0.05 * np.sin(2 * np.pi * (doy - 80) / 365.0)
    pet = p_factor * (0.46 * t + 8.13)
    pet = np.where(t <= 0, 0.0, pet)
    return pet


def crop_et(et0, kc):
    """Crop evapotranspiration  ETc = Kc * ET0  [mm/day or mm/month]."""
    return np.asarray(et0, float) * np.asarray(kc, float)


def irrigation_water_requirement(etc, precip_eff):
    """Irrigation water requirement  IWR = ETc - Peff  [mm]."""
    return np.maximum(np.asarray(etc, float) - np.asarray(precip_eff, float), 0.0)


def wei(abstraction, renewable):
    """Water Exploitation Index  WEI = abstraction / renewable * 100  [%]."""
    a = np.asarray(abstraction, float)
    r = np.asarray(renewable, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / r * 100.0
    return np.where(r > 0, out, np.nan)


# ---------------------------------------------------------------- water quality
def wqi(concentrations, standards=None, weights=None, param_names=None):
    """Weighted Arithmetic Water Quality Index.

    concentrations : array-like of measured values (one per parameter)
    standards      : array-like of permissible limits (same order). If None,
                     uses BIS IS 10500:2012 (India) standards.
    weights        : optional array-like of weights (default: 1/standard)
    param_names    : optional list of parameter names (used to look up BIS standards)
    Returns a scalar WQI (or array if inputs are 2-D).
    """
    c = np.asarray(concentrations, float)
    if standards is None:
        if param_names is None:
            param_names = list(BIS_WQI_STANDARDS.keys())
        standards = np.array([BIS_WQI_STANDARDS[p] for p in param_names], dtype=float)
    s = np.asarray(standards, float)
    if weights is None:
        w = 1.0 / s
    else:
        w = np.asarray(weights, float)
    q = c / s * 100.0
    if c.ndim == 1:
        return float(np.sum(q * w) / np.sum(w))
    return np.sum(q * w, axis=-1) / np.sum(w)


def exceedance(concentration, standard):
    """Binary exceedance flag: 1 if concentration > standard, else 0."""
    return (np.asarray(concentration, float) > np.asarray(standard, float)).astype(int)


# ---------------------------------------------------------------- drought (SPEI)

def ols_confidence_interval(years, depth, alpha=0.05):
    """Confidence interval for the OLS slope of depth vs year.

    Returns (slope, ci_low, ci_high, p_value). NaNs if < 3 points.
    """
    x = np.asarray(years, float)
    y = np.asarray(depth, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.std(x) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    res = stats.linregress(x, y)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 2)
    se = res.stderr
    return (float(res.slope), float(res.slope - tcrit * se),
            float(res.slope + tcrit * se), float(res.pvalue))


def mann_kendall(series):
    """Mann-Kendall trend test (non-parametric).

    Returns (S, Z, p_value, trend) where trend is 'increasing', 'decreasing', or 'no trend'.
    """
    x = np.asarray(series, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return (np.nan, np.nan, np.nan, "no trend")
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0:
                S += 1
            elif diff < 0:
                S -= 1
    # variance with tie correction
    from collections import Counter
    counts = Counter(x)
    tie_sum = sum(v * (v - 1) * (2 * v + 5) for v in counts.values() if v > 1)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18.0
    if var_s == 0:
        return (float(S), 0.0, 1.0, "no trend")
    if S > 0:
        Z = (S - 1) / np.sqrt(var_s)
    elif S < 0:
        Z = (S + 1) / np.sqrt(var_s)
    else:
        Z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(Z)))
    if p < 0.05:
        trend = "increasing" if S > 0 else "decreasing"
    else:
        trend = "no trend"
    return (float(S), float(Z), float(p), trend)
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


FALKENMARK_THRESHOLDS = (500, 1000, 1700)  # absolute scarcity / scarcity / stress (m3/person/yr)


def per_capita_classify(values):
    """Per-capita water availability class (Indian water resources context).

    Based on renewable water availability per person (m3/yr). India uses
    per-capita water availability as a key indicator (National Water Policy).
    < 500  -> absolute scarcity
    500-1000 -> scarcity
    1000-1700 -> stress
    >= 1700 -> no stress
    """
    s = pd.Series(values, dtype=float)
    out = pd.Series(np.select([s < 500, s < 1000, s < 1700], ["Absolute scarcity", "Scarcity", "Stress"],
                              "No stress"), index=s.index)
    return out.where(s.notna())


# Backward-compatible alias
falkenmark_classify = per_capita_classify
CATEGORY_COLORS = {"Low": "#2a9d8f", "Moderate": "#e9c46a", "High": "#f4a261", "Critical": "#c1121f"}
CATEGORY_ORDER = ["Low", "Moderate", "High", "Critical"]


def classify(score, cuts=(35, 50, 65)):
    """Category from the index:  Low < c1 <= Moderate < c2 <= High < c3 <= Critical."""
    s = pd.Series(score, dtype=float)
    c1, c2, c3 = cuts
    out = pd.Series(np.select([s >= c3, s >= c2, s >= c1], ["Critical", "High", "Moderate"], "Low"), index=s.index)
    return out.where(s.notna())
