"""Data-analysis pipeline.
Reads the two raw datasets and writes processed tables used by the Streamlit app.

    python src/build_data.py

Raw inputs  (data/raw):
  * india_groundwater_level_data_cgwb_1970-2025.csv   CGWB station readings (depth to water, m bgl)
  * district_wise_rainfall_normal.xlsx                IMD district rainfall normals (mm)
Outputs (data/processed):
  * district_metrics.csv   one row per district (groundwater + rainfall metrics)
  * district_year.csv      district x year pre/post-monsoon medians (for time series)
  * stations.csv           station-level recent depth and trend (for the map)
"""
import re
import difflib
from pathlib import Path
import numpy as np
import pandas as pd

import formulas as F

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)
RECENT = (2021, 2025)
TREND_FROM = 2010
MIN_TREND_YEARS = 8
MAX_DEPTH = 100

MONTH_PAT = re.compile(r"^(19|20)\d{2}-(0[1-9]|1[0-2])$")  # strict "YYYY-MM" columns

# ------------------------------------------------------------------ 1. groundwater
gw = pd.read_csv(RAW / "india_groundwater_level_data_cgwb_1970-2025.csv", low_memory=False)
month_cols = [c for c in gw.columns if MONTH_PAT.match(str(c))]
if not month_cols:
    raise ValueError("No YYYY-MM month columns found in the groundwater CSV - schema changed?")
gw = gw.copy()
gw["sid"] = np.arange(len(gw))
gw["state"] = gw["state"].astype(str).str.strip()
gw["district"] = gw["district"].astype(str).str.strip().str.title()
gw["well_depth_n"] = pd.to_numeric(gw["well_depth"], errors="coerce")

long = gw.melt(id_vars="sid", value_vars=month_cols, var_name="ym", value_name="d").dropna(subset=["d"])
n_raw = len(long)
long = long[(long.d > 0) & (long.d <= MAX_DEPTH)]  # drop impossible readings
print(f"readings: {n_raw:,} -> {len(long):,} after cleaning (0 < depth <= {MAX_DEPTH} m)")
long["year"] = long.ym.str[:4].astype(int)
long["mo"] = long.ym.str[5:].astype(int)


def season(mask, name):
    return long[mask].groupby(["sid", "year"]).d.mean().rename(name)


sy = pd.concat([season(long.mo.isin([4, 5]), "pre"), season(long.mo == 11, "post")], axis=1).reset_index()
sy["rise"] = F.seasonal_rise(sy.pre, sy.post)
sy = sy.merge(gw[["sid", "state", "district", "latitude", "longitude", "well_depth_n"]], on="sid")

# ---- station level: OLS depletion rate 2010-2025 + recent means
# (single OLS implementation: formulas.depletion_rate, same as the app uses)
t = sy[(sy.year >= TREND_FROM) & sy.pre.notna()].copy()
slope = t.groupby("sid")[["year", "pre"]].apply(
    lambda g: F.depletion_rate(g["year"], g["pre"])).rename("slope")
n = t.groupby("sid").pre.size().rename("n")
g = pd.concat([slope, n], axis=1)
g = g[(g.n >= MIN_TREND_YEARS) & g.slope.between(-5, 5)]
rec = sy[sy.year.between(*RECENT)].groupby("sid").agg(pre=("pre", "mean"), post=("post", "mean"), rise=("rise", "mean"))
st = gw.set_index("sid")[["stationName", "state", "district", "latitude", "longitude", "well_depth_n"]].join(rec).join(g[["slope"]])
st = st.rename(columns={"pre": "pre_depth", "post": "post_depth", "slope": "trend"})
old = sy[sy.year.between(2005, 2011)].groupby("sid").pre.mean().rename("pre_2005_11")
st = st.join(old)

# ---- district level (median of stations)
dist = st.groupby(["state", "district"]).agg(
    stations=("pre_depth", "count"), pre_depth=("pre_depth", "median"), post_depth=("post_depth", "median"),
    rise_m=("rise", "median"), trend=("trend", "median"), trend_n=("trend", "count"),
    pre_2005_11=("pre_2005_11", "median"),
    well_depth=("well_depth_n", lambda s: s[(s > 0) & (s <= 500)].median()),
    lat=("latitude", "mean"), lon=("longitude", "mean")).reset_index()

# ---- district x year series (for time-series charts)
dy = sy.groupby(["state", "district", "year"]).agg(pre=("pre", "median"), post=("post", "median"),
                                                   stations=("pre", "count")).reset_index()

# ------------------------------------------------------------------ 2. rainfall
r = pd.read_excel(RAW / "district_wise_rainfall_normal.xlsx")
SMAP = {"ORISSA": "Odisha", "UTTARANCHAL": "Uttarakhand", "HIMACHAL": "Himachal Pradesh", "CHATISGARH": "Chhattisgarh",
        "PONDICHERRY": "Puducherry", "DADAR NAGAR HAVELI": "Dadra & Nagar Haveli", "JAMMU AND KASHMIR": "Jammu & Kashmir"}
r["state"] = r.STATE_UT_NAME.map(lambda s: SMAP.get(s.strip(), s.strip().title()))


def norm(s):
    return re.sub(r"[^A-Z]", "", re.sub(r"\(.*?\)", "", str(s).upper()))


r["key"] = r.DISTRICT.map(norm)

# CGWB district names vs IMD normals use different (old/abbreviated) spellings.
# These verified aliases turn known pairs into EXACT matches instead of relying on
# fuzzy matching, which once paired S.B.S. Nagar with the rainfall of SAS NAGAR
# (Mohali) - a different district. Checked against every fuzzy match found earlier.
DISTRICT_ALIASES = {
    "Cuddapah": "KUDDAPAH", "Jehanabad": "JAHANABAD", "Purnia": "PURNEA",
    "Janjgir - Champa": "JANJGIR-CHAMP", "Dang": "DANGS", "Sonipat": "SONEPAT(RTK)",
    "Rajauri": "RAJOURI", "Kodarma": "KODERMA", "Palamau": "PALAMU",
    "Bangalore Rural": "BANGALORE RUR", "Bangalore Urban": "BANGALORE URB",
    "Belgaum": "BELGAM", "Chamarajanagara": "CHAMARAJANAGA",
    "Chikkamagaluru": "CHIKMAGALUR", "Dakshin Kannada": "DAKSHIN KANDA",
    "Davanagere": "DAVANGERE", "Shivamogga": "SHIMOGA", "Uttara Kannada": "UTTAR KANNADA",
    "Kannur": "CANNUR", "Kasaragod": "KASARGOD", "Buldana": "BULDHANA",
    "Sindudurg": "SINDHUDURG", "Anugul": "ANGUL", "Balangir": "BOLANGIR",
    "Jagatsinghapur": "JAGATSINGHAPU", "Jajapur": "JAJPUR",
    "Nabarangapur": "NAWARANGPUR", "Sonapur": "SONEPUR", "Mukatsar": "MUKTSAR",
    "Sbs Nagar": "NAWANSHAHR", "Kancheepuram": "KANCHIPURAM",
    "Ramanathapuram": "RAMANATHAPURA", "Tiruchirappalli": "TIRUCHIRAPPAL",
    "Tiruppur": "TIRUPUR", "Tiruvannamalai": "TIRUVANNAMALA",
    "Mahbubnagar": "MAHABUBNAGAR", "Budaun": "BADAUN",
    "Jyotiba Phule Nagar": "JYOTIBA PHULE", "Raibareli": "RAE BARELI",
    "Sant Kabir Nagar": "SANT KABIR NGR", "Siddharthnagar": "SIDDHARTH NGR",
    "Udham Singh Nagar": "UDHAM SINGH N", "Darjiling": "DARJEELING",
    "Maldah": "MALDA", "Puruliya": "PURULIA", "Fatehgarh Sahib": "FATEHGARH SAH",
    "Ashok Nagar": "ASHOKNAGAR(GNA", "Pratapgarh": "PRATAPGARH(CHT",
}


def match(row):
    st_ = "Andhra Pradesh" if row.state == "Telangana" else row.state  # Telangana was part of AP in the normals
    cand = r[r.state == st_]
    k = norm(row.district)
    alias = DISTRICT_ALIASES.get(row.district)
    if alias:
        ex = cand[cand.key == norm(alias)]
        if len(ex):
            return ex.iloc[0].ANNUAL, ex.iloc[0]["Jun-Sep"], "district name"
    ex = cand[cand.key == k]
    if len(ex):
        return ex.iloc[0].ANNUAL, ex.iloc[0]["Jun-Sep"], "district name"
    mm = difflib.get_close_matches(k, list(cand.key), n=1, cutoff=0.82)
    if mm:
        b = cand[cand.key == mm[0]].iloc[0]
        return b.ANNUAL, b["Jun-Sep"], "fuzzy name"
    if len(cand):
        return cand.ANNUAL.mean(), cand["Jun-Sep"].mean(), "state average"
    return np.nan, np.nan, "none"


res = dist.apply(match, axis=1, result_type="expand")
res.columns = ["rain_annual", "rain_jjas", "rain_match"]
dist = pd.concat([dist, res], axis=1)
dist["monsoon_share"] = F.monsoon_share(dist.rain_jjas, dist.rain_annual)
print("rainfall match:\n", dist.rain_match.value_counts().to_string())

dist = dist[dist.stations >= 3].dropna(subset=["pre_depth", "trend", "rain_annual", "rise_m"]).reset_index(drop=True)
print("districts analysed:", len(dist))

dist.round(3).to_csv(OUT / "district_metrics.csv", index=False)
dy.merge(dist[["state", "district"]], on=["state", "district"]).round(3).to_csv(OUT / "district_year.csv", index=False)
st[st.pre_depth.notna()].reset_index(names="sid").round(4).to_csv(OUT / "stations.csv", index=False)
print("written:", [p.name for p in OUT.iterdir()])
