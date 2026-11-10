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
import json, math, re
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


def _station_stats(grp):
    sl, ic, r, p, se = F.trend_stats(grp["year"], grp["pre"])
    return pd.Series({"slope": sl, "pvalue": p, "r2": (r * r) if pd.notna(r) else np.nan,
                      "n": len(grp)})


stats_df = t.groupby("sid")[["year", "pre"]].apply(_station_stats)
g = stats_df[(stats_df["n"] >= MIN_TREND_YEARS) & stats_df["slope"].between(-5, 5)]
rec = sy[sy.year.between(*RECENT)].groupby("sid").agg(pre=("pre", "mean"), post=("post", "mean"), rise=("rise", "mean"))
st = gw.set_index("sid")[["stationName", "state", "district", "latitude", "longitude", "well_depth_n"]].join(rec).join(g[["slope", "pvalue", "r2"]])
st = st.rename(columns={"pre": "pre_depth", "post": "post_depth", "slope": "trend", "pvalue": "trend_pvalue", "r2": "trend_r2"})
old = sy[sy.year.between(2005, 2011)].groupby("sid").pre.mean().rename("pre_2005_11")
st = st.join(old)

# ---- district level (median of stations)
dist = st.groupby(["state", "district"]).agg(
    stations=("pre_depth", "count"), pre_depth=("pre_depth", "median"), post_depth=("post_depth", "median"),
    rise_m=("rise", "median"), trend=("trend", "median"), trend_n=("trend", "count"),
    trend_pvalue=("trend_pvalue", "median"), trend_r2=("trend_r2", "median"),
    trend_sig=("trend_pvalue", lambda s: float((s < 0.05).mean()) if len(s) else np.nan),
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

# ------------------------------------------------------------------ 3. Tier-1 enrichment: district area + geo_id (GeoJSON), SPI (optional).
GEO_PATH = ROOT / "data/geo/india_district.geojson"

# 3a. District area (km2) + geo_id from the GeoJSON (pure-Python spherical area).
def _spherical_area_km2(geom):
    """Area of a GeoJSON geometry in km2 (spherical trapezoid formula)."""
    def ring_area(coords):
        # coords: list of [lon, lat]
        R = 6371.0  # km
        if not coords:
            return 0.0
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        s = 0.0
        for i in range(len(coords) - 1):
            lon1, lat1 = math.radians(coords[i][0]), math.radians(coords[i][1])
            lon2, lat2 = math.radians(coords[i + 1][0]), math.radians(coords[i + 1][1])
            s += (lon2 - lon1) * (2.0 + math.sin(lat1) + math.sin(lat2))
        return abs(s) * R * R / 2.0
    gtype = geom["type"]
    coords = geom["coordinates"]
    if gtype == "Polygon":
        return ring_area(coords[0]) - sum(ring_area(r) for r in coords[1:])
    if gtype == "MultiPolygon":
        return sum(_spherical_area_km2({"type": "Polygon", "coordinates": p}) for p in coords)
    return 0.0


def _state_key(s):
    return (str(s).strip().title()
            .replace("Andaman And Nicobar Islands", "Andaman & Nicobar")
            .replace("Nct Of Delhi", "Delhi")
            .replace("Dadra And Nagar Haveli", "Dadra & Nagar Haveli")
            .replace("Daman And Diu", "Daman & Diu")
            .replace("Orissa", "Odisha")
            .replace("Pondicherry", "Puducherry")
            .replace("Uttaranchal", "Uttarakhand")
            .replace("Jammu And Kashmir", "Jammu & Kashmir"))


GEO_ALIASES = dict(DISTRICT_ALIASES)
GEO_ALIASES.update({"Telangana": "Andhra Pradesh"})
GEO_ALIASES.update({
    "Gurugram": "Gurgaon", "Ballari": "Bellary", "Kalaburagi": "Gulbarga",
    "Prayagraj": "Allahabad", "Ayodhya": "Faizabad", "Tuticorin": "Thoothukudi",
    "Mohali": "Rupnagar", "Sbs Nagar": "Nawan Shehar",
    "Cooch Behar": "Kochbihar", "Howrah": "Haora",
    "North 24 Parganas": "North 24 Parganas",
    "South 24 Parganas": "South 24 Parganas",
    "G.B. Nagar": "Gautam Buddha Nagar", "Trichur": "Thrissur",
    "Cuddapah": "Cuddapah", "Nellore": "Nellore",
    "Kanker": "Kanker", "Dang": "The Dangs",
    "Charkhi Dadri": "Bhiwani", "Nuh": "Gurgaon", "Agar Malwa": "Shajapur",
    "Palghar": "Thane", "Sonapur": "Subarnapur", "Fazilka": "Firozpur",
    "Pathankot": "Gurdaspur", "Ropar": "Rupnagar",
    "Praygraj": "Allahabad", "Jhargram": "West Midnapore",
    "North Dinajpur": "Uttar Dinajpur", "Paschim Bardhaman": "Barddhaman",
    "West Medinipur": "West Midnapore", "Gaurella Pendra Marwahi": "Bilaspur",
    "Kheri": "Lakhimpur Kheri", "Jyotiba Phule Nagar": "Jyotiba Phule Nagar",
    "Ayodhya": "Faizabad", "G.B. Nagar": "Gautam Buddha Nagar",
    "Mau": "Mau", "Champawat": "Champawat", "Thrissur": "Thrissur",
    "Kaimur": "Bhabua", "Kabirdham": "Kawardha", "Dohad": "Dahod",
    "Narsimhapur": "Narsinghpur", "Baudh": "Boudh", "Debagarh": "Deogarh",
    "Barnala": "Sangrur", "Tarn Taran": "Amritsar", "Dholpur": "Dhaulpur",
    "Krishnagiri": "Dharmapuri", "Tirunelveli": "Tirunelveli Kattabo",
    "Tiruppur": "Coimbatore", "Tuticorin": "Thoothukudi",
    "Warangal Rural": "Warangal", "Warangal Urban": "Warangal",
    "Kanpur Nagar": "Kanpur", "East Medinipur": "East Midnapore",
    "Purba Bardhaman": "Barddhaman", "South Dinajpur": "Dakshin Dinajpur",
    "Beed": "Bid", "Mumbai City": "Greater Bombay", "Mumbai Suburban": "Greater Bombay",
    "Raigad": "Raigarh", "Kullu": "Mandi", "Reasi": "Udhampur",
    "Hamirpur": "Hamirpur", "Nuh": "Gurgaon", "Palwal": "Faridabad",
    "Alirajpur": "Jhabua", "Singrauli": "Sidhi", "Pratapgarh": "Chittaurgarh",
    "Jhargram": "West Midnapore",
})


def _match_geo(state, district):
    target_state = "Andhra Pradesh" if state == "Telangana" else ("Daman And Diu" if state == "Gujarat" and district == "Daman" else state)
    cands = [f for f in geo_features if f["state"] == target_state]
    if not cands:
        return None
    if target_state == "Delhi":
        return cands[0]
    k = norm(district)
    a = GEO_ALIASES.get(district)
    if a:
        for f in cands:
            if norm(a) == norm(f["district"]):
                return f
    for f in cands:
        if norm(f["district"]) == k:
            return f
    keys = [norm(f["district"]) for f in cands]
    mm = difflib.get_close_matches(k, keys, n=1, cutoff=0.82)
    if mm:
        idx = keys.index(mm[0])
        return cands[idx]
    return None


geo_features = []
if GEO_PATH.exists():
    gj = json.load(open(GEO_PATH))
    for f in gj["features"]:
        p = f["properties"]
        geo_features.append({
            "id1": p["ID_1"], "id2": p["ID_2"],
            "state": _state_key(p["NAME_1"]),
            "district": p["NAME_2"],
            "area_km2": round(_spherical_area_km2(f["geometry"]), 1),
        })
    geo_id_map = {}
    area_map = {}
    for row in dist.itertuples(index=False):
        gf = _match_geo(row.state, row.district)
        if gf is not None:
            gid = f"{gf['id1']}_{gf['id2']}"
            geo_id_map[(row.state, row.district)] = gid
            area_map[(row.state, row.district)] = gf["area_km2"]
    dist["geo_id"] = dist.apply(lambda r: geo_id_map.get((r.state, r.district)), axis=1)
    dist["area_km2"] = dist.apply(lambda r: area_map.get((r.state, r.district)), axis=1)
    print("geo matched:", dist.geo_id.notna().sum(), "of", len(dist))
else:
    dist["geo_id"] = np.nan
    dist["area_km2"] = np.nan
    print("WARNING: geojson not found at", GEO_PATH, "- choropleth disabled")

# 3d. Specific-yield hint by aquifer type (state-level heuristic, CGWB guidance).
HARD_ROCK_STATES = {"Karnataka", "Tamil Nadu", "Maharashtra", "Madhya Pradesh",
                    "Rajasthan", "Andhra Pradesh", "Telangana", "Odisha", "Jharkhand",
                    "Chhattisgarh", "Kerala", "Himachal Pradesh", "Uttarakhand",
                    "Jammu & Kashmir", "Gujarat"}
dist["sy_hint"] = dist["state"].map(lambda s: 0.04 if s in HARD_ROCK_STATES else 0.12)
dist["aquifer_type"] = dist["state"].map(lambda s: "hard rock" if s in HARD_ROCK_STATES else "alluvial")

# 3e. SPI (Standardised Precipitation Index) from district monthly rainfall (optional).
SPI_PATH = RAW / "district_rainfall_monthly.csv"
if SPI_PATH.exists():
    mrf = pd.read_csv(SPI_PATH)
    mrf["state"] = mrf["state"].astype(str).str.strip().str.title().replace(
        {"Orissa": "Odisha", "Uttaranchal": "Uttarakhand", "Pondicherry": "Puducherry"})
    mrf["district"] = mrf["district"].astype(str).str.strip().str.title()
    spi_summary_rows = []
    spi_ts_frames = []
    for (st_name, dist_name), g in mrf.groupby(["state", "district"]):
        g = g.sort_values(["year", "month"])
        rain = g["rainfall_mm"].to_numpy(float)
        s3 = F.spi(rain, 3)
        s12 = F.spi(rain, 12)
        valid3 = s3[~np.isnan(s3)]
        valid12 = s12[~np.isnan(s12)]
        spi_summary_rows.append({
            "state": st_name, "district": dist_name,
            "spi_3_latest": float(valid3[-1]) if len(valid3) else np.nan,
            "spi_12_latest": float(valid12[-1]) if len(valid12) else np.nan,
            "spi_12_mean_12m": (float(np.nanmean(s12[-12:])) if (len(s12) >= 12 and np.any(~np.isnan(s12[-12:]))) else np.nan),
        })
        spi_ts_frames.append(pd.DataFrame({
            "state": st_name, "district": dist_name,
            "year": g["year"].to_numpy(), "month": g["month"].to_numpy(),
            "spi_3": s3, "spi_12": s12,
        }))
    spi_summary = pd.DataFrame(spi_summary_rows)
    dist = dist.merge(spi_summary, on=["state", "district"], how="left")
    pd.concat(spi_ts_frames, ignore_index=True).to_csv(OUT / "district_spi.csv", index=False)
    print("SPI computed for", spi_summary.spi_3_latest.notna().sum(), "districts")
else:
    dist["spi_3_latest"] = np.nan
    dist["spi_12_latest"] = np.nan
    dist["spi_12_mean_12m"] = np.nan
    print("NOTE: district_rainfall_monthly.csv not found - SPI skipped (see README).")

dist = dist[dist.stations >= 3].dropna(subset=["pre_depth", "trend", "rain_annual", "rise_m"]).reset_index(drop=True)
print("districts analysed:", len(dist))

dist.round(3).to_csv(OUT / "district_metrics.csv", index=False)
dy.merge(dist[["state", "district"]], on=["state", "district"]).round(3).to_csv(OUT / "district_year.csv", index=False)
st[st.pre_depth.notna()].reset_index(names="sid").round(4).to_csv(OUT / "stations.csv", index=False)
print("written:", [p.name for p in OUT.iterdir()])
