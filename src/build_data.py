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

# ------------------------------------------------------------------ 3. Tier-1 enrichment: population (Census 2011), district area +
#    Falkenmark water-per-capita (from the district GeoJSON), stage of groundwater
#    extraction (CGWB, state level) and a specific-yield hint (aquifer type).
GEO_PATH = ROOT / "data/geo/india_district.geojson"
POP_PATH = RAW / "india_census_2011.csv"

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

# 3b. Population (Census 2011) + Falkenmark water per capita.
POP_ALIASES = dict(DISTRICT_ALIASES)
POP_ALIASES.update({
    "Gurugram": "Gurgaon", "Ballari": "Bellary", "Kalaburagi": "Gulbarga",
    "Prayagraj": "Allahabad", "Ayodhya": "Faizabad", "Tuticorin": "Thoothukkudi",
    "Mohali": "Sahibzada Ajit Singh Nagar", "Sbs Nagar": "Shahid Bhagat Singh Nagar",
    "Cooch Behar": "Koch Bihar", "Howrah": "Haora",
    "North 24 Parganas": "North Twenty Four Parganas",
    "South 24 Parganas": "South Twenty Four Parganas",
    "G.B. Nagar": "Gautam Buddha Nagar", "Trichur": "Thrissur",
    "Kabirdham": "Kabeerdham", "Beed": "Bid", "Raigad": "Raigarh",
    "Cuddapah": "Y.S.R.", "Nellore": "Sri Potti Sriramulu Nellore",
    "Kanker": "Uttar Bastar Kanker", "Dang": "The Dangs",
    "Charkhi Dadri": "Bhiwani", "Nuh": "Mewat", "Agar Malwa": "Shajapur",
    "Palghar": "Thane", "Sonapur": "Subarnapur", "Fazilka": "Firozpur",
    "Pathankot": "Gurdaspur", "Ropar": "Rupnagar",
    "Tuticorin": "Thoothukkudi", "Praygraj": "Allahabad",
    "Jhargram": "Paschim Medinipur", "North Dinajpur": "Uttar Dinajpur",
    "Paschim Bardhaman": "Barddhaman", "West Medinipur": "Paschim Medinipur",
    "Gaurella Pendra Marwahi": "Bilaspur", "Kheri": "Lakhimpur Kheri",
    "Jyotiba Phule Nagar": "Amroha", "Ayodhya": "Faizabad",
    "G.B. Nagar": "Gautam Buddha Nagar", "Mau": "Mau",
    "Champawat": "Champawat", "Thrissur": "Thrissur",
})


def _match_pop(state, district):
    target_state = "Andhra Pradesh" if state == "Telangana" else ("Daman And Diu" if state == "Gujarat" and district == "Daman" else state)
    cand = pop[pop.state_n == target_state]
    k = norm(district)
    a = POP_ALIASES.get(district)
    if a:
        ex = cand[cand.dname_n == norm(a)]
        if len(ex):
            return int(ex.iloc[0].Population)
    ex = cand[cand.dname_n == k]
    if len(ex):
        return int(ex.iloc[0].Population)
    mm = difflib.get_close_matches(k, list(cand.dname_n), n=1, cutoff=0.75)
    if mm:
        return int(cand[cand.dname_n == mm[0]].iloc[0].Population)
    return np.nan


if POP_PATH.exists():
    pop = pd.read_csv(POP_PATH)
    pop["state_n"] = pop["State name"].map(_state_key).replace("Telangana", "Andhra Pradesh")
    pop["dname_n"] = pop["District name"].map(norm)
    dist["population"] = dist.apply(lambda r: _match_pop(r.state, r.district), axis=1)
    # Falkenmark-style: renewable water (rainfall volume) per person, m3/person/yr
    dist["rainfall_volume_mcm"] = dist["rain_annual"] / 1000.0 * dist["area_km2"]  # million m3 (mm->m * km2)
    dist["falkenmark_m3_per_person"] = (dist["rainfall_volume_mcm"] * 1e6) / dist["population"]
    print("population matched:", dist.population.notna().sum(), "of", len(dist))
else:
    dist["population"] = np.nan
    dist["rainfall_volume_mcm"] = np.nan
    dist["falkenmark_m3_per_person"] = np.nan
    print("WARNING: census file not found at", POP_PATH)

# 3c. Stage of groundwater extraction (CGWB, state level, 2024-25 assessment).
STAGE_OF_EXTRACTION = {
    "Andhra Pradesh": 31.51, "Arunachal Pradesh": 0.41, "Assam": 10.0, "Bihar": 35.0,
    "Chhattisgarh": 48.18, "Goa": 15.0, "Gujarat": 62.0, "Haryana": 136.75,
    "Himachal Pradesh": 38.50, "Jharkhand": 32.89, "Jammu & Kashmir": 30.0,
    "Karnataka": 66.49, "Kerala": 49.95, "Madhya Pradesh": 59.32, "Maharashtra": 51.79,
    "Manipur": 9.09, "Meghalaya": 5.24, "Mizoram": 5.0, "Nagaland": 4.72,
    "Odisha": 48.75, "Punjab": 145.0, "Rajasthan": 147.11, "Sikkim": 5.0,
    "Tamil Nadu": 73.50, "Telangana": 46.69, "Tripura": 10.06, "Uttar Pradesh": 70.0,
    "Uttarakhand": 45.0, "West Bengal": 45.19, "Delhi": 92.10, "Chandigarh": 67.0,
    "Puducherry": 50.0, "Andaman & Nicobar": 5.0, "Dadra & Nagar Haveli": 30.0,
    "Daman & Diu": 30.0, "Lakshadweep": 5.0,
}
# 3c/3f. Stage of groundwater extraction: prefer district-level (CGWB 2025 report);
# fall back to state-level STAGE_OF_EXTRACTION where district-level is missing.
EXT_PATH = RAW / "district_extraction.csv"
STAGE_OF_EXTRACTION = {
    "Andhra Pradesh": 31.51, "Arunachal Pradesh": 0.41, "Assam": 10.0, "Bihar": 35.0,
    "Chhattisgarh": 48.18, "Goa": 15.0, "Gujarat": 62.0, "Haryana": 136.75,
    "Himachal Pradesh": 38.50, "Jharkhand": 32.89, "Jammu & Kashmir": 30.0,
    "Karnataka": 66.49, "Kerala": 49.95, "Madhya Pradesh": 59.32, "Maharashtra": 51.79,
    "Manipur": 9.09, "Meghalaya": 5.24, "Mizoram": 5.0, "Nagaland": 4.72,
    "Odisha": 48.75, "Punjab": 145.0, "Rajasthan": 147.11, "Sikkim": 5.0,
    "Tamil Nadu": 73.50, "Telangana": 46.69, "Tripura": 10.06, "Uttar Pradesh": 70.0,
    "Uttarakhand": 45.0, "West Bengal": 45.19, "Delhi": 92.10, "Chandigarh": 67.0,
    "Puducherry": 50.0, "Andaman & Nicobar": 5.0, "Dadra & Nagar Haveli": 30.0,
    "Daman & Diu": 30.0, "Lakshadweep": 5.0,
}
if EXT_PATH.exists():
    ext = pd.read_csv(EXT_PATH)
    keep = ["state", "district", "stage_pct", "category_cgwb", "recharge_ham",
            "extractable_ham", "extraction_ham", "extraction_irrigation_ham",
            "extraction_industrial_ham", "extraction_domestic_ham",
            "net_availability_ham"]
    keep = [c for c in keep if c in ext.columns]
    ext = ext[keep].copy()
    ext = ext.drop_duplicates(subset=["state", "district"], keep="first")
    dist = dist.merge(ext, on=["state", "district"], how="left", suffixes=("", "_cgwb"))
    print("district extraction matched:", dist.stage_pct.notna().sum(), "of", len(dist))
else:
    dist["stage_pct"] = np.nan
    dist["category_cgwb"] = None
    print("NOTE: district_extraction.csv not found - stage of extraction will be state-level only.")

dist["stage_pct"] = dist["stage_pct"].fillna(dist["state"].map(STAGE_OF_EXTRACTION))


def _cgwb_cat(p):
    if pd.isna(p):
        return None
    if p > 100:
        return "Over-exploited"
    if p >= 90:
        return "Critical"
    if p >= 70:
        return "Semi-critical"
    return "Safe"


if "category_cgwb" not in dist.columns:
    dist["category_cgwb"] = None
mask = dist["category_cgwb"].isna() & dist["stage_pct"].notna()
dist.loc[mask, "category_cgwb"] = dist.loc[mask, "stage_pct"].apply(_cgwb_cat)

if "stage_extraction_pct" in dist.columns:
    dist = dist.drop(columns=["stage_extraction_pct"])

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
            "spi_12_mean_12m": float(np.nanmean(s12[-12:])) if len(s12) >= 12 else np.nan,
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

# 3g. Reservoir storage (CWC, March 2024) - median storage % per district/state.
RES_PATH = RAW / "reservoir_storage.csv"
if RES_PATH.exists():
    res = pd.read_csv(RES_PATH)
    # district-level: median storage_pct across reservoirs mapped to that district
    dist_res = res.groupby(["state", "district"], as_index=False).agg(
        reservoir_storage_pct=("storage_pct_mean", "median"),
        reservoir_count=("Reservoir_name", "nunique"))
    dist = dist.merge(dist_res, on=["state", "district"], how="left")
    # state-level fallback for districts without a reservoir
    state_res = res.groupby("state", as_index=False).agg(
        reservoir_storage_pct_state=("storage_pct_mean", "median"))
    dist = dist.merge(state_res, on="state", how="left")
    mask = dist["reservoir_storage_pct"].isna()
    dist.loc[mask, "reservoir_storage_pct"] = dist.loc[mask, "reservoir_storage_pct_state"]
    dist = dist.drop(columns=["reservoir_storage_pct_state"])
    # reservoir stress: 0 (full) .. 1 (empty)
    dist["reservoir_stress"] = 1 - (dist["reservoir_storage_pct"] / 100.0)
    dist["reservoir_stress"] = dist["reservoir_stress"].clip(0, 1)
    print("reservoir storage matched:", dist.reservoir_storage_pct.notna().sum(), "of", len(dist),
          "(district+state fallback)")
else:
    dist["reservoir_storage_pct"] = np.nan
    dist["reservoir_stress"] = np.nan
    dist["reservoir_count"] = 0
    print("NOTE: reservoir_storage.csv not found - reservoir stress skipped.")

dist = dist[dist.stations >= 3].dropna(subset=["pre_depth", "trend", "rain_annual", "rise_m"]).reset_index(drop=True)
print("districts analysed:", len(dist))

dist.round(3).to_csv(OUT / "district_metrics.csv", index=False)
dy.merge(dist[["state", "district"]], on=["state", "district"]).round(3).to_csv(OUT / "district_year.csv", index=False)
st[st.pre_depth.notna()].reset_index(names="sid").round(4).to_csv(OUT / "stations.csv", index=False)
print("written:", [p.name for p in OUT.iterdir()])
