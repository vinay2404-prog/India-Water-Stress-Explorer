"""India Water Stress Explorer  -  Streamlit app.

Run:  streamlit run app.py
Data: built by `python src/build_data.py` from CGWB groundwater levels + IMD district rainfall normals.
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import formulas as F  # noqa: E402

st.set_page_config(page_title="India Water Stress Explorer", page_icon="💧", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    p = ROOT / "data/processed"
    return (pd.read_csv(p / "district_metrics.csv"),
            pd.read_csv(p / "district_year.csv"),
            pd.read_csv(p / "stations.csv"))


@st.cache_data
def load_geo():
    """District boundaries (GeoJSON) with a stable geo_id on each feature."""
    p = ROOT / "data/geo/india_district.geojson"
    if not p.exists():
        # Fallback: fetch from the public source once and cache locally.
        try:
            import urllib.request
            url = ("https://raw.githubusercontent.com/geohacker/india/"
                   "c3df722d136666b6b663f9359336ae1543809a48/district/india_district.geojson")
            p.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, p)
        except Exception:
            return None
    if not p.exists():
        return None
    gj = json.load(open(p))
    for f in gj["features"]:
        f["properties"]["geo_id"] = f"{f['properties']['ID_1']}_{f['properties']['ID_2']}"
    return gj


dist_raw, dy, stations = load()
geojson = load_geo()


def compute(df, sy, weights, cuts):
    """Apply every formula to the district table (normalisation uses ALL districts so scores
    stay comparable when the user filters)."""
    df = df.copy()
    df["recharge_mm"] = F.recharge_wtf(sy, df["rise_m"])                          # R = Sy x dh x 1000
    df["recharge_ratio"] = F.recharge_ratio(df["recharge_mm"], df["rain_annual"])  # R / P x 100
    if "monsoon_share" not in df.columns:   # older processed files pre-built without it
        df["monsoon_share"] = F.monsoon_share(df["rain_jjas"], df["rain_annual"])
    comps = F.stress_components(df)
    for k in comps:
        df["c_" + k] = comps[k]
    df["wsi"] = F.weighted_index(comps, weights)   # reuse comps (computed once)
    df["category"] = F.classify(df["wsi"], cuts)
    df["depth_change"] = df["pre_depth"] - df["pre_2005_11"]
    yrs = F.years_to_reach(df["pre_depth"], df["trend"], df["well_depth"])
    df["years_to_well_depth"] = np.where(yrs > 200, np.nan, yrs)
    return df


# ------------------------------------------------------------------ sidebar
st.sidebar.title("💧 Model settings")
st.sidebar.caption("Every number in the app is recomputed from these settings.")
st.sidebar.subheader("Index weights")
w = {
    "depth": st.sidebar.slider("Depth to water", 0.0, 1.0, 0.35, 0.05),
    "trend": st.sidebar.slider("Depletion rate", 0.0, 1.0, 0.30, 0.05),
    "rain": st.sidebar.slider("Rainfall deficit", 0.0, 1.0, 0.20, 0.05),
    "recharge": st.sidebar.slider("Recharge deficit", 0.0, 1.0, 0.15, 0.05),
}
if sum(w.values()) == 0:
    st.sidebar.error("At least one weight must be above zero.")
    st.stop()
st.sidebar.subheader("Recharge (WTF method)")
sy = st.sidebar.slider("Specific yield, Sy", 0.01, 0.25, 0.05, 0.01,
                       help="Fraction of aquifer volume that drains freely. ~0.02-0.03 hard rock, ~0.10-0.16 alluvium.")
with st.sidebar.expander("Category cut-offs"):
    c1 = st.slider("Moderate from", 0, 100, 35)
    c2 = st.slider("High from", 0, 100, 50)
    c3 = st.slider("Critical from", 0, 100, 65)
if not (c1 <= c2 <= c3):
    st.sidebar.error("Cut-offs must be ordered: Moderate ≤ High ≤ Critical.")
    st.stop()
st.sidebar.subheader("Filters")
all_states = sorted(dist_raw["state"].unique())
sel_states = st.sidebar.multiselect("States", all_states, default=all_states)
min_st = st.sidebar.slider("Minimum monitoring stations", 3, 30, 3)

df_all = compute(dist_raw, sy, w, (c1, c2, c3))
df = df_all[df_all["state"].isin(sel_states) & (df_all["stations"] >= min_st)].copy()
if df.empty:
    st.warning("No districts match the filters.")
    st.stop()

# ------------------------------------------------------------------ header
st.title("India Water Stress Explorer")
st.caption("Groundwater levels (CGWB, 1970-2025) combined with district rainfall normals (IMD). "
           "Depth values are metres below ground level - bigger means a deeper water table.")
tabs = st.tabs(["Overview", "Rankings", "District explorer", "Rainfall vs groundwater", "Method & data"])

# ------------------------------------------------------------------ 1 overview
with tabs[0]:
    k = st.columns(6)
    k[0].metric("Districts analysed", f"{len(df)}")
    k[1].metric("Critical / High", f"{int(df['category'].isin(['Critical', 'High']).sum())}")
    k[2].metric("Median depletion rate", f"{df['trend'].median():+.3f} m/yr")
    k[3].metric("Districts deepening >0.3 m/yr", f"{int((df['trend'] > 0.3).sum())}")
    k[4].metric("Median pre-monsoon depth", f"{df['pre_depth'].median():.1f} m")
    fk = df["falkenmark_m3_per_person"].median()
    k[5].metric("Median Falkenmark", "n/a" if pd.isna(fk) else f"{fk:,.0f} m\u00b3/p/yr")

    left, right = st.columns([2, 1])
    with left:
        if geojson is not None and "geo_id" in df.columns:
            fig = px.choropleth(
                df, geojson=geojson, locations="geo_id", featureidkey="properties.geo_id",
                color="wsi", color_continuous_scale="RdYlGn_r", range_color=(0, 100),
                hover_name="district",
                hover_data={"state": True, "wsi": ":.1f", "pre_depth": ":.1f", "trend": ":.2f",
                            "rain_annual": ":.0f", "stations": True, "geo_id": False},
                labels={"wsi": "Stress index"})
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width='stretch')
        else:
            fig = px.scatter_geo(df, lat="lat", lon="lon", color="wsi", size="stations", size_max=14,
                                 color_continuous_scale="RdYlGn_r", range_color=(0, 100),
                                 hover_name="district",
                                 hover_data={"state": True, "wsi": ":.1f", "pre_depth": ":.1f", "trend": ":.2f",
                                             "rain_annual": ":.0f", "lat": False, "lon": False, "stations": True},
                                 labels={"wsi": "Stress index"})
            fig.update_geos(scope="asia", lonaxis_range=[67, 98], lataxis_range=[6, 37], showcountries=True,
                            showland=True, landcolor="#f3f1ea", countrycolor="#999")
            fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width='stretch')
    with right:
        cnt = (df["category"].value_counts().reindex(F.CATEGORY_ORDER).fillna(0).rename_axis("category")
               .reset_index(name="districts"))
        fig = px.bar(cnt, x="category", y="districts", color="category", color_discrete_map=F.CATEGORY_COLORS,
                     text="districts")
        fig.update_layout(showlegend=False, height=270, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width='stretch')
        fig = px.histogram(df, x="trend", nbins=40, labels={"trend": "Depletion rate (m/yr)"})
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(height=270, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width='stretch')
        st.caption("Right of the dashed line = water table getting deeper; left = recovering.")

# ------------------------------------------------------------------ 2 rankings
with tabs[1]:
    n = st.slider("Show top N districts", 5, 50, 20)
    top = df.sort_values("wsi", ascending=False).head(n).sort_values("wsi")
    fig = px.bar(top, x="wsi", y="district", color="category", color_discrete_map=F.CATEGORY_COLORS,
                 orientation="h", hover_data=["state", "pre_depth", "trend", "rain_annual", "recharge_ratio"],
                 labels={"wsi": "Water Stress Index", "district": ""})
    fig.update_layout(height=max(400, 22 * n), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width='stretch')

    st.subheader("State summary")
    ss = (df.groupby("state").agg(districts=("district", "size"), mean_wsi=("wsi", "mean"),
                                  median_depth=("pre_depth", "median"), median_trend=("trend", "median"),
                                  rainfall=("rain_annual", "median")).reset_index())
    ss = ss[ss["districts"] >= 3].sort_values("mean_wsi", ascending=False)
    fig = px.bar(ss, x="state", y="mean_wsi", color="mean_wsi", color_continuous_scale="RdYlGn_r",
                 range_color=(0, 100), labels={"mean_wsi": "Mean index", "state": ""})
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch')
    st.caption("States with fewer than 3 analysed districts are omitted.")
    st.dataframe(ss.round(2), width='stretch', hide_index=True)

# ------------------------------------------------------------------ 3 district explorer
with tabs[2]:
    c = st.columns(2)
    state = c[0].selectbox("State", sorted(df["state"].unique()))
    names = sorted(df.loc[df["state"] == state, "district"])
    name = c[1].selectbox("District", names)
    r = df[(df["state"] == state) & (df["district"] == name)].iloc[0]

    m = st.columns(6)
    m[0].metric("Stress index", f"{r.wsi:.0f} / 100", r.category, delta_color="off")
    m[1].metric("Pre-monsoon depth", f"{r.pre_depth:.1f} m")
    m[2].metric("Depletion rate", f"{r.trend:+.2f} m/yr")
    m[3].metric("Annual rainfall", f"{r.rain_annual:.0f} mm")
    m[4].metric("Recharge (WTF)", f"{r.recharge_mm:.0f} mm")
    m[5].metric("Stations", f"{int(r.stations)}")

    m2 = st.columns(4)
    pop = r.get("population", None)
    m2[0].metric("Population (2011)", "n/a" if pd.isna(pop) else f"{int(pop):,}")
    fk = r.get("falkenmark_m3_per_person", None)
    fclass = F.falkenmark_classify(pd.Series([fk])).iloc[0] if pd.notna(fk) else "n/a"
    m2[1].metric("Falkenmark", "n/a" if pd.isna(fk) else f"{fk:,.0f} m\u00b3/p/yr", fclass, delta_color="off")
    se = r.get("stage_extraction_pct", None)
    m2[2].metric("Stage of extraction (state)", "n/a" if pd.isna(se) else f"{se:.1f} %")
    syh = r.get("sy_hint", None)
    m2[3].metric("Aquifer type", str(r.get("aquifer_type", "n/a")),
                 f"suggested Sy \u2248 {syh:.2f}" if pd.notna(syh) else None, delta_color="off")

    tp = r.get("trend_pvalue", None)
    tr = r.get("trend_r2", None)
    ts = r.get("trend_sig", None)
    if pd.notna(tp):
        st.caption(f"Trend significance: median p = {tp:.3g}, median R\u00b2 = {tr:.2f}, "
                   f"stations with p<0.05 = {0 if pd.isna(ts) else ts*100:.0f}% "
                   f"(of {int(r.get('trend_n', 0))} stations).")

    series = dy[(dy["state"] == state) & (dy["district"] == name)].sort_values("year")
    left, right = st.columns([3, 2])
    with left:
        horizon = st.slider("Project forward (years)", 0, 30, 10)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series["year"], y=series["pre"], mode="lines+markers", name="Pre-monsoon"))
        fig.add_trace(go.Scatter(x=series["year"], y=series["post"], mode="lines+markers", name="Post-monsoon"))
        recent = series[(series["year"] >= 2010) & series["pre"].notna()]
        if len(recent) >= 3:
            b = F.depletion_rate(recent["year"], recent["pre"])
            a = recent["pre"].mean() - b * recent["year"].mean()
            xs = np.arange(2010, int(series["year"].max()) + horizon + 1)
            fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", name=f"Trend {b:+.2f} m/yr",
                                     line=dict(dash="dash", color="#c1121f")))
        fig.update_yaxes(autorange="reversed", title="Depth to water (m)")
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
        st.plotly_chart(fig, width='stretch')
        st.caption("Y-axis reversed: lines going down = water table getting deeper.")
    with right:
        wn = pd.Series(w) / sum(w.values())
        contrib = pd.DataFrame({"component": [F.COMPONENTS[k] for k in wn.index],
                                "points": [100 * wn[k] * r["c_" + k] for k in wn.index]})
        fig = px.bar(contrib, x="points", y="component", orientation="h",
                     labels={"points": "Points of the index", "component": ""})
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width='stretch')

    st.subheader("The formulas, with this district's numbers")
    d_now = float(r.pre_depth)
    st.latex(rf"\text{{Recharge}} = S_y \times \Delta h \times 1000 = {sy:.2f} \times {r.rise_m:.2f} \times 1000 = {r.recharge_mm:.0f}\ \text{{mm}}")
    st.latex(rf"\text{{Recharge ratio}} = R / P \times 100 = {r.recharge_mm:.0f} / {r.rain_annual:.0f} \times 100 = {r.recharge_ratio:.1f}\%")
    st.latex(rf"D(t) = D_0 + \text{{rate}} \times t \;\Rightarrow\; D({horizon}) = {d_now:.1f} + ({r.trend:+.2f}) \times {horizon} = {d_now + r.trend * horizon:.1f}\ \text{{m}}")
    if pd.notna(r.years_to_well_depth) and pd.notna(r.well_depth):
        st.latex(rf"t = (D_\mathrm{{well}} - D_0)/\text{{rate}} = ({r.well_depth:.0f} - {d_now:.1f}) / {r.trend:.2f} = {r.years_to_well_depth:.0f}\ \text{{years}}")
    else:
        st.info("No years-to-depletion estimate here: the water table is not getting deeper "
                "(rate \u2264 0), it is already deeper than the well depth, or the projection "
                "exceeds 200 years.")
    st.caption(f"Rainfall matched by: {r.rain_match}. Monsoon (Jun-Sep) share of rainfall: {r.monsoon_share:.0f}%.")

# ------------------------------------------------------------------ 4 rainfall vs groundwater
with tabs[3]:
    ycol = st.selectbox("Groundwater measure (y-axis)", ["pre_depth", "trend", "recharge_ratio", "depth_change"],
                        format_func=lambda x: {"pre_depth": "Pre-monsoon depth (m)", "trend": "Depletion rate (m/yr)",
                                               "recharge_ratio": "Recharge ratio (% of rainfall)",
                                               "depth_change": "Depth change since 2005-11 (m)"}[x])
    sub = df.dropna(subset=["rain_annual", ycol])
    fig = px.scatter(sub, x="rain_annual", y=ycol, color="category", color_discrete_map=F.CATEGORY_COLORS,
                     hover_name="district", hover_data=["state"], category_orders={"category": F.CATEGORY_ORDER},
                     labels={"rain_annual": "Annual rainfall normal (mm)"})
    if len(sub) > 3:
        b, a = np.polyfit(sub["rain_annual"], sub[ycol], 1)
        xs = np.linspace(sub["rain_annual"].min(), sub["rain_annual"].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", name="Linear fit", line=dict(color="black", dash="dash")))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width='stretch')
    rho = np.nan
    valid = sub.dropna(subset=["rain_annual", ycol])
    if len(valid) >= 3:   # correlation is meaningless below 3 points
        rho = valid["rain_annual"].rank().corr(valid[ycol].rank())
    st.metric("Spearman rank correlation with rainfall",
              "n/a (too few districts)" if pd.isna(rho) else f"{rho:+.2f}")
    st.caption("Spearman rho = Pearson correlation of the ranks, so it is not distorted by outliers. "
               "Weak correlations mean rainfall alone does not explain groundwater stress - "
               "extraction for irrigation matters a great deal.")

# ------------------------------------------------------------------ 5 method
with tabs[4]:
    st.header("Formulas")
    st.latex(r"\text{rate} = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}\quad[\text{m/yr, OLS of pre-monsoon depth on year, 2010-2025}]")
    st.latex(r"\Delta h = \max(D_\text{pre} - D_\text{post},\,0)")
    st.latex(r"R = S_y \times \Delta h \times 1000 \quad[\text{mm, water-table-fluctuation method}]")
    st.latex(r"\text{Recharge ratio} = R / P_\text{annual} \times 100")
    st.latex(r"z = \operatorname{clip}\!\left(\frac{x - P_5}{P_{95} - P_5},\,0,\,1\right)")
    st.latex(r"\text{WSI} = 100 \times \frac{w_d z_\text{depth} + w_t z_\text{trend} + w_r (1 - z_\text{rain}) + w_c (1 - z_\text{recharge})}{w_d + w_t + w_r + w_c}")
    st.latex(r"t_\text{depletion} = \frac{D_\text{well} - D_0}{\text{rate}}\quad(\text{rate} > 0)")
    st.header("Data and cleaning")
    st.markdown(
        "- **Groundwater:** CGWB station readings 1970-2025. Readings outside 0-100 m were removed. "
        "Pre-monsoon = mean of April and May; post-monsoon = November. District values are medians across stations "
        "(recent window 2021-25; trend 2010-25, at least 8 years per station, at least 3 stations per district).\n"
        "- **Rainfall:** IMD district normals (annual and Jun-Sep). Joined by state + district name.\n"
        "- **Population:** Census of India 2011 (district level) - used for the Falkenmark indicator "
        "(rainfall volume per person). Telangana districts use the 2011 Andhra Pradesh figures.\n"
        "- **District area:** derived from the district GeoJSON (spherical polygon area) - used to convert rainfall (mm) "
        "into a volume for the Falkenmark indicator.\n"
        "- **Stage of extraction:** CGWB Dynamic Ground Water Resources assessment (state level, 2024-25).\n"
        "- **Specific yield hint:** CGWB guidance (hard-rock \u2248 0.02-0.04 vs alluvial \u2248 0.10-0.16), by state; "
        "the slider still overrides it for the recharge calculation.")
    mt = dist_raw["rain_match"].value_counts().rename_axis("how rainfall was matched").reset_index(name="districts")
    st.dataframe(mt, hide_index=True)
    st.subheader("Geo / population match quality")
    gm = pd.DataFrame({"dataset": ["District boundaries (GeoJSON)", "Census 2011 population"],
                       "matched": [int(dist_raw["geo_id"].notna().sum()), int(dist_raw["population"].notna().sum())],
                       "of": [len(dist_raw), len(dist_raw)]})
    st.dataframe(gm, hide_index=True)
    st.header("Limitations")
    st.markdown(
        "- The index is **relative** (districts ranked against each other), not an absolute scarcity measure.\n"
        "- **Stage of extraction** is shown at **state level only** (CGWB publishes district-level assessments as PDFs).\n"
        "- The WTF recharge depends strongly on the specific yield you pick and ignores monsoon pumping.\n"
        "- The Falkenmark indicator here is a **rainfall-volume proxy** (not blue-water availability), using 2011 population.\n"
        "- Districts with state-average rainfall (old vs new district names) are less precise.\n"
        "- Station networks changed over time, so district trends reflect the stations that reported.\n"
        "- District boundaries are from a 2011-era GeoJSON; newer (post-2011) districts inherit parent geometries where matched.")
    st.download_button("Download district results (CSV)", df.round(3).to_csv(index=False).encode(),
                       file_name="water_stress_results.csv", mime="text/csv")
