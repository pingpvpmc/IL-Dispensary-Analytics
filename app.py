# Illinois Wholesale Market Intelligence — Brand Capture (Aeriz-ready)
# Single-file Streamlit app. Designed for wholesalers to prioritize sell-in.
# Auto-pulls IL dispensaries (OSM), optionally scans websites/menus for brand mentions
# and promotions, and computes Brand Whitespace Scores per area & store.

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import requests, time, re, math
from typing import List, Dict, Any
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="IL Wholesale Market Intelligence (Brand Capture)", layout="wide", page_icon="📍")

# -------------------------- Sidebar Controls --------------------------
with st.sidebar:
    st.header("Controls")
    target_brand = st.text_input("Target brand", value="Aeriz").strip()
    refresh_btn = st.button("↻ Refresh IL dispensary list (OSM)")
    st.divider()
    st.markdown("**Signals (optional web scan)**")
    enable_brand_scan = st.toggle("Scan websites for target brand", value=True,
                                  help="Light keyword scan (respect robots/ToS).")
    enable_promo_scan = st.toggle("Scan for promo keywords", value=True,
                                  help="BOGO/discount/special/promo/sale.")
    scan_limit = st.number_input("Max websites to scan (0 = all)", min_value=0, value=60, step=10)
    st.caption("Tip: start with 25–60 on free hosts; increase if stable.")
    st.divider()
    st.markdown("**Filters**")
    f_city = st.text_input("City contains")
    f_owner = st.text_input("Owner contains")
    f_has_brand = st.selectbox("Brand presence", ["Any","Present","Absent"])
    f_has_promos = st.selectbox("Promos detected", ["Any","Yes","No"])

# -------------------------- Constants & Regex --------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127 Safari/537.36"
}

IL_BBOX = (36.9703, -91.5131, 42.5083, -87.0199)
IL_RELATION_ID = 114692

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

PROMO_RE = r"(?:deal|discount|special|promo|promotion|bogo|bundle|sale|offer)"
# Common menu platforms (helps us find brand words even when main page is a menu host link)
MENU_HINTS = [r"leafly", r"weedmaps", r"dutchie", r"iheartjane", r"treez", r"flowhub", r"woocommerce"]

CATEGORY_HINTS = {
    "flower": r"(flower|eighth|oz|ounce|pre[- ]?roll|preroll)",
    "vapes": r"(vape|cart|cartridge|510|disposable)",
    "edibles": r"(edible|gummy|chocolate|chew|drink|beverage)",
    "concentrates": r"(concentrate|shatter|wax|rosin|resin|dab|badder|sauce)"
}

# -------------------------- Data Fetchers --------------------------
@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_osm_dispensaries() -> pd.DataFrame:
    """Fetch all IL dispensaries from OSM with mirror & bbox fallbacks."""
    s,w,n,e = IL_BBOX
    q_rel = f"""
    [out:json][timeout:60];
    rel({IL_RELATION_ID}); map_to_area->.il;
    ( node["shop"="cannabis"](area.il);
      way["shop"="cannabis"](area.il);
      relation["shop"="cannabis"](area.il); );
    out center tags;
    """
    q_box = f"""
    [out:json][timeout:60];
    ( node["shop"="cannabis"]({s},{w},{n},{e});
      way["shop"="cannabis"]({s},{w},{n},{e});
      relation["shop"="cannabis"]({s},{w},{n},{e}); );
    out center tags;
    """
    def _try(q: str) -> List[Dict[str,Any]]:
        for url in OVERPASS_MIRRORS:
            try:
                r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=90)
                if r.status_code == 429:  # rate-limited
                    continue
                r.raise_for_status()
                els = r.json().get("elements", [])
                if els: return els
            except Exception:
                continue
        return []
    elements = _try(q_rel) or _try(q_box)
    rows=[]
    for el in elements:
        tags = el.get("tags", {})
        lat, lon = (el.get("lat"), el.get("lon")) if "lat" in el else (el.get("center",{}).get("lat"), el.get("center",{}).get("lon"))
        if lat is None or lon is None: continue
        rows.append({
            "id": el.get("id"),
            "name": tags.get("name","(unnamed)"),
            "owner": tags.get("operator", tags.get("brand","")),
            "lat": lat, "lon": lon,
            "city": tags.get("addr:city",""),
            "county": tags.get("addr:county",""),
            "website": tags.get("website", tags.get("contact:website",""))
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["name","lat","lon"]).reset_index(drop=True)
    df = df[(df["lat"]>=s)&(df["lat"]<=n)&(df["lon"]>=w)&(df["lon"]<=e)]
    return df

def _safe_get(url: str, timeout: int=20) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type",""):
            return r.text[:300_000]
    except Exception:
        pass
    return ""

def scan_signals(df: pd.DataFrame, brand: str, scan_brand: bool, scan_promo: bool, limit: int=0) -> pd.DataFrame:
    """Lightweight website scan for brand presence, promos, categories, and menu links."""
    rows=[]; count=0
    brand_pat = re.compile(re.escape(brand), flags=re.I) if brand else None
    promo_pat = re.compile(PROMO_RE, flags=re.I)
    menu_pat  = re.compile("|".join(MENU_HINTS), flags=re.I)
    cat_pats = {k: re.compile(v, flags=re.I) for k,v in CATEGORY_HINTS.items()}
    for r in df.itertuples(index=False):
        url = (r.website or "").strip()
        has_brand = False; has_promo=False; menu=False
        cats=set()
        if url.startswith("http") and (scan_brand or scan_promo):
            if limit and count>=limit:
                pass
            else:
                html = _safe_get(url)
                if html:
                    if brand_pat and scan_brand: has_brand = bool(brand_pat.search(html))
                    if scan_promo: has_promo = bool(promo_pat.search(html))
                    menu = bool(menu_pat.search(html))
                    for c,pat in cat_pats.items():
                        if pat.search(html): cats.add(c)
                count+=1
                time.sleep(0.15)
        rows.append({**r._asdict(),
                     "has_brand": has_brand,
                     "has_promos": has_promo,
                     "menu_platform": menu,
                     "cat_signals": sorted(list(cats))})
    out = pd.DataFrame(rows)
    # defaults if not scanned
    if not scan_brand: out["has_brand"] = False
    if not scan_promo: out["has_promos"] = False
    out["intensity"] = out["cat_signals"].apply(lambda L: 0.0 if not L else round(1 - math.exp(-0.25*len(L)),3))
    return out

# -------------------------- Market Math --------------------------
def gridize(df: pd.DataFrame, cell_deg: float=0.25) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["g_lat","g_lon","disp","brand_rate","promo_rate","avg_intensity","BWS"])
    g_lat = (df["lat"] // cell_deg) * cell_deg
    g_lon = (df["lon"] // cell_deg) * cell_deg
    g = df.assign(g_lat=g_lat, g_lon=g_lon).groupby(["g_lat","g_lon"], as_index=False).agg(
        disp=("id","count"),
        brand_rate=("has_brand","mean"),
        promo_rate=("has_promos","mean"),
        avg_intensity=("intensity","mean")
    )
    # Brand Whitespace Score: more stores, lower brand presence, lower promo pressure
    # BWS ∈ [0,1+] (we cap later for readability)
    g["BWS"] = (g["disp"]/g["disp"].max().clip(lower=1)) * (1 - g["brand_rate"]) * (1 - 0.6*g["promo_rate"]) * (1 - 0.4*g["avg_intensity"])
    g["BWS"] = g["BWS"].clip(lower=0).round(3)
    return g.sort_values("BWS", ascending=False)

def local_rate(df: pd.DataFrame, lat: float, lon: float, r_deg: float=0.12) -> dict:
    loc = df[(abs(df["lat"]-lat)<=r_deg) & (abs(df["lon"]-lon)<=r_deg)]
    if loc.empty: 
        return {"brand_rate":0.0,"promo_rate":0.0,"density":0}
    return {
        "brand_rate": float(loc["has_brand"].mean()),
        "promo_rate": float(loc["has_promos"].mean()),
        "density": int(len(loc))
    }

def store_whitespace_score(store: pd.Series, df: pd.DataFrame) -> float:
    loc = local_rate(df, store["lat"], store["lon"])
    # higher if area dense, brand absent, low promo pressure
    score = ( (loc["density"]/max(1, df.shape[0])) * (0 if store.get("has_brand") else 1) *
              (1 - 0.6*loc["promo_rate"]) )
    return round(score,3)

def recommend_sell_in(store: pd.Series, df: pd.DataFrame, brand: str) -> List[str]:
    loc = local_rate(df, store["lat"], store["lon"])
    recs = []
    if not store.get("has_brand", False):
        recs.append(f"Open **{brand}** with a limited **intro assortment** (2–3 SKUs).")
    # Category focus based on local signals
    cats = pd.Series([c for lst in df[(abs(df["lat"]-store['lat'])<=0.12)&(abs(df['lon']-store['lon'])<=0.12)]["cat_signals"] for c in (lst or [])])
    if cats.empty:
        cat_recs = ["flower","vapes"]
    else:
        counts = cats.value_counts()
        # push into underweighted categories
        cat_recs = [c for c in ["flower","vapes","edibles","concentrates"] if counts.get(c,0)==0][:2] or list(counts.sort_values().index[:2])
    if loc["promo_rate"] < 0.35:
        recs.append("Win quickly with **BOGO/bundle** on launch SKUs (2 weeks).")
    else:
        recs.append("Aim for **mid-week price dip** or **bundle** to avoid promo spikes.")
    recs.append(f"Lead with **{cat_recs[0]}** and **{cat_recs[1]}** SKUs to fill local gaps.")
    return recs[:3]

# -------------------------- UI / App Flow --------------------------
st.title("Illinois Wholesale Market Intelligence — Brand Capture")
st.caption("For wholesalers (e.g., Aeriz). Finds high-opportunity areas and stores with weak brand presence and low promo pressure.")

with st.spinner("Fetching IL dispensaries…"):
    if refresh_btn: fetch_osm_dispensaries.clear()
    base = fetch_osm_dispensaries()

if base.empty:
    st.error("No dispensaries returned from OpenStreetMap. Tap refresh, or try again shortly.")
    st.stop()

with st.spinner("Scanning websites (optional signals)…"):
    df = scan_signals(base, brand=target_brand, scan_brand=enable_brand_scan, scan_promo=enable_promo_scan, limit=int(scan_limit or 0))

# Filters
flt = df.copy()
if f_city:  flt = flt[flt["city"].str.contains(f_city, case=False, na=False)]
if f_owner: flt = flt[flt["owner"].str.contains(f_owner, case=False, na=False)]
if f_has_brand == "Present": flt = flt[flt["has_brand"]==True]
elif f_has_brand == "Absent": flt = flt[flt["has_brand"]==False]
if f_has_promos == "Yes": flt = flt[flt["has_promos"]==True]
elif f_has_promos == "No": flt = flt[flt["has_promos"]==False]

# KPIs
c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("Dispensaries (IL)", len(df))
with c2: st.metric(f"Brand presence ({target_brand})", f"{int(df['has_brand'].sum())} ({round(100*df['has_brand'].mean(),1)}%)")
with c3: st.metric("Promo rate (detected)", f"{round(100*df['has_promos'].mean(),1)}%")
with c4: st.metric("Filtered set", len(flt))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Map", "Targets", "Recommendations"])

with tab1:
    st.subheader("Brand Whitespace — by Area")
    grid = gridize(df)
    st.caption("BWS = higher where brand presence is low, promo pressure is low, and store density is higher.")
    st.dataframe(grid.head(30), use_container_width=True)

with tab2:
    st.subheader("Interactive Map")
    m = folium.Map(location=(40.0,-89.3), zoom_start=6, control_scale=True)
    for r in flt.itertuples():
        brand = "✅ Brand present" if r.has_brand else "— Brand absent"
        promo = "✅ Promos detected" if r.has_promos else "— No promos"
        cats = ", ".join(r.cat_signals or []) if r.cat_signals else "—"
        pop = f"""
        <div style='font-size:14px;line-height:1.35'>
          <b>{r.name}</b><br/>
          Owner: {r.owner or '—'}<br/>
          City/County: {r.city or '—'} / {r.county or '—'}<br/>
          Website: {r.website or '—'}<br/>
          {brand} · {promo}<br/>
          Category signals: {cats}
        </div>
        """
        color = "#2ca25f" if r.has_brand else "#de2d26"
        folium.CircleMarker(location=(r.lat,r.lon), radius=7, color=color, fill=True, fill_opacity=0.85,
                            tooltip=r.name, popup=folium.Popup(pop, max_width=360)).add_to(m)
    st_folium(m, width=None, height=560)

with tab3:
    st.subheader("Top Store Targets (Brand Capture)")
    # score every store
    flt["BWS_store"] = flt.apply(lambda x: store_whitespace_score(x, df), axis=1)
    # brand-absent stores first
    targets = flt.sort_values(["has_brand","BWS_store"], ascending=[True, False]).head(50)
    st.dataframe(
        targets[["name","owner","city","website","has_brand","has_promos","BWS_store"]],
        use_container_width=True, hide_index=True
    )

with tab4:
    st.subheader("Actionable Sell-In Recommendations")
    rows=[]
    for r in flt.sort_values(["has_brand","city"]).itertuples():
        recs = recommend_sell_in(pd.Series(r._asdict()), df, brand=target_brand)
        rows.append({
            "name": r.name, "owner": r.owner or "", "city": r.city or "",
            "brand_present": "Yes" if r.has_brand else "No",
            "local_density(~10km)": local_rate(df, r.lat, r.lon)["density"],
            "rec_1": recs[0] if len(recs)>0 else "",
            "rec_2": recs[1] if len(recs)>1 else "",
            "rec_3": recs[2] if len(recs)>2 else "",
        })
    rec_df = pd.DataFrame(rows)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

st.caption("Notes: Signals are derived via OpenStreetMap + optional light HTML scans (best-effort; respect robots/ToS). Scores are heuristic; plug in POS/sell-in to refine.")
