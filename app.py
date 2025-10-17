You got it. Here’s a single-file Streamlit app that:
	•	Automatically pulls every Illinois dispensary (no upload needed) using OpenStreetMap (Overpass API).
	•	(Optional) Auto-checks each website for deals/promos with a safe keyword scan (BOGO/discount/special/etc).
Toggle is on by default; turn off if a site blocks scraping.
	•	Shows a cleaner UI: map + cards + per-dispensary recommendations.
	•	Gives suggested promotions for each dispensary (even if no deals are found), based on local competition density and category gaps.
	•	Highlights “deal deserts” by area.

Note: Website scraping must respect site Terms/robots.txt. This code uses very light HTML reads; you control the toggle.

⸻

Drop-in code (save as app.py)

Paste this into your repo (or Replit) as app.py and run/deploy like before.

# Illinois Dispensary Intelligence — auto-fetch all IL dispensaries + per-store promo suggestions
# One-file Streamlit app. No uploads required.
#
# Data sources:
# - OpenStreetMap (Overpass API): shop=cannabis within Illinois boundary -> name, operator (owner), website, lat/lon
# - OPTIONAL: lightweight HTML scan of each website for deal-like keywords (toggleable)
#
# What it does:
# - Fetches ALL dispensaries in IL automatically
# - Builds a clean map + cards UI
# - Detects deal deserts and makes per-dispensary suggestions
#
# ⚠️ Legal/compliance: respect each website's Terms/robots.txt before enabling scraping.

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import time, re, json, math
from typing import List, Dict, Any, Optional
import requests

# Map
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Illinois Dispensary Intelligence", layout="wide")

# ------------------------------- Controls / Sidebar -------------------------------

with st.sidebar:
    st.header("Settings")
    auto_refresh = st.button("↻ Refresh live data (OSM)", help="Refetch the latest IL dispensary list from OpenStreetMap.")
    enable_scrape = st.toggle("Scan websites for deals (experimental)", value=True,
        help="Lightweight keyword scan of each site for 'deal/discount/bogo/special/promo'. Respect robots.txt/ToS.")
    scrape_limit = st.number_input("Max websites to scan (0 = all)", min_value=0, value=0, step=1)
    st.caption("Tip: if a build stalls due to site timeouts, set a small limit (e.g. 25).")
    st.divider()
    st.markdown("**Filters**")
    f_owner = st.text_input("Owner contains")
    f_city = st.text_input("City contains")
    f_has_deals = st.selectbox("Has live deals?", ["Any", "Yes", "No"])
    f_cat = st.multiselect("Deal category", ["flower", "concentrates", "vapes", "edibles", "accessories", "mixed"])

# ------------------------------- Constants -------------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127 Safari/537.36"
}
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Illinois boundary relation in OSM (state of Illinois). If Overpass updates IDs, we also fall back to bbox.
IL_RELATION_ID = 114692
IL_BBOX = (36.9703, -91.5131, 42.5083, -87.0199)  # (south, west, north, east) rough bbox of IL

DEAL_KEYWORDS = r"(?:deal|discount|special|promo|promotion|bogo|BOGO|bundle|sale)"
CAT_KEYWORDS = {
    "flower": r"(flower|eighth|oz|ounce|pre[- ]?roll|preroll)",
    "concentrates": r"(concentrate|shatter|wax|rosin|resin|dab|badder|sauce)",
    "vapes": r"(vape|cart|cartridge|510|disposable)",
    "edibles": r"(edible|gummy|chocolate|chew|drink|beverage)",
    "accessories": r"(accessor|battery|rig|pipe|bong|papers)"
}
CATEGORY_ORDER = ["flower", "vapes", "edibles", "concentrates", "accessories", "mixed"]

# ------------------------------- Data Fetchers -------------------------------

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_osm_dispensaries() -> pd.DataFrame:
    """
    Query Overpass for all shop=cannabis within Illinois.
    Returns DataFrame: id, name, owner (operator), lat, lon, city, county, website
    """
    # Try relation-based query first (more precise)
    q1 = f"""
    [out:json][timeout:60];
    rel({IL_RELATION_ID});
    map_to_area->.il;
    (
      node["shop"="cannabis"](area.il);
      way["shop"="cannabis"](area.il);
      relation["shop"="cannabis"](area.il);
    );
    out center tags;
    """
    try:
        r = requests.post(OVERPASS_URL, data={"data": q1}, headers=HEADERS, timeout=90)
        r.raise_for_status()
        data = r.json().get("elements", [])
    except Exception:
        # BBOX fallback
        s,w,n,e = IL_BBOX
        q2 = f"""
        [out:json][timeout:60];
        (
          node["shop"="cannabis"]({s},{w},{n},{e});
          way["shop"="cannabis"]({s},{w},{n},{e});
          relation["shop"="cannabis"]({s},{w},{n},{e});
        );
        out center tags;
        """
        r = requests.post(OVERPASS_URL, data={"data": q2}, headers=HEADERS, timeout=90)
        r.raise_for_status()
        data = r.json().get("elements", [])

    rows = []
    for el in data:
        tags = el.get("tags", {})
        lat, lon = None, None
        if "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"].get("lat"), el["center"].get("lon")

        if lat is None or lon is None: 
            continue

        rows.append({
            "id": el.get("id"),
            "name": tags.get("name", "(unnamed)"),
            "owner": tags.get("operator", tags.get("brand", "")),
            "lat": lat, "lon": lon,
            "city": tags.get("addr:city", ""),
            "county": tags.get("addr:county", ""),
            "website": tags.get("website", tags.get("contact:website", "")),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["name","lat","lon"]).reset_index(drop=True)
    # best-effort Illinois filter: remove obvious out-of-state if bbox fallback overshoots
    df = df[(df["lat"]>=IL_BBOX[0]) & (df["lat"]<=IL_BBOX[2]) & (df["lon"]>=IL_BBOX[1]) & (df["lon"]<=IL_BBOX[3])]
    return df

def _safe_get(url: str, timeout: int=20) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type",""):
            return r.text[:250000]  # cap for safety
    except Exception:
        pass
    return ""

def detect_deals(html: str) -> Dict[str, Any]:
    """Return has_deals + detected categories based on keywords."""
    if not html:
        return {"has_deals": False, "categories": [], "snippets": []}
    has = re.search(DEAL_KEYWORDS, html, flags=re.I)
    cats = []
    for cat, pattern in CAT_KEYWORDS.items():
        if re.search(pattern, html, flags=re.I):
            cats.append(cat)
    # gather a few short snippets around keywords
    snips = []
    for m in re.finditer(DEAL_KEYWORDS, html, flags=re.I):
        start = max(0, m.start()-60); end = min(len(html), m.end()+80)
        snippet = re.sub(r"\s+", " ", html[start:end]).strip()
        if 12 <= len(snippet) <= 200:
            snips.append(snippet)
        if len(snips) >= 5:
            break
    return {"has_deals": bool(has), "categories": sorted(set(cats), key=CATEGORY_ORDER.index if hasattr(CATEGORY_ORDER, "index") else None), "snippets": snips}

@st.cache_data(ttl=60*15, show_spinner=False)
def scan_deals_for(df: pd.DataFrame, limit: int = 0) -> pd.DataFrame:
    """Scan each dispensary website for deals. Returns df with has_deals, deal_categories, deal_snippets."""
    rows = []
    count = 0
    for r in df.itertuples(index=False):
        if limit and count >= limit:
            rows.append({**r._asdict(), "has_deals": False, "deal_categories": [], "deal_snippets": []})
            continue
        url = (r.website or "").strip()
        if not url or not url.startswith("http"):
            rows.append({**r._asdict(), "has_deals": False, "deal_categories": [], "deal_snippets": []})
            continue
        html = _safe_get(url)
        info = detect_deals(html)
        rows.append({**r._asdict(),
                     "has_deals": info["has_deals"], 
                     "deal_categories": info["categories"], 
                     "deal_snippets": info["snippets"]})
        count += 1
        time.sleep(0.2)  # be gentle
    return pd.DataFrame(rows)

# ------------------------------- Analytics / Suggestions -------------------------------

def deal_intensity(categories: List[str]) -> float:
    if not categories: return 0.0
    w = {"flower":1.0, "concentrates":1.1, "vapes":0.9, "edibles":0.8, "accessories":0.6, "mixed":0.7}
    score = sum(w.get(c, 0.7) for c in categories)
    return round(1 - math.exp(-0.25*score), 3)

def grid_hotspots(df: pd.DataFrame, cell_deg: float = 0.25) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["g_lat","g_lon","disp","with_deals","avg_intensity","deal_rate","gap_score"])
    g_lat = (df["lat"] // cell_deg) * cell_deg
    g_lon = (df["lon"] // cell_deg) * cell_deg
    grp = df.assign(g_lat=g_lat, g_lon=g_lon).groupby(["g_lat","g_lon"], as_index=False).agg(
        disp=("id","count"),
        with_deals=("has_deals","sum"),
        avg_intensity=("intensity","mean")
    )
    grp["deal_rate"] = grp["with_deals"] / grp["disp"].replace(0,1)
    grp["gap_score"] = (1 - grp["deal_rate"]) * (1 - grp["avg_intensity"])
    return grp.sort_values("gap_score", ascending=False)

def suggest_for_store(store: pd.Series, peers: pd.DataFrame) -> List[str]:
    """
    Generate 3 tailored suggestions per dispensary based on:
    - Whether nearby peers show deals
    - The categories peers run vs this location
    - Overall gap in local grid cell
    """
    # peers within ~10km
    R = 0.1  # ~0.1 deg ~ 11km lat; rough for quick density
    local = peers[(abs(peers["lat"]-store["lat"])<=R) & (abs(peers["lon"]-store["lon"])<=R)]
    local_rate = 0.0 if local.empty else (local["has_deals"].mean())
    # underserved categories locally
    cats = pd.Series([c for lst in local["deal_categories"] for c in (lst or [])])
    if cats.empty:
        underserved = ["flower","edibles","vapes"]
    else:
        counts = cats.value_counts()
        # recommend the rarer categories first
        underserved = [c for c in CATEGORY_ORDER if counts.get(c,0)==0][:2] or list(counts.sort_values().index[:2])
        if len(underserved)<2:
            underserved = (underserved + ["flower","edibles","vapes"])[:2]
    recs = []
    if not store.get("has_deals", False):
        recs.append("Launch a clear, always-on **first-time patient** discount (10–20%).")
    if local_rate < 0.35:
        recs.append("Own the area with a weekly **BOGO** or **bundle** promo (limit to house/featured SKUs).")
    else:
        recs.append("Compete mid-week: **Wed/Thu price drops** when nearby promos spike.")
    recs.append(f"Lean into **{underserved[0]}** and **{underserved[1]}** offers to fill local gaps.")
    return recs[:3]

# ------------------------------- Fetch + Prepare -------------------------------

st.title("Illinois Dispensary Intelligence")
st.caption("Auto-fetches all IL dispensaries from OpenStreetMap. Optional website scans for live deals. Produces per-store promo suggestions and highlights deal deserts.")

with st.spinner("Fetching Illinois dispensaries from OpenStreetMap…"):
    if auto_refresh:
        fetch_osm_dispensaries.clear()  # clear cache if user clicked refresh
    base_df = fetch_osm_dispensaries()

if base_df.empty:
    st.error("No dispensaries found from OpenStreetMap. Try again in a bit.")
    st.stop()

# optionally scan websites
if enable_scrape:
    with st.spinner("Scanning dispensary websites for deals (lightweight)…"):
        lim = int(scrape_limit) if scrape_limit
