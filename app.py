# Illinois Dispensary Intelligence — auto-fetch all IL dispensaries + per-store promo suggestions
# One-file Streamlit app. No uploads required.
#
# Data sources:
# - OpenStreetMap (Overpass API): shop=cannabis within Illinois boundary -> name, operator (owner), website, lat/lon
# - OPTIONAL: lightweight HTML scan of each website for deal-like keywords (BOGO/discount/special/etc.)
#
# ⚠️ Please respect each website's Terms of Service and robots.txt before enabling scraping.

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import time, re, math
from typing import List, Dict, Any
import requests
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Illinois Dispensary Intelligence", layout="wide")

# --------------------------------- Sidebar ---------------------------------

with st.sidebar:
    st.header("Settings")
    refresh_btn = st.button("↻ Refresh live data (OSM)", help="Refetch latest Illinois dispensary list.")
    enable_scrape = st.toggle(
        "Scan websites for deals (experimental)", value=True,
        help="Lightweight keyword scan for 'deal/discount/bogo/special/promo'."
    )
    scrape_limit = st.number_input("Max websites to scan (0 = all)", min_value=0, value=0, step=1)
    st.caption("If a site is slow, set a small limit (e.g., 25).")
    st.divider()
    st.markdown("**Filters**")
    f_owner = st.text_input("Owner contains")
    f_city = st.text_input("City contains")
    f_has_deals = st.selectbox("Has detected deals?", ["Any", "Yes", "No"])
    f_cat = st.multiselect("Deal category", ["flower", "vapes", "edibles", "concentrates", "accessories", "mixed"])

# --------------------------------- Constants ---------------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127 Safari/537.36"
}

# Illinois bbox (south, west, north, east) and OSM relation id
IL_BBOX = (36.9703, -91.5131, 42.5083, -87.0199)
IL_RELATION_ID = 114692

# Overpass mirrors (we’ll try each until one responds with data)
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

DEAL_KEYWORDS = r"(?:deal|discount|special|promo|promotion|bogo|bundle|sale)"
CAT_KEYWORDS = {
    "flower": r"(flower|eighth|oz|ounce|pre[- ]?roll|preroll)",
    "concentrates": r"(concentrate|shatter|wax|rosin|resin|dab|badder|sauce)",
    "vapes": r"(vape|cart|cartridge|510|disposable)",
    "edibles": r"(edible|gummy|chocolate|chew|drink|beverage)",
    "accessories": r"(accessor|battery|rig|pipe|bong|papers)"
}
CATEGORY_ORDER = ["flower", "vapes", "edibles", "concentrates", "accessories", "mixed"]

# --------------------------------- Helpers ---------------------------------

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_osm_dispensaries() -> pd.DataFrame:
    """
    Query multiple Overpass mirrors for shop=cannabis in Illinois.
    Tries state relation first; falls back to bbox; returns a clean DataFrame.
    """
    s, w, n, e = IL_BBOX

    q_relation = f"""
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

    q_bbox = f"""
    [out:json][timeout:60];
    (
      node["shop"="cannabis"]({s},{w},{n},{e});
      way["shop"="cannabis"]({s},{w},{n},{e});
      relation["shop"="cannabis"]({s},{w},{n},{e});
    );
    out center tags;
    """

    def _try(query: str) -> List[Dict[str, Any]]:
        for url in OVERPASS_MIRRORS:
            try:
                r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=90)
                # 429 == rate limited; try next mirror
                if r.status_code == 429:
                    continue
                r.raise_for_status()
                els = r.json().get("elements", [])
                if els:
                    return els
            except Exception:
                continue
        return []

    elements = _try(q_relation)
    if not elements:
        elements = _try(q_bbox)

    rows: List[Dict[str, Any]] = []
    for el in elements:
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
    # prune any out-of-state strays if bbox was used
    df = df[(df["lat"]>=s) & (df["lat"]<=n) & (df["lon"]>=w) & (df["lon"]<=e)]
    return df

def _safe_get(url: str, timeout: int = 20) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type",""):
            return r.text[:250000]  # cap to be safe
    except Exception:
        pass
    return ""

def detect_deals(html: str) -> Dict[str, Any]:
    """Return has_deals + detected categories and a few keyword snippets."""
    if not html:
        return {"has_deals": False, "categories": [], "snippets": []}
    has = re.search(DEAL_KEYWORDS, html, flags=re.I)
    cats = []
    for cat, pattern in CAT_KEYWORDS.items():
        if re.search(pattern, html, flags=re.I):
            cats.append(cat)
    snips = []
    for m in re.finditer(DEAL_KEYWORDS, html, flags=re.I):
        start = max(0, m.start()-60); end = min(len(html), m.end()+80)
        s = re.sub(r"\s+", " ", html[start:end]).strip()
        if 12 <= len(s) <= 200:
            snips.append(s)
        if len(snips) >= 5:
            break
    # Keep a stable category order
    order = {k:i for i,k in enumerate(CATEGORY_ORDER)}
    cats = sorted(set(cats), key=lambda x: order.get(x, 999))
    return {"has_deals": bool(has), "categories": cats, "snippets": snips}

@st.cache_data(ttl=60*15, show_spinner=False)
def scan_deals_for(df: pd.DataFrame, limit: int = 0) -> pd.DataFrame:
    """Scan each dispensary website for deals. Returns df with has_deals, deal_categories, deal_snippets."""
    out = []
    count = 0
    for r in df.itertuples(index=False):
        if limit and count >= limit:
            out.append({**r._asdict(), "has_deals": False, "deal_categories": [], "deal_snippets": []})
            continue
        url = (r.website or "").strip()
        if not url or not url.startswith("http"):
            out.append({**r._asdict(), "has_deals": False, "deal_categories": [], "deal_snippets": []})
            continue
        html = _safe_get(url)
        info = detect_deals(html)
        out.append({**r._asdict(),
                    "has_deals": info["has_deals"],
                    "deal_categories": info["categories"],
                    "deal_snippets": info["snippets"]})
        count += 1
        time.sleep(0.2)  # be gentle
    return pd.DataFrame(out)

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

def suggest_for_store(store: pd.Series, all_df: pd.DataFrame) -> List[str]:
    """
    Generate 3 tailored suggestions per dispensary based on:
    - Whether nearby peers show deals
    - The categories peers run vs this location
    - Overall gap in local grid cell
    """
    # peers within ~10km (rough)
    R = 0.1  # ~0.1 deg ~ 11km lat
    local = all_df[(abs(all_df["lat"]-store["lat"])<=R) & (abs(all_df["lon"]-store["lon"])<=R)]
    local_rate = 0.0 if local.empty else float(local["has_deals"].mean())

    cats_series = pd.Series([c for lst in local["deal_categories"] for c in (lst or [])])
    if cats_series.empty:
        underserved = ["flower","edibles","vapes"]
    else:
        counts = cats_series.value_counts()
        # recommend rarer categories first in that area
        underserved = [c for c in CATEGORY_ORDER if counts.get(c,0)==0][:2] or list(counts.sort_values().index[:2])
        if len(underserved) < 2:
            underserved = (underserved + ["flower","edibles","vapes"])[:2]

    recs: List[str] = []
    if not store.get("has_deals", False):
        recs.append("Launch a clear, always-on **first-time patient** discount (10–20%).")
    if local_rate < 0.35:
        recs.append("Own the area with a weekly **BOGO** or **bundle** promo (focus on house/featured SKUs).")
    else:
        recs.append("Compete mid-week with **Wed/Thu price drops** when nearby promos spike.")
    recs.append(f"Lean into **{underserved[0]}** and **{underserved[1]}** offers to fill local gaps.")
    return recs[:3]

# --------------------------------- Fetch & Prepare ---------------------------------

st.title("Illinois Dispensary Intelligence")
st.caption("Auto-fetches all IL dispensaries from OpenStreetMap (multi-mirror). Optional website scans for live deals. Produces per-store promo suggestions and highlights deal deserts.")

with st.spinner("Fetching Illinois dispensaries from OpenStreetMap…"):
    if refresh_btn:
        fetch_osm_dispensaries.clear()  # clear cache on demand
    base_df = fetch_osm_dispensaries()

if base_df.empty:
    st.error("No dispensaries found from OpenStreetMap. Tap **↻ Refresh live data (OSM)** in the sidebar or try again shortly.")
    st.stop()

if enable_scrape:
    with st.spinner("Scanning dispensary websites for deals (lightweight)…"):
        lim = int(scrape_limit) if scrape_limit else 0
        df = scan_deals_for(base_df, limit=lim)
else:
    df = base_df.assign(has_deals=False, deal_categories=[[]]*len(base_df), deal_snippets=[[]]*len(base_df))

df["intensity"] = df["deal_categories"].apply(deal_intensity)

# Filters
flt = df.copy()
if f_owner:
    flt = flt[flt["owner"].str.contains(f_owner, case=False, na=False)]
if f_city:
    flt = flt[flt["city"].str.contains(f_city, case=False, na=False)]
if f_has_deals == "Yes":
    flt = flt[flt["has_deals"]==True]
elif f_has_deals == "No":
    flt = flt[flt["has_deals"]==False]
if f_cat:
    flt = flt[flt["deal_categories"].apply(lambda L: any(c in (L or []) for c in f_cat))]

# --------------------------------- Map ---------------------------------

st.subheader("Interactive Map")
m = folium.Map(location=(40.0, -89.3), zoom_start=6, control_scale=True)
for r in flt.itertuples():
    cat_txt = ", ".join(r.deal_categories or []) if r.deal_categories else "—"
    deals_flag = "✅ Has deals" if r.has_deals else "— No deals detected"
    pop = f"""
    <div style='font-size:14px;line-height:1.35'>
      <b>{r.name}</b><br/>
      Owner: {r.owner or '—'}<br/>
      City/County: {r.city or '—'} / {r.county or '—'}<br/>
      Website: {r.website or '—'}<br/>
      {deals_flag}<br/>
      Categories: {cat_txt}<br/>
    </div>
    """
    folium.CircleMarker(
        location=(r.lat, r.lon),
        radius=7,
        color="#2c7fb8",
        fill=True, fill_opacity=0.85,
        tooltip=r.name,
        popup=folium.Popup(pop, max_width=360),
    ).add_to(m)
st_folium(m, width=None, height=520)

# --------------------------------- KPIs ---------------------------------

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Dispensaries (IL)", len(df))
with c2: st.metric("With detected deals", int(df["has_deals"].sum()))
with c3: st.metric("Avg deal intensity", round(df["intensity"].mean(), 3))
with c4: st.metric("Cities covered", df["city"].replace("", np.nan).nunique())

# --------------------------------- Deal deserts & Suggestions ---------------------------------

st.subheader("Deal Deserts (by area)")
grid = grid_hotspots(df)
st.caption("Higher 'gap score' = fewer/weaker promos relative to local density.")
st.dataframe(grid.head(25), use_container_width=True)

st.subheader("Per-Dispensary Recommendations")
rows = []
for r in flt.itertuples():
    recs = suggest_for_store(pd.Series(r._asdict()), df)
    rows.append({
        "name": r.name,
        "owner": r.owner or "",
        "city": r.city or "",
        "website": r.website or "",
        "has_deals": "Yes" if r.has_deals else "No",
        "suggestion_1": recs[0] if len(recs)>0 else "",
        "suggestion_2": recs[1] if len(recs)>1 else "",
        "suggestion_3": recs[2] if len(recs)>2 else "",
    })
suggest_df = pd.DataFrame(rows)
st.dataframe(suggest_df, use_container_width=True, hide_index=True)

# Pretty card view for readability
st.markdown("### Clean Cards")
for r in flt.itertuples():
    recs = suggest_for_store(pd.Series(r._asdict()), df)
    st.markdown(
        f"""
<div style="border:1px solid #e6e6e6;border-radius:10px;padding:14px;margin-bottom:10px;background:#fafafa">
  <div style="font-size:16px;font-weight:700">{r.name}</div>
  <div style="color:#555;margin:2px 0 8px 0">{(r.owner or '—')} · {(r.city or '—')}</div>
  <div style="font-size:13px;margin-bottom:6px">
    <b>Deals detected:</b> {"✅ Yes" if r.has_deals else "— No"} &nbsp;|&nbsp;
    <b>Categories:</b> {(", ".join(r.deal_categories) if r.deal_categories else "—")}
  </div>
  <div style="font-size:13px"><b>Suggested promotions:</b>
    <ul style="margin:6px 0 0 18px">
      <li>{recs[0] if len(recs)>0 else ""}</li>
      <li>{recs[1] if len(recs)>1 else ""}</li>
      <li>{recs[2] if len(recs)>2 else ""}</li>
    </ul>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

st.info("Suggestions are heuristic: we consider local competition density, nearby categories, and area gap scores. For higher accuracy, keep website scanning enabled or integrate official promo feeds if available.")

st.caption("Sources: OpenStreetMap (shop=cannabis) via multiple Overpass mirrors. Website scanning is optional and best-effort.")
