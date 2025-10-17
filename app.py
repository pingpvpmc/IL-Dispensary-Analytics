# Illinois Wholesale Market Intelligence — single-file Streamlit app
# Wholesaler POV (e.g., aeriz). Finds IL dispensaries, scans websites/menus for deals and brand presence,
# ranks targets by opportunity, shows map/whitespace, and gives per-store action plans.

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import time, re, math
from typing import List, Dict, Any, Tuple
import requests
import folium
from streamlit_folium import st_folium
from urllib.parse import urljoin, urlparse

st.set_page_config(page_title="IL Wholesale Market Intelligence", layout="wide")

# ----------------------------- Sidebar (Wholesaler) -----------------------------

with st.sidebar:
    st.header("Wholesaler Settings")
    brand = st.text_input("Focus brand", value="aeriz").strip()
    brand_aliases_input = st.text_input(
        "Brand aliases (comma-separated)",
        value="aeriz,aerīz,aeriz cannabis"
    )
    target_cats = st.multiselect(
        "Target categories",
        ["flower", "vapes", "edibles", "concentrates", "accessories", "mixed"],
        default=["flower", "concentrates"]
    )

    st.divider()
    st.header("Data & Scan")
    refresh_btn = st.button("↻ Refresh dispensary list (OSM)")
    enable_scrape = st.toggle(
        "Scan websites & menus (deals + brand)", value=True,
        help="Lightweight HTML scan (respect robots.txt/ToS)."
    )
    scrape_limit = st.number_input("Max sites to scan (0 = all)", min_value=0, value=0, step=1)
    per_site_timeout = st.slider("Per-page timeout (sec)", 5, 30, 15)

    st.divider()
    st.header("View Filters")
    f_city = st.text_input("City contains")
    f_owner = st.text_input("Owner contains")
    f_has_deals = st.selectbox("Detected deals?", ["Any", "Yes", "No"])

# ----------------------------- Constants / Regex -----------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127 Safari/537.36"
}

# IL bbox (south, west, north, east) & relation id
IL_BBOX = (36.9703, -91.5131, 42.5083, -87.0199)
IL_RELATION_ID = 114692

# Overpass mirrors
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# Deal & category keyword patterns
DEAL_KEYWORDS = r"(?:deal|discount|special|promo|promotion|bogo|bundle|sale)"
CAT_KEYWORDS = {
    "flower": r"(flower|eighth|oz|ounce|pre[- ]?roll|preroll)",
    "concentrates": r"(concentrate|shatter|wax|rosin|resin|dab|badder|sauce)",
    "vapes": r"(vape|cart|cartridge|510|disposable)",
    "edibles": r"(edible|gummy|chocolate|chew|drink|beverage)",
    "accessories": r"(accessor|battery|rig|pipe|bong|papers)",
}
CATEGORY_ORDER = ["flower","vapes","edibles","concentrates","accessories","mixed"]

PLATFORM_HINTS = [
    "dutchie.com", "iheartjane.com", "leafly.com", "weedmaps.com", "flowhub.com", "treez.io"
]
COMMON_MENU_PATHS = ["/menu", "/menus", "/shop", "/products", "/deals", "/promotions", "/specials"]

# ----------------------------- Utilities -----------------------------

def norm_text(x: str) -> str:
    x = re.sub(r"[\W_]+", "", x or "", flags=re.I).lower()
    return x

def build_brand_regex(brand: str, aliases_csv: str) -> re.Pattern:
    aliases = [a.strip() for a in (aliases_csv or "").split(",") if a.strip()]
    names = sorted(set([brand] + aliases), key=len, reverse=True)
    # Normalize diacritics by allowing optional accents/marks to be ignored (roughly via \W*)
    escaped = [re.escape(n) for n in names if n]
    pattern = r"(?:" + "|".join(escaped) + r")"
    return re.compile(pattern, flags=re.I)

# ----------------------------- Data Fetch (OSM) -----------------------------

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_osm_dispensaries() -> pd.DataFrame:
    """Query multiple Overpass mirrors for IL dispensaries."""
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
                if r.status_code == 429:
                    continue
                r.raise_for_status()
                els = r.json().get("elements", [])
                if els: return els
            except Exception:
                continue
        return []

    elements = _try(q_relation) or _try(q_bbox)

    rows = []
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
            "owner": tags.get("operator", tags.get("brand","")),
            "lat": lat, "lon": lon,
            "city": tags.get("addr:city",""),
            "county": tags.get("addr:county",""),
            "website": tags.get("website", tags.get("contact:website","")),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["name","lat","lon"]).reset_index(drop=True)
    df = df[(df["lat"]>=s) & (df["lat"]<=n) & (df["lon"]>=w) & (df["lon"]<=e)]
    return df

def _safe_get(url: str, timeout: int) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type",""):
            # basic sanitization: replace tags we care about with text visible to regex
            text = r.text
            # pull title/alt/aria/links into text
            extras = []
            for pat in [r"<title[^>]*>(.*?)</title>",
                        r"alt=['\"](.*?)['\"]",
                        r"aria-label=['\"](.*?)['\"]",
                        r">([^<]{2,60})<"]:
                extras.extend(re.findall(pat, text, flags=re.I|re.S))
            blob = " ".join(extras) + " " + text
            return blob[:300000]  # cap size
    except Exception:
        pass
    return ""

def _extract_menu_urls(base_url: str, html: str) -> List[str]:
    urls = set()
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}"

    # Known platform links within html
    for host in PLATFORM_HINTS:
        for m in re.finditer(rf"https?://[^\s\"']*{re.escape(host)}/[^\s\"'>)]+", html, flags=re.I):
            urls.add(m.group(0))

    # Common local paths
    for p in COMMON_MENU_PATHS:
        urls.add(urljoin(root, p))

    return list(urls)[:6]  # keep it small

def detect_from_blobs(blobs: List[str], brand_re: re.Pattern) -> Dict[str, Any]:
    """Combine multiple HTML/text blobs into one signal."""
    combined = " ".join([b for b in blobs if b])
    has_deal = bool(re.search(DEAL_KEYWORDS, combined, flags=re.I))
    cats = []
    for cat, pat in CAT_KEYWORDS.items():
        if re.search(pat, combined, flags=re.I):
            cats.append(cat)
    # brand presence across pages
    brand_present = bool(brand_re.search(combined))
    # snippets around deals (for debug/insight)
    snips = []
    for m in re.finditer(DEAL_KEYWORDS, combined, flags=re.I):
        s = max(0, m.start()-60); e = min(len(combined), m.end()+80)
        sn = re.sub(r"\s+"," ", combined[s:e]).strip()
        if 12 <= len(sn) <= 200:
            snips.append(sn)
        if len(snips) >= 5: break
    # stable category order
    order = {k:i for i,k in enumerate(CATEGORY_ORDER)}
    cats = sorted(set(cats), key=lambda x: order.get(x, 999))
    return {"has_deals": has_deal, "categories": cats, "brand_present": brand_present, "snippets": snips}

@st.cache_data(ttl=60*15, show_spinner=False)
def scan_sites(df: pd.DataFrame, brand: str, brand_aliases: str, limit: int, timeout: int) -> pd.DataFrame:
    """Scan website + likely menu pages for deals and brand presence."""
    out = []
    brand_re = build_brand_regex(brand, brand_aliases)
    count = 0
    for r in df.itertuples(index=False):
        base = (r.website or "").strip()
        if limit and count >= limit:
            out.append({**r._asdict(), "has_deals": False, "deal_categories": [], "deal_snippets": [], "brand_present": False})
            continue

        blobs = []
        if base.startswith("http"):
            blobs.append(_safe_get(base, timeout))
            # menu/platform pages
            for murl in _extract_menu_urls(base, blobs[-1] or ""):
                blobs.append(_safe_get(murl, timeout))

        info = detect_from_blobs(blobs, brand_re)
        out.append({**r._asdict(),
                    "has_deals": info["has_deals"],
                    "deal_categories": info["categories"],
                    "deal_snippets": info["snippets"],
                    "brand_present": info["brand_present"]})
        count += 1
        time.sleep(0.2)
    return pd.DataFrame(out)

# ----------------------------- Analytics (Wholesaler) -----------------------------

def deal_intensity(categories: List[str]) -> float:
    if not categories: return 0.0
    w = {"flower":1.0, "concentrates":1.1, "vapes":0.9, "edibles":0.8, "accessories":0.6, "mixed":0.7}
    score = sum(w.get(c,0.7) for c in categories)
    return round(1 - math.exp(-0.25*score), 3)

def local_underserved_categories(store: pd.Series, all_df: pd.DataFrame) -> List[str]:
    R = 0.1  # ~10–11 km
    local = all_df[(abs(all_df["lat"]-store["lat"])<=R) & (abs(all_df["lon"]-store["lon"])<=R)]
    cats = pd.Series([c for lst in local["deal_categories"] for c in (lst or [])])
    if cats.empty:
        return ["flower","edibles","vapes"]
    counts = cats.value_counts()
    rare = [c for c in CATEGORY_ORDER if counts.get(c,0)==0][:2] or list(counts.sort_values().index[:2])
    if len(rare) < 2:
        rare = (rare + ["flower","edibles","vapes"])[:2]
    return rare

def opportunity_score(store: pd.Series, peers: pd.DataFrame, brand: str, target_cats: List[str]) -> float:
    """
    Composite brand opportunity score (0..1).
    Components:
      - Market potential (nearby density)           0.20
      - Local promo gap (inverse intensity)         0.25
      - Brand absence (no mentions)                 0.25
      - Category fit with local gaps                0.20
      - Store promo posture (not running deals)     0.10
    """
    R = 0.1
    local = peers[(abs(peers["lat"]-store["lat"])<=R) & (abs(peers["lon"]-store["lon"])<=R)]
    mp = min(len(local)/10, 1.0)
    pg = 1.0 - float(local["intensity"].mean() if not local.empty else 0.0)
    ba = 0.0 if store.get("brand_present", False) else 1.0

    underserved = set(local_underserved_categories(store, peers))
    tc = set([c.lower() for c in target_cats])
    cf = 1.0 if underserved.intersection(tc) else 0.5

    sp = 1.0 if not store.get("has_deals", False) else 0.4

    w_mp, w_pg, w_ba, w_cf, w_sp = 0.20, 0.25, 0.25, 0.20, 0.10
    score = w_mp*mp + w_pg*pg + w_ba*ba + w_cf*cf + w_sp*sp
    return round(float(score), 3)

def promo_recommendations(store: pd.Series, peers: pd.DataFrame, brand: str, target_cats: List[str]) -> List[str]:
    rare = local_underserved_categories(store, peers)
    recs = []
    if not store.get("brand_present", False):
        recs.append(f"Introduce **{brand}** with a 2-week **feature endcap** + staff education (sell-through guarantee).")
    if not store.get("has_deals", False):
        recs.append("Launch **first-time buyer** discount on branded SKUs (10–20%) to seed repeat rate.")
    else:
        recs.append("Run **mid-week price drops** on branded SKUs (Wed/Thu) to intercept competitor promos.")
    tc = [c for c in target_cats if c in rare] or rare[:2]
    if tc:
        recs.append(f"Push **bundle/BOGO** in **{tc[0]}**, and **mix-&-match** in **{tc[1] if len(tc)>1 else 'a second core'}**.")
    return recs[:3]

def whitespace_grid(df: pd.DataFrame, cell_deg: float=0.25) -> pd.DataFrame:
    """Regional grid for 'brand whitespace' (no brand mentions + weak promos)."""
    if df.empty:
        return pd.DataFrame(columns=["g_lat","g_lon","disp","with_deals","avg_intensity","promo_rate","brand_rate","BWS"])
    g_lat = (df["lat"] // cell_deg) * cell_deg
    g_lon = (df["lon"] // cell_deg) * cell_deg
    g = df.assign(g_lat=g_lat, g_lon=g_lon).groupby(["g_lat","g_lon"], as_index=False).agg(
        disp=("id","count"),
        with_deals=("has_deals","sum"),
        avg_intensity=("intensity","mean"),
        brand_hits=("brand_present","sum")
    )
    g["promo_rate"] = g["with_deals"] / g["disp"].replace(0, 1)
    g["brand_rate"] = g["brand_hits"] / g["disp"].replace(0, 1)
    # FIX: avoid Series.clip(lower=...) on a scalar. Use safe scaler instead.
    max_disp = int(g["disp"].max()) if len(g) else 1
    max_disp = max(1, max_disp)
    density = g["disp"] / max_disp
    # Brand WhiteSpace (higher is better target): density * (no brand) * (weak promos) * (low intensity)
    g["BWS"] = density * (1 - g["brand_rate"]) * (1 - 0.6*g["promo_rate"]) * (1 - 0.4*g["avg_intensity"].fillna(0))
    return g.sort_values("BWS", ascending=False)

# ----------------------------- Fetch & Prepare -----------------------------

st.title("Illinois Wholesale Market Intelligence")
st.caption("Wholesaler lens (e.g., aeriz): identify under-served areas, rank target partners, and generate brand-specific action plans.")

with st.spinner("Fetching Illinois dispensaries (OpenStreetMap)…"):
    if refresh_btn:
        fetch_osm_dispensaries.clear()
    base_df = fetch_osm_dispensaries()

if base_df.empty:
    st.error("No dispensaries found from OpenStreetMap. Tap **↻ Refresh dispensary list (OSM)** or try again shortly.")
    st.stop()

if enable_scrape:
    with st.spinner("Scanning websites & menus for deals and brand mentions…"):
        lim = int(scrape_limit) if scrape_limit else 0
        df = scan_sites(base_df, brand=brand, brand_aliases=brand_aliases_input, limit=lim, timeout=int(per_site_timeout))
else:
    df = base_df.assign(has_deals=False, deal_categories=[[]]*len(base_df),
                        deal_snippets=[[]]*len(base_df), brand_present=False)

df["intensity"] = df["deal_categories"].apply(deal_intensity)

# View filters
flt = df.copy()
if f_city:  flt = flt[flt["city"].str.contains(f_city, case=False, na=False)]
if f_owner: flt = flt[flt["owner"].str.contains(f_owner, case=False, na=False)]
if f_has_deals == "Yes": flt = flt[flt["has_deals"]==True]
elif f_has_deals == "No": flt = flt[flt["has_deals"]==False]

# ----------------------------- KPIs -----------------------------

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("IL Dispensaries", len(df))
with c2: st.metric("Detected promo stores", int(df["has_deals"].sum()))
with c3: st.metric("Avg promo intensity", round(df["intensity"].mean(),3))
with c4: st.metric(f"Sites mentioning '{brand}'", int(df["brand_present"].sum()))
with c5: st.metric("Cities covered", df["city"].replace("", np.nan).nunique())

# ----------------------------- Tabs -----------------------------

tabs = st.tabs(["Overview", "Map", "Targets", "Recommendations"])

# ---- Overview
with tabs[0]:
    st.subheader("Brand Whitespace — by Area")
    grid = whitespace_grid(df)
    st.caption("BWS = density × (1−brand rate) × (1−0.6·promo rate) × (1−0.4·avg intensity). Higher = stronger whitespace.")
    st.dataframe(grid.head(30), use_container_width=True)

    st.subheader("Detected Brand Presence (debug)")
    dbg = flt[["name","city","website","brand_present","has_deals","deal_categories"]].copy()
    st.dataframe(dbg.sort_values(["brand_present","has_deals"], ascending=False), use_container_width=True)

# ---- Map
with tabs[1]:
    st.subheader("Opportunity Map (color = brand opportunity)")
    # Score per store (brand opportunity)
    scored_full = flt.copy()
    scored_full["opportunity"] = scored_full.apply(lambda r: opportunity_score(pd.Series(r), df, brand, target_cats), axis=1)
    m = folium.Map(location=(40.0,-89.3), zoom_start=6, control_scale=True)
    for r in scored_full.itertuples():
        op = r.opportunity
        hue = int(120 * op)  # green high → red low via HSL
        color = f"hsl({hue}, 75%, 45%)"
        cat_txt = ", ".join(r.deal_categories or []) if r.deal_categories else "—"
        deals_flag = "✅ deals" if r.has_deals else "— no deals"
        brand_flag = f"🔎 brand: {'Yes' if r.brand_present else 'No'}"
        pop = f"""
        <div style='font-size:14px;line-height:1.35'>
          <b>{r.name}</b><br/>
          Owner: {r.owner or '—'}<br/>
          City/County: {r.city or '—'} / {r.county or '—'}<br/>
          Website: {r.website or '—'}<br/>
          {deals_flag} | {brand_flag}<br/>
          Categories: {cat_txt}<br/>
          Opportunity: <b>{op:.2f}</b>
        </div>
        """
        folium.CircleMarker(
            location=(r.lat, r.lon),
            radius=7,
            color=color, fill=True, fill_opacity=0.9,
            tooltip=f"{r.name} — {op:.2f}",
            popup=folium.Popup(pop, max_width=360),
        ).add_to(m)
    st_folium(m, width=None, height=520)

# ---- Targets
with tabs[2]:
    st.subheader(f"Top Targets for '{brand}'")
    scored = flt.copy()
    scored["opportunity"] = scored.apply(lambda r: opportunity_score(pd.Series(r), df, brand, target_cats), axis=1)
    scored = scored.sort_values("opportunity", ascending=False).reset_index(drop=True)
    top_n = min(60, len(scored))
    top_table = scored.head(top_n)[[
        "name","owner","city","website","has_deals","brand_present","intensity","opportunity"
    ]].rename(columns={"has_deals":"has_deals_now", "brand_present":f"{brand}_present"})
    st.dataframe(top_table, use_container_width=True, hide_index=True)

# ---- Recommendations
with tabs[3]:
    st.subheader("Per-Store Action Plans")
    cards_to_show = min(40, len(flt))
    # use overall df for context in recommendations / competition
    all_df = df
    # compute once for speed
    for r in scored.head(cards_to_show).itertuples():
        recs = promo_recommendations(pd.Series(r._asdict()), all_df, brand, target_cats)
        st.markdown(
            f"""
<div style="border:1px solid #e6e6e6;border-radius:10px;padding:14px;margin-bottom:12px;background:#fafafa">
  <div style="font-size:16px;font-weight:700">{r.name}</div>
  <div style="color:#555;margin:2px 0 8px 0">{(r.owner or '—')} · {(r.city or '—')}</div>
  <div style="font-size:13px;margin-bottom:6px">
    <b>Now running promos:</b> {"✅ Yes" if r.has_deals else "— No"}
    &nbsp;|&nbsp; <b>Brand present:</b> {"✅ Yes" if r.brand_present else "— No"}
    &nbsp;|&nbsp; <b>Opportunity:</b> {r.opportunity:.2f}
  </div>
  <div style="font-size:13px"><b>Suggested playbook:</b>
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

# ----------------------------- Footer -----------------------------
st.info("Brand presence is inferred from public site/menu text. Results are best-effort and depend on each site’s content/robots. \
For accuracy, tune aliases (sidebar) and keep scanning enabled. You can export tables from the three tabs via the overflow menu.")
