import streamlit as st, pandas as pd, numpy as np, folium, re, time
from typing import List, Dict, Any
from streamlit_folium import st_folium

st.set_page_config(page_title="Illinois Dispensary Deals Intelligence", layout="wide")
st.title("Illinois Dispensary Deals Intelligence")
st.caption("Interactive Illinois map with owner labels, live/placeholder deals, whitespace analytics, and suggestions.")

# ---------------- Mock data ----------------
data = [
    (1,"Ascend - Collinsville","Ascend Wellness Holdings",38.6726,-90.0012,"Collinsville","Madison"),
    (2,"Sunnyside - Chicago Lakeview","Cresco Labs",41.9413,-87.6535,"Chicago","Cook"),
    (3,"Verilife - Arlington Heights","PharmaCann",42.0884,-87.9811,"Arlington Heights","Cook"),
    (4,"Enlightened - Mount Prospect","Ayr Wellness",42.0664,-87.9373,"Mount Prospect","Cook"),
    (5,"NuEra - Urbana","nuEra",40.1106,-88.2073,"Urbana","Champaign"),
    (6,"RISE - Niles","Green Thumb Industries",42.0189,-87.8029,"Niles","Cook"),
    (7,"Zen Leaf - Aurora","Verano",41.7590,-88.2901,"Aurora","Kane"),
    (8,"Maribis - Springfield","Maribis",39.7817,-89.6501,"Springfield","Sangamon"),
    (9,"Consume - Oakbrook Terrace","Progressive Treatment Solutions",41.85,-87.9647,"Oakbrook Terrace","DuPage"),
    (10,"Star Buds - Burbank","Schwazze",41.7464,-87.7703,"Burbank","Cook"),
]
cols=["id","name","owner","lat","lon","city","county"]
disp=pd.DataFrame(data,columns=cols)
mock={
 "1":{"deals":[{"category":"flower","title":"20% off select eighths"}]},
 "2":{"deals":[{"category":"vapes","title":"$10 off carts"}]},
 "3":{"deals":[]},
 "4":{"deals":[{"category":"concentrates","title":"15% off shatter"}]},
 "5":{"deals":[]},
 "6":{"deals":[{"category":"flower","title":"$99 ounce special"}]},
 "7":{"deals":[{"category":"vapes","title":"Cartridge bundles"}]},
 "8":{"deals":[]},
 "9":{"deals":[{"category":"edibles","title":"Weekend brownie promo"}]},
 "10":{"deals":[]}
}

# ---------------- Helpers ----------------
def intensity(deals:List[Dict[str,Any]]):
    if not deals:return 0.0
    w={"flower":1,"concentrates":1.1,"vapes":0.9,"edibles":0.8,"mixed":0.8}
    s=sum(w.get(d.get("category","mixed"),.8) for d in deals)
    return round(1-np.exp(-0.25*s),3)
def grid(df:pd.DataFrame,deg=.25):
    g_lat=(df.lat//deg)*deg;g_lon=(df.lon//deg)*deg
    g=df.assign(g_lat=g_lat,g_lon=g_lon).groupby(["g_lat","g_lon"],as_index=False).agg(
        disp=("id","count"),deals=("has","sum"),avg=("intensity","mean"))
    g["rate"]=g.deals/g.disp;g["gap"]=(1-g.rate)*(1-g.avg)
    return g.sort_values("gap",ascending=False)
def suggest(df:pd.DataFrame):
    cats=[c for r in df.top for c in r]
    if not cats:return["Flower discounts","BOGO edibles","Bundle carts"]
    s=pd.Series(cats).value_counts().index[:3]
    return [f"More {c} promos" for c in s]
def summary(deals):
    if not deals:return"—"
    txt=[f"• {d['category']}: {d['title']}" for d in deals[:3]]
    if len(deals)>3:txt.append(f"...and {len(deals)-3} more")
    return"<br/>".join(txt)

# ---------------- Merge mock data ----------------
rows=[]
for r in disp.to_dict(orient="records"):
    d=mock.get(str(r["id"]),{"deals":[]})["deals"]
    rows.append({**r,"has":bool(d),"intensity":intensity(d),
                 "top":[x.get("category","mixed")for x in d],"sum":summary(d)})
df=pd.DataFrame(rows)

# ---------------- Filters ----------------
with st.sidebar:
    owner=st.text_input("Owner contains")
    city=st.text_input("City contains")
    cat=st.multiselect("Deal category",["flower","concentrates","vapes","edibles","mixed"])
    show=st.toggle("Show gap grid",True)
if owner:df=df[df.owner.str.contains(owner,case=False,na=False)]
if city:df=df[df.city.str.contains(city,case=False,na=False)]
if cat:df=df[df.top.apply(lambda L:any(c in L for c in cat))]

# ---------------- Map ----------------
m=folium.Map(location=(40,-89.3),zoom_start=6,control_scale=True)
for r in df.itertuples():
    folium.CircleMarker(location=(r.lat,r.lon),radius=8,tooltip=r.name,
        popup=folium.Popup(f"<b>{r.name}</b><br/>Owner:{r.owner}<br/>City:{r.city}<br/>{r.sum}",max_width=320)).add_to(m)
st_folium(m,width=1000,height=650)

# ---------------- Metrics & suggestions ----------------
c1,c2,c3,c4=st.columns(4)
c1.metric("Dispensaries",len(df));c2.metric("With Deals",int(df["has"].sum()))
c3.metric("Avg Intensity",round(df.intensity.mean(),3));c4.metric("Counties",df.county.nunique())
g=grid(df)
if show:st.subheader("Whitespace Grid");st.dataframe(g.head(20))
st.subheader("Opportunity Recommendations")
if not g.empty:
    top=g.iloc[0];mask=((df.lat//.25)*.25==top.g_lat)&((df.lon//.25)*.25==top.g_lon)
    loc=df[mask];rec=suggest(loc)
    st.markdown(f"**Hotspot:** lat≈{top.g_lat:.2f}, lon≈{top.g_lon:.2f} | {int(top.disp)} dispensaries, DealRate={top.rate:.2f}")
    for r in rec:st.markdown(f"- {r}")
else:st.write("No data.")
