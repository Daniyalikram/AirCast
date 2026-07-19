"""
AirCast — Islamabad PM2.5 Day-Ahead Forecaster
================================================
Atmospheric, glassmorphism-styled Streamlit dashboard.

  1) Explore & Results — historical air-quality patterns + honest model performance.
  2) Live Forecast     — pulls the latest Islamabad data from Open-Meteo (no API key)
                         and predicts PM2.5 for the next 24 hours on demand.

Data : Copernicus CAMS (air quality) + ERA5 / Open-Meteo (weather).  PM2.5 is modelled,
       not sensor-measured.  Model: scikit-learn RandomForestRegressor.
Author: Muhammad Danyal Ikram
"""

from pathlib import Path
import numpy as np
import random
import time
import pandas as pd
import requests
import streamlit as st
import joblib
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AirCast — Islamabad PM2.5 Forecast",
                   page_icon="\U0001F32B\uFE0F", layout="wide",
                   initial_sidebar_state="collapsed")

LAT, LON = 33.6844, 73.0479
TZ = "Asia/Karachi"
HORIZON = 24

MODEL_PATH    = Path("models/rf_pm25.pkl")
CLEAN_PATH    = Path("data/islamabad_clean.csv")
FEATURES_PATH = Path("data/islamabad_features_v2.csv")

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                "surface_pressure", "wind_speed_10m", "wind_direction_10m", "precipitation"]
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"
WHO_GUIDELINE = 15.0

# AQI palette (hex with #, used in HTML/Plotly)
AQI = [("Good", 0, 12, "#22c55e"), ("Moderate", 12, 35.4, "#eab308"),
       ("Unhealthy (sensitive)", 35.4, 55.4, "#f97316"), ("Unhealthy", 55.4, 150.4, "#ef4444"),
       ("Very Unhealthy", 150.4, 250.4, "#a855f7"), ("Hazardous", 250.4, 500, "#7f1d1d")]

def pm25_category(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "Unknown", "#94a3b8", "No data available."
    advice = {
        "Good": "Air quality is satisfactory \u2014 enjoy the outdoors.",
        "Moderate": "Acceptable; unusually sensitive people should limit long outdoor exertion.",
        "Unhealthy (sensitive)": "Sensitive groups should reduce prolonged outdoor exertion.",
        "Unhealthy": "Everyone may feel effects \u2014 limit outdoor activity.",
        "Very Unhealthy": "Health alert \u2014 avoid outdoor activity.",
        "Hazardous": "Emergency conditions \u2014 stay indoors.",
    }
    for name, lo, hi, col in AQI:
        if v <= hi:
            return name, col, advice[name]
    return "Hazardous", "#7f1d1d", advice["Hazardous"]

# ---------------------------------------------------------------------------
# Theme (CSS)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root{ --ink:#0f2233; --muted:#57708a; --teal:#0ea5a5; --line:rgba(15,34,51,.08); }

.stApp{
  background:
    radial-gradient(1200px 600px at 15% -5%, #cfe8ff 0%, rgba(207,232,255,0) 55%),
    radial-gradient(1000px 500px at 90% 0%, #d7f5ee 0%, rgba(215,245,238,0) 50%),
    linear-gradient(180deg, #eaf4ff 0%, #f2f8ff 40%, #f7fafc 100%);
  background-attachment: fixed;
}
/* drifting atmospheric haze */
.stApp::before, .stApp::after{
  content:""; position:fixed; border-radius:50%; filter:blur(70px);
  pointer-events:none; z-index:0; opacity:.5;
}
.stApp::before{ width:520px;height:520px; top:-120px; left:-100px;
  background:radial-gradient(circle,#bfe3ff,rgba(191,227,255,0)); animation:drift1 26s ease-in-out infinite; }
.stApp::after{ width:460px;height:460px; top:20%; right:-120px;
  background:radial-gradient(circle,#bff0e4,rgba(191,240,228,0)); animation:drift2 32s ease-in-out infinite; }
@keyframes drift1{0%,100%{transform:translate(0,0)}50%{transform:translate(60px,50px)}}
@keyframes drift2{0%,100%{transform:translate(0,0)}50%{transform:translate(-50px,40px)}}

header[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
[data-testid="stToolbar"]{ visibility:hidden; }
.block-container{ padding-top:1.4rem; padding-bottom:3rem; max-width:1200px; position:relative; z-index:1; }

html, body, [class*="css"]{ font-family:'Inter',sans-serif; color:var(--ink); }
h1,h2,h3{ font-family:'Space Grotesk',sans-serif; }

/* hero */
.hero{ margin:.2rem 0 1.1rem; }
.hero .pill{ display:inline-block; font-family:'Space Grotesk'; font-size:.72rem; letter-spacing:.12em;
  font-weight:600; color:var(--teal); background:rgba(14,165,165,.10); border:1px solid rgba(14,165,165,.25);
  padding:.28rem .7rem; border-radius:999px; }
.hero h1{ font-size:3.3rem; font-weight:700; margin:.5rem 0 .2rem; letter-spacing:-.02em; color:var(--ink); }
.hero p{ font-size:1.12rem; color:var(--muted); margin:0; max-width:760px; }

/* glass */
.glass{ background:rgba(255,255,255,.55); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  border:1px solid rgba(255,255,255,.65); border-radius:20px;
  box-shadow:0 10px 30px rgba(31,45,61,.10), inset 0 1px 0 rgba(255,255,255,.6); }

.sec{ margin:1.8rem 0 .6rem; }
.sec .eyebrow{ display:inline-block; font-family:'Space Grotesk'; font-size:.72rem; letter-spacing:.14em; font-weight:600;
  color:#0d8f8f; text-transform:uppercase; background:rgba(255,255,255,.62); border:1px solid rgba(255,255,255,.6);
  backdrop-filter:blur(8px); padding:.18rem .6rem; border-radius:9px; }
.sec .sec-title{ display:inline-block; font-family:'Space Grotesk'; font-size:1.5rem; font-weight:700; color:var(--ink);
  background:rgba(255,255,255,.66); border:1px solid rgba(255,255,255,.65); backdrop-filter:blur(10px);
  padding:.2rem .75rem; border-radius:14px; margin-top:.4rem; box-shadow:0 4px 16px rgba(31,45,61,.06); }

/* metric card */
.metric{ padding:1.05rem 1.2rem; min-height:118px; }
.metric .m-label{ font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-weight:600; }
.metric .m-value{ font-family:'Space Grotesk'; font-size:2.15rem; font-weight:700; line-height:1.15; margin:.15rem 0; }
.metric .m-sub{ font-size:.82rem; color:var(--muted); }

/* aqi hero */
.aqi{ padding:1.4rem 1.6rem; display:flex; align-items:center; gap:1.6rem; }
.aqi .ring{ width:120px;height:120px;border-radius:50%; display:flex;flex-direction:column;
  align-items:center;justify-content:center; color:#fff; flex:0 0 auto;
  box-shadow:0 10px 26px rgba(31,45,61,.18); }
.aqi .ring .v{ font-family:'Space Grotesk'; font-size:2.1rem; font-weight:700; line-height:1; }
.aqi .ring .u{ font-size:.72rem; opacity:.9; }
.aqi .txt .badge{ display:inline-block; padding:.22rem .7rem; border-radius:999px; color:#fff;
  font-weight:600; font-size:.85rem; }
.aqi .txt h3{ margin:.5rem 0 .2rem; font-size:1.25rem; color:var(--ink); }
.aqi .txt p{ margin:0; color:var(--muted); }

/* legend chip */
.chip{ display:inline-flex; align-items:center; gap:.4rem; margin:.15rem .5rem .15rem 0; font-size:.82rem; color:var(--muted); }
.chip i{ width:14px;height:14px;border-radius:4px; display:inline-block; }

/* tabs */
.stTabs [data-baseweb="tab-list"]{ gap:.4rem; background:transparent; border-bottom:none; }
.stTabs [data-baseweb="tab"]{ background:rgba(255,255,255,.5); border:1px solid rgba(255,255,255,.7);
  border-radius:12px; padding:.55rem 1.1rem; font-weight:600; color:var(--muted); }
.stTabs [aria-selected="true"]{ background:var(--ink) !important; color:#fff !important; border-color:var(--ink) !important; }

/* button */
.stButton>button{ background:linear-gradient(135deg,#0ea5a5,#0891b2); color:#fff; border:none;
  border-radius:12px; padding:.6rem 1.3rem; font-weight:600; box-shadow:0 8px 20px rgba(8,145,178,.25); }
.stButton>button:hover{ filter:brightness(1.05); }

.cap{ display:inline-block; font-size:.82rem; color:#42576c; font-style:italic; margin-top:.25rem;
  background:rgba(255,255,255,.6); border:1px solid rgba(255,255,255,.6); backdrop-filter:blur(6px);
  padding:.1rem .55rem; border-radius:9px; }
.chart-h{ display:inline-block; font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.05rem; color:var(--ink);
  background:rgba(255,255,255,.66); border:1px solid rgba(255,255,255,.6); backdrop-filter:blur(10px);
  padding:.16rem .65rem; border-radius:11px; margin:.5rem 0 .2rem; }

/* ---- ambient effect layers: fill the viewport, ALWAYS behind content (z-index:-1) ---- */
.fxbg{ position:fixed; inset:0; z-index:-1; pointer-events:none; overflow:hidden; }
.fx-cloud{ position:absolute; border-radius:50%; filter:blur(10px);
  background:radial-gradient(closest-side, rgba(255,255,255,.72), rgba(255,255,255,0)); animation:fxDrift linear infinite; }
.fx-smoke{ position:absolute; border-radius:50%; filter:blur(22px); animation:fxSmoke ease-in-out infinite; }
.fx-pt{ position:absolute; border-radius:50%; filter:blur(.3px);
  animation-iteration-count:infinite; animation-timing-function:ease-in-out; }
.fx-rd{ position:absolute; top:-12%; width:2.5px; border-radius:3px;
  background:linear-gradient(transparent, rgba(96,150,215,.95)); animation:fxFall linear infinite; }
.fx-sun{ position:absolute; border-radius:50%; animation:fxPulse 6s ease-in-out infinite; }
@keyframes fxDrift{ from{transform:translateX(-32vw)} to{transform:translateX(132vw)} }
@keyframes fxSmoke{ 0%{transform:translate(0,0) scale(1);opacity:0} 22%{opacity:.7} 100%{transform:translate(7vw,-16vh) scale(1.7);opacity:0} }
@keyframes fxFloaty{ 0%,100%{transform:translate(0,0)} 50%{transform:translate(20px,-38px)} }
@keyframes fxFloatx{ 0%{transform:translateX(-7vw)} 100%{transform:translateX(107vw)} }
@keyframes fxFall{ to{transform:translateY(120vh)} }
@keyframes fxPulse{ 0%,100%{transform:scale(1);opacity:.85} 50%{transform:scale(1.09);opacity:1} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@st.cache_data
def load_clean():
    return pd.read_csv(CLEAN_PATH, parse_dates=["time"]) if CLEAN_PATH.exists() else None

@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_PATH, parse_dates=["time"]) if FEATURES_PATH.exists() else None

def _get(url, params):
    # Retry a few times with backoff — venue Wi-Fi is often flaky.
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30,
                             headers={"User-Agent": "AirCast/1.0"})
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last

@st.cache_data(ttl=1800)
def fetch_live():
    aq = _get(AQ_URL, {"latitude": LAT, "longitude": LON, "hourly": "pm2_5,pm10,dust",
                       "past_days": 5, "forecast_days": 1, "timezone": TZ})
    aq_df = pd.DataFrame(aq["hourly"]); aq_df["time"] = pd.to_datetime(aq_df["time"])
    wx = _get(WX_URL, {"latitude": LAT, "longitude": LON, "hourly": ",".join(WEATHER_VARS),
                       "past_days": 5, "forecast_days": 2, "timezone": TZ})
    wx_df = pd.DataFrame(wx["hourly"]); wx_df["time"] = pd.to_datetime(wx_df["time"])
    return aq_df, wx_df

def build_live_features(aq_df, wx_df):
    df = pd.merge(wx_df, aq_df, on="time", how="left").sort_values("time").reset_index(drop=True)
    df["hour"]=df["time"].dt.hour; df["dayofweek"]=df["time"].dt.dayofweek
    df["month"]=df["time"].dt.month; df["dayofyear"]=df["time"].dt.dayofyear
    df["is_weekend"]=(df["dayofweek"]>=5).astype(int)
    df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24); df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"]=np.sin(2*np.pi*df["month"]/12); df["month_cos"]=np.cos(2*np.pi*df["month"]/12)
    for lag in [1,2,3,6,12,24,48]: df[f"pm25_lag_{lag}"]=df["pm2_5"].shift(lag)
    df["pm25_roll_mean_3"]=df["pm2_5"].rolling(3).mean()
    df["pm25_roll_mean_24"]=df["pm2_5"].rolling(24).mean()
    df["pm25_roll_std_24"]=df["pm2_5"].rolling(24).std()
    for w in WEATHER_VARS: df[f"fc_{w}"]=df[w].shift(-HORIZON)
    return df

def predict_next_24h(model, feat_names, df):
    obs = df.loc[df["pm2_5"].notna(), "time"]
    if obs.empty: return None, None
    now_t = obs.max()
    origins = df[(df["time"]>=now_t-pd.Timedelta(hours=HORIZON-1)) & (df["time"]<=now_t)].copy()
    X = origins[feat_names]; ok = X.notna().all(axis=1); X, origins = X[ok], origins[ok]
    if X.empty: return now_t, None
    preds = model.predict(X)
    out = pd.DataFrame({"forecast_time": pd.to_datetime(origins["time"].values)+pd.Timedelta(hours=HORIZON),
                        "predicted_pm25": np.round(preds,1)}).reset_index(drop=True)
    return now_t, out

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def section(eyebrow, title):
    st.markdown(f"<div class='sec'><div class='eyebrow'>{eyebrow}</div>"
                f"<div class='sec-title'>{title}</div></div>", unsafe_allow_html=True)

def metric_card(label, value, sub, color="#0f2233"):
    st.markdown(f"<div class='glass metric'><div class='m-label'>{label}</div>"
                f"<div class='m-value' style='color:{color}'>{value}</div>"
                f"<div class='m-sub'>{sub}</div></div>", unsafe_allow_html=True)

BASE_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   font=dict(family="Inter, sans-serif", color="#0f2233", size=13),
                   margin=dict(l=10, r=10, t=14, b=28), hovermode="x unified",
                   showlegend=False,
                   hoverlabel=dict(bgcolor="rgba(255,255,255,.96)", bordercolor="rgba(15,34,51,.10)",
                                   font=dict(family="Inter, sans-serif", color="#0f2233", size=12)))

def chart_head(title, sub=None):
    st.markdown(f"<div class='chart-h'>{title}</div>" + (f"<div class='cap'>{sub}</div>" if sub else ""), unsafe_allow_html=True)

def style_axes(fig, spikes=False):
    fig.update_xaxes(automargin=True, showgrid=False, showline=False, ticks="",
                     tickfont=dict(color="#7089a0", size=12), title_font=dict(color="#7089a0", size=12))
    fig.update_yaxes(automargin=True, title_standoff=14, showline=False, ticks="",
                     gridcolor="rgba(15,34,51,.08)", griddash="dot", zeroline=False,
                     tickfont=dict(color="#7089a0", size=12), title_font=dict(color="#7089a0", size=12))
    if spikes:
        fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1,
                         spikecolor="rgba(15,34,51,.22)", spikedash="dot")
    return fig

def aqi_bands(fig, ymax):
    for name, lo, hi, col in AQI:
        if lo > ymax: break
        fig.add_hrect(y0=lo, y1=min(hi, ymax*1.05), fillcolor=col, opacity=0.06,
                      line_width=0, layer="below")

# ---------------------------------------------------------------------------
# Ambient effects (CSS-only, lightweight) — visible, but ALWAYS behind content
# ---------------------------------------------------------------------------
def cat_index(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 1
    for i, (n, lo, hi, c) in enumerate(AQI):
        if v <= hi:
            return i
    return 5

def _fx_clouds(n=4, dur=(80, 150)):
    o = ""
    for _ in range(n):
        top = random.uniform(2, 34); w = random.randint(260, 480); h = int(w*0.36)
        d = random.uniform(*dur); delay = -random.uniform(0, d)
        o += (f"<div class='fx-cloud' style='top:{top:.0f}%;width:{w}px;height:{h}px;"
              f"animation-duration:{d:.0f}s;animation-delay:{delay:.0f}s'></div>")
    return o

def _fx_smoke(n=7, color="rgba(140,132,116,.34)", dur=(24, 42)):
    o = ""
    for _ in range(n):
        left = random.uniform(0, 100); w = random.randint(200, 360)
        d = random.uniform(*dur); delay = -random.uniform(0, d)
        o += (f"<div class='fx-smoke' style='left:{left:.0f}%;bottom:-12%;width:{w}px;height:{w}px;"
              f"background:radial-gradient(closest-side,{color},rgba(0,0,0,0));"
              f"animation-duration:{d:.0f}s;animation-delay:{delay:.0f}s'></div>")
    return o

def _fx_particles(n, dot, anim="fxFloaty", dur=(14, 26), size=(6, 14)):
    o = ""
    for _ in range(n):
        sz = random.randint(*size)
        o += (f"<span class='fx-pt' style='left:{random.uniform(0,100):.1f}%;top:{random.uniform(0,100):.1f}%;"
              f"width:{sz}px;height:{sz}px;background:{dot};"
              f"animation-name:{anim};animation-duration:{random.uniform(*dur):.1f}s;"
              f"animation-delay:{random.uniform(0,8):.1f}s;opacity:{random.uniform(.3,.6):.2f}'></span>")
    return o

def _fx_rain(n=70):
    o = ""
    for _ in range(n):
        o += (f"<span class='fx-rd' style='left:{random.uniform(0,100):.1f}%;height:{random.randint(55,110)}px;"
              f"opacity:{random.uniform(.35,.65):.2f};animation-duration:{random.uniform(.5,.95):.2f}s;"
              f"animation-delay:{random.uniform(0,2):.2f}s'></span>")
    return o

def _fx_sun():
    return ("<div class='fx-sun' style='top:-140px;right:-120px;width:520px;height:520px;"
            "background:radial-gradient(circle,rgba(255,209,110,.62),rgba(255,209,110,0) 66%)'></div>")

def analysis_fx():
    """Tab 1 — visible drifting dust & smoke haze (calm, still behind content)."""
    return ("<div class='fxbg'>" + _fx_smoke(7, "rgba(150,140,120,.30)", (26, 46))
            + _fx_particles(26, "rgba(150,138,112,.6)", anim="fxFloatx", dur=(24, 44))
            + _fx_clouds(3) + "</div>")

def live_fx(peak, is_rain, wind_high, hour):
    """Tab 2 — reactive atmosphere driven by the forecast."""
    ci = cat_index(peak)
    layers = "<div class='fxbg'>"
    if is_rain:
        layers += _fx_rain(72) + _fx_clouds(3)
    elif ci >= 3:                      # unhealthy or worse -> smog
        layers += (_fx_smoke(9, "rgba(115,105,90,.40)", (22, 38))
                   + _fx_particles(42, "rgba(92,82,66,.6)",
                                   anim=("fxFloatx" if wind_high else "fxFloaty"),
                                   dur=((7, 13) if wind_high else (15, 28))))
    elif ci == 2:                      # unhealthy (sensitive) -> dust haze
        layers += _fx_smoke(6, "rgba(150,140,118,.32)", (26, 44)) + _fx_particles(28, "rgba(150,138,110,.6)")
    elif 7 <= hour <= 18:              # clean daytime -> sunny
        layers += _fx_sun() + _fx_clouds(5)
    else:                              # calm / clean
        layers += _fx_particles(22, "rgba(120,195,205,.6)", dur=(20, 34)) + _fx_clouds(3)
    layers += "</div>"
    return layers

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='hero'><span class='pill'>\u25CF LIVE \u00B7 AUTONOMOUS AIR-QUALITY FORECASTING</span>"
    "<h1>AirCast</h1>"
    "<p>Forecasting Islamabad\u2019s PM2.5 air pollution a full day ahead \u2014 built from free, "
    "open climate data. PM2.5 is modelled (Copernicus CAMS), not sensor-measured.</p></div>",
    unsafe_allow_html=True)

bundle, clean, features = load_model(), load_clean(), load_features()
tab_explore, tab_live = st.tabs(["\U0001F4CA  Explore & Results", "\U0001F52E  Live Forecast"])

# ===========================================================================
# TAB 1 — Explore & Results
# ===========================================================================
with tab_explore:
    st.markdown(analysis_fx(), unsafe_allow_html=True)
    if clean is None:
        st.warning("`data/islamabad_clean.csv` not found. Add it to the repo to see this tab.")
    else:
        section("The problem", "How bad is Islamabad\u2019s air?")
        pm = clean["pm2_5"].dropna()
        pct_who = (pm > WHO_GUIDELINE).mean()*100
        pct_un  = (pm > 55.4).mean()*100
        c = st.columns(4)
        with c[0]: metric_card("Average PM2.5", f"{pm.mean():.1f}", f"{pm.mean()/WHO_GUIDELINE:.1f}\u00D7 WHO safe limit", "#f97316")
        with c[1]: metric_card("Worst hour", f"{pm.max():.0f}", "\u00B5g/m\u00B3 \u2014 very unhealthy", "#ef4444")
        with c[2]: metric_card("Hours above WHO", f"{pct_who:.0f}%", "of all hours breach the limit", "#0f2233")
        with c[3]: metric_card("Hours 'Unhealthy'", f"{pct_un:.0f}%", "above 55 \u00B5g/m\u00B3", "#0f2233")

        # AQI legend chips
        chips = "".join(f"<span class='chip'><i style='background:{col}'></i>{name}</span>"
                        for name, lo, hi, col in AQI)
        st.markdown(f"<div style='margin:.7rem 0 .2rem'>{chips}</div>", unsafe_allow_html=True)

        section("Patterns", "When is the air worst?")
        L, R = st.columns(2)
        with L:
            chart_head("Average PM2.5 by month", "Winter and mid-summer are the worst.")
            monthly = clean.assign(m=clean["time"].dt.month).groupby("m")["pm2_5"].mean()
            names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            fig = go.Figure(go.Bar(x=names, y=monthly.values,
                    marker=dict(color=monthly.values, colorscale=[[0,"#22c55e"],[0.5,"#f97316"],[1,"#ef4444"]],
                                cornerradius=7, line_width=0),
                    hovertemplate="%{x}: %{y:.0f} \u00B5g/m\u00B3<extra></extra>"))
            fig.add_hline(y=WHO_GUIDELINE, line=dict(color="#16a34a", dash="dash"),
                          annotation_text="WHO 15", annotation_position="top left",
                          annotation_font_color="#16a34a")
            fig.update_layout(height=300, **BASE_LAYOUT)
            style_axes(fig)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
        with R:
            chart_head("Average PM2.5 by hour of day", "Cleanest at midday; builds again after dark.")
            hourly = clean.assign(h=clean["time"].dt.hour).groupby("h")["pm2_5"].mean()
            fig = go.Figure(go.Scatter(x=hourly.index, y=hourly.values, mode="lines",
                    line=dict(color="#0ea5a5", width=3, shape="spline"), fill="tozeroy",
                    fillgradient=dict(type="vertical", colorscale=[[0,"rgba(14,165,165,0)"],[1,"rgba(14,165,165,.42)"]]),
                    hovertemplate="%{x}:00 \u2014 %{y:.0f} \u00B5g/m\u00B3<extra></extra>"))
            fig.update_layout(height=300, xaxis_title="Hour", **BASE_LAYOUT)
            style_axes(fig, spikes=True)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

        section("The model", "Does it actually work?")
        if bundle is None or features is None:
            st.info("Add `models/rf_pm25.pkl` and `data/islamabad_features_v2.csv` to see model results.")
        else:
            model, feat, split = bundle["model"], bundle["features"], bundle["split_index"]
            m = features.sort_values("time").reset_index(drop=True)
            Xte, yte, tte = m[feat].iloc[split:], m["target_pm25"].iloc[split:], m["time"].iloc[split:]
            pred = model.predict(Xte)
            base = Xte["pm2_5"].values
            mae_m = np.mean(np.abs(yte.values-pred)); mae_b = np.mean(np.abs(yte.values-base))
            rmse_m = np.sqrt(np.mean((yte.values-pred)**2))
            from sklearn.metrics import r2_score
            r2 = r2_score(yte, pred)
            c = st.columns(4)
            with c[0]: metric_card("Test MAE", f"{mae_m:.2f}", "\u00B5g/m\u00B3 average error", "#0ea5a5")
            with c[1]: metric_card("Test RMSE", f"{rmse_m:.2f}", "\u00B5g/m\u00B3", "#0ea5a5")
            with c[2]: metric_card("R\u00B2", f"{r2:.3f}", "variation explained", "#0ea5a5")
            with c[3]: metric_card("vs. baseline", f"+{(mae_b-mae_m)/mae_b*100:.1f}%", "better than na\u00EFve guess", "#0891b2")

            st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
            chart_head("Day-ahead forecast vs actual \u2014 unseen test period",
                       "Orange = AirCast\u2019s forecast \u00B7 grey = actual \u00B7 shaded bands = AQI health levels.")
            t = pd.to_datetime(tte.values)
            fig = go.Figure()
            aqi_bands(fig, float(np.nanmax(yte.values)))
            fig.add_trace(go.Scatter(x=t, y=yte.values, mode="lines", name="Actual",
                          line=dict(color="#94a3b8", width=1)))
            fig.add_trace(go.Scatter(x=t, y=pred, mode="lines", name="Predicted (24h ahead)",
                          line=dict(color="#f97316", width=1.4)))
            fig.add_hline(y=WHO_GUIDELINE, line=dict(color="#16a34a", dash="dash"))
            fig.update_layout(height=380, yaxis_title="PM2.5 (\u00B5g/m\u00B3)", **BASE_LAYOUT)
            fig.update_layout(showlegend=True, legend=dict(orientation="h", x=0, y=1.08,
                              yanchor="bottom", bgcolor="rgba(0,0,0,0)"), margin=dict(l=10, r=10, t=34, b=28))
            style_axes(fig, spikes=True)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

            chart_head("What drives the forecast \u2014 top features",
                       "Recent pollution plus tomorrow\u2019s wind & humidity lead \u2014 matching the physics.")
            imp = pd.Series(model.feature_importances_, index=feat).sort_values(ascending=False).head(10)[::-1]
            cols = ["#0ea5a5"]*len(imp); cols[-1] = "#ef4444"
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                          marker=dict(color=cols, cornerradius=6, line_width=0),
                          hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            fig.update_layout(height=360, **BASE_LAYOUT)
            style_axes(fig)
            fig.update_yaxes(tickfont=dict(color="#334155"))
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

# ===========================================================================
# TAB 2 — Live Forecast
# ===========================================================================
with tab_live:
    section("Live", "The next 24 hours in Islamabad")
    if bundle is None:
        st.markdown(live_fx(30, False, False, 12), unsafe_allow_html=True)
        st.warning("`models/rf_pm25.pkl` not found. Add it to the repo to enable the live forecast.")
    else:
        model, feat = bundle["model"], bundle["features"]
        if st.button("\U0001F504  Fetch latest data & forecast", type="primary"):
            fetch_live.clear()
        try:
            with st.spinner("Pulling live Islamabad data from Open-Meteo\u2026"):
                aq_df, wx_df = fetch_live()
                live = build_live_features(aq_df, wx_df)
                now_t, fc = predict_next_24h(model, feat, live)
            if fc is None or fc.empty:
                st.markdown(live_fx(30, False, False, 12), unsafe_allow_html=True)
                st.error("Could not build a forecast from the latest data. Try again shortly.")
            else:
                cur = live.loc[live["time"] == now_t, "pm2_5"]
                cur_val = float(cur.iloc[0]) if not cur.empty else float("nan")
                peak = fc.loc[fc["predicted_pm25"].idxmax()]
                pcat, pcol, advice = pm25_category(peak["predicted_pm25"])
                fut = live[(live["time"] > now_t) & (live["time"] <= now_t + pd.Timedelta(hours=HORIZON))]
                wind_avg = fut["wind_speed_10m"].mean(); rain_sum = fut["precipitation"].sum()

                precip_12h = fut["precipitation"].head(12).sum()
                is_rain = precip_12h >= 0.5
                wind_high = wind_avg >= 15
                st.markdown(live_fx(float(peak["predicted_pm25"]), is_rain, wind_high, now_t.hour),
                            unsafe_allow_html=True)

                st.markdown(
                    f"<div class='glass aqi'>"
                    f"<div class='ring' style='background:linear-gradient(135deg,{pcol},{pcol}cc)'>"
                    f"<div class='v'>{peak['predicted_pm25']:.0f}</div><div class='u'>\u00B5g/m\u00B3 peak</div></div>"
                    f"<div class='txt'><span class='badge' style='background:{pcol}'>{pcat}</span>"
                    f"<h3>Predicted peak in the next 24 hours</h3>"
                    f"<p>{advice}</p></div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='cap'>Latest observed data: {now_t:%Y-%m-%d %H:%M} PKT \u00B7 forecast horizon: next {HORIZON} h</div>", unsafe_allow_html=True)

                st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
                ncat, ncol, _ = pm25_category(cur_val)
                c = st.columns(3)
                with c[0]: metric_card("Now (latest observed)", f"{cur_val:.0f}", ncat, ncol)
                with c[1]: metric_card("Forecast wind", f"{wind_avg:.1f}", "km/h avg \u2014 clears the air", "#0ea5a5")
                with c[2]: metric_card("Forecast rain", f"{rain_sum:.1f}", "mm total \u2014 washes out PM", "#0891b2")

                chart_head("24-hour PM2.5 forecast",
                           "Shaded bands = AQI health levels \u00B7 uses Open-Meteo\u2019s weather forecast for the target hours.")
                fcx = pd.to_datetime(fc["forecast_time"])
                fig = go.Figure()
                aqi_bands(fig, float(fc["predicted_pm25"].max())*1.15)
                fig.add_trace(go.Scatter(x=fcx, y=fc["predicted_pm25"], mode="lines+markers",
                              name="Predicted PM2.5", line=dict(color="#0ea5a5", width=3, shape="spline"),
                              fill="tozeroy",
                              fillgradient=dict(type="vertical", colorscale=[[0,"rgba(14,165,165,0)"],[1,"rgba(14,165,165,.42)"]]),
                              marker=dict(size=7, color="#0ea5a5", line=dict(color="white", width=1.5)),
                              hovertemplate="%{x|%a %H:%M}: %{y:.0f} \u00B5g/m\u00B3<extra></extra>"))
                fig.add_hline(y=WHO_GUIDELINE, line=dict(color="#16a34a", dash="dash"),
                              annotation_text="WHO 15", annotation_position="top left", annotation_font_color="#16a34a")
                fig.update_layout(height=380, yaxis_title="PM2.5 (\u00B5g/m\u00B3)", **BASE_LAYOUT)
                style_axes(fig, spikes=True)
                st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
        except requests.RequestException as e:
            st.markdown(live_fx(30, False, False, 12), unsafe_allow_html=True)
            st.error("Couldn't reach Open-Meteo after 3 tries. This is a connection issue, not the app.")
            st.caption(f"Technical detail: {type(e).__name__} \u2014 {e}")
            st.info("Try again in a moment. If it keeps failing, your network may be blocking the API "
                    "(common on university / public Wi-Fi) \u2014 a phone hotspot usually fixes it.")
        except Exception as e:
            st.markdown(live_fx(30, False, False, 12), unsafe_allow_html=True)
            st.error("Something went wrong building the live forecast.")
            st.caption(f"Technical detail: {type(e).__name__} \u2014 {e}")

st.markdown(
    "<div style='margin-top:2rem;text-align:center;color:#57708a;font-size:.8rem'>"
    "AirCast \u00B7 Data \u00A9 Copernicus CAMS & ECMWF ERA5 via Open-Meteo (CC BY 4.0) \u00B7 "
    "Built by Muhammad Danyal Ikram</div>", unsafe_allow_html=True)
