"""
AirCast — Islamabad PM2.5 Day-Ahead Forecaster
================================================
A Streamlit dashboard with two tabs:
  1) Explore & Results — historical air-quality patterns for Islamabad plus the
     trained model's honest performance on an unseen (time-separated) test set.
  2) Live Forecast     — pulls the LATEST Islamabad data from Open-Meteo (no API
     key) and predicts PM2.5 for the next 24 hours on demand.

Data : Copernicus CAMS (air quality) + ERA5 / Open-Meteo (weather) via Open-Meteo.
Model: scikit-learn RandomForestRegressor trained in the accompanying notebook.
Author: Muhammad Danyal Ikram

NOTE (state this openly): PM2.5 here is *modelled* output from Copernicus CAMS,
not a physical ground sensor. For the live forecast, tomorrow's weather comes
from Open-Meteo's weather forecast — which is exactly how real operational
air-quality forecasts work.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AirCast — Islamabad PM2.5 Forecast",
                   page_icon="\U0001F32B\uFE0F", layout="wide")

LAT, LON = 33.6844, 73.0479           # Islamabad
TZ = "Asia/Karachi"
HORIZON = 24                          # hours ahead — MUST match training

MODEL_PATH    = Path("models/rf_pm25.pkl")
CLEAN_PATH    = Path("data/islamabad_clean.csv")
FEATURES_PATH = Path("data/islamabad_features_v2.csv")

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                "surface_pressure", "wind_speed_10m", "wind_direction_10m",
                "precipitation"]

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"

WHO_GUIDELINE = 15.0                  # µg/m³ reference line

# ---------------------------------------------------------------------------
# Health category — US EPA PM2.5 breakpoints (µg/m³)
# ---------------------------------------------------------------------------
def pm25_category(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "Unknown", "#9e9e9e", "No data available."
    if v <= 12:    return "Good", "#2ecc71", "Air quality is satisfactory."
    if v <= 35.4:  return "Moderate", "#f1c40f", "Acceptable; unusually sensitive people should limit long outdoor exertion."
    if v <= 55.4:  return "Unhealthy for Sensitive Groups", "#e67e22", "Sensitive groups should reduce prolonged outdoor exertion."
    if v <= 150.4: return "Unhealthy", "#e74c3c", "Everyone may begin to feel effects; limit outdoor activity."
    if v <= 250.4: return "Very Unhealthy", "#8e44ad", "Health alert; avoid outdoor activity."
    return "Hazardous", "#7b241c", "Emergency conditions; stay indoors."


def badge(label, color):
    return (f"<span style='background:{color};color:white;padding:4px 12px;"
            f"border-radius:12px;font-weight:600;font-size:0.9rem'>{label}</span>")

# ---------------------------------------------------------------------------
# Cached loaders
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

# ---------------------------------------------------------------------------
# Live data fetch (no API key)
# ---------------------------------------------------------------------------
def _get(url, params):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=1800)  # refresh at most every 30 minutes
def fetch_live():
    aq = _get(AQ_URL, {
        "latitude": LAT, "longitude": LON,
        "hourly": "pm2_5,pm10,dust",
        "past_days": 5, "forecast_days": 1, "timezone": TZ,
    })
    aq_df = pd.DataFrame(aq["hourly"])
    aq_df["time"] = pd.to_datetime(aq_df["time"])

    wx = _get(WX_URL, {
        "latitude": LAT, "longitude": LON,
        "hourly": ",".join(WEATHER_VARS),
        "past_days": 5, "forecast_days": 2, "timezone": TZ,
    })
    wx_df = pd.DataFrame(wx["hourly"])
    wx_df["time"] = pd.to_datetime(wx_df["time"])
    return aq_df, wx_df

# ---------------------------------------------------------------------------
# Rebuild the EXACT training features on a live timeline
# ---------------------------------------------------------------------------
def build_live_features(aq_df, wx_df):
    # Weather is the spine (it extends into the future); air quality left-joined.
    df = pd.merge(wx_df, aq_df, on="time", how="left").sort_values("time").reset_index(drop=True)

    df["hour"]       = df["time"].dt.hour
    df["dayofweek"]  = df["time"].dt.dayofweek
    df["month"]      = df["time"].dt.month
    df["dayofyear"]  = df["time"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f"pm25_lag_{lag}"] = df["pm2_5"].shift(lag)
    df["pm25_roll_mean_3"]  = df["pm2_5"].rolling(3).mean()
    df["pm25_roll_mean_24"] = df["pm2_5"].rolling(24).mean()
    df["pm25_roll_std_24"]  = df["pm2_5"].rolling(24).std()

    for w in WEATHER_VARS:                       # forecast weather = weather at t+HORIZON
        df[f"fc_{w}"] = df[w].shift(-HORIZON)
    return df


def predict_next_24h(model, feat_names, df):
    obs = df.loc[df["pm2_5"].notna(), "time"]
    if obs.empty:
        return None, None
    now_t = obs.max()
    origins = df[(df["time"] >= now_t - pd.Timedelta(hours=HORIZON - 1)) &
                 (df["time"] <= now_t)].copy()
    X = origins[feat_names]
    ok = X.notna().all(axis=1)
    X, origins = X[ok], origins[ok]
    if X.empty:
        return now_t, None
    preds = model.predict(X)
    out = pd.DataFrame({
        "forecast_time": pd.to_datetime(origins["time"].values) + pd.Timedelta(hours=HORIZON),
        "predicted_pm25": np.round(preds, 1),
    }).reset_index(drop=True)
    return now_t, out

# ===========================================================================
# UI
# ===========================================================================
st.title("\U0001F32B\uFE0F AirCast — Islamabad PM2.5 Day-Ahead Forecaster")
st.caption("Forecasting fine-particle air pollution 24 hours ahead for Islamabad. "
           "Data: Copernicus CAMS + Open-Meteo (CC BY 4.0). PM2.5 is modelled, not sensor-measured.")

bundle   = load_model()
clean    = load_clean()
features = load_features()

tab_explore, tab_live = st.tabs(["\U0001F4CA  Explore & Results", "\U0001F52E  Live Forecast"])

# ---------------------------------------------------------------------------
# TAB 1 — Explore & Results
# ---------------------------------------------------------------------------
with tab_explore:
    if clean is None:
        st.warning("`data/islamabad_clean.csv` not found. Add it to the repo to see this tab.")
    else:
        st.subheader("How bad is Islamabad's air?")
        pm = clean["pm2_5"].dropna()
        pct_who = (pm > WHO_GUIDELINE).mean() * 100
        pct_un  = (pm > 55.4).mean() * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average PM2.5", f"{pm.mean():.1f} µg/m³", f"{pm.mean()/WHO_GUIDELINE:.1f}× WHO limit")
        c2.metric("Worst hour", f"{pm.max():.0f} µg/m³")
        c3.metric("Hours above WHO limit", f"{pct_who:.0f}%")
        c4.metric("Hours 'Unhealthy' (>55)", f"{pct_un:.0f}%")

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            st.markdown("**Average PM2.5 by month** — the winter smog season")
            monthly = clean.assign(month=clean["time"].dt.month).groupby("month")["pm2_5"].mean()
            monthly.index = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            st.bar_chart(monthly, color="#c0392b")
        with right:
            st.markdown("**Average PM2.5 by hour** — the daily rhythm")
            hourly = clean.assign(hour=clean["time"].dt.hour).groupby("hour")["pm2_5"].mean()
            st.line_chart(hourly, color="#2c3e50")

        st.markdown("---")
        st.subheader("Does the model actually work?")
        if bundle is None or features is None:
            st.info("Add `models/rf_pm25.pkl` and `data/islamabad_features_v2.csv` to see model results.")
        else:
            model, feat, split = bundle["model"], bundle["features"], bundle["split_index"]
            m = features.sort_values("time").reset_index(drop=True)
            Xte, yte, tte = m[feat].iloc[split:], m["target_pm25"].iloc[split:], m["time"].iloc[split:]
            pred = model.predict(Xte)
            base = Xte["pm2_5"].values
            mae_m  = np.mean(np.abs(yte.values - pred))
            mae_b  = np.mean(np.abs(yte.values - base))
            rmse_m = np.sqrt(np.mean((yte.values - pred) ** 2))
            from sklearn.metrics import r2_score
            r2 = r2_score(yte, pred)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Test MAE", f"{mae_m:.2f} µg/m³")
            k2.metric("Test RMSE", f"{rmse_m:.2f} µg/m³")
            k3.metric("R²", f"{r2:.3f}")
            k4.metric("vs. naïve baseline", f"{(mae_b - mae_m)/mae_b*100:.1f}% better")

            st.markdown("**Forecast vs actual — the unseen test period**")
            chart_df = pd.DataFrame({"Actual": yte.values, "Predicted (24h ahead)": pred},
                                    index=pd.to_datetime(tte.values))
            st.line_chart(chart_df, color=["#34495e", "#e67e22"])

            st.markdown("**What the model leans on — top features**")
            imp = (pd.Series(model.feature_importances_, index=feat)
                   .sort_values(ascending=False).head(12))
            st.bar_chart(imp, color="#2c3e50", horizontal=True)

# ---------------------------------------------------------------------------
# TAB 2 — Live Forecast
# ---------------------------------------------------------------------------
with tab_live:
    st.subheader("Next 24 hours — live forecast for Islamabad")
    if bundle is None:
        st.warning("`models/rf_pm25.pkl` not found. Add it to the repo to enable the live forecast.")
    else:
        model, feat = bundle["model"], bundle["features"]
        colb, _ = st.columns([1, 3])
        refresh = colb.button("\U0001F504 Fetch latest data & forecast", type="primary")
        if refresh:
            fetch_live.clear()  # bust the cache so the button truly refreshes

        try:
            with st.spinner("Pulling live Islamabad data from Open-Meteo…"):
                aq_df, wx_df = fetch_live()
                live = build_live_features(aq_df, wx_df)
                now_t, fc = predict_next_24h(model, feat, live)

            if fc is None or fc.empty:
                st.error("Could not build a forecast from the latest data. Try again shortly.")
            else:
                cur = live.loc[live["time"] == now_t, "pm2_5"]
                cur_val = float(cur.iloc[0]) if not cur.empty else np.nan
                cat_now, col_now, _ = pm25_category(cur_val)

                peak_row = fc.loc[fc["predicted_pm25"].idxmax()]
                cat_pk, col_pk, advice = pm25_category(peak_row["predicted_pm25"])

                # forecast-weather drivers over the horizon
                fut = live[(live["time"] > now_t) & (live["time"] <= now_t + pd.Timedelta(hours=HORIZON))]
                wind_avg = fut["wind_speed_10m"].mean()
                rain_sum = fut["precipitation"].sum()

                st.caption(f"Latest observed data: {now_t:%Y-%m-%d %H:%M} PKT  ·  "
                           f"forecast horizon: next {HORIZON} h")

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown("**Now (latest observed)**")
                    st.markdown(f"### {cur_val:.0f} µg/m³")
                    st.markdown(badge(cat_now, col_now), unsafe_allow_html=True)
                with m2:
                    st.markdown("**Predicted peak (next 24 h)**")
                    st.markdown(f"### {peak_row['predicted_pm25']:.0f} µg/m³")
                    st.markdown(badge(cat_pk, col_pk), unsafe_allow_html=True)
                    st.caption(f"around {pd.to_datetime(peak_row['forecast_time']):%a %H:%M}")
                with m3:
                    st.markdown("**Forecast drivers**")
                    st.markdown(f"Wind ≈ **{wind_avg:.1f}** km/h  \nRain ≈ **{rain_sum:.1f}** mm")
                    st.caption("Wind and rain clear the air — the model's key signals.")

                st.info(f"**Health guidance:** {advice}")

                fc_chart = fc.set_index("forecast_time").rename(
                    columns={"predicted_pm25": "Predicted PM2.5"})
                st.line_chart(fc_chart, color="#e67e22")
                st.caption("Dashed reference: WHO 24-h guideline is 15 µg/m³. "
                           "Forecast uses Open-Meteo's weather forecast for the target hours.")
        except requests.RequestException:
            st.error("Network error reaching Open-Meteo. Check your connection and try again.")

st.markdown("---")
st.caption("AirCast · Data © Copernicus CAMS & Open-Meteo (CC BY 4.0) · "
           "Model: RandomForest, day-ahead PM2.5 · Built by Muhammad Danyal Ikram")
