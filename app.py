"""
PREDAQI dashboard.

Run locally:   streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from aqi import CATEGORIES, category_color  # noqa: E402
from config import CITIES, LABELS, RESULTS_DIR, UNITS  # noqa: E402
from forecast import (  # noqa: E402
    current_aqi, data_period, explain_change, forecast_aqi, load_city, valid_start_times,
)

st.set_page_config(page_title="PREDAQI", page_icon="🌫️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1300px;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 16px;}
.card .label {font-size: 13px; opacity: .7;}
.card .value {font-size: 30px; font-weight: 700; line-height: 1.2;}
.card .sub {font-size: 13px; font-weight: 600;}
.note {font-size: 13px; opacity: .75;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers

COLORS = {"pm25": "#2563eb", "pm10": "#d97706", "no2": "#dc2626",
          "so2": "#7c3aed", "co": "#475569", "o3": "#059669"}


def color(pollutant):
    return COLORS.get(pollutant, "#888888")


def name(pollutant):
    return LABELS.get(pollutant, str(pollutant))


def card(label, value, sub="", color=None):
    style = f"color:{color};" if color else ""
    st.markdown(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub" style="{style}">{sub}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def read_result(file):
    path = RESULTS_DIR / file
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def available_cities():
    out = []
    for city in CITIES:
        try:
            if len(load_city(city).columns):
                out.append(city)
        except FileNotFoundError:
            pass
    return out


def add_category_bands(fig, y_max):
    lower = 0
    for upper, cat_name, color in CATEGORIES:
        if lower > y_max:
            break
        fig.add_hrect(y0=lower, y1=upper, fillcolor=color, opacity=0.08, line_width=0,
                      annotation_text=cat_name, annotation_position="top left",
                      annotation_font_size=10, annotation_opacity=0.6)
        lower = upper


def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified")
    return fig


cities = available_cities()

st.title("PREDAQI")
st.caption("48-hour Air Quality Index forecasting for Indian cities · "
           "Linear Regression (gradient descent), Neural Network and PCA")

tab_demo, tab_acc, tab_model, tab_study, tab_about = st.tabs(
    ["Forecast demo", "Model accuracy", "How the model decides", "Pollutant study", "About"]
)

# =================================================================== DEMO
with tab_demo:
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 0.8, 1.4])
    city = c1.selectbox("City", cities)
    starts = valid_start_times(city)
    first, last = starts.min(), starts.max()

    # Default: the day before Diwali 2024 (a big pollution event)
    default = pd.Timestamp("2024-10-30 09:00")
    if not (first <= default <= last):
        default = last - pd.Timedelta(days=3)
    day = c2.date_input("Forecast start date", value=default.date(),
                        min_value=first.date(), max_value=last.date())
    hour = c3.selectbox("Hour", list(range(24)), index=9, format_func=lambda h: f"{h:02d}:00")
    model_choice = c4.selectbox(
        "Model", ["best", "sgd", "mlp"],
        format_func={"best": "Best per pollutant (validation)",
                     "sgd": "Linear Regression (SGD)",
                     "mlp": "Neural Network (MLP)"}.get,
    )

    start = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    st.markdown(
        f'<div class="note">The model only uses readings up to <b>{start:%d %b %Y, %H:%M}</b> '
        f"and forecasts the next 48 hours. Data available from {first:%d %b %Y} to "
        f"{last:%d %b %Y}. Where real readings exist, they are shown for comparison.</div>",
        unsafe_allow_html=True,
    )

    period = data_period(city, start)
    period_text = {
        "training": "⚠️ This date is in the <b>training</b> period, so the model has seen it before.",
        "gap": "This date is in the 48-hour gap between training and validation data.",
        "validation": "This date is in the <b>validation</b> period: the model was not trained on it.",
        "test": "✅ This date is in the <b>test</b> period: the model never saw it.",
    }[period]
    st.markdown(f'<div class="note">{period_text}</div>', unsafe_allow_html=True)

    if start not in starts:
        st.warning("Some sensor readings are missing just before this time, so a forecast "
                   "can't be made. Please pick another hour or date.")
        st.stop()

    result, used = forecast_aqi(city, start, model_choice)
    now = current_aqi(city, start)
    pollutants = list(load_city(city).columns)
    has_actual = result["aqi_actual"].notna().any()

    if result["aqi_pred"].isna().all():
        st.error("This city does not measure enough pollutants for an official AQI "
                 "(CPCB needs at least 3, including PM2.5 or PM10).")
        st.stop()

    peak = result.loc[result["aqi_pred"].idxmax()]
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        cat = now["category"]
        card("AQI now (measured)", f"{now['aqi']:.0f}" if pd.notna(now["aqi"]) else "–",
             f"{cat} · {name(now['main_pollutant'])}", category_color(cat))
    for col, h in [(k2, 24), (k3, 48)]:
        row = result.iloc[h - 1]
        with col:
            card(f"Forecast in {h} hours", f"{row['aqi_pred']:.0f}",
                 f"{row['category_pred']} · {name(row['main_pollutant_pred'])}",
                 category_color(row["category_pred"]))
    with k4:
        card("Highest forecast AQI", f"{peak['aqi_pred']:.0f}",
             f"{peak.name:%d %b %H:%M} · {name(peak['main_pollutant_pred'])}",
             category_color(peak["category_pred"]))

    # --- AQI chart
    fig = go.Figure()
    y_max = max(result["aqi_pred"].max(), result["aqi_actual"].max(skipna=True) or 0) * 1.15
    add_category_bands(fig, y_max)
    fig.add_trace(go.Scatter(x=result.index, y=result["aqi_pred"], name="Forecast AQI",
                             line=dict(width=3, color="#2563eb")))
    if has_actual:
        fig.add_trace(go.Scatter(x=result.index, y=result["aqi_actual"], name="Actual AQI",
                                 line=dict(width=2, dash="dot", color="#111827")))
    fig.update_yaxes(range=[0, y_max], title="AQI")
    fig.update_layout(title="AQI over the next 48 hours")
    st.plotly_chart(style(fig), use_container_width=True)

    if has_actual:
        err = (result["aqi_pred"] - result["aqi_actual"]).abs().mean()
        cat_ok = (result["category_pred"] == result["category_actual"]).mean() * 100
        st.markdown(f"**For this forecast:** average AQI error **{err:.1f} points**, "
                    f"correct AQI category in **{cat_ok:.0f}%** of the 48 hours.")
    else:
        st.info("These hours are after the last reading in our data, so this is a true "
                "forecast with nothing to compare against.")

    # --- Why
    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("What drives the change?")
        drivers, _ = explain_change(city, start, result)
        top = drivers.iloc[0]
        direction = "rise" if peak["aqi_pred"] > now["aqi"] else "stay at or below its current level"
        st.markdown(
            f"The AQI is forecast to **{direction}** (now {now['aqi']:.0f}, peak "
            f"{peak['aqi_pred']:.0f}). The AQI is set by the pollutant with the highest "
            f"sub-index: at the peak that is **{name(peak['main_pollutant_pred'])}**. "
            f"The biggest sub-index increase comes from **{name(top['pollutant'])}** "
            f"({top['change']:+.0f} points)."
        )
        show = drivers.copy()
        show["pollutant"] = show["pollutant"].map(name)
        show.columns = ["Pollutant", "Sub-index now", "Sub-index at peak", "Change"]
        st.dataframe(show.round(0), hide_index=True, use_container_width=True)
        st.markdown('<div class="note">Models used: ' +
                    ", ".join(f"{name(p)} → {m}" for p, m in used.items()) + "</div>",
                    unsafe_allow_html=True)

    with right:
        st.subheader("Sub-index of each pollutant")
        fig = go.Figure()
        for p in pollutants:
            fig.add_trace(go.Scatter(x=result.index, y=result[f"{p}_si_pred"], name=name(p),
                                     line=dict(color=color(p))))
        fig.update_yaxes(title="Sub-index (AQI scale)")
        st.plotly_chart(style(fig, 330), use_container_width=True)

    # --- Breakdown table
    st.subheader("Hour-by-hour breakdown")
    st.markdown('<div class="note">Forecast concentration (actual in brackets). '
                "The AQI is the highest pollutant sub-index, as per CPCB.</div>",
                unsafe_allow_html=True)
    table = pd.DataFrame({"Time": result.index.strftime("%d %b %H:%M"),
                          "Hours ahead": result["hours_ahead"]})
    for p in pollutants:
        pred, act = result[f"{p}_pred"], result[f"{p}_actual"]
        decimals = 2 if p == "co" else 1
        table[f"{name(p)} ({UNITS[p]})"] = [
            f"{a:.{decimals}f}" + (f" ({b:.{decimals}f})" if pd.notna(b) else "")
            for a, b in zip(pred, act)
        ]
    table["AQI forecast"] = result["aqi_pred"].values
    table["Category"] = result["category_pred"].values
    table["Main pollutant"] = result["main_pollutant_pred"].map(name).values
    if has_actual:
        table["AQI actual"] = result["aqi_actual"].values
        table["Actual category"] = result["category_actual"].values
    st.dataframe(table, hide_index=True, use_container_width=True, height=420)
    st.download_button("Download this forecast (CSV)", result.to_csv().encode(),
                       f"predaqi_{city.lower()}_{start:%Y%m%d_%H}.csv", "text/csv")

# =============================================================== ACCURACY
with tab_acc:
    aqi_summary = read_result("aqi_summary.csv")
    summary = read_result("model_summary.csv")
    by_hour = read_result("model_accuracy_by_hour.csv")

    st.subheader("Final AQI forecast accuracy (test period)")
    st.markdown('<div class="note">Test period = the last 15% of each city\'s data, never '
                "used for training. A forecast was started every 6 hours and compared "
                "with the real AQI. Baseline = \"AQI stays as it is now\".</div>",
                unsafe_allow_html=True)
    if not aqi_summary.empty:
        t = aqi_summary.copy()
        t["improvement"] = (1 - t["aqi_mae"] / t["aqi_mae_persistence"]) * 100
        t = t[["city", "forecasts_checked", "average_actual_aqi", "aqi_mae",
               "aqi_mae_persistence", "improvement", "category_accuracy",
               "category_accuracy_persistence", "main_pollutant_accuracy"]]
        for c in ["category_accuracy", "category_accuracy_persistence", "main_pollutant_accuracy"]:
            t[c] = t[c] * 100
        t.columns = ["City", "Forecasts checked", "Average AQI", "AQI error (MAE)",
                     "Baseline error", "Better than baseline (%)", "Category correct (%)",
                     "Baseline category correct (%)", "Main pollutant correct (%)"]
        st.dataframe(t.round(1), hide_index=True, use_container_width=True)

        aqi_hour = read_result("aqi_accuracy_by_hour.csv")
        city_a = st.selectbox("City", aqi_hour["city"].unique(), key="acc_city")
        d = aqi_hour[aqi_hour["city"] == city_a]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["hours_ahead"], y=d["aqi_mae"], name="PREDAQI forecast"))
        fig.add_trace(go.Scatter(x=d["hours_ahead"], y=d["aqi_mae_persistence"],
                                 name="Baseline (no change)", line=dict(dash="dot")))
        fig.update_xaxes(title="Hours ahead")
        fig.update_yaxes(title="AQI error (MAE)", rangemode="tozero")
        fig.update_layout(title=f"{city_a}: AQI error grows the further ahead we forecast")
        st.plotly_chart(style(fig), use_container_width=True)
    else:
        st.info("AQI accuracy results not found. Run src/evaluate_aqi.py.")

    st.subheader("Each pollutant: Linear Regression vs Neural Network vs baseline")
    if not summary.empty:
        piv = summary.pivot_table(index=["city", "pollutant"], columns="model",
                                  values="test_mae").reset_index()
        piv["pollutant"] = piv["pollutant"].map(name)
        base = "Persistence baseline"
        for m in ["Linear Regression (SGD)", "Neural Network (MLP)"]:
            piv[f"{m.split(' (')[0]} vs baseline (%)"] = (1 - piv[m] / piv[base]) * 100
        st.markdown('<div class="note">Average error (MAE) over all 48 forecast hours, in each '
                    "pollutant's own unit. Positive % = better than the baseline.</div>",
                    unsafe_allow_html=True)
        st.dataframe(piv.round(2), hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        city_p = c1.selectbox("City", summary["city"].unique(), key="p_city")
        pol_p = c2.selectbox("Pollutant", summary[summary["city"] == city_p]["pollutant"].unique(),
                             format_func=name, key="p_pol")
        d = by_hour[(by_hour["city"] == city_p) & (by_hour["pollutant"] == pol_p)]
        fig = go.Figure()
        for m in d["model"].unique():
            dm = d[d["model"] == m]
            fig.add_trace(go.Scatter(x=dm["hours_ahead"], y=dm["mae"], name=m,
                                     line=dict(dash="dot") if "Persistence" in m else None))
        fig.update_xaxes(title="Hours ahead")
        fig.update_yaxes(title=f"MAE ({UNITS[pol_p]})", rangemode="tozero")
        st.plotly_chart(style(fig), use_container_width=True)

        st.subheader("Did PCA help?")
        p = summary[summary["model"] != base][["city", "pollutant", "model", "pca", "val_mae", "test_mae"]].copy()
        p["pollutant"] = p["pollutant"].map(name)
        p.columns = ["City", "Pollutant", "Model", "Chosen on validation", "Validation MAE", "Test MAE"]
        st.markdown('<div class="note">Each model was trained with and without PCA (keeping 95% '
                    "of the variance). The version with the lower validation error was kept.</div>",
                    unsafe_allow_html=True)
        st.dataframe(p.round(2), hide_index=True, use_container_width=True)

# ============================================================ HOW IT DECIDES
with tab_model:
    weights = read_result("model_weights.csv")
    st.markdown(
        "Each pollutant has a **Linear Regression** model trained with gradient descent. "
        "It predicts the **% change** from the current value, as a weighted sum of the inputs. "
        "Inputs are standardised, so a bigger bar means a bigger influence. A **positive** "
        "weight pushes the forecast **up**, a **negative** weight pushes it **down**."
    )
    if not weights.empty:
        c1, c2, c3 = st.columns(3)
        city_w = c1.selectbox("City", weights["city"].unique(), key="w_city")
        pol_w = c2.selectbox("Pollutant", weights[weights["city"] == city_w]["pollutant"].unique(),
                             format_func=name, key="w_pol")
        h_w = c3.selectbox("Hours ahead", sorted(weights["hours_ahead"].unique()), index=2, key="w_h")
        d = weights[(weights["city"] == city_w) & (weights["pollutant"] == pol_w)
                    & (weights["hours_ahead"] == h_w)].copy()
        d = d.reindex(d["weight"].abs().sort_values().index)

        def describe(f):
            if f.startswith("lag_"):
                return f"{name(pol_w)} {f[4:-1]} h ago (vs now)"
            if f == "avg_6h":
                return "6-hour average (vs now)"
            if f == "avg_24h":
                return "24-hour average (vs now)"
            if f.endswith("_vs_avg"):
                return f"{name(f.replace('_vs_avg', ''))} above its daily average"
            return {"hour_sin": "Time of day (sin)", "hour_cos": "Time of day (cos)",
                    "weekend": "Weekend"}.get(f, f)

        d["label"] = d["feature"].map(describe)
        fig = go.Figure(go.Bar(x=d["weight"], y=d["label"], orientation="h",
                               marker_color=["#dc2626" if w > 0 else "#2563eb" for w in d["weight"]]))
        fig.update_layout(title=f"{city_w} · {name(pol_w)} · {h_w} hours ahead",
                          height=max(320, 26 * len(d)), margin=dict(l=10, r=10, t=40, b=10))
        fig.update_xaxes(title="Weight (red = pushes forecast up, blue = down)")
        st.plotly_chart(fig, use_container_width=True)

        top = d.iloc[-1]
        st.markdown(
            f"**Reading this chart:** the most important input is *{top['label']}*. "
            "For example, a strongly positive weight on the 24-hour average means: if the "
            "current value is well below the day's average, the model expects it to climb "
            "back up (pollution tends to return to its normal level). Time-of-day inputs "
            "let it learn daily cycles such as night-time build-up and rush hours."
        )
    st.markdown(
        "**How the AQI forecast is made:** (1) each pollutant is forecast for 48 hours; "
        "(2) the forecasts are averaged the CPCB way (24-hour average for PM2.5, PM10, NO₂, "
        "SO₂; 8-hour for CO and O₃); (3) each average becomes a sub-index; (4) the **AQI is "
        "the highest sub-index**, and that pollutant is the **main pollutant**. So the model "
        "decides the AQI will rise when the forecast of the main pollutant rises, or when "
        "another pollutant's sub-index climbs above it."
    )

# ================================================================== STUDY
with tab_study:
    S = "study/"
    dom = read_result(S + "dominant_pollutant.csv")
    rise = read_result(S + "aqi_rise_drivers.csv")
    week = read_result(S + "weekday_vs_weekend.csv")
    prof = read_result(S + "hourly_profile.csv")
    month = read_result(S + "monthly.csv")
    hol = read_result(S + "holidays.csv")
    st.markdown('<div class="note">All results on this page come from the real measured data. '
                "They show patterns that occur together, not proof of cause.</div>",
                unsafe_allow_html=True)

    def stacked(df, value, title):
        fig = go.Figure()
        for p in df["pollutant"].unique():
            dp = df[df["pollutant"] == p]
            fig.add_trace(go.Bar(x=dp["city"], y=dp[value], name=name(p), marker_color=color(p)))
        fig.update_layout(barmode="stack", title=title)
        fig.update_yaxes(title="%", range=[0, 100])
        return style(fig, 360)

    c1, c2 = st.columns(2)
    if not dom.empty:
        c1.plotly_chart(stacked(dom, "percent_of_hours",
                                "1. Which pollutant sets the AQI (% of hours)"),
                        use_container_width=True)
    if not rise.empty:
        c2.plotly_chart(stacked(rise, "percent_of_rising_days",
                                "2. Who caused the rise on days the AQI went up 10+"),
                        use_container_width=True)

    if not week.empty:
        st.subheader("3. Traffic signal: weekday vs weekend")
        st.markdown('<div class="note">NO₂ and CO come mostly from vehicles. If they fall on '
                    "weekends, traffic is a likely contributor.</div>", unsafe_allow_html=True)
        fig = go.Figure()
        for p in week["pollutant"].unique():
            dp = week[week["pollutant"] == p]
            fig.add_trace(go.Bar(x=dp["city"], y=dp["weekend_change_percent"], name=name(p),
                                 marker_color=color(p)))
        fig.update_yaxes(title="Weekend vs weekday (%)")
        st.plotly_chart(style(fig, 340), use_container_width=True)

    if not prof.empty:
        st.subheader("4. Daily cycle")
        c1, c2 = st.columns([1, 3])
        city_s = c1.selectbox("City", prof["city"].unique(), key="s_city")
        c1.markdown('<div class="note">100 = the pollutant\'s daily average. Peaks around '
                    "8–10 am and 7–10 pm point to rush-hour traffic; O₃ peaks in the afternoon "
                    "because sunlight creates it.</div>", unsafe_allow_html=True)
        d = prof[prof["city"] == city_s]
        fig = go.Figure()
        for p in d["pollutant"].unique():
            dp = d[d["pollutant"] == p]
            fig.add_trace(go.Scatter(x=dp["hour"], y=dp["percent_of_daily_avg"], name=name(p),
                                     line=dict(color=color(p))))
        fig.update_xaxes(title="Hour of day", dtick=2)
        fig.update_yaxes(title="% of daily average")
        c2.plotly_chart(style(fig, 340), use_container_width=True)

    if not month.empty:
        st.subheader("5. Month by month")
        fig = go.Figure()
        for c in month["city"].unique():
            dc = month[month["city"] == c]
            fig.add_trace(go.Scatter(x=dc["month"], y=dc["aqi"] if dc["aqi"].notna().any() else dc["pm25"],
                                     name=c, mode="lines+markers"))
        fig.update_yaxes(title="Monthly average AQI")
        st.plotly_chart(style(fig, 340), use_container_width=True)
        st.markdown('<div class="note">Pollution falls in the monsoon (June to September), when '
                    "rain washes particles out of the air, and rises sharply from October to "
                    "December, when cold, still air traps pollution near the ground.</div>",
                    unsafe_allow_html=True)

    if not hol.empty:
        st.subheader("6. Festivals and holidays")
        h = hol.copy()
        h["pollutant"] = h["pollutant"].map(name)
        wide = h.pivot_table(index=["city", "date", "holiday"], columns="pollutant",
                             values="change_percent").reset_index()
        st.markdown('<div class="note">% change on the holiday compared with normal days in '
                    "the week before and after. Positive = more pollution on the holiday.</div>",
                    unsafe_allow_html=True)
        wide = wide.rename(columns={"city": "City", "date": "Date", "holiday": "Festival / holiday"})
        st.dataframe(wide.round(1), hide_index=True, use_container_width=True)

# ================================================================== ABOUT
with tab_about:
    st.markdown(f"""
**Data.** Hourly readings from CPCB monitoring stations, via the India Air Quality Database
(airquality.xkdr.org), for {", ".join(cities)}, for the full year 2024. We use 2024 because
the database has NO₂, SO₂, CO and Ozone only up to 31 Dec 2024, and all six pollutants are
needed for the official AQI. All stations in a city are combined using the **median**, so one
faulty sensor can't distort the city value. Gaps of up to 3 hours are filled by interpolation;
longer gaps are left empty and never used for training.

**Models (from the FOML syllabus).**
- **Linear Regression trained with gradient descent** (`SGDRegressor`).
- **Feedforward Neural Network** (`MLPRegressor`, 64 → 32 neurons, trained by backpropagation).
- **PCA**: each model is tried with and without PCA (95% variance); the validation set decides.
- **Persistence baseline**: "the future equals now". A model is only useful if it beats this.

One model per pollutant per city predicts all 48 hours at once. Models predict the
**% change** from the current value, because pollution levels shift strongly between seasons
(winter vs monsoon), while hourly patterns stay similar.

**Evaluation.** Data is split by time: 70% training, 15% validation, 15% test, with 48-hour
gaps between them. The test set is used only once, for the final numbers.

**AQI.** Calculated with the official CPCB National AQI method: 24-hour averages
(8-hour for CO and O₃), breakpoint tables to sub-indices, AQI = highest sub-index. Valid only
with at least 3 pollutants, including PM2.5 or PM10.

**Limits.** Festival spikes such as Diwali fireworks are hard to forecast: the model has no
input that says "festival tomorrow", and one year of data has only one Diwali. The data source has complete pollutant data only up to 31 Dec 2024, so the demo
forecasts dates in 2024 (and up to 48 hours beyond the last reading). No weather data is used, so sudden changes from
rain or wind are hard to predict. Study results show patterns, not proven causes.
""")
