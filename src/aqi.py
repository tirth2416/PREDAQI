"""
PREDAQI - Indian National Air Quality Index (CPCB) calculation.

How CPCB calculates AQI:
1. Each pollutant is averaged over a fixed period:
      PM2.5, PM10, NO2, SO2 -> 24-hour average
      CO, O3                -> 8-hour average
2. Each average is turned into a "sub-index" (0-500) using the
   official breakpoint table below.
3. The AQI is the HIGHEST sub-index. The pollutant with the highest
   sub-index is the "main (prominent) pollutant".
4. An AQI is only valid if at least 3 pollutants are available and
   one of them is PM2.5 or PM10.
"""

import numpy as np
import pandas as pd

# Concentration breakpoints for each AQI band.
# AQI band:      0    50   100   200   300   400   500
BREAKPOINTS = {
    "pm25": [0, 30, 60, 90, 120, 250, 380],
    "pm10": [0, 50, 100, 250, 350, 430, 510],
    "no2": [0, 40, 80, 180, 280, 400, 520],
    "so2": [0, 40, 80, 380, 800, 1600, 2100],
    "co": [0, 1.0, 2.0, 10, 17, 34, 51],      # mg/m3
    "o3": [0, 50, 100, 168, 208, 748, 1000],
}
AQI_LEVELS = [0, 50, 100, 200, 300, 400, 500]

# Averaging period in hours for each pollutant
AVERAGING_HOURS = {
    "pm25": 24,
    "pm10": 24,
    "no2": 24,
    "so2": 24,
    "co": 8,
    "o3": 8,
}

CATEGORIES = [
    (50, "Good", "#2e9e5b"),
    (100, "Satisfactory", "#8cc152"),
    (200, "Moderate", "#e6b800"),
    (300, "Poor", "#f08a24"),
    (400, "Very Poor", "#d9342b"),
    (500, "Severe", "#8b1a1a"),
]


def sub_index(pollutant, concentration):
    """Convert a concentration into a sub-index (0-500)."""
    values = np.asarray(concentration, dtype=float)
    result = np.interp(values, BREAKPOINTS[pollutant], AQI_LEVELS)
    return np.where(np.isnan(values), np.nan, result)


def category(aqi):
    """Return the AQI category name, e.g. 'Moderate'."""
    if pd.isna(aqi):
        return "Not available"
    for upper, name, _ in CATEGORIES:
        if aqi <= upper:
            return name
    return "Severe"


def category_color(name):
    for _, cat_name, color in CATEGORIES:
        if cat_name == name:
            return color
    return "#888888"


def averaged(series, pollutant):
    """CPCB averaging: 24-hour or 8-hour rolling mean of hourly values."""
    hours = AVERAGING_HOURS[pollutant]
    # CPCB needs at least ~2/3 of the hours to be present
    minimum = 16 if hours == 24 else 6
    return series.rolling(hours, min_periods=minimum).mean()


def aqi_table(hourly):
    """
    hourly: DataFrame indexed by hour, one column per pollutant
            (hourly concentrations, may contain gaps).

    Returns a DataFrame with, for every hour:
      <pollutant>_avg  the averaged concentration
      <pollutant>_si   the sub-index
      aqi, main_pollutant, category
    """
    out = pd.DataFrame(index=hourly.index)
    si_columns = []

    for pollutant in BREAKPOINTS:
        if pollutant not in hourly.columns:
            continue
        avg = averaged(hourly[pollutant], pollutant)
        out[f"{pollutant}_avg"] = avg
        out[f"{pollutant}_si"] = sub_index(pollutant, avg)
        si_columns.append(f"{pollutant}_si")

    if not si_columns:
        out["aqi"] = np.nan
        out["main_pollutant"] = None
        out["category"] = "Not available"
        return out

    sub_indices = out[si_columns]

    # Validity rule: >= 3 pollutants and at least one PM pollutant
    count = sub_indices.notna().sum(axis=1)
    pm_columns = [c for c in ["pm25_si", "pm10_si"] if c in si_columns]
    has_pm = out[pm_columns].notna().any(axis=1) if pm_columns else False
    valid = (count >= 3) & has_pm

    out["aqi"] = sub_indices.max(axis=1).where(valid).round()
    main = sub_indices.fillna(-1).idxmax(axis=1).str.replace("_si", "")
    out["main_pollutant"] = main.where(valid)
    out["category"] = out["aqi"].apply(category)
    return out
