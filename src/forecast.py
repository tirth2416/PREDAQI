"""
PREDAQI - make a 48-hour AQI forecast for a city, starting at any hour.

Steps:
1. For each pollutant, the saved model predicts the next 48 hourly values.
2. The forecast hours are joined onto the real past readings, because
   CPCB's 24-hour and 8-hour averages need past hours too.
3. The official CPCB method (aqi.py) turns that into sub-indices,
   the AQI and the main pollutant for every forecast hour.
4. If real readings exist for those hours (they do for any date in
   our data), the actual AQI is calculated too, for comparison.

Used by the dashboard (app.py) and by evaluate_aqi.py.
"""

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from aqi import aqi_table
from config import HORIZON, MODEL_DIR, PROCESSED_DIR
from features import from_change, make_inputs


@lru_cache(maxsize=None)
def load_city(city):
    """Hourly table for a city, with only the pollutants that have models."""
    table = pd.read_csv(PROCESSED_DIR / f"{city.lower()}_hourly.csv",
                        index_col=0, parse_dates=True)
    modelled = [p for p in table.columns
                if (MODEL_DIR / f"{city.lower()}_{p}_sgd.joblib").exists()]
    return table[modelled]


@lru_cache(maxsize=None)
def load_model(city, pollutant, kind):
    return joblib.load(MODEL_DIR / f"{city.lower()}_{pollutant}_{kind}.joblib")


def best_kind(city, pollutant):
    """The model type (sgd or mlp) with the lower VALIDATION error."""
    sgd = load_model(city, pollutant, "sgd")
    mlp = load_model(city, pollutant, "mlp")
    return "sgd" if sgd["val_mae"] <= mlp["val_mae"] else "mlp"


@lru_cache(maxsize=None)
def city_inputs(city, pollutant):
    """Model inputs for every hour (cached, so the app stays fast)."""
    return make_inputs(load_city(city), pollutant)


def valid_start_times(city):
    """Hours at which every pollutant model has complete inputs."""
    table = load_city(city)
    ok = pd.Series(True, index=table.index)
    for pollutant in table.columns:
        ok &= city_inputs(city, pollutant).notna().all(axis=1)
        ok &= table[pollutant].notna()
    return table.index[ok]


def forecast_pollutants(city, start, model="best"):
    """
    Predicted concentrations for the 48 hours after `start`.
    model: "best", "sgd" or "mlp".
    Returns (DataFrame indexed by future hour, dict of model used per pollutant).
    """
    table = load_city(city)
    start = pd.Timestamp(start)
    future_hours = pd.date_range(start + pd.Timedelta(hours=1), periods=HORIZON, freq="h")

    predictions, used = {}, {}
    for pollutant in table.columns:
        kind = best_kind(city, pollutant) if model == "best" else model
        artifact = load_model(city, pollutant, kind)
        X = city_inputs(city, pollutant).loc[[start], artifact["features"]]
        if X.isna().any(axis=None):
            raise ValueError(f"Not enough past data at {start} for {pollutant}.")
        change = artifact["model"].predict(X)
        now = table.loc[[start], pollutant].values
        predictions[pollutant] = from_change(change, now)[0]
        used[pollutant] = artifact["model_name"]

    return pd.DataFrame(predictions, index=future_hours), used


def forecast_aqi(city, start, model="best"):
    """
    Full 48-hour forecast with the AQI breakdown.
    Returns a DataFrame (one row per forecast hour) and the models used.
    """
    table = load_city(city)
    start = pd.Timestamp(start)
    predicted, used = forecast_pollutants(city, start, model)

    # Past 24 hours of real readings + the 48 forecast hours
    history = table.loc[start - pd.Timedelta(hours=30):start]
    combined = pd.concat([history, predicted])
    predicted_aqi = aqi_table(combined).loc[predicted.index]

    # Actual readings and AQI for the same hours (if we have them)
    actual_aqi = aqi_table(table).reindex(predicted.index)
    actual = table.reindex(predicted.index)

    out = pd.DataFrame(index=predicted.index)
    out.index.name = "time"
    out["hours_ahead"] = np.arange(1, HORIZON + 1)
    for pollutant in table.columns:
        out[f"{pollutant}_pred"] = predicted[pollutant]
        out[f"{pollutant}_actual"] = actual[pollutant]
        out[f"{pollutant}_si_pred"] = predicted_aqi[f"{pollutant}_si"]
    out["aqi_pred"] = predicted_aqi["aqi"]
    out["category_pred"] = predicted_aqi["category"]
    out["main_pollutant_pred"] = predicted_aqi["main_pollutant"]
    out["aqi_actual"] = actual_aqi["aqi"]
    out["category_actual"] = actual_aqi["category"]
    out["main_pollutant_actual"] = actual_aqi["main_pollutant"]
    return out, used


def data_period(city, time):
    """Was this hour used for training, validation or testing?"""
    artifact = load_model(city, "pm25", "sgd")
    time = pd.Timestamp(time)
    if time <= artifact["train_end"]:
        return "training"
    if time < artifact["test_start"]:
        return "validation" if time >= artifact["val_start"] else "gap"
    return "test"


def current_aqi(city, start):
    """The AQI breakdown at the start hour (from real readings)."""
    table = load_city(city)
    return aqi_table(table.loc[:pd.Timestamp(start)]).iloc[-1]


def explain_change(city, start, result):
    """
    Which pollutant pushes the AQI up or down over the forecast?
    Compares each pollutant's sub-index now with its forecast at the
    hour of the highest predicted AQI.
    """
    now = current_aqi(city, start)
    peak_row = result.loc[result["aqi_pred"].idxmax()] if result["aqi_pred"].notna().any() else None
    rows = []
    for pollutant in load_city(city).columns:
        si_now = now.get(f"{pollutant}_si", np.nan)
        si_peak = peak_row[f"{pollutant}_si_pred"] if peak_row is not None else np.nan
        rows.append({"pollutant": pollutant, "sub_index_now": si_now,
                     "sub_index_at_peak": si_peak, "change": si_peak - si_now})
    return pd.DataFrame(rows).sort_values("change", ascending=False), now
