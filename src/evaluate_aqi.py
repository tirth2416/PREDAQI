"""
PREDAQI - Step 4: How accurate is the final AQI forecast?

Over each city's TEST period (data the models never trained on), we
start a 48-hour forecast every 6 hours and compare it with the real
AQI. We also check a persistence baseline ("AQI stays as it is now").

Outputs:
  results/aqi_accuracy_by_hour.csv  AQI error for each hour ahead
  results/aqi_summary.csv           one line per city
"""

import warnings

import numpy as np
import pandas as pd

from config import CITIES, PROCESSED_DIR, RESULTS_DIR
from forecast import current_aqi, forecast_aqi, load_city, load_model, valid_start_times

warnings.filterwarnings("ignore")


def evaluate_city(city):
    table = load_city(city)
    # Start of the test period = latest test start among this city's models
    test_start = max(load_model(city, p, "sgd")["test_start"] for p in table.columns)
    starts = [t for t in valid_start_times(city) if t >= test_start and t.hour % 6 == 0]
    # The 48 forecast hours must lie inside our data
    starts = [t for t in starts if t + pd.Timedelta(hours=48) <= table.index.max()]

    rows = []
    for start in starts:
        result, _ = forecast_aqi(city, start)
        now = current_aqi(city, start)
        for _, r in result.iterrows():
            if pd.isna(r["aqi_actual"]) or pd.isna(r["aqi_pred"]):
                continue
            rows.append({
                "city": city,
                "start": start,
                "hours_ahead": r["hours_ahead"],
                "aqi_actual": r["aqi_actual"],
                "aqi_pred": r["aqi_pred"],
                "aqi_persistence": now["aqi"],
                "category_ok": r["category_pred"] == r["category_actual"],
                "category_ok_persistence": now["category"] == r["category_actual"],
                "main_ok": r["main_pollutant_pred"] == r["main_pollutant_actual"],
            })
    return pd.DataFrame(rows)


def main():
    frames = []
    for city in CITIES:
        if not (PROCESSED_DIR / f"{city.lower()}_hourly.csv").exists():
            continue
        try:
            df = evaluate_city(city)
        except FileNotFoundError:
            continue
        if df.empty:
            print(f"{city}: AQI not available (needs 3 pollutants incl. PM)")
            continue
        frames.append(df)
        print(f"{city}: {df['start'].nunique()} forecasts checked")

    if not frames:
        print("No AQI results.")
        return

    all_rows = pd.concat(frames)
    all_rows["error"] = (all_rows["aqi_pred"] - all_rows["aqi_actual"]).abs()
    all_rows["error_persistence"] = (all_rows["aqi_persistence"] - all_rows["aqi_actual"]).abs()

    by_hour = all_rows.groupby(["city", "hours_ahead"]).agg(
        aqi_mae=("error", "mean"),
        aqi_mae_persistence=("error_persistence", "mean"),
        category_accuracy=("category_ok", "mean"),
        category_accuracy_persistence=("category_ok_persistence", "mean"),
        main_pollutant_accuracy=("main_ok", "mean"),
    ).reset_index()
    by_hour.round(3).to_csv(RESULTS_DIR / "aqi_accuracy_by_hour.csv", index=False)

    summary = all_rows.groupby("city").agg(
        forecasts_checked=("start", "nunique"),
        average_actual_aqi=("aqi_actual", "mean"),
        aqi_mae=("error", "mean"),
        aqi_mae_persistence=("error_persistence", "mean"),
        category_accuracy=("category_ok", "mean"),
        category_accuracy_persistence=("category_ok_persistence", "mean"),
        main_pollutant_accuracy=("main_ok", "mean"),
    ).reset_index()
    summary.round(3).to_csv(RESULTS_DIR / "aqi_summary.csv", index=False)

    print("\nAQI forecast accuracy on the test period:")
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
