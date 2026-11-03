"""
PREDAQI - Step 2: Clean the raw data and make one hourly table per city.

Input : data/raw/<city>_<pollutant>_2024.csv   (one row per station per hour)
Output: data/processed/<city>_hourly.csv       (one row per hour,
                                                one column per pollutant)

Cleaning rules:
- remove negative readings and impossible values (sensor errors)
- convert CO to mg/m3 if the API gives it in µg/m3
- combine stations using the MEDIAN, so one faulty sensor
  cannot distort the whole city's value
- fill very short gaps (up to 3 hours) by straight-line interpolation;
  longer gaps stay empty
- data/processed/data_quality.csv records what was removed
"""

import pandas as pd

from config import CITIES, MAX_VALID, POLLUTANTS, PROCESSED_DIR, RAW_DIR, YEAR

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_pollutant(city, pollutant):
    path = RAW_DIR / f"{city.lower()}_{pollutant}_{YEAR}.csv"
    if not path.exists():
        return None, None

    df = pd.read_csv(path)
    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df = df.dropna(subset=["period_start", "mean"])

    # Keep only the chosen year
    df = df[df["period_start"].dt.year == YEAR]

    report = {"city": city, "pollutant": pollutant, "raw_readings": len(df)}

    # CO must be in mg/m3 for the AQI table
    if pollutant == "co" and "unit" in df.columns:
        in_micrograms = df["unit"].astype(str).str.contains("µg|ug", regex=True)
        df.loc[in_micrograms, "mean"] = df.loc[in_micrograms, "mean"] / 1000

    bad = (df["mean"] < 0) | (df["mean"] > MAX_VALID[pollutant])
    report["removed_invalid"] = int(bad.sum())
    df = df[~bad]

    # One value per hour for the whole city
    hourly = df.groupby("period_start")["mean"].median()
    report["stations"] = df["station_id"].nunique()
    report["hours_with_data"] = len(hourly)
    return hourly, report


def main():
    reports = []

    for city in CITIES:
        columns = {}
        for pollutant in POLLUTANTS:
            hourly, report = load_pollutant(city, pollutant)
            if hourly is None or hourly.empty:
                print(f"{city:10s} {pollutant:5s} no data")
                continue
            columns[pollutant] = hourly
            reports.append(report)

        if "pm25" not in columns:
            print(f"{city}: no PM2.5 data, skipping city")
            continue

        table = pd.DataFrame(columns)

        # Complete hourly timeline (missing hours become empty rows)
        full_index = pd.date_range(table.index.min(), table.index.max(), freq="h")
        table = table.reindex(full_index)
        table.index.name = "time"

        # Fill gaps of up to 3 hours only
        table = table.interpolate(limit=3, limit_area="inside")

        path = PROCESSED_DIR / f"{city.lower()}_hourly.csv"
        table.round(3).to_csv(path)

        coverage = (table.notna().mean() * 100).round(1).to_dict()
        print(f"{city:10s} {table.index.min()} to {table.index.max()} "
              f"| % hours with data: {coverage}")

    pd.DataFrame(reports).to_csv(PROCESSED_DIR / "data_quality.csv", index=False)
    print("\nSaved hourly tables to data/processed/")


if __name__ == "__main__":
    main()
