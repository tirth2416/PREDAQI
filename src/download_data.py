"""
PREDAQI - Step 1: Download hourly pollutant data

Downloads hourly readings for 6 pollutants (PM2.5, PM10, NO2, SO2, CO, O3)
for 5 cities from the India Air Quality Database (airquality.xkdr.org),
for the full year 2024.

Why 2024: the database has NO2, SO2, CO and Ozone only up to
31 Dec 2024 (PM2.5 and PM10 continue into 2025). All 6 pollutants are
needed for the official AQI, so 2024 is the latest complete year.

Run from the project folder:
    python src/download_data.py

Your API key must be in the .env file:
    AIR_QUALITY_API_KEY=your_key_here

Files are saved as data/raw/<city>_<pollutant>_2024.csv
If a file already covers the full date range, it is skipped,
so you can safely run this script again if it stops halfway.
"""

import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")

API_KEY = os.getenv("AIR_QUALITY_API_KEY")
BASE_URL = "https://airquality.xkdr.org/v1/measurements"

CITIES = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai"]

# Our short name for each pollutant -> the name used by the API
# (checked with the API's /v1/parameters list)
POLLUTANTS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "so2": "SO2",
    "co": "CO",       # in mg/m3
    "o3": "Ozone",
}

YEAR = 2024
START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 1, 1)   # end date is not included

BATCH_DAYS = 7                # ask for one week at a time
MAX_RETRIES = 3

OUTPUT_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns we keep (keeps the files small)
KEEP_COLUMNS = ["station_id", "parameter_name", "unit", "period_start", "mean"]

if not API_KEY:
    raise ValueError("API key missing. Add AIR_QUALITY_API_KEY to your .env file.")

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {API_KEY}"})


def get_records(response_json):
    """The API returns a list of records (or a dict containing one)."""
    if isinstance(response_json, list):
        return response_json
    if isinstance(response_json, dict):
        for key in ("data", "measurements", "results"):
            if isinstance(response_json.get(key), list):
                return response_json[key]
    return []


def fetch(city, api_name, start, end):
    """
    Download one date range. Returns (records, ok).
    If the API truncates the answer or fails, the range is split
    into single days and downloaded again.
    """
    params = {
        "city": city,
        "parameter": api_name,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "agg": "hourly",
        "format": "json",
        "limit": 10000,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(BASE_URL, params=params, timeout=90)
            response.raise_for_status()
            records = get_records(response.json())
            truncated = response.headers.get("X-Truncated", "false").lower() == "true"

            if truncated and (end - start).days > 1:
                print("    response truncated, downloading day by day...")
                return fetch_day_by_day(city, api_name, start, end)

            return records, True

        except (requests.RequestException, ValueError) as error:
            print(f"    attempt {attempt} failed: {error}")
            time.sleep(2 * attempt)

    if (end - start).days > 1:
        print("    retrying this week day by day...")
        return fetch_day_by_day(city, api_name, start, end)

    return [], False


def fetch_day_by_day(city, api_name, start, end):
    all_records = []
    all_ok = True
    day = start
    while day < end:
        records, ok = fetch(city, api_name, day, day + timedelta(days=1))
        all_records.extend(records)
        all_ok = all_ok and ok
        day += timedelta(days=1)
    return all_records, all_ok


def file_is_complete(path):
    """True if the file already has data up to the end date."""
    if not path.exists():
        return False
    try:
        dates = pd.read_csv(path, usecols=["period_start"])["period_start"]
        latest = pd.to_datetime(dates, errors="coerce").max()
        return latest.date() >= END_DATE - timedelta(days=1)
    except Exception:
        return False


def download(city, pollutant, api_name):
    path = OUTPUT_DIR / f"{city.lower()}_{pollutant}_{YEAR}.csv"

    if file_is_complete(path):
        print(f"{city} {pollutant}: already downloaded, skipping")
        return

    print(f"{city} {pollutant}: downloading...")
    all_records = []
    failed = 0
    start = START_DATE

    while start < END_DATE:
        end = min(start + timedelta(days=BATCH_DAYS), END_DATE)
        records, ok = fetch(city, api_name, start, end)
        all_records.extend(records)
        if not ok:
            failed += 1
        start = end
        time.sleep(0.2)

    if not all_records:
        print(f"  no data for {city} {pollutant}")
        return

    df = pd.DataFrame(all_records)
    df = df[[c for c in KEEP_COLUMNS if c in df.columns]]
    df = df.drop_duplicates(subset=["station_id", "period_start"])
    df = df.sort_values(["station_id", "period_start"])
    df.to_csv(path, index=False)

    print(f"  saved {len(df):,} rows from {df['station_id'].nunique()} stations"
          f" ({df['period_start'].min()} to {df['period_start'].max()})")
    if failed:
        print(f"  WARNING: {failed} batches failed. Run the script again to retry.")


def main():
    print("PREDAQI download")
    print(f"Period: {START_DATE} to {END_DATE}\n")

    for city in CITIES:
        for pollutant, api_name in POLLUTANTS.items():
            download(city, pollutant, api_name)

    print("\nDone. Files are in data/raw/")


if __name__ == "__main__":
    main()
