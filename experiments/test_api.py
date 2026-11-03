import os
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")

API_KEY = os.getenv("AIR_QUALITY_API_KEY")
BASE_URL = "https://airquality.xkdr.org/v1/measurements"

if not API_KEY:
    raise ValueError("API key missing. Check your .env file.")

headers = {"Authorization": f"Bearer {API_KEY}"}

cities = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai"]

for city in cities:
    print(f"\n{'=' * 45}")
    print(f"Checking historical PM2.5 data: {city}")

    params = {
        "city": city,
        "parameter": "PM2.5",
        "start": "2025-04-01",
        "end": "2025-04-07",
        "agg": "hourly",
        "format": "json",
        "limit": 10000,
    }

    try:
        response = requests.get(
            BASE_URL,
            headers=headers,
            params=params,
            timeout=90,
        )

        print("HTTP status:", response.status_code)

        if not response.ok:
            print("API response:", response.text[:300])
            continue

        print("Rows reported by API:",
              response.headers.get("X-Row-Count", "Not provided"))
        print("Truncated:",
              response.headers.get("X-Truncated", "Not provided"))

        data = response.json()

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = next(
                (
                    data[key]
                    for key in ("data", "measurements", "results")
                    if isinstance(data.get(key), list)
                ),
                [],
            )
        else:
            records = []

        print("Records returned:", len(records))

        if records:
            print("First record:", records[0])
        else:
            print("No records found in the expected JSON structure.")
            print("Response keys:",
                  list(data.keys()) if isinstance(data, dict)
                  else type(data).__name__)

    except requests.RequestException as error:
        print("Request failed:", error)