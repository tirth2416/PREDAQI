"""
PREDAQI - shared settings used by every script.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Year of data used (the latest year with all 6 pollutants in the database)
YEAR = 2024

CITIES = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai"]

# Short names used in file and column names
POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

# Names shown in the dashboard
LABELS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO₂",
    "so2": "SO₂",
    "co": "CO",
    "o3": "O₃",
}

UNITS = {
    "pm25": "µg/m³",
    "pm10": "µg/m³",
    "no2": "µg/m³",
    "so2": "µg/m³",
    "co": "mg/m³",
    "o3": "µg/m³",
}

# Readings above these values are treated as sensor errors
MAX_VALID = {
    "pm25": 999,
    "pm10": 1500,
    "no2": 1000,
    "so2": 1000,
    "co": 50,
    "o3": 1000,
}

# How many hours ahead we forecast
HORIZON = 48

# Past readings (in hours) given to the model as inputs
LAGS = [1, 2, 3, 6, 12, 24]
