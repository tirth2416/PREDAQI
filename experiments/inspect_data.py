import os
import pandas as pd

DATA_DIR = "data/raw"

cities = ["delhi", "mumbai", "bengaluru", "kolkata", "chennai"]

for city in cities:
    file_path = os.path.join(DATA_DIR, f"{city}_pm25_2025.csv")

    print("\n" + "=" * 55)
    print(f"CITY: {city.upper()}")
    print("=" * 55)

    if not os.path.exists(file_path):
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(file_path)
    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")

    print("Total records:", len(df))
    print("Unique stations:", df["station_id"].nunique())
    print("Start date:", df["period_start"].min())
    print("End date:", df["period_start"].max())
    print("Missing timestamps:", df["period_start"].isna().sum())
    print("Missing PM2.5 values:", df["mean"].isna().sum())
    print("Duplicate station-time records:",
          df.duplicated(["station_id", "period_start"]).sum())

    print("\nPM2.5 statistics:")
    print(df["mean"].describe().round(2))

    print("\nRecords by month:")
    print(df.groupby(df["period_start"].dt.to_period("M")).size())

    print("\nRecords by station (top 5):")
    print(df["station_id"].value_counts().head(5))