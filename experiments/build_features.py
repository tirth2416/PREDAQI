
import os
import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/all_cities_hourly.csv"
OUTPUT_DIR = "data/processed/features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

df["period_start"] = pd.to_datetime(df["period_start"])
df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

df = df.dropna(subset=["period_start", "pm25"])
df = df.sort_values(["city", "period_start"])

# Forecast every hour from 1 to 48 hours ahead
HORIZONS = list(range(1, 49))

# Historical PM2.5 lag features
LAGS = [1, 2, 3, 6, 12, 24, 48]

all_features = []

for city in sorted(df["city"].unique()):
    print(f"\nPreparing features for {city}")

    city_df = (
        df[df["city"] == city]
        .sort_values("period_start")
        .set_index("period_start")
    )

    # Insert missing hours as NaN
    full_index = pd.date_range(
        start=city_df.index.min(),
        end=city_df.index.max(),
        freq="h"
    )

    city_df = city_df.reindex(full_index)
    city_df.index.name = "period_start"
    city_df["city"] = city

    # Assign continuous-run identifiers
    city_df["run_id"] = city_df["pm25"].isna().cumsum()

    valid = city_df["pm25"].notna()

    pieces = []

    # Construct features separately for continuous runs
    for run_id, run in city_df.loc[valid].groupby("run_id"):
        run = run.copy()

        if len(run) < 100:
            continue

        # Historical lag features
        for lag in LAGS:
            run[f"pm25_lag_{lag}"] = run["pm25"].shift(lag)

        # Rolling averages use past values only
        run["rolling_mean_3h"] = (
            run["pm25"].shift(1).rolling(3).mean()
        )

        run["rolling_mean_6h"] = (
            run["pm25"].shift(1).rolling(6).mean()
        )

        run["rolling_mean_24h"] = (
            run["pm25"].shift(1).rolling(24).mean()
        )

        # Current time features
        run["hour"] = run.index.hour
        run["day_of_week"] = run.index.dayofweek
        run["month"] = run.index.month

        # Cyclical encoding of current hour
        run["hour_sin"] = np.sin(
            2 * np.pi * run["hour"] / 24
        )

        run["hour_cos"] = np.cos(
            2 * np.pi * run["hour"] / 24
        )

        # Future target-time features for each horizon
        for horizon in HORIZONS:
            future_index = run.index + pd.Timedelta(
                hours=horizon
            )

            future_hour = future_index.hour
            future_day_of_week = future_index.dayofweek
            future_month = future_index.month

            run[f"target_{horizon}h"] = (
                run["pm25"].shift(-horizon)
            )

            run[f"target_hour_{horizon}h"] = future_hour

            run[f"target_hour_sin_{horizon}h"] = np.sin(
                2 * np.pi * future_hour / 24
            )

            run[f"target_hour_cos_{horizon}h"] = np.cos(
                2 * np.pi * future_hour / 24
            )

            run[f"target_day_of_week_{horizon}h"] = (
                future_day_of_week
            )

            run[f"target_month_{horizon}h"] = (
                future_month
            )

        pieces.append(run.reset_index())

    if not pieces:
        print(
            "No continuous runs long enough "
            "for feature creation."
        )
        continue

    features = pd.concat(
        pieces,
        ignore_index=True
    )

    # Required historical features
    required_columns = (
        [f"pm25_lag_{lag}" for lag in LAGS]
        + [
            "rolling_mean_3h",
            "rolling_mean_6h",
            "rolling_mean_24h"
        ]
    )

    # Require all 48 future targets
    required_columns += [
        f"target_{horizon}h"
        for horizon in HORIZONS
    ]

    features = features.dropna(
        subset=required_columns
    )

    features["city"] = city

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{city.lower()}_features.csv"
    )

    features.to_csv(
        output_path,
        index=False
    )

    print("Feature rows:", len(features))

    print(
        "Date range:",
        features["period_start"].min(),
        "to",
        features["period_start"].max()
    )

    print("Saved:", output_path)

    all_features.append(features)


# Save combined dataset
if all_features:
    combined = pd.concat(
        all_features,
        ignore_index=True
    )

    combined_path = os.path.join(
        OUTPUT_DIR,
        "all_cities_features.csv"
    )

    combined.to_csv(
        combined_path,
        index=False
    )

    print("\n" + "=" * 55)
    print("FEATURE PREPARATION COMPLETE")
    print("=" * 55)

    print("Total feature rows:", len(combined))
    print("Cities:", combined["city"].nunique())
    print("Saved:", combined_path)

    print("\nRows per city:")
    print(combined.groupby("city").size())

else:
    print("No feature datasets were created.")
