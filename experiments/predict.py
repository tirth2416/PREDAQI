from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "data" / "processed" / "features"
MODEL_DIR = ROOT / "results" / "final" / "models"
PREDICTION_DIR = ROOT / "results" / "predictions"

PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = range(1, 49)


def safe_city_name(city):
    return "".join(
        character
        if character.isalnum() or character in ("-", "_")
        else "_"
        for character in str(city)
    )


def add_target_time_features(row, forecast_time, horizon):
    """
    Update the calendar features for the future forecast timestamp.
    """

    hour = forecast_time.hour
    day_of_week = forecast_time.dayofweek
    month = forecast_time.month

    row[f"target_hour_sin_{horizon}h"] = np.sin(
        2 * np.pi * hour / 24
    )

    row[f"target_hour_cos_{horizon}h"] = np.cos(
        2 * np.pi * hour / 24
    )

    row[f"target_day_of_week_{horizon}h"] = day_of_week
    row[f"target_month_{horizon}h"] = month

    return row


def load_latest_features(feature_file):
    """
    Load the latest usable historical feature row for a city.
    """

    df = pd.read_csv(feature_file)

    if "period_start" not in df.columns:
        raise ValueError(
            f"period_start missing in {feature_file.name}"
        )

    df["period_start"] = pd.to_datetime(
        df["period_start"]
    )

    df = df.sort_values("period_start").reset_index(drop=True)

    if "city" in df.columns:
        city = df["city"].iloc[0]
    else:
        city = feature_file.stem.replace(
            "_features", ""
        ).capitalize()

    return df, city


def predict_city(df, city):
    """
    Generate forecasts for all 48 hours using saved models.
    """

    safe_city = safe_city_name(city)

    predictions = []

    print("\n" + "=" * 60)
    print(f"Generating predictions for {city}")
    print("=" * 60)

    for horizon in HORIZONS:

        target_time_columns = [
            f"target_hour_sin_{horizon}h",
            f"target_hour_cos_{horizon}h",
            f"target_day_of_week_{horizon}h",
            f"target_month_{horizon}h"
        ]

        for model_type in ["sgd", "mlp"]:

            model_path = (
                MODEL_DIR
                / f"{safe_city}_{model_type}_{horizon}h.joblib"
            )

            if not model_path.exists():
                print(f"Model not found: {model_path.name}")
                continue

            artifact = joblib.load(model_path)

            model = artifact["model"]
            feature_columns = artifact["feature_columns"]

            # Historical features that must be present.
            historical_columns = [
                column
                for column in feature_columns
                if column not in target_time_columns
            ]

            # Find the most recent row with complete historical features.
            usable_data = df.dropna(
                subset=historical_columns
            ).copy()

            if usable_data.empty:
                print(
                    f"No usable features for {city}, "
                    f"{horizon}h, {model_type}"
                )
                continue

            latest_row = usable_data.iloc[-1]

            latest_timestamp = pd.Timestamp(
                latest_row["period_start"]
            )

            forecast_time = (
                latest_timestamp
                + pd.Timedelta(hours=horizon)
            )

            # Build input row.
            X = latest_row[feature_columns].copy()

            # Update future calendar features.
            X = add_target_time_features(
                X,
                forecast_time,
                horizon
            )

            X = pd.DataFrame(
                [X],
                columns=feature_columns
            )

            # Predict PM2.5.
            prediction = model.predict(X)[0]

            # PM2.5 cannot be negative.
            prediction = max(0, float(prediction))

            predictions.append({
                "city": city,
                "last_observation_time": latest_timestamp,
                "forecast_time": forecast_time,
                "horizon_hours": horizon,
                "model": model_type.upper(),
                "predicted_pm25": round(prediction, 2),
                "selected_pca": artifact["pca_label"],
                "validation_mae": artifact["validation_mae"],
                "test_mae": artifact["test_mae"],
                "test_rmse": artifact["test_rmse"]
            })

    return predictions


def main():

    feature_files = sorted(
        FEATURE_DIR.glob("*_features.csv")
    )

    feature_files = [
        file
        for file in feature_files
        if file.name != "all_cities_features.csv"
    ]

    if not feature_files:
        print("No city feature files found.")
        return

    all_predictions = []

    for feature_file in feature_files:

        df, city = load_latest_features(
            feature_file
        )

        city_predictions = predict_city(
            df,
            city
        )

        all_predictions.extend(
            city_predictions
        )

    if not all_predictions:
        print("No predictions generated.")
        return

    predictions_df = pd.DataFrame(
        all_predictions
    )

    output_path = (
        PREDICTION_DIR
        / "pm25_48_hour_forecasts.csv"
    )

    predictions_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)

    print(
        predictions_df[
            [
                "city",
                "forecast_time",
                "horizon_hours",
                "model",
                "predicted_pm25"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nPredictions saved to: {output_path}"
    )


if __name__ == "__main__":
    main()