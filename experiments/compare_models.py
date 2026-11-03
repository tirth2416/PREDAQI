
import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input file
INPUT_FILE = PROJECT_ROOT / "results" / "predictions" / "pm25_48_hour_forecasts.csv"

# Load predictions and evaluation metrics
df = pd.read_csv(INPUT_FILE)

# Calculate average metrics for each city and model
comparison = (
    df.groupby(["city", "model"])
    .agg(
        Average_MAE=("test_mae", "mean"),
        Average_RMSE=("test_rmse", "mean")
    )
    .reset_index()
)

# Round results
comparison["Average_MAE"] = comparison["Average_MAE"].round(3)
comparison["Average_RMSE"] = comparison["Average_RMSE"].round(3)

print("\n========== MODEL PERFORMANCE BY CITY ==========\n")

print(comparison.to_string(index=False))

# Identify best model for each city
print("\n========== BEST MODEL FOR EACH CITY ==========\n")

for city in comparison["city"].unique():

    city_data = comparison[comparison["city"] == city]

    best_mae = city_data.loc[city_data["Average_MAE"].idxmin()]
    best_rmse = city_data.loc[city_data["Average_RMSE"].idxmin()]

    print(f"City: {city}")
    print(
        f"Best model by MAE: {best_mae['model']} "
        f"(MAE = {best_mae['Average_MAE']})"
    )
    print(
        f"Best model by RMSE: {best_rmse['model']} "
        f"(RMSE = {best_rmse['Average_RMSE']})"
    )
    print()

# Overall performance across cities
overall = (
    df.groupby("model")
    .agg(
        Overall_MAE=("test_mae", "mean"),
        Overall_RMSE=("test_rmse", "mean")
    )
    .reset_index()
)

overall["Overall_MAE"] = overall["Overall_MAE"].round(3)
overall["Overall_RMSE"] = overall["Overall_RMSE"].round(3)

print("\n========== OVERALL MODEL PERFORMANCE ==========\n")
print(overall.to_string(index=False))

# Save results
OUTPUT_FILE = PROJECT_ROOT / "results" / "model_comparison.csv"

comparison.to_csv(OUTPUT_FILE, index=False)

print(f"\nComparison saved to: {OUTPUT_FILE}")
