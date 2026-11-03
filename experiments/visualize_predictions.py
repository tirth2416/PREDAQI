
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input CSV
INPUT_FILE = PROJECT_ROOT / "results" / "predictions" / "pm25_48_hour_forecasts.csv"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "results" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("Loaded CSV:", INPUT_FILE)
print("Output directory:", OUTPUT_DIR)

for city in df["city"].unique():

    city_df = df[df["city"] == city]

    fig, ax = plt.subplots(figsize=(12, 6))

    for model in city_df["model"].unique():

        model_df = city_df[city_df["model"] == model]
        model_df = model_df.sort_values("horizon_hours")

        ax.plot(
            model_df["horizon_hours"],
            model_df["predicted_pm25"],
            marker="o",
            markersize=3,
            label=model
        )

    ax.set_title(f"48-Hour PM2.5 Forecast - {city}")
    ax.set_xlabel("Forecast Horizon (Hours)")
    ax.set_ylabel("Predicted PM2.5")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_file = OUTPUT_DIR / f"{city.lower()}_forecast.png"

    fig.savefig(output_file, format="png", dpi=300)

    print(f"Saved: {output_file}")
    print(f"File exists: {output_file.exists()}")
    print(f"File size: {output_file.stat().st_size} bytes")

    plt.close(fig)

print("\nVisualization generation completed.")
