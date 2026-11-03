import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/all_cities_hourly.csv"
OUTPUT_DIR = "results/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["period_start"] = pd.to_datetime(df["period_start"])
df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

df = df.dropna(subset=["period_start", "pm25"])
df = df.sort_values(["city", "period_start"])

sns.set_theme(style="whitegrid")


# 1. Average PM2.5 by city
city_avg = df.groupby("city")["pm25"].mean().sort_values()

plt.figure(figsize=(10, 6))
city_avg.plot(kind="bar")
plt.title("Average PM2.5 Concentration by City")
plt.xlabel("City")
plt.ylabel("PM2.5 (µg/m³)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/city_comparison.png", dpi=300)
plt.close()


# 2. Hourly PM2.5 trends
# Insert missing timestamps so lines break at gaps instead of
# connecting observations that are several hours apart.

plt.figure(figsize=(15, 7))

for city in sorted(df["city"].unique()):
    city_data = (
        df[df["city"] == city]
        .set_index("period_start")
        .sort_index()
    )

    # Create a complete hourly index for this city's own date range.
    full_index = pd.date_range(
        start=city_data.index.min(),
        end=city_data.index.max(),
        freq="h"
    )

    city_data = city_data.reindex(full_index)

    # Break the line when there are more than 1 hour between readings.
    plt.plot(
        city_data.index,
        city_data["pm25"],
        label=city,
        linewidth=0.8,
        alpha=0.8
    )

plt.title("Hourly PM2.5 Trends Across Five Cities")
plt.xlabel("Date")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/hourly_trends.png", dpi=300)
plt.close()


# 3. Monthly comparison
df["month"] = df["period_start"].dt.to_period("M").astype(str)

monthly = (
    df.groupby(["city", "month"])["pm25"]
    .mean()
    .reset_index()
)

# Main comparison: April to August, when all five cities have data.
common_months = ["2025-04", "2025-05", "2025-06", "2025-07", "2025-08"]
monthly_common = monthly[monthly["month"].isin(common_months)].copy()

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=monthly_common,
    x="month",
    y="pm25",
    hue="city",
    marker="o"
)
plt.title("Monthly Average PM2.5 by City (April–August 2025)")
plt.xlabel("Month")
plt.ylabel("PM2.5 (µg/m³)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/monthly_comparison.png", dpi=300)
plt.close()


# 4. Distribution of PM2.5
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="city", y="pm25")
plt.title("Distribution of PM2.5 Concentrations by City")
plt.xlabel("City")
plt.ylabel("PM2.5 (µg/m³)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pm25_distribution.png", dpi=300)
plt.close()


# 5. Average PM2.5 by hour of day
df["hour"] = df["period_start"].dt.hour

hourly_pattern = (
    df.groupby(["city", "hour"])["pm25"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=hourly_pattern,
    x="hour",
    y="pm25",
    hue="city",
    marker="o"
)
plt.title("Average PM2.5 by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("PM2.5 (µg/m³)")
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/hourly_pattern.png", dpi=300)
plt.close()


# 6. Coverage and missing-hour audit
print("\nDATA COVERAGE AUDIT")
print("=" * 55)

for city in sorted(df["city"].unique()):
    city_data = df[df["city"] == city].sort_values("period_start")

    start = city_data["period_start"].min()
    end = city_data["period_start"].max()

    expected_hours = int((end - start).total_seconds() / 3600) + 1
    observed_hours = city_data["period_start"].nunique()
    missing_hours = expected_hours - observed_hours

    print(f"\n{city}")
    print("Start:", start)
    print("End:", end)
    print("Observed hours:", observed_hours)
    print("Expected hours:", expected_hours)
    print("Missing hours:", missing_hours)


# 7. Summary statistics
print("\nCITY SUMMARY")
print("=" * 55)

summary = df.groupby("city")["pm25"].agg(
    ["count", "mean", "median", "std", "min", "max"]
).round(2)

print(summary)
summary.to_csv(f"{OUTPUT_DIR}/city_summary.csv")


# 8. Data coverage by city and month
coverage = (
    df.groupby(["city", "month"])["period_start"]
    .nunique()
    .reset_index(name="observed_hours")
)

coverage.to_csv(f"{OUTPUT_DIR}/monthly_coverage.csv", index=False)

print("\nEDA completed.")
print("Charts and summaries saved in:", OUTPUT_DIR)