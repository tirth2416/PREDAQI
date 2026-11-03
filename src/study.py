"""
PREDAQI - Step 5: Pollutant study (uses the REAL measured data).

Questions answered:
 1. Which pollutant decides the AQI most often in each city?
 2. When the AQI goes up from one day to the next, which pollutant
    is responsible for the rise?
 3. Do traffic pollutants (NO2, CO) drop on weekends?
 4. How does each pollutant change through the day (rush hours)?
 5. How do pollutants change month by month (e.g. monsoon)?
 6. What happens to pollution on festivals and public holidays?

These are PATTERNS in the data (correlations), not proof of cause.

Outputs go to results/study/
"""

import numpy as np
import pandas as pd

from aqi import aqi_table
from config import CITIES, PROCESSED_DIR, RESULTS_DIR

OUT = RESULTS_DIR / "study"
OUT.mkdir(parents=True, exist_ok=True)

# Festivals and public holidays in 2024
HOLIDAYS = {
    "2024-01-26": "Republic Day",
    "2024-03-25": "Holi",
    "2024-04-11": "Eid ul-Fitr",
    "2024-06-17": "Eid al-Adha",
    "2024-08-15": "Independence Day",
    "2024-08-26": "Janmashtami",
    "2024-09-07": "Ganesh Chaturthi",
    "2024-10-02": "Gandhi Jayanti",
    "2024-10-11": "Durga Puja (Navami)",
    "2024-10-12": "Dussehra",
    "2024-10-31": "Diwali",
    "2024-11-01": "Day after Diwali",
    "2024-11-15": "Guru Nanak Jayanti",
    "2024-12-25": "Christmas",
}


def load(city):
    path = PROCESSED_DIR / f"{city.lower()}_hourly.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def dominant_pollutant(city, aqi):
    counts = aqi["main_pollutant"].value_counts(normalize=True) * 100
    return [{"city": city, "pollutant": p, "percent_of_hours": round(v, 1)}
            for p, v in counts.items()]


def rise_drivers(city, aqi, pollutants):
    """Daily AQI: on days where AQI rose by 10+ points, whose sub-index rose most?"""
    daily = aqi.resample("D").mean(numeric_only=True)
    si_cols = [f"{p}_si" for p in pollutants if f"{p}_si" in daily]
    change = daily[si_cols + ["aqi"]].diff()
    rising = change[change["aqi"] >= 10].dropna(subset=si_cols, how="all")

    rows = []
    if len(rising):
        winners = rising[si_cols].idxmax(axis=1).str.replace("_si", "")
        shares = winners.value_counts(normalize=True) * 100
        for p, v in shares.items():
            rows.append({"city": city, "pollutant": p, "percent_of_rising_days": round(v, 1),
                         "rising_days": len(rising)})

    # How closely each sub-index moves with the AQI (day-to-day changes)
    corr = change[si_cols].corrwith(change["aqi"])
    corr_rows = [{"city": city, "pollutant": c.replace("_si", ""),
                  "correlation_with_aqi_change": round(v, 2)} for c, v in corr.items()]
    return rows, corr_rows


def weekday_weekend(city, table):
    weekend = table.index.dayofweek >= 5
    rows = []
    for p in table.columns:
        wd, we = table.loc[~weekend, p].mean(), table.loc[weekend, p].mean()
        rows.append({"city": city, "pollutant": p, "weekday_mean": round(wd, 3),
                     "weekend_mean": round(we, 3),
                     "weekend_change_percent": round((we - wd) / wd * 100, 1)})
    return rows


def hourly_profile(city, table):
    profile = table.groupby(table.index.hour).mean()
    # As % of the daily average, so different pollutants can share one chart
    relative = profile / profile.mean() * 100
    rows = []
    for hour in profile.index:
        for p in table.columns:
            rows.append({"city": city, "pollutant": p, "hour": hour,
                         "mean": round(profile.loc[hour, p], 3),
                         "percent_of_daily_avg": round(relative.loc[hour, p], 1)})
    return rows


def monthly(city, table, aqi):
    m = table.resample("MS").mean()
    m["aqi"] = aqi["aqi"].resample("MS").mean()
    m["hours_with_data"] = table.notna().all(axis=1).resample("MS").sum()
    m.insert(0, "city", city)
    m.index = m.index.strftime("%Y-%m")
    m.index.name = "month"
    return m.reset_index()


def holidays(city, table):
    """Holiday average vs the normal days in the 7 days before and after."""
    rows = []
    for date, name in HOLIDAYS.items():
        day = pd.Timestamp(date)
        on_day = table.loc[day:day + pd.Timedelta(hours=23)]
        around = table.loc[day - pd.Timedelta(days=7):day + pd.Timedelta(days=8)]
        around = around[(around.index < day - pd.Timedelta(days=1)) |
                        (around.index >= day + pd.Timedelta(days=2))]
        if on_day.notna().sum().max() < 18 or around.notna().sum().max() < 150:
            continue
        for p in table.columns:
            holiday_mean, normal_mean = on_day[p].mean(), around[p].mean()
            if pd.isna(holiday_mean) or pd.isna(normal_mean) or normal_mean == 0:
                continue
            rows.append({"city": city, "date": date, "holiday": name, "pollutant": p,
                         "holiday_mean": round(holiday_mean, 3),
                         "normal_days_mean": round(normal_mean, 3),
                         "change_percent": round((holiday_mean - normal_mean) / normal_mean * 100, 1)})
    return rows


def main():
    dom, rise, corr, week, hours, months, hol, coverage = [], [], [], [], [], [], [], []

    for city in CITIES:
        table = load(city)
        if table is None:
            continue
        aqi = aqi_table(table)
        pollutants = list(table.columns)
        coverage.append({"city": city, "pollutants": ", ".join(pollutants),
                         "hours_with_valid_aqi": int(aqi["aqi"].notna().sum()),
                         "average_aqi": round(aqi["aqi"].mean(), 1)})

        dom += dominant_pollutant(city, aqi)
        r, c = rise_drivers(city, aqi, pollutants)
        rise += r
        corr += c
        week += weekday_weekend(city, table)
        hours += hourly_profile(city, table)
        months.append(monthly(city, table, aqi))
        hol += holidays(city, table)
        print(f"{city}: done")

    pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False)
    pd.DataFrame(dom).to_csv(OUT / "dominant_pollutant.csv", index=False)
    pd.DataFrame(rise).to_csv(OUT / "aqi_rise_drivers.csv", index=False)
    pd.DataFrame(corr).to_csv(OUT / "aqi_correlation.csv", index=False)
    pd.DataFrame(week).to_csv(OUT / "weekday_vs_weekend.csv", index=False)
    pd.DataFrame(hours).to_csv(OUT / "hourly_profile.csv", index=False)
    pd.concat(months).round(2).to_csv(OUT / "monthly.csv", index=False)
    pd.DataFrame(hol).to_csv(OUT / "holidays.csv", index=False)
    print("\nStudy results saved to results/study/")


if __name__ == "__main__":
    main()
