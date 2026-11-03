"""
PREDAQI - turn an hourly table into model inputs (X) and targets.

Key idea: the models predict how much a pollutant will CHANGE,
not its raw level.

Why: our data runs from winter into the monsoon, so pollution levels
in the test months are much lower than in the training months.
A model that learned "PM2.5 is usually about 60" would be wrong in
August. But the way pollution rises and falls through the day is
similar all year. So all inputs and targets are written relative to
the current value, on a log scale (log changes are percentage changes):

  input  lag_6h   = log(value 6 hours ago) - log(value now)
  input  avg_24h  = log(24-hour average)   - log(value now)
  input  no2_vs_avg = log(NO2 now) - log(NO2 24-hour average)   (other pollutants)
  input  hour_sin, hour_cos, weekend
  target h24      = log(value in 24 hours) - log(value now)

Forecast = value now x exp(predicted change).
One model predicts all 48 future hours at once.
"""

import numpy as np
import pandas as pd

from config import HORIZON, LAGS


def log(values):
    """log(1 + x), safe for zero values."""
    return np.log1p(values.clip(lower=0))


def make_inputs(table, pollutant):
    """Input features for every hour in the table (same index)."""
    series = table[pollutant]
    now = log(series)
    X = pd.DataFrame(index=table.index)

    for lag in LAGS:
        X[f"lag_{lag}h"] = log(series.shift(lag)) - now

    X["avg_6h"] = log(series.rolling(6).mean()) - now
    X["avg_24h"] = log(series.rolling(24).mean()) - now

    # Is each other pollutant currently above or below its own daily average?
    for other in table.columns:
        if other != pollutant:
            X[f"{other}_vs_avg"] = log(table[other]) - log(table[other].rolling(24).mean())

    hour = table.index.hour
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["weekend"] = (table.index.dayofweek >= 5).astype(int)
    return X


def make_targets(table, pollutant):
    """Actual values 1..48 hours ahead (columns 'h1' ... 'h48')."""
    series = table[pollutant]
    return pd.DataFrame(
        {f"h{h}": series.shift(-h) for h in range(1, HORIZON + 1)},
        index=table.index,
    )


def to_change(y, now):
    """Turn future values into log-changes from the current value."""
    return log(y).sub(log(now), axis=0)


def from_change(predicted_change, now):
    """Turn predicted log-changes back into concentrations."""
    now_log = log(pd.Series(now)).values.reshape(-1, 1)
    return np.expm1(predicted_change + now_log).clip(min=0)


def make_dataset(table, pollutant):
    """
    Rows where every input and all 48 targets are known.
    Returns X (inputs), y (future values) and now (current value).
    """
    X = make_inputs(table, pollutant)
    y = make_targets(table, pollutant)
    now = table[pollutant]
    complete = X.notna().all(axis=1) & y.notna().all(axis=1) & now.notna()
    return X[complete], y[complete], now[complete]


def pollutants_for_city(table, min_coverage=0.6):
    """Pollutants with enough data (at least 60% of hours) to model."""
    coverage = table.notna().mean()
    return [p for p in table.columns if coverage[p] >= min_coverage]
