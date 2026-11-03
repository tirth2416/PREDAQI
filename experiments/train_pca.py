
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/processed/features/all_cities_features.csv"
OUTPUT_DIR = "results/pca"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["period_start"] = pd.to_datetime(df["period_start"])

LAGS = [1, 2, 3, 6, 12, 24, 48]
FEATURES = (
    [f"pm25_lag_{lag}" for lag in LAGS]
    + [
        "rolling_mean_3h",
        "rolling_mean_6h",
        "rolling_mean_24h",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "month",
    ]
)

HORIZONS = [6, 12, 24, 48]
PCA_COMPONENTS = [0.90, 0.95, 0.99]

results = []

def make_model(model_name):
    if model_name == "Linear Regression (SGD)":
        return SGDRegressor(
            loss="squared_error",
            penalty=None,
            max_iter=3000,
            tol=1e-5,
            random_state=42
        )

    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42
    )


for city in sorted(df["city"].unique()):
    city_df = (
        df[df["city"] == city]
        .sort_values("period_start")
        .reset_index(drop=True)
    )

    n = len(city_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = city_df.iloc[:train_end]
    val_df = city_df.iloc[train_end:val_end]
    test_df = city_df.iloc[val_end:]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[FEATURES])
    X_val = scaler.transform(val_df[FEATURES])
    X_test = scaler.transform(test_df[FEATURES])

    print("\n" + "=" * 55)
    print("CITY:", city)
    print("=" * 55)

    for horizon in HORIZONS:
        target = f"target_{horizon}h"

        y_train = train_df[target].values
        y_val = val_df[target].values
        y_test = test_df[target].values

        for model_name in [
            "Linear Regression (SGD)",
            "Feedforward Neural Network"
        ]:
            candidates = [("No PCA", None)]
            candidates += [
                (f"PCA {int(v * 100)}%", v)
                for v in PCA_COMPONENTS
            ]

            best = None

            for label, variance in candidates:
                if variance is None:
                    train_features = X_train
                    val_features = X_val
                    test_features = X_test
                    n_components = len(FEATURES)
                else:
                    pca = PCA(
                        n_components=variance,
                        svd_solver="full"
                    )

                    train_features = pca.fit_transform(X_train)
                    val_features = pca.transform(X_val)
                    test_features = pca.transform(X_test)
                    n_components = pca.n_components_

                model = make_model(model_name)

                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", ConvergenceWarning
                    )
                    model.fit(train_features, y_train)

                val_predictions = model.predict(val_features)
                val_mae = mean_absolute_error(
                    y_val, val_predictions
                )

                if best is None or val_mae < best["val_mae"]:
                    best = {
                        "configuration": label,
                        "components": n_components,
                        "val_mae": val_mae,
                        "model": model,
                        "test_features": test_features
                    }

            test_predictions = best["model"].predict(
                best["test_features"]
            )

            test_mae = mean_absolute_error(
                y_test, test_predictions
            )
            test_rmse = np.sqrt(
                mean_squared_error(y_test, test_predictions)
            )

            results.append({
                "city": city,
                "horizon_hours": horizon,
                "model": model_name,
                "configuration": best["configuration"],
                "components": best["components"],
                "validation_mae": best["val_mae"],
                "test_mae": test_mae,
                "test_rmse": test_rmse
            })

            print(
                f"{horizon}h | {model_name} | "
                f"{best['configuration']} | "
                f"Components: {best['components']} | "
                f"Val MAE: {best['val_mae']:.2f} | "
                f"Test MAE: {test_mae:.2f}"
            )


results_df = pd.DataFrame(results)

results_path = os.path.join(
    OUTPUT_DIR,
    "pca_comparison.csv"
)
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 55)
print("PCA EXPERIMENT COMPLETE")
print("=" * 55)

print("\nSelected configurations:")
print(
    results_df.groupby(
        ["model", "configuration"]
    ).size()
)

print("\nAverage test performance:")
print(
    results_df.groupby("model")[
        ["test_mae", "test_rmse"]
    ].mean().round(2)
)

print("\nResults saved to:", results_path)
