import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/processed/features/all_cities_features.csv"
OUTPUT_DIR = "results/models"

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

results = []

for city in sorted(df["city"].unique()):
    city_df = (
        df[df["city"] == city]
        .sort_values("period_start")
        .reset_index(drop=True)
    )

    # Chronological split: first 70% train, next 15% validation,
    # final 15% test.
    n = len(city_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = city_df.iloc[:train_end]
    val_df = city_df.iloc[train_end:val_end]
    test_df = city_df.iloc[val_end:]

    print("\n" + "=" * 60)
    print(f"CITY: {city}")
    print("=" * 60)
    print(f"Training rows:   {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows:       {len(test_df)}")

    for horizon in HORIZONS:
        target = f"target_{horizon}h"

        X_train = train_df[FEATURES]
        y_train = train_df[target]

        X_val = val_df[FEATURES]
        y_val = val_df[target]

        X_test = test_df[FEATURES]
        y_test = test_df[target]

        # Scale features using training data only.
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Model 1: Linear Regression trained with Gradient Descent.
        linear_model = SGDRegressor(
            loss="squared_error",
            penalty=None,
            max_iter=3000,
            tol=1e-5,
            random_state=42
        )

        linear_model.fit(X_train_scaled, y_train)

        linear_val_pred = linear_model.predict(X_val_scaled)
        linear_val_mae = mean_absolute_error(
            y_val, linear_val_pred
        )

        # Model 2: Feedforward Neural Network.
        nn_model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42
        )

        nn_model.fit(X_train_scaled, y_train)

        nn_val_pred = nn_model.predict(X_val_scaled)
        nn_val_mae = mean_absolute_error(y_val, nn_val_pred)

        # Select the model using validation MAE only.
        if linear_val_mae <= nn_val_mae:
            selected_name = "Linear Regression (SGD)"
            selected_model = linear_model
        else:
            selected_name = "Feedforward Neural Network"
            selected_model = nn_model

        # Evaluate both models on the untouched test set.
        linear_test_pred = linear_model.predict(X_test_scaled)
        nn_test_pred = nn_model.predict(X_test_scaled)

        selected_test_pred = selected_model.predict(X_test_scaled)

        for model_name, predictions in [
            ("Linear Regression (SGD)", linear_test_pred),
            ("Feedforward Neural Network", nn_test_pred)
        ]:
            results.append({
                "city": city,
                "horizon_hours": horizon,
                "model": model_name,
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": np.sqrt(
                    mean_squared_error(y_test, predictions)
                ),
                "validation_mae": (
                    linear_val_mae
                    if model_name == "Linear Regression (SGD)"
                    else nn_val_mae
                ),
                "selected_by_validation": (
                    model_name == selected_name
                )
            })

        print(f"\nForecast horizon: {horizon} hours")
        print(f"Linear validation MAE: {linear_val_mae:.2f}")
        print(f"Neural network validation MAE: {nn_val_mae:.2f}")
        print(f"Selected using validation: {selected_name}")
        print(
            "Selected model test MAE:",
            f"{mean_absolute_error(y_test, selected_test_pred):.2f}"
        )
        print(
            "Selected model test RMSE:",
            f"{np.sqrt(mean_squared_error(y_test, selected_test_pred)):.2f}"
        )

results_df = pd.DataFrame(results)

results_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETE")
print("=" * 60)

print("\nOverall results:")
print(
    results_df.groupby(["model", "horizon_hours"])[
        ["mae", "rmse"]
    ].mean().round(2)
)

print("\nResults saved to:", results_path)