import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.decomposition import PCA
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "data" / "processed" / "features"
RESULT_DIR = ROOT / "results" / "final"
MODEL_DIR = RESULT_DIR / "models"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 49))
PCA_OPTIONS = [None, 0.95, 0.99]
PURGE_HOURS = 48


def build_model(model_name, pca_variance):
    steps = [("scaler", StandardScaler())]

    if pca_variance is not None:
        steps.append(
            ("pca", PCA(n_components=pca_variance, random_state=42))
        )

    if model_name == "Linear Regression (SGD)":
        estimator = SGDRegressor(
            loss="squared_error",
            penalty=None,
            max_iter=3000,
            tol=1e-5,
            random_state=42
        )
    else:
        estimator = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=False,
            random_state=42
        )

    steps.append(("model", estimator))

    return Pipeline(steps)


def get_feature_columns(df, horizon):
    excluded = {
        "city",
        "period_start",
        "timestamp",
        "run_id",
        "segment_id",
        "gap_id"
    }

    feature_columns = []

    for column in df.columns:
        if column in excluded:
            continue

        if column.startswith("target_"):
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)

    target_time_columns = [
        f"target_hour_sin_{horizon}h",
        f"target_hour_cos_{horizon}h",
        f"target_day_of_week_{horizon}h",
        f"target_month_{horizon}h"
    ]

    for column in target_time_columns:
        if column in df.columns:
            feature_columns.append(column)

    return feature_columns


def evaluate_city_horizon(df, city, horizon):
    target_column = f"target_{horizon}h"

    if target_column not in df.columns:
        print(f"Skipping {city} {horizon}h: target missing.")
        return []

    feature_columns = get_feature_columns(df, horizon)

    persistence_column = "pm25_lag_1"

    if persistence_column not in feature_columns:
        print(
            f"Skipping {city} {horizon}h: "
            "pm25_lag_1 missing."
        )
        return []

    data = df.dropna(
        subset=feature_columns + [target_column]
    ).copy()

    data = data.sort_values(
        "period_start"
    ).reset_index(drop=True)

    if len(data) < 200:
        print(
            f"Skipping {city} {horizon}h: "
            f"insufficient rows ({len(data)})."
        )
        return []

    train_cut_index = int(len(data) * 0.70)
    test_cut_index = int(len(data) * 0.85)

    validation_start = data.loc[
        train_cut_index, "period_start"
    ]

    test_start = data.loc[
        test_cut_index, "period_start"
    ]

    train_end = validation_start - pd.Timedelta(
        hours=PURGE_HOURS
    )

    validation_end = test_start - pd.Timedelta(
        hours=PURGE_HOURS
    )

    train = data[
        data["period_start"] < train_end
    ]

    validation = data[
        (data["period_start"] >= validation_start)
        & (data["period_start"] < validation_end)
    ]

    test = data[
        data["period_start"] >= test_start
    ]

    if min(len(train), len(validation), len(test)) < 50:
        print(
            f"Skipping {city} {horizon}h: "
            f"split too small "
            f"(train={len(train)}, "
            f"val={len(validation)}, "
            f"test={len(test)})."
        )
        return []

    X_train = train[feature_columns]
    y_train = train[target_column]

    X_val = validation[feature_columns]
    y_val = validation[target_column]

    X_test = test[feature_columns]
    y_test = test[target_column]

    results = []

    print(
        f"\n{city} | Horizon: {horizon}h | "
        f"Train: {len(train)}, "
        f"Validation: {len(validation)}, "
        f"Test: {len(test)}"
    )

    # -----------------------------------------
    # Persistence baseline
    # -----------------------------------------

    baseline_predictions = test[persistence_column]

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_predictions
        )
    )

    results.append({
        "city": city,
        "horizon_hours": horizon,
        "model": "Persistence Baseline",
        "selected_pca": "Not Applicable",
        "validation_mae": np.nan,
        "test_mae": baseline_mae,
        "test_rmse": baseline_rmse,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test)
    })

    print(
        f"Persistence Baseline | "
        f"MAE: {baseline_mae:.2f} | "
        f"RMSE: {baseline_rmse:.2f}"
    )

    # -----------------------------------------
    # SGD and MLP model comparison
    # -----------------------------------------

    model_names = [
        "Linear Regression (SGD)",
        "Feedforward Neural Network"
    ]

    for model_name in model_names:
        candidates = []

        for pca_variance in PCA_OPTIONS:
            model = build_model(
                model_name,
                pca_variance
            )

            try:
                model.fit(
                    X_train,
                    y_train
                )

                validation_predictions = model.predict(
                    X_val
                )

                val_mae = mean_absolute_error(
                    y_val,
                    validation_predictions
                )

                candidates.append({
                    "model": model,
                    "pca_variance": pca_variance,
                    "validation_mae": val_mae
                })

            except Exception as error:
                print(
                    f"Candidate failed: {model_name}, "
                    f"PCA={pca_variance}: {error}"
                )

        if not candidates:
            print(f"No successful candidates for {model_name}.")
            continue

        best_candidate = min(
            candidates,
            key=lambda item: item["validation_mae"]
        )

        best_model = best_candidate["model"]

        test_predictions = best_model.predict(
            X_test
        )

        test_mae = mean_absolute_error(
            y_test,
            test_predictions
        )

        test_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                test_predictions
            )
        )

        pca_choice = best_candidate["pca_variance"]

        pca_label = (
            "No PCA"
            if pca_choice is None
            else f"PCA {pca_choice:.0%}"
        )

        results.append({
            "city": city,
            "horizon_hours": horizon,
            "model": model_name,
            "selected_pca": pca_label,
            "validation_mae": best_candidate["validation_mae"],
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test)
        })

        print(
            f"{model_name} | "
            f"{pca_label} | "
            f"Validation MAE: "
            f"{best_candidate['validation_mae']:.2f} | "
            f"Test MAE: {test_mae:.2f} | "
            f"Test RMSE: {test_rmse:.2f}"
        )

        # -----------------------------------------
        # Save deployment-ready model
        # -----------------------------------------

        deployment_data = pd.concat(
            [train, validation],
            axis=0
        ).sort_values("period_start")

        deployment_model = build_model(
            model_name,
            pca_choice
        )

        deployment_model.fit(
            deployment_data[feature_columns],
            deployment_data[target_column]
        )

        safe_city = "".join(
            character
            if character.isalnum() or character in ("-", "_")
            else "_"
            for character in str(city)
        )

        safe_model = (
            "sgd"
            if model_name == "Linear Regression (SGD)"
            else "mlp"
        )

        artifact_path = (
            MODEL_DIR
            / f"{safe_city}_{safe_model}_{horizon}h.joblib"
        )

        artifact = {
            "model": deployment_model,
            "city": city,
            "model_name": model_name,
            "horizon_hours": horizon,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "selected_pca": pca_choice,
            "pca_label": pca_label,
            "validation_mae": best_candidate["validation_mae"],
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "trained_through": deployment_data["period_start"].max(),
            "persistence_column": persistence_column
        }

        joblib.dump(
            artifact,
            artifact_path
        )

        print(f"Saved deployment model: {artifact_path}")

    return results


def main():
    files = sorted(
        file
        for file in FEATURE_DIR.glob("*_features.csv")
        if file.name != "all_cities_features.csv"
    )

    if not files:
        print("No individual city feature files found.")
        return

    all_results = []

    for file in files:
        df = pd.read_csv(file)

        if "period_start" not in df.columns:
            print(
                f"Skipping {file.name}: "
                "period_start column missing."
            )
            continue

        df["period_start"] = pd.to_datetime(
            df["period_start"]
        )

        city = (
            df["city"].iloc[0]
            if "city" in df.columns
            else file.stem.replace("_features", "").capitalize()
        )

        print("\n" + "=" * 60)
        print(f"Processing {city}")
        print("=" * 60)

        for horizon in HORIZONS:
            results = evaluate_city_horizon(
                df,
                city,
                horizon
            )

            all_results.extend(results)

    if not all_results:
        print("No results generated.")
        return

    results_df = pd.DataFrame(all_results)

    results_path = (
        RESULT_DIR
        / "hourly_model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    summary = (
        results_df.groupby(
            ["model", "horizon_hours"]
        )
        .agg(
            average_test_mae=("test_mae", "mean"),
            average_test_rmse=("test_rmse", "mean")
        )
        .reset_index()
    )

    summary_path = (
        RESULT_DIR
        / "hourly_model_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("HOURLY FORECAST MODEL SUMMARY")
    print("=" * 60)

    print(
        summary.to_string(
            index=False
        )
    )

    print(f"\nDetailed results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Saved model artifacts in: {MODEL_DIR}")


if __name__ == "__main__":
    main()