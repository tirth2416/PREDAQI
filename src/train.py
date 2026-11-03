"""
PREDAQI - Step 3: Train and compare forecasting models.

For every city and every pollutant we compare:
  1. Persistence baseline  - "the next 48 hours = the value right now"
  2. Linear Regression trained with gradient descent (SGDRegressor)
  3. Feedforward Neural Network (MLPRegressor, trained by backpropagation)
Models 2 and 3 are each tried with and without PCA (keeping 95% variance).

The data is split by time (never shuffled):
  first 70% -> training, next 15% -> validation, last 15% -> test,
with a 48-hour gap between the parts so no future value leaks backwards.

PCA on/off is chosen using the validation set. The test set is used
once at the end to report accuracy. The saved model is exactly the
model that was tested.

Outputs:
  models/<city>_<pollutant>_<sgd|mlp>.joblib
  results/model_accuracy_by_hour.csv  (MAE for every forecast hour)
  results/model_summary.csv           (one line per city/pollutant/model)
  results/model_weights.csv           (what the linear model looks at)
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CITIES, HORIZON, MODEL_DIR, PROCESSED_DIR, RESULTS_DIR
from features import from_change, make_dataset, pollutants_for_city, to_change

warnings.filterwarnings("ignore")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GAP_HOURS = 48
MODEL_NAMES = {"sgd": "Linear Regression (SGD)", "mlp": "Neural Network (MLP)"}
BASELINE = "Persistence baseline"


def build_model(kind, use_pca):
    steps = [("scale", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=42)))

    if kind == "sgd":
        # One gradient-descent linear model for each of the 48 hours
        model = MultiOutputRegressor(
            SGDRegressor(
                penalty="l2",
                alpha=0.001,
                max_iter=2000,
                tol=1e-5,
                random_state=42,
            )
        )
    else:
        # One neural network that outputs all 48 hours at once
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.01,
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )
    steps.append(("model", model))
    return Pipeline(steps)


def split_by_time(index):
    """70% train / 15% validation / 15% test, with 48-hour gaps."""
    val_start = index[int(len(index) * 0.70)]
    test_start = index[int(len(index) * 0.85)]
    gap = pd.Timedelta(hours=GAP_HOURS)

    train = index < val_start - gap
    val = (index >= val_start) & (index < test_start - gap)
    test = index >= test_start
    return train, val, test


def mae_per_hour(actual, predicted):
    """Mean absolute error for each of the 48 forecast hours."""
    return np.abs(np.asarray(actual) - np.asarray(predicted)).mean(axis=0)


def linear_weights(pipeline, feature_names):
    """
    Weight of each input in the linear model (inputs are standardised,
    so weights can be compared). If PCA was used, the weights are
    mapped back from the components to the original inputs.
    Returns a table with 48 rows (hours ahead) and one column per input.
    """
    estimators = pipeline.named_steps["model"].estimators_
    coefs = np.array([est.coef_ for est in estimators])
    if "pca" in pipeline.named_steps:
        coefs = coefs @ pipeline.named_steps["pca"].components_
    return pd.DataFrame(coefs, columns=feature_names)


def summary_row(city, pollutant, model, pca, val_mae, mae):
    return {
        "city": city, "pollutant": pollutant, "model": model, "pca": pca,
        "val_mae": val_mae, "test_mae": mae.mean(),
        "test_mae_1_6h": mae[:6].mean(), "test_mae_24h": mae[23],
        "test_mae_48h": mae[47],
    }


def train_one(city, pollutant, table):
    X, y, now = make_dataset(table, pollutant)
    if len(X) < 1000:
        print(f"  {pollutant:5s} only {len(X)} usable rows, skipped")
        return [], [], []

    train, val, test = split_by_time(X.index)
    hours = np.arange(1, HORIZON + 1)
    by_hour, summary, weights = [], [], []

    # 1. Persistence baseline: repeat the current value for 48 hours
    persistence = np.repeat(now[test].values.reshape(-1, 1), HORIZON, axis=1)
    base_mae = mae_per_hour(y[test], persistence)
    by_hour += [[city, pollutant, BASELINE, h, m] for h, m in zip(hours, base_mae)]
    summary.append(summary_row(city, pollutant, BASELINE, "-", np.nan, base_mae))
    line = (f"  {pollutant:5s} rows {train.sum()}/{val.sum()}/{test.sum()}"
            f" | baseline {base_mae.mean():6.2f}")

    # 2 and 3. Linear Regression (SGD) and Neural Network, with/without PCA
    for kind in ["sgd", "mlp"]:
        best = None
        for use_pca in [False, True]:
            model = build_model(kind, use_pca)
            model.fit(X[train], to_change(y[train], now[train]))
            val_pred = from_change(model.predict(X[val]), now[val])
            val_mae = mae_per_hour(y[val], val_pred).mean()
            if best is None or val_mae < best[2]:
                best = (model, use_pca, val_mae)

        model, use_pca, val_mae = best
        test_pred = from_change(model.predict(X[test]), now[test])
        test_mae = mae_per_hour(y[test], test_pred)

        pca_label = "No PCA"
        if use_pca:
            n = model.named_steps["pca"].n_components_
            pca_label = f"PCA 95% ({n} of {X.shape[1]} inputs)"

        by_hour += [[city, pollutant, MODEL_NAMES[kind], h, m] for h, m in zip(hours, test_mae)]
        summary.append(summary_row(city, pollutant, MODEL_NAMES[kind], pca_label, val_mae, test_mae))

        joblib.dump(
            {
                "model": model,
                "features": list(X.columns),
                "city": city,
                "pollutant": pollutant,
                "kind": kind,
                "model_name": MODEL_NAMES[kind],
                "pca": pca_label,
                "val_mae": val_mae,
                "test_mae": test_mae.mean(),
                "test_mae_by_hour": test_mae,
                "baseline_mae": base_mae.mean(),
                "train_end": X.index[train].max(),
                "val_start": X.index[val].min(),
                "test_start": X.index[test].min(),
            },
            MODEL_DIR / f"{city.lower()}_{pollutant}_{kind}.joblib",
            compress=3,
        )

        if kind == "sgd":
            w = linear_weights(model, list(X.columns))
            for h in [1, 6, 24, 48]:
                for feature, value in w.iloc[h - 1].items():
                    weights.append([city, pollutant, h, feature, value])

        line += f" | {kind} {test_mae.mean():6.2f} ({'PCA' if use_pca else 'no PCA'})"

    print(line)
    return by_hour, summary, weights


def main():
    all_by_hour, all_summary, all_weights = [], [], []

    for city in CITIES:
        path = PROCESSED_DIR / f"{city.lower()}_hourly.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path, index_col=0, parse_dates=True)
        pollutants = pollutants_for_city(table)
        table = table[pollutants]
        print(f"\n{city}: modelling {pollutants}")

        for pollutant in pollutants:
            by_hour, summary, weights = train_one(city, pollutant, table)
            all_by_hour += by_hour
            all_summary += summary
            all_weights += weights

    pd.DataFrame(
        all_by_hour, columns=["city", "pollutant", "model", "hours_ahead", "mae"]
    ).to_csv(RESULTS_DIR / "model_accuracy_by_hour.csv", index=False)

    summary = pd.DataFrame(all_summary).round(3)
    summary.to_csv(RESULTS_DIR / "model_summary.csv", index=False)

    pd.DataFrame(
        all_weights, columns=["city", "pollutant", "hours_ahead", "feature", "weight"]
    ).round(4).to_csv(RESULTS_DIR / "model_weights.csv", index=False)

    print("\nAverage test MAE over all 48 hours (lower is better):")
    table = summary.pivot_table(index=["city", "pollutant"], columns="model", values="test_mae")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
