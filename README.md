# PREDAQI: Predictive Air Quality Intelligence

48-hour **Air Quality Index (AQI)** forecasting for five Indian cities (Delhi, Mumbai,
Bengaluru, Kolkata, Chennai), with a full pollutant breakdown and a study of what drives
the AQI. Built for the *Foundations of Machine Learning* course.

**Course concepts used:** Linear Regression trained with Gradient Descent, Feedforward
Neural Network (backpropagation), Principal Component Analysis (PCA), and proper model
evaluation (time-based train/validation/test split, baseline comparison).

## What it does

1. **Forecasts 6 pollutants** (PM2.5, PM10, NO₂, SO₂, CO, O₃) for each of the next 48 hours.
2. **Calculates the official CPCB AQI** from those forecasts: 24-hour / 8-hour averages →
   sub-index for each pollutant → AQI = highest sub-index → main pollutant.
3. **Explains the forecast:** which pollutant sets the AQI, which one pushes it up, and which
   inputs the linear model relies on.
4. **Pollutant study** on real data: dominant pollutants, causes of AQI rises, weekday vs
   weekend (traffic), daily cycles, seasons, festivals (including Diwali).
5. **Dashboard demo:** pick a city, date and hour, and see the 48-hour forecast next to what
   actually happened.

## Data

Hourly readings from CPCB monitoring stations via the
[India Air Quality Database](https://airquality.xkdr.org) (XKDR Forum), **January to
December 2024**. 2024 is used because the database has NO₂, SO₂, CO and Ozone only up to
31 Dec 2024, and the official AQI needs at least three pollutants.

Cleaning: impossible values are removed, stations are combined with the **median** (robust
to a faulty sensor), and gaps of up to 3 hours are interpolated. Longer gaps are never used
for training.

## Method

| Step | Detail |
|---|---|
| Inputs | Past values (1–24 h ago), 6 h / 24 h averages, other pollutants vs their daily average, time of day, weekend |
| Target | Change over the next 1…48 hours, on a log scale (≈ % change) |
| Models | Persistence baseline · Linear Regression (`SGDRegressor`) · Neural Network (`MLPRegressor` 64→32) |
| PCA | Each model is tried with and without PCA (95% variance); the validation set decides |
| Split | By time: 70% train / 15% validation / 15% test, with 48-hour gaps |

**Why predict the change instead of the level?** Pollution levels shift strongly between
seasons (winter vs monsoon), so a model trained on one season's levels is biased in another.
Hour-to-hour changes behave much more consistently, so predicting them generalises better.

All results (`results/`) are on the **test set**, which is never used for training or model
selection. The saved model is exactly the one that was tested.

## Project structure

```
app.py                 Streamlit dashboard
src/
  config.py            settings (cities, pollutants, paths)
  download_data.py     1. download raw data from the API (needs API key in .env)
  prepare_data.py      2. clean and build hourly city tables
  features.py             model inputs and targets
  train.py             3. train and compare models, save to models/
  aqi.py                  official CPCB AQI calculation
  forecast.py             48-hour forecast + AQI breakdown (used by the app)
  evaluate_aqi.py      4. AQI forecast accuracy on the test period
  study.py             5. pollutant study
  run_all.py              runs steps 2–5
data/processed/        cleaned hourly data per city
models/                trained models (.joblib)
results/               accuracy tables and study results
experiments/           earlier experiments (PM2.5-only, 2025 data)
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py   # only needed to re-download (needs .env with AIR_QUALITY_API_KEY)
python src/run_all.py         # prepare -> train -> evaluate -> study
streamlit run app.py
```

## Deploy on Render

The repo includes `render.yaml`. On [render.com](https://render.com): **New → Blueprint**,
then pick this repository. Or create a **Web Service** manually with:

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- Environment variable: `PYTHON_VERSION = 3.11.9`

## Limitations

- Complete pollutant data ends on 31 Dec 2024, so the demo forecasts dates within 2024.
- No weather data is used, so sudden changes from rain or wind are hard to predict.
- The study shows patterns that occur together (correlations), not proven causes.
