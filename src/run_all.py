"""
PREDAQI - run the whole pipeline after downloading the data.

    python src/run_all.py

Steps: clean data -> train models -> check AQI accuracy -> pollutant study
"""

import evaluate_aqi
import prepare_data
import study
import train

print("=" * 60, "\n1. Preparing data\n" + "=" * 60)
prepare_data.main()

print("\n" + "=" * 60, "\n2. Training models\n" + "=" * 60)
train.main()

print("\n" + "=" * 60, "\n3. Checking AQI forecast accuracy\n" + "=" * 60)
evaluate_aqi.main()

print("\n" + "=" * 60, "\n4. Pollutant study\n" + "=" * 60)
study.main()

print("\nAll done. Start the dashboard with:  streamlit run app.py")
