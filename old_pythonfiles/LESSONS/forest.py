#!/usr/bin/env python3
import sys
import json
import urllib.request
import argparse
from pathlib import Path

import pandas as pd
from joblib import load

def main():
    # ─── Locate the script’s directory ───────────────────────────────────────
    BASE = Path(__file__).parent.resolve()

    # ─── Parse command‑line arguments ────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Fetch road weather data and predict driving conditions"
    )
    parser.add_argument(
        "-m", "--model",
        type=Path,
        default=BASE / "weather_decision_model.joblib",
        help=(
            "Path to weather_decision_model.joblib "
            f"[default: {BASE/'weather_decision_model.joblib'}]"
        )
    )
    args = parser.parse_args()

    # ─── Verify model file exists ────────────────────────────────────────────
    model_path = args.model.resolve()
    if not model_path.exists():
        sys.exit(f"ERROR: Model file not found: {model_path}")

    # ─── Load trained RandomForest model ───────────────────────────────
    model = load(str(model_path))

    # ─── Class labels in the order model was trained on ───────────────
    labels = ["Ice", "Normal", "Rain", "Snow"]

    # ─── Fetch the latest data from the Frostbit API ─────────────────────────
    API_URL = "https://edu.frostbit.fi/api/road_weather/2025/"
    try:
        with urllib.request.urlopen(API_URL) as resp:
            data = json.load(resp)
    except Exception as e:
        sys.exit(f"ERROR fetching API data: {e}")

    if not data:
        sys.exit("No data returned from the API.")

    # ─── Normalize JSON into a pandas DataFrame ─────────────────────────────
    records = data if isinstance(data, list) else [data]
    df = pd.DataFrame(records)

    # ─── Ensure all features model expects are present ─────────────────
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = df.columns.tolist()

    missing = [f for f in expected if f not in df.columns]
    if missing:
        print(f"WARNING: missing expected features: {missing}")
    X = df.reindex(columns=expected).fillna(0)

    # ─── Predict probabilities & class ──────────────────────────────────────
    proba = model.predict_proba(X)[0]
    preds = model.predict(X)[0]

    # ─── Print out each probability ─────────────────────────────────────────
    print("\nAll probabilities by category:")
    for idx, p in enumerate(proba):
        print(f" - {labels[idx]:<6}: {p*100:5.2f}%")

    result = labels[preds]
    print(f"\nCurrent weather condition: {result}")
    print("-------------------")

    # ─── Driving advice based on prediction ─────────────────────────────────
    advice = {
        "Normal": "Normal weather, drive with normal speed!",
        "Rain":   "Raining. Reduce speed by 20%.",
        "Snow":   "Snowing. Reduce speed by 40%.",
        "Ice":    "Icy! Reduce speed by 60%!",
    }
    print(advice.get(result, "No advice available."))

if __name__ == "__main__":
    main()
