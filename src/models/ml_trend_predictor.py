import pandas as pd
import joblib
import os
from functools import lru_cache


# -----------------------------
# Cached model loader
# -----------------------------
@lru_cache(maxsize=1)
def load_model():
    print("Loading trained model...")
    return joblib.load("models/trend_classifier.pkl")


def run_ml_trend_prediction():

    print("Loading engineered dataset...")

    df = pd.read_csv("data/processed_data/fashion_trend_engineered.csv")

    # Use cached model
    model = load_model()

    # Columns removed during training (to avoid leakage)
    drop_cols = [
        "trend_label",
        "next_week_search",
        "search_index",
        "search_growth",
        "momentum_4w"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print("Generating predictions...")

    predictions = model.predict(X)

    df["predicted_trend"] = predictions

    # Mapping trend numbers to readable labels
    trend_map = {
        0: "Declining",
        1: "Stable",
        2: "Rising"
    }

    category_columns = [
        "category_Jeans",
        "category_Kurti",
        "category_Saree",
        "category_Streetwear",
        "category_TShirt"
    ]

    category_results = {}

    for cat in category_columns:

        cat_rows = df[df[cat] == 1]

        if len(cat_rows) > 0:
            latest_prediction = cat_rows.iloc[-1]["predicted_trend"]
            category_results[cat.replace("category_", "")] = trend_map.get(latest_prediction, "Unknown")

    report_lines = []

    def log(line=""):
        print(line)
        report_lines.append(line)

    log("\nFashion Trend Forecast (Next Week)")
    log("----------------------------------\n")

    for cat, trend in category_results.items():
        log(f"{cat:12} → {trend}")

    output_path = "reports/ml_trend_prediction.txt"

    os.makedirs("reports", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")

    log(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    run_ml_trend_prediction()