import subprocess
import sys


def run_step(description, command):

    print("\n" + "="*50)
    print(f"Running: {description}")
    print("="*50)

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"\nError occurred while running: {description}")
        sys.exit(1)


def run_pipeline():

    print("\nFASHION TREND PREDICTOR PIPELINE")
    print("--------------------------------")

    # Step 1 — Computer Vision Analysis
    run_step(
        "Computer Vision Trend Analyzer",
        "python -m src.vision.visual_trend_analyzer"
    )

    # Step 2 — NLP Trend Analysis
    run_step(
        "NLP Trend Analysis",
        "python -m src.nlp.nlp_pipeline"
    )

    # Step 3 — ML Trend Prediction
    run_step(
        "ML Trend Prediction",
        "python -m src.models.ml_trend_predictor"
    )

    print("\nPipeline completed successfully.")

    print("\nGenerated Reports:")
    print("reports/visual_trend_summary.txt")
    print("reports/nlp_trend_summary.txt")
    print("reports/ml_trend_prediction.txt")


if __name__ == "__main__":
    run_pipeline()