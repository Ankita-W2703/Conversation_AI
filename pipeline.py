# pipeline.py

import json
import csv
import time
from pathlib import Path
from tqdm import tqdm

from qa_generation import generate_answer1
from evaluation import full_evaluation
from report import generate_pdf_report, generate_html_report

DATA_DIR = Path("data")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def run_pipeline():
    print("Starting Hybrid RAG Pipeline")

    # -------------------------
    # Load QA dataset
    # -------------------------
    with open(DATA_DIR / "qa_dataset_100.json", "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)

    print(f"Loaded {len(qa_dataset)} QA pairs")

    predictions = []

    # -------------------------
    # Inference phase (ONCE)
    # -------------------------
    for qa in tqdm(qa_dataset, desc="Generating answers"):
        start = time.time()
        pred, context = generate_answer1(qa["question"])
        latency = time.time() - start

        predictions.append({
            "question_id": qa["question_id"],
            "question": qa["question"],
            "prediction": pred,
            "ground_truth": qa["answer"],
            "source_url": qa["source_url"],
            "question_type": qa["question_type"],
            "latency_sec": round(latency, 3)
        })

    # -------------------------
    # Save predictions CSV
    # -------------------------
    csv_path = REPORT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Predictions saved → {csv_path}")

    # -------------------------
    # Evaluation phase (NO GENERATION)
    # -------------------------
    metrics = full_evaluation(qa_dataset)

    json_path = REPORT_DIR / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved → {json_path}")

    # -------------------------
    # Reports
    # -------------------------
    generate_pdf_report(metrics, REPORT_DIR / "report.pdf")
    generate_html_report(metrics, REPORT_DIR / "report.html")

    print("Reports generated")
    print(f"PDF  → reports/report.pdf")
    print(f"HTML → reports/report.html")
    print("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()
