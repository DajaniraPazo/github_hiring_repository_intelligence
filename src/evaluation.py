"""
Model evaluation, error analysis, and baseline comparison.

Loads the best available model (DistilBERT > sklearn) and computes:
  - Accuracy, Precision, Recall, F1 (macro and weighted)
  - Per-class breakdown
  - Confusion matrix
  - Top misclassification pairs (error analysis)
  - Baseline vs. alternative comparison
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import (
    setup_logging, SPLITS_DIR, MODELS_DIR, METRICS_DIR,
    CATEGORIES, ID2LABEL, LABEL2ID,
)

logger = setup_logging("evaluation")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def predict_bert(texts: list, model_dir: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    preds = []
    for i in range(0, len(texts), 16):
        batch = texts[i: i + 16]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        preds.extend(out.logits.argmax(dim=-1).tolist())
    return np.array(preds)


def predict_sklearn(texts: list, model_path: str) -> np.ndarray:
    import joblib
    pipe = joblib.load(model_path)
    return np.array(pipe.predict(texts))


def _load_best_model(texts: list) -> Tuple[Optional[np.ndarray], str]:
    bert_dir = MODELS_DIR / "bert_classifier"
    sklearn_path = MODELS_DIR / "sklearn_baseline.pkl"

    if TORCH_AVAILABLE and bert_dir.exists():
        try:
            return predict_bert(texts, str(bert_dir)), "DistilBERT"
        except Exception as e:
            logger.warning(f"BERT prediction failed: {e}")

    if sklearn_path.exists():
        try:
            return predict_sklearn(texts, str(sklearn_path)), "TF-IDF + Logistic Regression"
        except Exception as e:
            logger.warning(f"Sklearn prediction failed: {e}")

    return None, "none"


def compute_metrics_dict(true_labels, predictions) -> dict:
    labs = list(range(len(CATEGORIES)))
    pc_f1 = f1_score(true_labels, predictions, average=None, zero_division=0, labels=labs)
    pc_prec = precision_score(true_labels, predictions, average=None, zero_division=0, labels=labs)
    pc_rec = recall_score(true_labels, predictions, average=None, zero_division=0, labels=labs)
    cm = confusion_matrix(true_labels, predictions, labels=labs)

    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision_macro": float(precision_score(true_labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true_labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true_labels, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(true_labels, predictions, average="weighted", zero_division=0)),
        "per_class": {
            CATEGORIES[i]: {
                "precision": float(pc_prec[i]),
                "recall": float(pc_rec[i]),
                "f1": float(pc_f1[i]),
            }
            for i in range(len(CATEGORIES))
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            true_labels, predictions, target_names=CATEGORIES, zero_division=0
        ),
    }


def evaluate_model(test_df: pd.DataFrame = None) -> Tuple[dict, pd.DataFrame]:
    if test_df is None:
        test_path = SPLITS_DIR / "test.csv"
        if not test_path.exists():
            logger.error("No test split found — run training pipeline first.")
            return {}, pd.DataFrame()
        test_df = pd.read_csv(test_path)

    texts = test_df["text_summary"].fillna("").tolist()
    true_labels = test_df["label_id"].astype(int).values

    predictions, model_name = _load_best_model(texts)
    if predictions is None:
        logger.error("No trained model found.")
        return {}, test_df

    metrics = compute_metrics_dict(true_labels, predictions)
    metrics["model_used"] = model_name
    metrics["total_samples"] = len(true_labels)

    # Error analysis
    test_df = test_df.copy()
    test_df["predicted_id"] = predictions
    test_df["predicted_label"] = test_df["predicted_id"].map(ID2LABEL)
    test_df["is_correct"] = (test_df["label_id"] == test_df["predicted_id"]).astype(int)

    errors = test_df[test_df["is_correct"] == 0]
    if len(errors):
        top_err = (
            errors.groupby(["weak_label", "predicted_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
        metrics["top_errors"] = top_err.to_dict("records")
    else:
        metrics["top_errors"] = []

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(METRICS_DIR / "evaluation_results.json", "w") as f:
        json.dump(save_metrics, f, indent=2)
    with open(METRICS_DIR / "classification_report.txt", "w") as f:
        f.write(metrics["classification_report"])

    logger.info(f"Model: {model_name}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}  F1-macro: {metrics['f1_macro']:.4f}")
    logger.info(f"\n{metrics['classification_report']}")

    return metrics, test_df


def run_baseline_comparison(test_df: pd.DataFrame) -> dict:
    """Compare primary model against a majority-class baseline."""
    from collections import Counter
    true_labels = test_df["label_id"].astype(int).values

    # Majority class baseline
    majority = Counter(true_labels).most_common(1)[0][0]
    majority_preds = np.full_like(true_labels, majority)
    majority_metrics = compute_metrics_dict(true_labels, majority_preds)
    majority_metrics["model_used"] = "Majority Class Baseline"
    majority_metrics["total_samples"] = len(true_labels)

    # Load trained model predictions
    texts = test_df["text_summary"].fillna("").tolist()
    preds, model_name = _load_best_model(texts)
    if preds is not None:
        model_metrics = compute_metrics_dict(true_labels, preds)
        model_metrics["model_used"] = model_name
    else:
        model_metrics = {}

    comparison = {"baseline": majority_metrics, "model": model_metrics}
    with open(METRICS_DIR / "baseline_comparison.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("confusion_matrix", "classification_report")}
                   for k, v in comparison.items()}, f, indent=2)
    logger.info("Baseline comparison saved.")
    return comparison


if __name__ == "__main__":
    metrics, test_df = evaluate_model()
    if metrics and len(test_df):
        run_baseline_comparison(test_df)
