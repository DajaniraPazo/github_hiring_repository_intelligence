"""
Model training pipeline.

Primary:  DistilBERT fine-tuning via HuggingFace Transformers
Fallback: TF-IDF + Logistic Regression (sklearn) when PyTorch is unavailable

Input:  text_summary column from labeled dataset
Output: fine-tuned classifier in models/trained_models/

Training strategy:
  - 70/15/15 stratified split
  - AdamW optimizer with linear warmup (10% of steps)
  - Early stopping on val F1-macro (patience=2)
  - FP16 training when CUDA is available
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import (
    setup_logging, LABELED_DIR, SPLITS_DIR, MODELS_DIR, METRICS_DIR,
    CATEGORIES, LABEL2ID, ID2LABEL, load_config,
)

logger = setup_logging("train")

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch/Transformers not installed — will use sklearn fallback.")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ── Dataset class ────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class RepoDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=256):
            self.encodings = tokenizer(
                texts, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()} | {"labels": self.labels[idx]}


# ── Split ────────────────────────────────────────────────────────────────────

def split_dataset(
    input_path=None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_path = Path(input_path) if input_path else LABELED_DIR / "repos_labeled.csv"
    df = pd.read_csv(input_path)
    df = df[df["label_id"].notna() & (df["label_id"] >= 0)].copy()
    if "label_confidence" in df.columns:
        df = df[df["label_confidence"] >= 0.55].copy()
    logger.info(f"Samples after filtering: {len(df)}")

    y = df["label_id"].astype(int).values

    X_tr, X_tmp, y_tr, y_tmp, idx_tr, idx_tmp = train_test_split(
        df["text_summary"].values, y, df.index,
        test_size=test_size + val_size, random_state=seed, stratify=y,
    )
    rel_val = val_size / (test_size + val_size)
    X_v, X_te, y_v, y_te, idx_v, idx_te = train_test_split(
        X_tmp, y_tmp, idx_tmp,
        test_size=1 - rel_val, random_state=seed, stratify=y_tmp,
    )

    train_df = df.loc[idx_tr].reset_index(drop=True)
    val_df = df.loc[idx_v].reset_index(drop=True)
    test_df = df.loc[idx_te].reset_index(drop=True)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)
    logger.info(f"Split → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


# ── BERT training ─────────────────────────────────────────────────────────────

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def train_bert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
    output_dir: str = None,
) -> dict:
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch unavailable — switching to sklearn baseline.")
        return train_sklearn(train_df, val_df)

    out = output_dir or str(MODELS_DIR / "bert_classifier")
    logger.info(f"Loading {model_name} …")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(CATEGORIES), id2label=ID2LABEL, label2id=LABEL2ID,
    )

    tr_texts = train_df["text_summary"].fillna("").tolist()
    va_texts = val_df["text_summary"].fillna("").tolist()
    tr_labels = train_df["label_id"].astype(int).tolist()
    va_labels = val_df["label_id"].astype(int).tolist()

    tr_ds = RepoDataset(tr_texts, tr_labels, tokenizer, max_length)
    va_ds = RepoDataset(va_texts, va_labels, tokenizer, max_length)

    args = TrainingArguments(
        output_dir=out,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=50,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tr_ds, eval_dataset=va_ds,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Training DistilBERT …")
    trainer.train()
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    logger.info(f"Model saved → {out}")

    val_res = trainer.evaluate()
    logger.info(f"Val: {val_res}")
    return {"model_type": "bert", "model_name": model_name, "val_results": val_res, "output_dir": out}


# ── sklearn baseline ──────────────────────────────────────────────────────────

def train_sklearn(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib

    logger.info("Training TF-IDF + Logistic Regression baseline …")
    tr_texts = train_df["text_summary"].fillna("").tolist()
    va_texts = val_df["text_summary"].fillna("").tolist()
    tr_labels = train_df["label_id"].astype(int).tolist()
    va_labels = val_df["label_id"].astype(int).tolist()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(tr_texts, tr_labels)
    preds = pipe.predict(va_texts)

    acc = accuracy_score(va_labels, preds)
    f1 = f1_score(va_labels, preds, average="macro", zero_division=0)
    logger.info(f"Val accuracy={acc:.4f}  F1-macro={f1:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "sklearn_baseline.pkl"
    joblib.dump(pipe, model_path)
    logger.info(f"Sklearn model saved → {model_path}")

    return {"model_type": "sklearn", "val_accuracy": acc, "val_f1_macro": f1, "output_dir": str(MODELS_DIR)}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_training_pipeline(input_path=None):
    cfg = load_config()
    train_df, val_df, _ = split_dataset(
        input_path=input_path,
        test_size=cfg.get("test_size", 0.15),
        val_size=cfg.get("val_size", 0.15),
        seed=cfg.get("random_seed", 42),
    )

    if TORCH_AVAILABLE:
        results = train_bert(
            train_df, val_df,
            model_name=cfg.get("model_name", "distilbert-base-uncased"),
            num_epochs=cfg.get("num_epochs", 3),
            batch_size=cfg.get("batch_size", 16),
            lr=cfg.get("learning_rate", 2e-5),
            max_length=cfg.get("max_length", 256),
        )
    else:
        results = train_sklearn(train_df, val_df)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Training complete.")
    return results


if __name__ == "__main__":
    run_training_pipeline()
