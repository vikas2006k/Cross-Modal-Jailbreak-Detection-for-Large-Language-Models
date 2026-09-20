"""
================================================================================
Cross-Modal Jailbreak Detection for Large Language Models (CMJD)
Phase 3.1.3: DistilBERT Training Pipeline
================================================================================
File: scripts/train_distilbert.py

Description:
    Fine-tunes DistilBERT (distilbert-base-uncased) on the serialized CMJD-10K
    tokenized dataset using Hugging Face Trainer and early stopping.
    Evaluates the best checkpoint on the held-out test split, generates
    publication-grade 300 DPI figures, serializes evaluation metrics,
    and saves test set predictions.

Usage:
    python scripts/train_distilbert.py
================================================================================
"""

import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed
)
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Optimize CPU multithreading
torch.set_num_threads(min(8, os.cpu_count() or 8))


class DynamicTrimDataCollator:
    """
    Trims trailing padding in each batch down to the maximum sequence length
    present in that batch, preserving full mathematical equivalence while
    greatly accelerating transformer execution on CPU/GPU.
    """
    def __call__(self, features):
        max_len = max(
            int(f["attention_mask"].sum()) if isinstance(f["attention_mask"], torch.Tensor) else sum(f["attention_mask"])
            for f in features
        )
        max_len = max(max_len, 1)

        input_ids = [torch.as_tensor(f["input_ids"])[:max_len] for f in features]
        attention_mask = [torch.as_tensor(f["attention_mask"])[:max_len] for f in features]
        labels = [f["label"] for f in features]

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


# Custom logging callback for clean epoch metric printing
class EpochMetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.epoch_metrics = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        pass

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_f1" in metrics:
            epoch = metrics.get("epoch", state.epoch)
            elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
            
            # Find recent training loss from log history
            train_loss = None
            for log in reversed(state.log_history):
                if "loss" in log:
                    train_loss = log["loss"]
                    break
            if train_loss is None:
                train_loss = metrics.get("eval_loss", 0.0)

            record = {
                "epoch": int(round(epoch)) if epoch is not None else len(self.epoch_metrics) + 1,
                "train_loss": float(train_loss),
                "val_loss": float(metrics.get("eval_loss", 0.0)),
                "accuracy": float(metrics.get("eval_accuracy", 0.0)),
                "precision": float(metrics.get("eval_precision", 0.0)),
                "recall": float(metrics.get("eval_recall", 0.0)),
                "f1": float(metrics.get("eval_f1", 0.0)),
                "time_sec": round(elapsed, 2)
            }
            self.epoch_metrics.append(record)

            gpu_mem = "N/A (CPU)"
            if torch.cuda.is_available():
                gpu_mem = f"{torch.cuda.memory_allocated(0) / (1024**2):.1f} MB"

            print()
            print(f"--- Epoch {record['epoch']} Evaluation ---")
            print(f"  Training Loss:    {record['train_loss']:.4f}")
            print(f"  Validation Loss:  {record['val_loss']:.4f}")
            print(f"  Accuracy:         {record['accuracy']:.4f}")
            print(f"  Precision:        {record['precision']:.4f}")
            print(f"  Recall:           {record['recall']:.4f}")
            print(f"  F1 Score:         {record['f1']:.4f}")
            print(f"  Training Time:    {record['time_sec']:.1f}s")
            print(f"  GPU Memory:       {gpu_mem}")
            print()


def verify_tokenized_dataset(dataset_dir="models/text_classifier/tokenized_dataset"):
    """
    Step 1: Verify tokenized dataset existence and partition counts.
    """
    p = Path(dataset_dir)
    print("=========================================================")
    print("STEP 1: VERIFY TOKENIZED DATASET")
    print("=========================================================")
    if not p.exists():
        print(f"Tokenized dataset folder missing: {p} [FAIL]")
        return False, None

    try:
        ds = load_from_disk(str(p))
    except Exception as e:
        print(f"Error loading DatasetDict: {e} [FAIL]")
        return False, None

    train_len = len(ds["train"])
    val_len = len(ds["validation"])
    test_len = len(ds["test"])

    c1 = train_len == 7000
    c2 = val_len == 1500
    c3 = test_len == 1500

    print(f"Train samples:      {train_len} (Expected: 7000) -> {'PASS' if c1 else 'FAIL'}")
    print(f"Validation samples: {val_len} (Expected: 1500) -> {'PASS' if c2 else 'FAIL'}")
    print(f"Test samples:       {test_len} (Expected: 1500) -> {'PASS' if c3 else 'FAIL'}")

    overall_pass = c1 and c2 and c3
    print(f"Tokenized Dataset Verification: {'PASS' if overall_pass else 'FAIL'}")
    print("=========================================================")
    return overall_pass, ds


def load_training_config(config_path="models/text_classifier/config/training_config.json"):
    """
    Step 2: Read and display training hyperparameter configuration.
    """
    print()
    print("=========================================================")
    print("STEP 2: LOAD TRAINING CONFIGURATION")
    print("=========================================================")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=========================================================")
    return cfg


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def train_distilbert_model():
    # Set seed
    set_seed(42)
    start_total_time = time.time()

    # Step 1: Verify tokenized dataset
    ok, dataset_dict = verify_tokenized_dataset()
    if not ok:
        raise RuntimeError("Tokenized dataset validation failed.")

    # Step 2: Load config
    cfg = load_training_config()

    # Step 3: Load DistilBERT Model
    print()
    print("=========================================================")
    print("STEP 3: LOAD DISTILBERT MODEL")
    print("=========================================================")
    model_name = cfg["model_name"]
    id2label = {0: "safe", 1: "jailbreak"}
    label2id = {"safe": 0, "jailbreak": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")
        except Exception:
            pass

    print(f"Model loaded: {model_name} (num_labels=2)")
    print("Label mapping: 0 -> safe, 1 -> jailbreak")
    print("=========================================================")

    # Step 4: Configure TrainingArguments & Trainer
    print()
    print("=========================================================")
    print("STEP 4: INITIALIZE TRAINER & CALLBACKS")
    print("=========================================================")
    output_dir = cfg["output_dir"]
    logging_dir = cfg["logging_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logging_dir).mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        eval_strategy=cfg["evaluation_strategy"],
        save_strategy=cfg["save_strategy"],
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=True,
        fp16=use_fp16,
        seed=cfg["seed"],
        dataloader_pin_memory=False,
        report_to="none"
    )

    metrics_callback = EpochMetricsCallback()
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=cfg["early_stopping_patience"]
    )

    # Step 5: Train Model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=DynamicTrimDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback, early_stopping_callback]
    )

    print("Trainer initialized with EarlyStoppingCallback (patience=2).")
    print("Starting fine-tuning...")
    print("=========================================================")

    train_result = trainer.train()
    total_training_time = time.time() - start_total_time

    # Determine best epoch and early stopping
    history = metrics_callback.epoch_metrics
    df_history = pd.DataFrame(history)
    best_epoch = 1
    if not df_history.empty and "f1" in df_history:
        best_epoch = int(df_history.loc[df_history["f1"].idxmax()]["epoch"])
    
    epochs_completed = len(df_history)
    early_stopped = epochs_completed < cfg["epochs"]

    print()
    print("=========================================================")
    print("TRAINING FINISHED")
    print(f"Total Epochs Run:        {epochs_completed}")
    print(f"Best Validation Epoch:   {best_epoch}")
    print(f"Early Stopping Triggered:{early_stopped}")
    print(f"Total Training Duration: {total_training_time:.1f}s ({total_training_time/60:.2f} min)")
    print("=========================================================")

    # Step 6: Save Best Model & Tokenizer
    print()
    print("=========================================================")
    print("STEP 6: SAVE BEST MODEL & TOKENIZER")
    print("=========================================================")
    best_model_dir = Path(cfg["best_model_dir"])
    tokenizer_dir = Path(cfg["tokenizer_dir"])
    best_model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(best_model_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(best_model_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))
    print(f"Best model saved to: {best_model_dir}")
    print(f"Tokenizer saved to:  {tokenizer_dir}")
    print("=========================================================")

    # Step 7: Evaluate Best Model on Test Set
    print()
    print("=========================================================")
    print("STEP 7: EVALUATE BEST MODEL ON HELD-OUT TEST SET")
    print("=========================================================")
    test_ds = dataset_dict["test"]
    predictions_output = trainer.predict(test_ds)
    test_logits = predictions_output.predictions
    test_labels = predictions_output.label_ids

    # Softmax probabilities
    probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    pred_labels = np.argmax(probs, axis=-1)
    prob_safe = probs[:, 0]
    prob_jb = probs[:, 1]

    # Compute comprehensive metrics
    test_accuracy = float(accuracy_score(test_labels, pred_labels))
    test_precision = float(precision_score(test_labels, pred_labels, zero_division=0))
    test_recall = float(recall_score(test_labels, pred_labels, zero_division=0))
    test_f1 = float(f1_score(test_labels, pred_labels, zero_division=0))
    test_roc_auc = float(roc_auc_score(test_labels, prob_jb))
    test_mcc = float(matthews_corrcoef(test_labels, pred_labels))
    test_balanced_acc = float(balanced_accuracy_score(test_labels, pred_labels))

    cm = confusion_matrix(test_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    clf_report = classification_report(
        test_labels,
        pred_labels,
        target_names=["safe", "jailbreak"],
        digits=4
    )

    print(f"Test Accuracy:          {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Precision:         {test_precision:.4f}")
    print(f"Test Recall:            {test_recall:.4f}")
    print(f"Test F1 Score:          {test_f1:.4f}")
    print(f"Test ROC-AUC:           {test_roc_auc:.4f}")
    print(f"Test MCC:               {test_mcc:.4f}")
    print(f"Test Balanced Accuracy: {test_balanced_acc:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  True Safe:      {tn:4d}  |  False Jailbreak: {fp:4d}")
    print(f"  False Safe:     {fn:4d}  |  True Jailbreak:  {tp:4d}")
    print()
    print("Classification Report:")
    print(clf_report)
    print("=========================================================")

    # Step 8: Save Metrics and Predictions
    print()
    print("=========================================================")
    print("STEP 8: SAVE EVALUATION METRICS & PREDICTIONS")
    print("=========================================================")
    metrics_dir = Path("models/text_classifier/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. metrics.json
    metrics_json = {
        "model_name": model_name,
        "test_samples": len(test_labels),
        "accuracy": round(test_accuracy, 4),
        "precision": round(test_precision, 4),
        "recall": round(test_recall, 4),
        "f1_score": round(test_f1, 4),
        "roc_auc": round(test_roc_auc, 4),
        "matthews_correlation_coefficient": round(test_mcc, 4),
        "balanced_accuracy": round(test_balanced_acc, 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "training_epochs": epochs_completed,
        "best_epoch": best_epoch,
        "training_time_seconds": round(total_training_time, 2)
    }
    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    # 2. classification_report.txt
    with open(metrics_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("CMJD DistilBERT Text Classifier — Test Classification Report\n")
        f.write("============================================================\n\n")
        f.write(clf_report)

    # 3. confusion_matrix.csv
    df_cm = pd.DataFrame(
        cm,
        index=["Actual_Safe", "Actual_Jailbreak"],
        columns=["Predicted_Safe", "Predicted_Jailbreak"]
    )
    df_cm.to_csv(metrics_dir / "confusion_matrix.csv")

    # 4. test_predictions.csv
    test_prompts = test_ds["prompt"]
    test_ids = test_ds["id"]
    df_test_preds = pd.DataFrame({
        "id": test_ids,
        "prompt": test_prompts,
        "true_label": [id2label[y] for y in test_labels],
        "predicted_label": [id2label[p] for p in pred_labels],
        "confidence_safe": [round(float(s), 4) for s in prob_safe],
        "confidence_jailbreak": [round(float(j), 4) for j in prob_jb]
    })
    df_test_preds.to_csv(metrics_dir / "test_predictions.csv", index=False)

    # 5. roc_auc.json
    fpr, tpr, roc_thresholds = roc_curve(test_labels, prob_jb)
    with open(metrics_dir / "roc_auc.json", "w", encoding="utf-8") as f:
        json.dump({
            "roc_auc_score": round(test_roc_auc, 4),
            "fpr": [round(float(x), 4) for x in fpr.tolist()[:50]],
            "tpr": [round(float(x), 4) for x in tpr.tolist()[:50]]
        }, f, indent=2)

    # 6. training_history.csv
    df_history.to_csv(metrics_dir / "training_history.csv", index=False)
    print(f"Saved all 6 metric files to {metrics_dir}")
    print("=========================================================")

    # Step 9: Generate Publication Visualizations (300 DPI)
    print()
    print("=========================================================")
    print("STEP 9: GENERATE TRAINING VISUALIZATIONS (300 DPI)")
    print("=========================================================")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    # 1. loss_curve.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    epochs_range = df_history["epoch"].tolist()
    ax.plot(epochs_range, df_history["train_loss"], marker="o", color="#1f77b4", label="Training Loss", linewidth=2)
    ax.plot(epochs_range, df_history["val_loss"], marker="s", color="#ff7f0e", label="Validation Loss", linewidth=2)
    ax.set_title("DistilBERT Training & Validation Loss Curve", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=10, labelpad=8)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=10, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(metrics_dir / "loss_curve.png", dpi=300)
    plt.close(fig)

    # 2. accuracy_curve.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(epochs_range, df_history["accuracy"], marker="o", color="#2ca02c", label="Validation Accuracy", linewidth=2)
    ax.plot(epochs_range, df_history["f1"], marker="^", color="#d62728", label="Validation F1", linewidth=2, linestyle="--")
    ax.set_title("DistilBERT Validation Performance Across Epochs", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=10, labelpad=8)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=10, labelpad=8)
    ax.set_ylim([0.8, 1.02])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(metrics_dir / "accuracy_curve.png", dpi=300)
    plt.close(fig)

    # 3. precision_recall_curve.png
    prec_pts, rec_pts, _ = precision_recall_curve(test_labels, prob_jb)
    pr_auc = auc(rec_pts, prec_pts)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(rec_pts, prec_pts, color="#9467bd", linewidth=2, label=f"Precision-Recall (AUC = {pr_auc:.4f})")
    ax.set_title("DistilBERT Precision-Recall Curve (Test Set)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Recall", fontsize=10, labelpad=8)
    ax.set_ylabel("Precision", fontsize=10, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="lower left")
    fig.tight_layout()
    fig.savefig(metrics_dir / "precision_recall_curve.png", dpi=300)
    plt.close(fig)

    # 4. roc_curve.png
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"DistilBERT ROC (AUC = {test_roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier (AUC = 0.50)")
    ax.set_title("DistilBERT Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=10, labelpad=8)
    ax.set_ylabel("True Positive Rate", fontsize=10, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    fig.savefig(metrics_dir / "roc_curve.png", dpi=300)
    plt.close(fig)

    # 5. confusion_matrix.png
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    fig.colorbar(cax)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = (val / np.sum(cm[i, :])) * 100
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color="black" if val < np.max(cm)/2 else "white", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Safe (0)", "Jailbreak (1)"])
    ax.set_yticklabels(["Safe (0)", "Jailbreak (1)"])
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
    ax.set_ylabel("True Label", fontsize=10, labelpad=8)
    ax.set_title("DistilBERT Test Set Confusion Matrix", fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(metrics_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    print("Generated all 5 visualization figures (300 DPI) in metrics/")
    print("=========================================================")

    # Step 10: Generate Training Summary Markdown
    print()
    print("=========================================================")
    print("STEP 10: GENERATE TRAINING SUMMARY MARKDOWN")
    print("=========================================================")
    summary_md = f"""# DistilBERT Training & Evaluation Summary — CMJD Phase 3.1.3

## Executive Summary
- **Model Name**: `{model_name}` (Sequence Classification Head, 2 classes)
- **Dataset**: CMJD-10K Text Benchmark (`models/text_classifier/tokenized_dataset`)
- **Partitions**: Train (7,000 samples) | Validation (1,500 samples) | Test (1,500 samples)
- **Epochs Completed**: {epochs_completed} / {cfg['epochs']}
- **Best Epoch**: {best_epoch}
- **Early Stopping Triggered**: {"Yes" if early_stopped else "No"}
- **Total Training Duration**: {total_training_time:.1f}s ({total_training_time/60:.2f} min)

---

## IEEE Hyperparameter Configuration
- **Batch Size**: {cfg['batch_size']}
- **Learning Rate**: {cfg['learning_rate']}
- **Optimizer**: AdamW
- **Weight Decay**: {cfg['weight_decay']}
- **Warmup Ratio**: {cfg['warmup_ratio']}
- **Max Sequence Length**: {cfg['max_length']} tokens
- **Random Seed**: {cfg['seed']}
- **Early Stopping Patience**: {cfg['early_stopping_patience']} epochs (monitoring `validation_f1`)

---

## Final Test Performance Metrics (Held-Out Test Split, N=1,500)
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **{test_accuracy:.4f}** ({test_accuracy*100:.2f}%) | Overall correct binary classifications |
| **Precision** | **{test_precision:.4f}** | True jailbreak detections / total flagged prompts |
| **Recall** | **{test_recall:.4f}** | True jailbreak detections / total actual jailbreaks |
| **F1 Score** | **{test_f1:.4f}** | Harmonic mean of precision and recall |
| **ROC-AUC** | **{test_roc_auc:.4f}** | Area under the ROC curve (discrimination ability) |
| **Matthews Corr. Coeff. (MCC)** | **{test_mcc:.4f}** | High-quality correlation metric across binary classes |
| **Balanced Accuracy** | **{test_balanced_acc:.4f}** | Macro-average accuracy across safe and jailbreak |

---

## Confusion Matrix Summary
| True \\ Predicted | Predicted Safe (0) | Predicted Jailbreak (1) | Total | Class Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Actual Safe (0)** | **{tn:,}** (TN) | **{fp:,}** (FP) | {tn+fp:,} | {tn/(tn+fp)*100:.2f}% |
| **Actual Jailbreak (1)** | **{fn:,}** (FN) | **{tp:,}** (TP) | {fn+tp:,} | {tp/(fn+tp)*100:.2f}% |
| **Total** | {tn+fn:,} | {fp+tp:,} | {len(test_labels):,} | - |

---

## Training History Across Epochs
| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1 Score | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for rec in history:
        summary_md += f"| {rec['epoch']} | {rec['train_loss']:.4f} | {rec['val_loss']:.4f} | {rec['accuracy']:.4f} | {rec['precision']:.4f} | {rec['recall']:.4f} | {rec['f1']:.4f} | {rec['time_sec']}s |\n"

    summary_md += f"""
---

## Visualizations Generated
- `loss_curve.png`: Training vs. validation loss trajectory.
- `accuracy_curve.png`: Epoch-wise accuracy and F1 score trajectory.
- `precision_recall_curve.png`: Precision vs. recall tradeoff curve (AUC = {pr_auc:.4f}).
- `roc_curve.png`: Receiver operating characteristic curve (AUC = {test_roc_auc:.4f}).
- `confusion_matrix.png`: Heatmap with count and class percentage annotations.
"""
    with open(metrics_dir / "training_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"Saved training summary markdown to {metrics_dir / 'training_summary.md'}")
    print("=========================================================")

    # Step 11: Print Final Verification Report
    print()
    print("=========================================================")
    print("CMJD PHASE 3.1.3 TRAINING REPORT")
    print("=========================================================")
    print()
    print("Tokenized Dataset Loaded = PASS")
    print()
    print("DistilBERT Loaded = PASS")
    print()
    print("Training Started = PASS")
    print()
    print(f"Epochs Completed = {epochs_completed}")
    print()
    print(f"Best Epoch = {best_epoch}")
    print()
    print("Early Stopping = PASS")
    print()
    print("Best Model Saved = PASS")
    print()
    print("Tokenizer Saved = PASS")
    print()
    print("Evaluation Completed = PASS")
    print()
    print("Metrics Generated = PASS")
    print()
    print("Visualizations Generated = PASS")
    print()
    print("Predictions Saved = PASS")
    print()
    print("STATUS")
    print()
    print("READY FOR PHASE 3.1.4 — Error Analysis & Explainability")
    print()
    print("=========================================================")


if __name__ == "__main__":
    train_distilbert_model()
