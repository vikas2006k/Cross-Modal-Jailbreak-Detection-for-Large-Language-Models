from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from vision_detector.feature_extractor import CLIPFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_CSV = PROJECT_ROOT / "vision_detector" / "dataset" / "metadata.csv"
MODELS_DIR = PROJECT_ROOT / "vision_detector" / "models" / "best_model"
METRICS_DIR = PROJECT_ROOT / "vision_detector" / "metrics"
FIGURES_DIR = METRICS_DIR / "figures"
HISTORY_CSV = METRICS_DIR / "training_history.csv"


class CLIPClassifier(nn.Module):
    """
    Lightweight multilayer perceptron classifier operating on frozen CLIP 512-dim embeddings.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class EmbeddingDataset(Dataset):
    """PyTorch dataset for precomputed CLIP embeddings and binary labels."""

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


def extract_split_embeddings(
    extractor: CLIPFeatureExtractor,
    df_split: pd.DataFrame,
    batch_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts CLIP visual embeddings for all images in a given split DataFrame.
    """
    image_paths = [PROJECT_ROOT / rel_path for rel_path in df_split["file_path"]]
    labels = torch.tensor(df_split["label"].values, dtype=torch.long)

    print(f"[*] Extracting CLIP embeddings for {len(image_paths)} images...")
    embeddings = extractor.extract_features(image_paths, batch_size=batch_size)
    return embeddings, labels


def train_model(
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3
) -> Dict[str, Any]:
    """
    Executes training on pre-extracted CLIP embeddings.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {METADATA_CSV}. Please run dataset_builder.py first.")

    df = pd.read_csv(METADATA_CSV)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    print(f"[+] Dataset splits loaded: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = CLIPFeatureExtractor.get_instance(device=str(device))

    # Pre-extract embeddings for all splits
    train_embeds, train_labels = extract_split_embeddings(extractor, df_train)
    val_embeds, val_labels = extract_split_embeddings(extractor, df_val)
    test_embeds, test_labels = extract_split_embeddings(extractor, df_test)

    # Save test embeddings for independent evaluation and explainability
    torch.save({
        "test_embeds": test_embeds,
        "test_labels": test_labels,
        "df_test": df_test
    }, METRICS_DIR / "test_data.pt")

    train_loader = DataLoader(EmbeddingDataset(train_embeds, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(val_embeds, val_labels), batch_size=batch_size, shuffle=False)

    model = CLIPClassifier(input_dim=512, hidden_dim=256, num_classes=2, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    start_training_time = time.perf_counter()
    print("\n" + "=" * 65)
    print(f"| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |  Val F1  |")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        total_train_loss = 0.0
        train_preds, train_targets = [], []

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(y_b)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(y_b.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_embeds)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                total_val_loss += loss.item() * len(y_b)

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_b.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_embeds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="binary")

        scheduler.step(val_f1)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f1"].append(round(val_f1, 4))

        print(f"|  {epoch:02d}   |   {avg_train_loss:.4f}   |  {train_acc:.4f}   |  {avg_val_loss:.4f}  |  {val_acc:.4f}  |  {val_f1:.4f}  |")

        # Check for best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint
            torch.save(model.state_dict(), MODELS_DIR / "model.pt")
            config = {
                "input_dim": 512,
                "hidden_dim": 256,
                "num_classes": 2,
                "dropout": 0.2,
                "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "backbone": "openai/clip-vit-base-patch32"
            }
            with open(MODELS_DIR / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[*] Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
                break

    training_time = round(time.perf_counter() - start_training_time, 2)
    print("=" * 65)
    print(f"[+] Training completed in {training_time} seconds. Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_CSV, index=False)
    print(f"[+] Training history exported to: {HISTORY_CSV}")

    # Plot 300 DPI training curves
    _plot_curves(history_df)

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "training_time_seconds": training_time
    }


def _plot_curves(df: pd.DataFrame) -> None:
    """Exports IEEE-standard 300 DPI training and validation curves."""
    epochs = df["epoch"]

    # Loss curve
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(epochs, df["train_loss"], label="Train Loss", color="#1f77b4", marker="o", linewidth=2)
    ax.plot(epochs, df["val_loss"], label="Validation Loss", color="#ff7f0e", marker="s", linewidth=2)
    ax.set_title("Vision Detector — Cross-Entropy Loss Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_curve.png", dpi=300)
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(epochs, df["train_acc"], label="Train Accuracy", color="#2ca02c", marker="o", linewidth=2)
    ax.plot(epochs, df["val_acc"], label="Validation Accuracy", color="#d62728", marker="s", linewidth=2)
    ax.set_title("Vision Detector — Accuracy Convergence Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_curve.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    train_model(epochs=15, batch_size=32)
