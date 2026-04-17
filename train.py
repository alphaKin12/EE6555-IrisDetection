"""
train.py — Iris Detection Pipeline: Training
Data preparation, augmentation, DataLoader setup, and training loop.
"""

import os
import random

import cv2
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from model import DeepComplexIrisNet, RealisticComplexLoss, build_model, reset_weights
from utils import (
    load_dataset, detect_eyes, detect_iris, extract_roi,
    augment, save_dataset, IMG_SIZE
)

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_PATH = "/kaggle/input/datasets/sondosaabed/casia-iris-thousand/CASIA-Iris-Thousand/CASIA-Iris-Thousand"
OUTPUT_BASE = "/kaggle/working"
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-4
TEST_SIZE = 0.2


# ──────────────────────────────────────────────
# Directory setup
# ──────────────────────────────────────────────
def setup_dirs(output_base):
    for sub in ["final_casia", "edging_5", "eyes", "iris"]:
        os.makedirs(f"{output_base}/{sub}", exist_ok=True)


# ──────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────
def prepare_data(final_output, labels):
    """
    Apply CLAHE + z-score normalization + augmentation,
    then split into train/test DataLoaders.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    X_list, y_list = [], []

    for img, lbl in zip(final_output, labels):
        res = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.uint8)
        enhanced = clahe.apply(res).astype(np.float32)
        norm = (enhanced - np.mean(enhanced)) / (np.std(enhanced) + 1e-6)

        for aug in augment(norm):
            X_list.append(aug)
            y_list.append(lbl)

    X = np.expand_dims(np.array(X_list), axis=1)   # (N, 1, 64, 64)
    y = np.array(y_list)

    print(f"After augmentation: {len(X)} samples")
    print(f"Unique identities:  {len(np.unique(y))}")
    print(f"Samples per identity: {len(X) / len(np.unique(y)):.1f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    return train_loader, test_loader, X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────
def make_scheduler(optimizer, epochs, warmup=5):
    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (epochs - warmup)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train(model, train_loader, test_loader, device,
          epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
          save_path="best_model.pth"):
    """
    Full training loop with cosine-annealing LR, gradient clipping,
    and best-model checkpointing.

    Returns:
        train_losses (list of float)
        val_accs     (list of float)
        best_val_acc (float)
    """
    criterion = RealisticComplexLoss(alpha=0.5, margin=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, epochs)

    reset_weights(model)
    print(f"Model reset. Training on {device}...")

    train_losses, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, c_emb = model(inputs)
            loss, _, _ = criterion(logits, c_emb, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()

        # ── Validate ──
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                _, predicted = torch.max(logits, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_acc = 100 * correct / total
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Best: {best_val_acc:.2f}%"
            )

    print(f"\nBest Val Acc: {best_val_acc:.2f}%")
    return train_losses, val_accs, best_val_acc


# ──────────────────────────────────────────────
# Plot training curves
# ──────────────────────────────────────────────
def plot_training_curves(train_losses, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(val_accs, label="Val Accuracy", color="orange")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Training curves saved to training_curves.png")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    setup_dirs(OUTPUT_BASE)

    # Pipeline
    imgs, label_map = load_dataset(BASE_PATH, max_folders=100)
    eye_detected = detect_eyes(imgs, output_base=OUTPUT_BASE)
    iris_detected = detect_iris(eye_detected, output_base=OUTPUT_BASE)
    final_output, labels = extract_roi(iris_detected, output_base=OUTPUT_BASE)
    save_dataset(OUTPUT_BASE, final_output, labels)

    train_loader, test_loader, *_ = prepare_data(final_output, labels)

    num_classes = len(np.unique(labels))
    model = build_model(num_classes, device)

    train_losses, val_accs, best_acc = train(
        model, train_loader, test_loader, device,
        epochs=EPOCHS, save_path="best_model.pth"
    )
    plot_training_curves(train_losses, val_accs)


if __name__ == "__main__":
    main()
