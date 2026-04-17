"""
eval.py — Iris Detection Pipeline: Evaluation
Classification metrics, biometric verification (EER / TAR @ FAR),
confusion matrix, and single-image inference.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc,
    precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef,
)

from model import DeepComplexIrisNet
from utils import preprocess_image_bytes, label_to_name, build_inverse_label_map
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d


# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────
def extract_features(model, test_loader, device):
    """
    Run forward pass on the full test set.
    Returns (all_logits, all_embeddings, all_labels) as numpy arrays / tensors.
    """
    model.eval()
    all_logits, all_embeddings, all_labels = [], [], []

    with torch.no_grad():
        for inputs, labels_batch in test_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            logits, complex_embeddings = model(inputs)
            all_logits.append(logits.cpu())
            all_embeddings.append(complex_embeddings.cpu())
            all_labels.extend(labels_batch.cpu().numpy())

    all_logits = torch.cat(all_logits, dim=0)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = np.array(all_labels)
    return all_logits, all_embeddings, all_labels


# ──────────────────────────────────────────────
# Classification evaluation
# ──────────────────────────────────────────────
def evaluate_classification(all_logits, all_labels):
    """Print accuracy and per-class classification report."""
    _, preds = torch.max(all_logits, 1)
    preds = preds.numpy()

    print("========== PART 1: CLASSIFICATION PERFORMANCE ==========")
    print(f"Final Test Accuracy: {accuracy_score(all_labels, preds):.2%}")
    print("\nDetailed Classification Report:\n",
          classification_report(all_labels, preds, zero_division=0))

    macro_precision = precision_score(all_labels, preds, average='macro', zero_division=0)
    macro_recall    = recall_score(all_labels, preds, average='macro', zero_division=0)
    macro_f1        = f1_score(all_labels, preds, average='macro', zero_division=0)
    kappa           = cohen_kappa_score(all_labels, preds)
    mcc             = matthews_corrcoef(all_labels, preds)

    print(f"Macro-Average Precision: {macro_precision:.4f}")
    print(f"Macro-Average Recall:    {macro_recall:.4f}")
    print(f"Macro-Average F1-Score:  {macro_f1:.4f}")
    print(f"Cohen's Kappa:           {kappa:.4f}")
    print(f"Matthews Corr. Coef.:    {mcc:.4f}")

    return preds


# ──────────────────────────────────────────────
# Biometric verification (EER / TAR @ FAR)
# ──────────────────────────────────────────────
def evaluate_biometric(all_embeddings, all_labels):
    """
    Compute genuine/imposter score distributions, EER, and TAR @ FAR.
    Uses phase correlation as the similarity score.
    Returns fpr, tpr, thresholds, eer, eer_idx.
    """
    magnitude = torch.abs(all_embeddings) + 1e-8
    norm_emb = all_embeddings / magnitude
    phase = torch.angle(norm_emb).numpy()

    genuine_scores, imposter_scores = [], []
    n = len(all_labels)

    for i in range(n):
        for j in range(i + 1, n):
            score = float(np.cos(phase[i] - phase[j]).mean())
            if all_labels[i] == all_labels[j]:
                genuine_scores.append(score)
            else:
                imposter_scores.append(score)

    print("\n========== PART 2: BIOMETRIC VERIFICATION (EER/TAR) ==========")
    print(f"Genuine pairs:  {len(genuine_scores)}")
    print(f"Imposter pairs: {len(imposter_scores)}")
    print(f"Mean genuine score:  {np.mean(genuine_scores):.4f}")
    print(f"Mean imposter score: {np.mean(imposter_scores):.4f}")
    print(f"Separation gap: {np.mean(genuine_scores) - np.mean(imposter_scores):.4f}")

    y_true = [1] * len(genuine_scores) + [0] * len(imposter_scores)
    y_scores = genuine_scores + imposter_scores

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]

    print(f"\nEqual Error Rate (EER): {eer:.2%}")
    for target_far in [0.01, 0.001]:
        try:
            tar = tpr[np.where(fpr <= target_far)[0][-1]]
            print(f"TAR @ {target_far * 100}% FAR: {tar:.2%}")
        except IndexError:
            print(f"TAR @ {target_far * 100}% FAR: Insufficient data.")

    return fpr, tpr, thresholds, eer, eer_idx, genuine_scores, imposter_scores


# ──────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────
def plot_verification_results(fpr, tpr, thresholds, eer, eer_idx,
                               genuine_scores, imposter_scores):
    """Score distributions, ROC curve, and EER summary."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Score distributions
    ax = axes[0]
    sns.kdeplot(genuine_scores,  label='Genuine',  color='green', fill=True, ax=ax)
    sns.kdeplot(imposter_scores, label='Imposter', color='red',   fill=True, ax=ax)
    ax.axvline(thresholds[eer_idx], color='black', linestyle='--',
               label=f'EER={eer:.2%}')
    ax.set_title("Score Distributions")
    ax.set_xlabel("Phase Correlation Score")
    ax.legend()

    # ROC curve
    roc_auc = auc(fpr, tpr)
    ax = axes[1]
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.scatter(fpr[eer_idx], tpr[eer_idx], color='red', zorder=5,
               label=f"EER = {eer:.2%}")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    # EER summary bar
    ax = axes[2]
    ax.bar(['EER'], [eer * 100], color='coral')
    ax.set_ylim(0, 20)
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Equal Error Rate")

    plt.tight_layout()
    plt.savefig("verification_results.png", dpi=150)
    plt.show()
    print("Verification plots saved to verification_results.png")


def plot_confusion_matrix(all_labels, preds):
    """Heatmap of the confusion matrix (suitable for many classes)."""
    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True, square=True)
    plt.title('Confusion Matrix', fontsize=20, pad=20)
    plt.xlabel('Predicted Label', fontsize=15, labelpad=15)
    plt.ylabel('True Label',      fontsize=15, labelpad=15)
    plt.grid(visible=True, color='grey', linewidth=0.2, alpha=0.3)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved to confusion_matrix.png")


# ──────────────────────────────────────────────
# Single-image inference (closed-set)
# ──────────────────────────────────────────────
def get_embedding(model, img_array, device):
    """
    Extract the complex embedding for a single preprocessed image array.
    img_array: float32 numpy array, shape (H, W).
    """
    import cv2 as _cv2
    img_resized = _cv2.resize(img_array.astype(np.float32), (64, 64))
    tensor = (torch.tensor(img_resized, dtype=torch.complex64)
              .unsqueeze(0).unsqueeze(0).to(device))

    model.eval()
    with torch.no_grad():
        x = tensor
        x = complex_max_pool2d(complex_relu(model.bn1(model.conv1(x))), 2)
        x = complex_max_pool2d(complex_relu(model.bn2(model.conv2(x))), 2)
        x = complex_max_pool2d(complex_relu(model.bn3(model.conv3(x))), 2)
        x = complex_max_pool2d(complex_relu(model.bn4(model.conv4(x))), 2)
        x = x.view(x.size(0), -1)
        embedding = complex_relu(model.fc1(x))
    return embedding.cpu()


def verify_pair(emb1, emb2, threshold=0.96):
    """
    Compare two complex embeddings by phase correlation.
    Returns (score, is_match).
    """
    mag1 = torch.abs(emb1) + 1e-8
    mag2 = torch.abs(emb2) + 1e-8
    norm1 = emb1 / mag1
    norm2 = emb2 / mag2
    phase1 = torch.angle(norm1).numpy()
    phase2 = torch.angle(norm2).numpy()
    score = float(np.cos(phase1 - phase2).mean())
    return score, score >= threshold


def predict_identity(model, img_bytes, device, label_map, img_size=64):
    """
    Predict the identity label for a raw image (bytes).
    Returns (predicted_label, confidence, top5_labels, top5_probs).
    """
    import cv2 as _cv2

    img_array = preprocess_image_bytes(img_bytes, img_size=img_size)
    img_resized = _cv2.resize(img_array.astype(np.float32), (img_size, img_size))
    tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    top5_idx = np.argsort(probs)[::-1][:5]
    inverse_map = build_inverse_label_map(label_map)
    top5_labels = [label_to_name(i, inverse_map) for i in top5_idx]
    top5_probs  = probs[top5_idx]
    pred_label  = int(top5_idx[0])
    confidence  = float(top5_probs[0])

    return pred_label, confidence, top5_labels, top5_probs


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main(model, test_loader, device, label_map=None):
    """
    Run the full evaluation suite.
    Call after training is complete.
    """
    all_logits, all_embeddings, all_labels = extract_features(model, test_loader, device)

    # Classification
    preds = evaluate_classification(all_logits, all_labels)
    plot_confusion_matrix(all_labels, preds)

    # Biometric verification
    fpr, tpr, thresholds, eer, eer_idx, genuine, imposter = evaluate_biometric(
        all_embeddings, all_labels
    )
    plot_verification_results(fpr, tpr, thresholds, eer, eer_idx, genuine, imposter)


if __name__ == "__main__":
    # Standalone usage example — load model checkpoint and run eval.
    import sys
    from train import prepare_data, OUTPUT_BASE
    from utils import load_saved_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_output, labels = load_saved_dataset(OUTPUT_BASE)
    _, test_loader, *_ = prepare_data(final_output, labels)

    num_classes = len(np.unique(labels))
    model = DeepComplexIrisNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("Checkpoint loaded.")

    main(model, test_loader, device)
