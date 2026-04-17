"""
model.py — Iris Detection Pipeline: Model Architecture
Deep Complex-Valued Neural Network (CVNN) with Gabor-initialized first layer.
"""

import torch
import torch.nn as nn
import numpy as np
from complexPyTorch.complexLayers import ComplexConv2d, ComplexLinear, ComplexBatchNorm2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d


# ──────────────────────────────────────────────
# Gabor filter initialization
# ──────────────────────────────────────────────
def generate_true_complex_gabor(out_channels, kernel_size):
    """
    Generate a bank of complex-valued Gabor filters spanning orientations [0, π).
    Returns a tensor of shape (out_channels, 1, kernel_size, kernel_size, complex64).
    """
    filters = torch.zeros(out_channels, 1, kernel_size, kernel_size, dtype=torch.complex64)
    for i in range(out_channels):
        theta = (i / out_channels) * np.pi
        sigma, freq = 2.0, 0.5
        for u in range(kernel_size):
            for v in range(kernel_size):
                x = u - kernel_size // 2
                y = v - kernel_size // 2
                xp =  x * np.cos(theta) + y * np.sin(theta)
                yp = -x * np.sin(theta) + y * np.cos(theta)
                real = np.exp(-(xp ** 2 + yp ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * freq * xp)
                imag = np.exp(-(xp ** 2 + yp ** 2) / (2 * sigma ** 2)) * np.sin(2 * np.pi * freq * xp)
                filters[i, 0, u, v] = complex(real, imag)
    return filters


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
class DeepComplexIrisNet(nn.Module):
    """
    5-layer Deep Complex-Valued CNN for iris recognition.

    Architecture:
        Conv1 (Gabor init, 11×11) → BN → ReLU → MaxPool
        Conv2 (5×5)               → BN → ReLU → MaxPool
        Conv3 (3×3)               → BN → ReLU → MaxPool
        Conv4 (3×3)               → BN → ReLU → MaxPool
        FC1 (complex, 1024→256)   → |·| magnitude → Dropout
        FC2 (real,    256→num_classes)

    Input:  (N, 1, 64, 64) float32  — cast to complex64 in forward()
    Output: logits (N, num_classes), complex_embeddings (N, 256, complex64)
    """

    def __init__(self, num_classes: int):
        super(DeepComplexIrisNet, self).__init__()

        # ── Layer 1: Gabor-initialized complex conv ──
        self.conv1 = ComplexConv2d(1, 16, kernel_size=11, padding=5)
        gabor_init = generate_true_complex_gabor(16, 11)
        self.conv1.conv_r.weight.data = gabor_init.real
        self.conv1.conv_i.weight.data = gabor_init.imag
        self.bn1 = ComplexBatchNorm2d(16)

        # ── Layer 2 ──
        self.conv2 = ComplexConv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = ComplexBatchNorm2d(32)

        # ── Layer 3 ──
        self.conv3 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(64)

        # ── Layer 4 ──
        self.conv4 = ComplexConv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = ComplexBatchNorm2d(64)

        # ── Classifier head ──
        # After 4× MaxPool(2): 64 / 2^4 = 4  →  64 * 4 * 4 = 1024
        self.fc1 = ComplexLinear(1024, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.type(torch.complex64)

        x = complex_max_pool2d(complex_relu(self.bn1(self.conv1(x))), 2)
        x = complex_max_pool2d(complex_relu(self.bn2(self.conv2(x))), 2)
        x = complex_max_pool2d(complex_relu(self.bn3(self.conv3(x))), 2)
        x = complex_max_pool2d(complex_relu(self.bn4(self.conv4(x))), 2)

        x = x.view(x.size(0), -1)
        complex_embeddings = complex_relu(self.fc1(x))

        mag = torch.abs(complex_embeddings)
        mag = self.dropout(mag)
        logits = self.fc2(mag)

        return logits, complex_embeddings


# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────
class RealisticComplexLoss(nn.Module):
    """
    Combined Cross-Entropy + phase-coherence loss for complex embeddings.

    CE loss encourages correct classification.
    Phase loss encourages same-identity embeddings to be phase-aligned
    and different-identity embeddings to be phase-orthogonal.

    Args:
        alpha  : weight of the phase loss term (default 0.5)
        margin : soft target margin for the similarity matrix (default 0.2)
    """

    def __init__(self, alpha: float = 0.5, margin: float = 0.2):
        super(RealisticComplexLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.alpha = alpha
        self.margin = margin

    def forward(self, logits, complex_embeddings, labels):
        loss_ce = self.ce(logits, labels)

        magnitude = torch.abs(complex_embeddings) + 1e-8
        norm_emb = complex_embeddings / magnitude
        sim_matrix = torch.real(
            torch.matmul(norm_emb, norm_emb.conj().T)
        )
        is_same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        target = torch.where(
            is_same == 1,
            1.0 - self.margin,
            0.0 + self.margin
        )
        loss_phase = torch.nn.functional.mse_loss(sim_matrix, target)

        total = loss_ce + self.alpha * loss_phase
        return total, loss_ce.item(), loss_phase.item()


# ──────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────
def build_model(num_classes: int, device: torch.device) -> DeepComplexIrisNet:
    model = DeepComplexIrisNet(num_classes=num_classes).to(device)
    print("5-Layer Deep CVNN Initialized.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def reset_weights(model: nn.Module) -> None:
    """Re-initialize all layers that support reset_parameters()."""
    def _reset(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    model.apply(_reset)
    print("Model weights reset.")
