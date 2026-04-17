"""
utils.py — Iris Detection Pipeline: Utilities
Data loading, preprocessing, augmentation, normalization, and I/O helpers.
"""

import cv2
import numpy as np
import os
import pickle
import random
from glob import glob

# ──────────────────────────────────────────────
# Global constants
# ──────────────────────────────────────────────
IMG_SIZE = 64
NORM_W, NORM_H = 360, 64
kernel = np.ones((5, 5), np.uint8)

# Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)


# ──────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────
def load_dataset(base_path, max_folders=100):
    """
    Load CASIA-Iris-Thousand dataset.
    Returns list of [grayscale_img, file_path, label] and label_map.
    """
    imgs = []
    label_counter = 0
    label_map = {}

    folders = sorted([f for f in glob(base_path + "/*") if os.path.isdir(f)])
    folders = folders[:max_folders]

    for folder in folders:
        for side in ['L', 'R']:
            side_path = os.path.join(folder, side)
            if not os.path.exists(side_path):
                continue

            identity_key = (folder, side)
            if identity_key not in label_map:
                label_map[identity_key] = label_counter
                label_counter += 1
            label = label_map[identity_key]

            image_paths = glob(side_path + "/*.jpg")
            for file_path in image_paths:
                img = cv2.imread(file_path)
                if img is None:
                    continue
                img = cv2.resize(img, (200, 150))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgs.append([gray, file_path, label])

    print(f"Total images: {len(imgs)}")
    print(f"Total unique identities: {label_counter}")
    return imgs, label_map


# ──────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────
def preprocess(img):
    """Gaussian blur + histogram equalization."""
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.equalizeHist(img)
    return img


def transform_image(img, thresh):
    """Binary threshold + morphological open/close."""
    _, threshold = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_or(opening, closing)


# ──────────────────────────────────────────────
# Eye & iris detection
# ──────────────────────────────────────────────
def detect_eyes(imgs, output_base=None):
    """Run Haar cascade eye detection on a list of [img, path, label]."""
    eye_detected = []

    for idx, (img, path, label) in enumerate(imgs):
        resized = cv2.resize(img, (400, 300))
        eyes = eye_cascade.detectMultiScale(
            resized, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        if len(eyes) == 0:
            continue

        x, y, w, h = max(eyes, key=lambda e: e[2] * e[3])
        eye_crop = resized[y:y + h, x:x + w]

        if output_base:
            cv2.imwrite(f"{output_base}/eyes/{label}_{idx}.jpg", eye_crop)

        eye_detected.append([eye_crop, path, label])

    print(f"Eyes detected: {len(eye_detected)}")
    return eye_detected


def detect_iris(eye_detected, output_base=None):
    """Run Hough Circle Transform for iris detection."""
    iris_detected = []

    for idx, (img, path, label) in enumerate(eye_detected):
        proc = preprocess(img)
        circles = cv2.HoughCircles(
            proc, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=30,
            param1=50, param2=15,
            minRadius=10, maxRadius=150
        )

        if circles is None:
            h, w = img.shape[:2]
            cx, cy, r = w // 2, h // 2, min(w, h) // 3
        else:
            circles = np.round(circles[0]).astype("int")
            cx, cy, r = max(circles, key=lambda c: c[2])

        if output_base:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
            cv2.imwrite(f"{output_base}/iris/{label}_{idx}.jpg", vis)

        iris_detected.append([img, (cx, cy, r), label])

    print(f"Iris detected: {len(iris_detected)}")
    print(f"Recovery rate: {len(iris_detected) / len(eye_detected) * 100:.1f}%")
    return iris_detected


# ──────────────────────────────────────────────
# Daugman normalization
# ──────────────────────────────────────────────
def daugman_normalize(img, cx, cy, r_inner, r_outer, width=360, height=64):
    """Unwrap the iris annulus into a rectangular strip (Daugman rubber-sheet)."""
    theta = np.linspace(0, 2 * np.pi, width, endpoint=False)
    rho = np.linspace(0, 1, height, endpoint=False)
    THETA, RHO = np.meshgrid(theta, rho)
    R = r_inner + RHO * (r_outer - r_inner)
    X = (cx + R * np.cos(THETA)).astype(np.float32)
    Y = (cy + R * np.sin(THETA)).astype(np.float32)
    return cv2.remap(img, X, Y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def normalize_iris(iris_detected):
    """Apply Daugman normalization + CLAHE to each detected iris."""
    iris_normalized = []
    for img, (cx, cy, r_iris), label in iris_detected:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        r_pupil = int(r_iris * 0.45)
        norm = daugman_normalize(gray, cx, cy, r_pupil, r_iris, NORM_W, NORM_H)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm = clahe.apply(norm)
        iris_normalized.append((norm, label))
    print(f"Normalized: {len(iris_normalized)} — each strip: {NORM_H}×{NORM_W}")
    return iris_normalized


# ──────────────────────────────────────────────
# ROI extraction
# ──────────────────────────────────────────────
def extract_roi(iris_detected, output_base=None):
    """Crop and resize padded iris ROI to IMG_SIZE×IMG_SIZE."""
    random.shuffle(iris_detected)
    final_output, labels = [], []

    for idx, (img, (cx, cy, r), label) in enumerate(iris_detected):
        padding = int(r * 0.15)
        r_padded = r + padding

        x1 = max(cx - r_padded, 0)
        y1 = max(cy - r_padded, 0)
        x2 = min(cx + r_padded, img.shape[1])
        y2 = min(cy + r_padded, img.shape[0])

        roi = img[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            roi = img

        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

        if output_base:
            cv2.imwrite(f"{output_base}/final_casia/{label}_{idx}.jpg", roi)

        final_output.append(roi)
        labels.append(label)

    final_output = np.array(final_output)
    labels = np.array(labels)
    print(f"Final dataset size: {len(final_output)}")
    print(f"Unique identities: {len(np.unique(labels))}")
    print(f"Samples per identity: {len(final_output) / len(np.unique(labels)):.1f}")
    return final_output, labels


# ──────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────
def augment(img, img_size=IMG_SIZE):
    """Returns original + 3 augmented versions of each image."""
    results = [img]
    # Horizontal flip
    results.append(np.fliplr(img).copy())
    # Slight rotation (-10 to +10 degrees)
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((img_size // 2, img_size // 2), angle, 1.0)
    results.append(cv2.warpAffine(img, M, (img_size, img_size)))
    # Brightness shift
    shifted = np.clip(img + random.uniform(-0.3, 0.3), -3, 3)
    results.append(shifted)
    return results


# ──────────────────────────────────────────────
# Preprocessing for inference
# ──────────────────────────────────────────────
def preprocess_image_bytes(img_bytes, img_size=IMG_SIZE):
    """
    Full pipeline for a raw image (bytes) at inference time.
    Returns a normalized float32 numpy array of shape (img_size, img_size).
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image.")

    resized = cv2.resize(img, (400, 300))
    eyes = eye_cascade.detectMultiScale(resized, 1.1, 5, minSize=(50, 50))

    if len(eyes) > 0:
        x, y, w, h = max(eyes, key=lambda e: e[2] * e[3])
        eye_crop = resized[y:y + h, x:x + w]
    else:
        eye_crop = resized

    proc = cv2.GaussianBlur(eye_crop, (7, 7), 0)
    proc = cv2.equalizeHist(proc)

    circles = cv2.HoughCircles(
        proc, cv2.HOUGH_GRADIENT, 1.2, 30,
        param1=50, param2=15, minRadius=10, maxRadius=150
    )

    h2, w2 = eye_crop.shape[:2]
    if circles is not None:
        circles = np.round(circles[0]).astype("int")
        cx, cy, r = max(circles, key=lambda c: c[2])
    else:
        cx, cy, r = w2 // 2, h2 // 2, min(w2, h2) // 3

    pad = int(r * 0.15)
    r_padded = r + pad
    x1 = max(cx - r_padded, 0)
    y1 = max(cy - r_padded, 0)
    x2 = min(cx + r_padded, w2)
    y2 = min(cy + r_padded, h2)

    roi = eye_crop[y1:y2, x1:x2]
    if roi.size == 0:
        roi = eye_crop

    roi = cv2.resize(roi, (img_size, img_size)).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi).astype(np.float32)
    norm = (enhanced - np.mean(enhanced)) / (np.std(enhanced) + 1e-6)
    return norm


# ──────────────────────────────────────────────
# Pickle I/O
# ──────────────────────────────────────────────
def save_dataset(output_base, final_output, labels, test_imgs=None):
    with open(f"{output_base}/features.pickle", "wb") as f:
        pickle.dump(final_output, f)
    with open(f"{output_base}/labels.pickle", "wb") as f:
        pickle.dump(labels, f)
    if test_imgs is not None:
        with open(f"{output_base}/test_imgs.pickle", "wb") as f:
            pickle.dump(test_imgs, f)
    print("Dataset saved.")


def load_saved_dataset(output_base):
    with open(f"{output_base}/features.pickle", "rb") as f:
        final_output = pickle.load(f)
    with open(f"{output_base}/labels.pickle", "rb") as f:
        labels = pickle.load(f)
    print("Dataset loaded from disk.")
    return final_output, labels


# ──────────────────────────────────────────────
# Label utilities
# ──────────────────────────────────────────────
def build_inverse_label_map(label_map):
    return {v: k for k, v in label_map.items()}


def label_to_name(lbl, inverse_label_map):
    if lbl not in inverse_label_map:
        return f"Label {lbl}"
    folder, side = inverse_label_map[lbl]
    person_name = os.path.basename(folder)
    return f"{person_name} ({side})"
