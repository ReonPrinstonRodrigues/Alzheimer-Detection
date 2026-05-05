"""
create_notebook.py — Generates ml_pipeline.ipynb for Google Colab
Run this script locally to create the Jupyter notebook, then upload it to Colab.
"""

import json

def make_md(source):
    """Create a markdown cell."""
    lines = [line + "\n" for line in source.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def make_code(source):
    """Create a code cell."""
    lines = [line + "\n" for line in source.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "metadata": {}, "source": lines, "execution_count": None, "outputs": []}

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": []
}

cells = notebook["cells"]

# ══════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════
cells.append(make_md("""# 🧠 Alzheimer's Disease Detection — ML Pipeline
## Multi-Class Classification of MRI Brain Scans (OASIS Dataset)

**Classes:** Non Demented | Very Mild Demented | Mild Demented | Moderate Demented

**Models Trained:**
1. Custom CNN
2. MLP (Multi-Layer Perceptron)
3. VGG16 (Transfer Learning)
4. ResNet50 (Transfer Learning)
5. EfficientNetB0 (Transfer Learning)
6. MobileNetV2 (Transfer Learning)

---"""))

# ══════════════════════════════════════════════════════════════
# Step 0: Setup
# ══════════════════════════════════════════════════════════════
cells.append(make_md("## ⚙️ Step 0 — Environment Setup"))

cells.append(make_code("""# Install required packages
!pip install -q tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python pillow tqdm

import os
import cv2
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                      Dropout, BatchNormalization, GlobalAveragePooling2D,
                                      Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0, MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("Setup complete! ✓")"""))

# ══════════════════════════════════════════════════════════════
# Step 0.5: Upload Data
# ══════════════════════════════════════════════════════════════
cells.append(make_md("""## 📦 Upload Dataset

**Option A:** Extract manually uploaded `Data.zip` from your local machine.
**Option B:** Mount Google Drive if you've uploaded the zip there.
**Option C:** Download directly from Kaggle (Requires Kaggle credentials).

Choose ONE option below and run that cell."""))

cells.append(make_code("""# === OPTION A: Extract Data.zip (Uploaded manually) ===
# Ensure you have uploaded 'Data.zip' using the file browser on the left before running this cell.

import os
import zipfile

dataset_zip = '/content/Data.zip'

if not os.path.exists(dataset_zip):
    print(f"Error: {dataset_zip} not found! Please upload it via the left panel.")
else:
    print(f"Extracting {dataset_zip}...")
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall('/content/')
    
    print("\\n✓ Data extracted successfully!")
    !ls /content/Data/"""))

cells.append(make_code("""# === OPTION B: Mount Google Drive (if Data.zip is in Drive) ===
# Uncomment and run this cell instead if your data is in Google Drive

# from google.colab import drive
# drive.mount('/content/drive')
# import zipfile
# zip_path = '/content/drive/MyDrive/Data.zip'  # <-- Update this path
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall('/content/')
# print("✓ Data extracted from Drive!")
# !ls /content/Data/"""))

cells.append(make_code("""# === OPTION C: Download directly from Kaggle ===
# Uses the official kagglehub library to download the imagesoasis dataset.
# Run this cell to automatically download the dataset (no authentication required).

!pip install -q kagglehub
import kagglehub
import os
import shutil

print("Downloading dataset...")
path = kagglehub.dataset_download("ninadaithal/imagesoasis")
print("Downloaded to:", path)

target_path = '/content/Data'

# Clean up existing Data dir if needed
if os.path.exists(target_path) and os.path.islink(target_path):
    os.unlink(target_path)
elif os.path.exists(target_path):
    shutil.rmtree(target_path)

# Ensure the paths align correctly based on dataset structure
if os.path.exists(os.path.join(path, 'Data')):
    os.symlink(os.path.join(path, 'Data'), target_path)
else:
    os.symlink(path, target_path)

print(f"\\n✓ Data ready at {target_path}!")
!ls /content/Data/
"""))

# ══════════════════════════════════════════════════════════════
# Step 1: Problem Analysis
# ══════════════════════════════════════════════════════════════
cells.append(make_md("## 📋 Step 1 — Problem Analysis"))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# STEP 1: PROBLEM ANALYSIS
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("  ALZHEIMER'S DISEASE DETECTION — PROBLEM ANALYSIS")
print("=" * 70)

print(\"\"\"
PROBLEM: Multi-class classification of Alzheimer's disease severity
         from MRI brain scan images.

DATASET: OASIS (Open Access Series of Imaging Studies) Brain MRI Dataset

CLASSES (4):
  1. Non Demented      — Normal cognitive function, no signs of dementia
  2. Very Mild Demented — Early memory lapses, subtle cognitive changes
  3. Mild Demented      — Noticeable memory loss, difficulty with daily tasks
  4. Moderate Demented  — Significant cognitive decline, requires full-time care

CLINICAL SIGNIFICANCE:
  Early detection of Alzheimer's enables timely intervention, potentially
  slowing disease progression and improving patient quality of life.
  AI-assisted MRI analysis can help neurologists with faster, more
  consistent screening.

CHALLENGES:
  • Severe class imbalance (Moderate class has ~138x fewer samples than Non Demented)
  • Subtle visual differences between early stages
  • Need for high sensitivity to avoid missing positive cases
\"\"\")"""))

# ══════════════════════════════════════════════════════════════
# Step 2: Data Collection
# ══════════════════════════════════════════════════════════════
cells.append(make_md("## 📂 Step 2 — Data Collection"))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# STEP 2: DATA COLLECTION
# ═══════════════════════════════════════════════════════════════

DATA_DIR = '/content/Data'
IMG_SIZE = 224
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Map folder names to standardized class names
CLASS_MAP = {
    'Mild Dementia': 'Mild Demented',
    'Moderate Dementia': 'Moderate Demented',
    'Non Demented': 'Non Demented',
    'Very mild Dementia': 'Very Mild Demented'
}

# Collect all image paths and labels
image_paths = []
image_labels = []

print("Scanning dataset directory...")
for folder_name in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    class_name = CLASS_MAP.get(folder_name, folder_name)
    count = 0
    for fname in os.listdir(folder_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext in VALID_EXTENSIONS:
            image_paths.append(os.path.join(folder_path, fname))
            image_labels.append(class_name)
            count += 1

    print(f"  {class_name}: {count} images")

print(f"\\nTotal images found: {len(image_paths)}")

# Class distribution
class_counts = Counter(image_labels)
print("\\nClass Distribution:")
for cls, count in sorted(class_counts.items()):
    pct = (count / len(image_labels)) * 100
    print(f"  {cls}: {count} ({pct:.1f}%)")"""))

# ══════════════════════════════════════════════════════════════
# Step 3: Data Cleaning
# ══════════════════════════════════════════════════════════════
cells.append(make_md("## 🧹 Step 3 — Data Cleaning"))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# STEP 3: DATA CLEANING
# ═══════════════════════════════════════════════════════════════

print("Starting data cleaning pipeline...")
print("-" * 50)

clean_paths = []
clean_labels = []
corrupted_count = 0
duplicate_count = 0
hash_set = set()

for i, (path, label) in enumerate(tqdm(zip(image_paths, image_labels),
                                         total=len(image_paths),
                                         desc="Cleaning")):
    # Check if image is readable
    try:
        img = cv2.imread(path)
        if img is None:
            corrupted_count += 1
            continue
    except Exception:
        corrupted_count += 1
        continue

    # MD5 hash for deduplication
    img_bytes = cv2.resize(img, (64, 64)).tobytes()  # Hash at lower res for speed
    img_hash = hashlib.md5(img_bytes).hexdigest()

    if img_hash in hash_set:
        duplicate_count += 1
        continue

    hash_set.add(img_hash)
    clean_paths.append(path)
    clean_labels.append(label)

print(f"\\n{'=' * 50}")
print(f"CLEANING REPORT")
print(f"{'=' * 50}")
print(f"  Original images:   {len(image_paths)}")
print(f"  Corrupted removed: {corrupted_count}")
print(f"  Duplicates removed: {duplicate_count}")
print(f"  Clean images:      {len(clean_paths)}")
print(f"{'=' * 50}")

# Update counts
clean_counts = Counter(clean_labels)
print("\\nClean Class Distribution:")
for cls, count in sorted(clean_counts.items()):
    pct = (count / len(clean_labels)) * 100
    print(f"  {cls}: {count} ({pct:.1f}%)")"""))

# ══════════════════════════════════════════════════════════════
# Step 4: Train/Test Split
# ══════════════════════════════════════════════════════════════
cells.append(make_md("## ✂️ Step 4 — Train / Validation / Test Split"))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# STEP 4: TRAIN / VALIDATION / TEST SPLIT (70 / 15 / 15)
# ═══════════════════════════════════════════════════════════════

# Encode labels
CLASS_NAMES = sorted(list(set(clean_labels)))
label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
encoded_labels = np.array([label_to_idx[l] for l in clean_labels])
paths_array = np.array(clean_paths)

print(f"Class mapping: {label_to_idx}")

# First split: 70% train, 30% temp
X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(
    paths_array, encoded_labels,
    test_size=0.30,
    stratify=encoded_labels,
    random_state=42
)

# Second split: 50/50 of temp → 15% val, 15% test
X_val_paths, X_test_paths, y_val, y_test = train_test_split(
    X_temp_paths, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print(f"\\nSplit sizes:")
print(f"  Training:   {len(X_train_paths)} images ({len(X_train_paths)/len(paths_array)*100:.1f}%)")
print(f"  Validation: {len(X_val_paths)} images ({len(X_val_paths)/len(paths_array)*100:.1f}%)")
print(f"  Test:       {len(X_test_paths)} images ({len(X_test_paths)/len(paths_array)*100:.1f}%)")

# Print per-class distribution in each split
for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    counts = Counter(labels)
    dist = ", ".join([f"{CLASS_NAMES[i]}: {counts.get(i, 0)}" for i in range(len(CLASS_NAMES))])
    print(f"  {name}: {dist}")"""))

# ══════════════════════════════════════════════════════════════
# Step 5: Load images + Data Balancing
# ══════════════════════════════════════════════════════════════
cells.append(make_md("## ⚖️ Step 5 — Data Loading & Balancing"))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# STEP 5: LOAD IMAGES INTO ARRAYS & DATA BALANCING
# ═══════════════════════════════════════════════════════════════

def load_images(paths, labels, img_size=224, desc="Loading"):
    \"\"\"Load and preprocess images from file paths.\"\"\"
    images = []
    valid_labels = []
    for path, label in tqdm(zip(paths, labels), total=len(paths), desc=desc):
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            valid_labels.append(label)
        except Exception:
            continue
    return np.array(images, dtype='float32') / 255.0, np.array(valid_labels)

# Load all sets
print("Loading training images...")
X_train, y_train_loaded = load_images(X_train_paths, y_train, desc="Train")
print(f"  Train shape: {X_train.shape}")

print("\\nLoading validation images...")
X_val, y_val_loaded = load_images(X_val_paths, y_val, desc="Val")
print(f"  Val shape: {X_val.shape}")

print("\\nLoading test images...")
X_test, y_test_loaded = load_images(X_test_paths, y_test, desc="Test")
print(f"  Test shape: {X_test.shape}")

# One-hot encode labels
num_classes = len(CLASS_NAMES)
y_train_cat = to_categorical(y_train_loaded, num_classes)
y_val_cat = to_categorical(y_val_loaded, num_classes)
y_test_cat = to_categorical(y_test_loaded, num_classes)

print(f"\\nLabels one-hot encoded: {num_classes} classes")"""))

cells.append(make_code("""# ─── Class Weights & Data Augmentation ───────────────────────
# Compute class weights to handle imbalance
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_loaded),
    y=y_train_loaded
)
class_weights = dict(enumerate(class_weights_array))

print("Class Weights (to handle imbalance):")
for idx, weight in class_weights.items():
    print(f"  {CLASS_NAMES[idx]}: {weight:.4f}")

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

# No augmentation for validation/test
val_datagen = ImageDataGenerator()

# Create generators
BATCH_SIZE = 32
train_generator = train_datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val_cat, batch_size=BATCH_SIZE)

print(f"\\nData augmentation configured ✓")
print(f"Batch size: {BATCH_SIZE}")"""))

cells.append(make_code("""# ─── Visualize Class Distribution ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before balancing (raw counts)
colors = ['#2ecc71', '#e67e22', '#e74c3c', '#f39c12']
train_counts = Counter(y_train_loaded)
classes = [CLASS_NAMES[i] for i in range(num_classes)]
counts = [train_counts.get(i, 0) for i in range(num_classes)]

axes[0].bar(classes, counts, color=colors, edgecolor='white', linewidth=0.5)
axes[0].set_title('Class Distribution (Training Set)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Images')
axes[0].tick_params(axis='x', rotation=15)
for i, (c, v) in enumerate(zip(classes, counts)):
    axes[0].text(i, v + max(counts)*0.01, str(v), ha='center', fontsize=9, fontweight='bold')

# Class weights visualization
weights = [class_weights[i] for i in range(num_classes)]
axes[1].bar(classes, weights, color=colors, edgecolor='white', linewidth=0.5)
axes[1].set_title('Class Weights (Balancing Factor)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Weight')
axes[1].tick_params(axis='x', rotation=15)
for i, (c, v) in enumerate(zip(classes, weights)):
    axes[1].text(i, v + max(weights)*0.01, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
os.makedirs('/content/plots', exist_ok=True)
plt.savefig('/content/plots/class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: plots/class_distribution.png")"""))

# ══════════════════════════════════════════════════════════════
# Step 6: Training — Helper functions
# ══════════════════════════════════════════════════════════════
cells.append(make_md("""## 🏋️ Step 6 — Model Training

Training 6 models with:
- Adam optimizer, categorical crossentropy
- EarlyStopping (patience=5) + ReduceLROnPlateau (patience=3)
- Class weights for imbalance
- 20 epochs each"""))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# TRAINING HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

os.makedirs('/content/models', exist_ok=True)
os.makedirs('/content/plots', exist_ok=True)

EPOCHS = 20

def get_callbacks():
    \"\"\"Standard callbacks for all models.\"\"\"
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]

def plot_training_history(history, model_name):
    \"\"\"Plot and save accuracy/loss curves.\"\"\"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='#4dabf7')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, color='#ff6b6b')
    axes[0].set_title(f'{model_name} — Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2, color='#4dabf7')
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='#ff6b6b')
    axes[1].set_title(f'{model_name} — Loss', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'/content/plots/{model_name}_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: plots/{model_name}_accuracy.png")

def plot_confusion_matrix(y_true, y_pred, model_name):
    \"\"\"Plot and save confusion matrix.\"\"\"
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'{model_name} — Confusion Matrix', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'/content/plots/{model_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: plots/{model_name}_confusion_matrix.png")

def train_and_evaluate(model, model_name, model_key):
    \"\"\"Train a model and generate all evaluation artifacts.\"\"\"
    print(f"\\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Total parameters: {model.count_params():,}")
    print(f"Training with {EPOCHS} epochs, batch size {BATCH_SIZE}...")

    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(X_val) // BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    # Classification report
    print(f"\\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # F1 Score
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Plots
    plot_training_history(history, model_key)
    plot_confusion_matrix(y_true, y_pred, model_key)

    # Save model
    model_path = f'/content/models/{model_key}_model.h5'
    model.save(model_path)
    print(f"✓ Model saved: {model_path}")

    # Return metrics
    train_acc = max(history.history['accuracy'])
    val_acc = max(history.history['val_accuracy'])

    return {
        'Model': model_name,
        'Key': model_key,
        'Train Acc': round(train_acc, 4),
        'Val Acc': round(val_acc, 4),
        'Test Acc': round(test_acc, 4),
        'F1-Score': round(f1, 4),
        'Params': model.count_params()
    }

# Store results for all models
all_results = []
print("Helper functions ready! ✓")"""))

# ──── Model 1: CNN ────
cells.append(make_md("### 🔷 Model 1: Custom CNN"))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# MODEL 1: CUSTOM CNN
# ═══════════════════════════════════════════════════════════════

cnn_model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    # Classifier
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
], name='Custom_CNN')

cnn_model.summary()
result = train_and_evaluate(cnn_model, 'Custom CNN', 'cnn')
all_results.append(result)"""))

# ──── Model 2: MLP ────
cells.append(make_md("### 🔷 Model 2: MLP (Multi-Layer Perceptron)"))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# MODEL 2: MLP
# ═══════════════════════════════════════════════════════════════

mlp_model = Sequential([
    Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
], name='MLP')

mlp_model.summary()
result = train_and_evaluate(mlp_model, 'MLP', 'mlp')
all_results.append(result)"""))

# ──── Model 3: VGG16 ────
cells.append(make_md("### 🔷 Model 3: VGG16 (Transfer Learning)"))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# MODEL 3: VGG16
# ═══════════════════════════════════════════════════════════════

vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
vgg_base.trainable = False  # Freeze base layers

vgg_model = Sequential([
    vgg_base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
], name='VGG16_Transfer')

vgg_model.summary()
result = train_and_evaluate(vgg_model, 'VGG16', 'vgg16')
all_results.append(result)"""))

# ──── Model 4: ResNet50 ────
cells.append(make_md("### 🔷 Model 4: ResNet50 (Transfer Learning)"))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# MODEL 4: RESNET50
# ═══════════════════════════════════════════════════════════════

resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
resnet_base.trainable = False

resnet_model = Sequential([
    resnet_base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
], name='ResNet50_Transfer')

resnet_model.summary()
result = train_and_evaluate(resnet_model, 'ResNet50', 'resnet50')
all_results.append(result)"""))

# ──── Model 5: EfficientNetB0 ────
cells.append(make_md("### 🔷 Model 5: EfficientNetB0 (Transfer Learning)"))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# MODEL 5: EFFICIENTNETB0
# ═══════════════════════════════════════════════════════════════

effnet_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
effnet_base.trainable = False

effnet_model = Sequential([
    effnet_base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
], name='EfficientNetB0_Transfer')

effnet_model.summary()
result = train_and_evaluate(effnet_model, 'EfficientNetB0', 'efficientnet')
all_results.append(result)"""))

# ──── Model 6: MobileNetV2 ────
cells.append(make_md("### 🔷 Model 6: MobileNetV2 (Transfer Learning)"))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# MODEL 6: MOBILENETV2
# ═══════════════════════════════════════════════════════════════

mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
mobilenet_base.trainable = False

mobilenet_model = Sequential([
    mobilenet_base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
], name='MobileNetV2_Transfer')

mobilenet_model.summary()
result = train_and_evaluate(mobilenet_model, 'MobileNetV2', 'mobilenet')
all_results.append(result)"""))

# ══════════════════════════════════════════════════════════════
# Step 7: Model Comparison
# ══════════════════════════════════════════════════════════════
cells.append(make_md("## 🏆 Step 7 — Model Comparison"))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# STEP 7: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════

results_df = pd.DataFrame(all_results)
results_df['Params'] = results_df['Params'].apply(lambda x: f"{x:,}")

print("\\n" + "=" * 90)
print("  MODEL COMPARISON — ALL 6 ARCHITECTURES")
print("=" * 90)
print(results_df[['Model', 'Train Acc', 'Val Acc', 'Test Acc', 'F1-Score', 'Params']].to_string(index=False))
print("=" * 90)

# Find best model
best_idx = results_df['Test Acc'].astype(float).idxmax()
best_model_info = results_df.iloc[best_idx]
print(f"\\n🏆 Best Model: {best_model_info['Model']}")
print(f"   Test Accuracy: {best_model_info['Test Acc']}")
print(f"   F1-Score: {best_model_info['F1-Score']}")

# Copy best model
import shutil
best_key = best_model_info['Key']
best_src = f'/content/models/{best_key}_model.h5'
best_dst = '/content/models/best_model.h5'
shutil.copy2(best_src, best_dst)
print(f"\\n✓ Best model saved as: models/best_model.h5")

# Comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.25

bars1 = ax.bar(x - width, results_df['Train Acc'].astype(float), width, label='Train Acc', color='#4dabf7')
bars2 = ax.bar(x, results_df['Val Acc'].astype(float), width, label='Val Acc', color='#9775fa')
bars3 = ax.bar(x + width, results_df['Test Acc'].astype(float), width, label='Test Acc', color='#ff6b6b')

ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Model Comparison — Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=15)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.1)

for bar in bars1 + bars2 + bars3:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('/content/plots/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: plots/model_comparison.png")"""))

# ══════════════════════════════════════════════════════════════
# Download
# ══════════════════════════════════════════════════════════════
cells.append(make_md("""## 📥 Download Models & Plots
Run the cell below to download all trained models and generated plots as a zip file.
Place the contents in your local `alzheimer_project/` directory."""))

cells.append(make_code("""# ═══════════════════════════════════════════════════════════════
# DOWNLOAD ALL ARTIFACTS
# ═══════════════════════════════════════════════════════════════

import shutil
from google.colab import files

# Create zip of models and plots
shutil.make_archive('/content/alzheimer_artifacts', 'zip', '/content', 'models')

# Also zip plots
shutil.make_archive('/content/alzheimer_plots', 'zip', '/content', 'plots')

print("Downloading models...")
files.download('/content/alzheimer_artifacts.zip')

print("Downloading plots...")
files.download('/content/alzheimer_plots.zip')

print("\\n✓ Download complete!")
print("\\nNext steps:")
print("  1. Extract alzheimer_artifacts.zip → place .h5 files in alzheimer_project/models/")
print("  2. Extract alzheimer_plots.zip → place .png files in alzheimer_project/plots/")
print("  3. Run: python app.py")
print("  4. Open: http://localhost:5050")"""))

# ──── Write notebook ────
output_path = r'c:\Users\Anvisha\Brain\alzheimer_project\ml_pipeline.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✓ Notebook created: {output_path}")
print(f"  Total cells: {len(cells)}")
