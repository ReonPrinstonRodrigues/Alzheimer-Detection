"""
ml_pipeline.py — Alzheimer's Disease Detection ML Pipeline
Full training script for local execution (requires GPU recommended).
For Colab training, use ml_pipeline.ipynb instead.
"""

import os
import cv2
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0, MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import shutil

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
# Fallback: if dataset/ doesn't exist, use the parent Data/ directory
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Data')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

CLASS_MAP = {
    'Mild Dementia': 'Mild Demented',
    'Moderate Dementia': 'Moderate Demented',
    'Non Demented': 'Non Demented',
    'Very mild Dementia': 'Very Mild Demented'
}

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# STEP 1: PROBLEM ANALYSIS
# ═══════════════════════════════════════════════════════════════

def step1_problem_analysis():
    print("=" * 70)
    print("  STEP 1: PROBLEM ANALYSIS")
    print("=" * 70)
    print("""
PROBLEM: Multi-class classification of Alzheimer's disease severity
         from MRI brain scan images.

DATASET: OASIS (Open Access Series of Imaging Studies) Brain MRI Dataset

CLASSES (4):
  1. Non Demented      — Normal cognitive function
  2. Very Mild Demented — Early memory lapses
  3. Mild Demented      — Noticeable memory loss
  4. Moderate Demented  — Significant cognitive decline

CHALLENGES:
  • Severe class imbalance
  • Subtle visual differences between early stages
  • Need for high sensitivity
    """)


# ═══════════════════════════════════════════════════════════════
# STEP 2: DATA COLLECTION
# ═══════════════════════════════════════════════════════════════

def step2_data_collection():
    print("=" * 70)
    print("  STEP 2: DATA COLLECTION")
    print("=" * 70)

    image_paths = []
    image_labels = []

    print(f"Scanning: {DATA_DIR}")
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

    print(f"\nTotal images: {len(image_paths)}")
    return image_paths, image_labels


# ═══════════════════════════════════════════════════════════════
# STEP 3: DATA CLEANING
# ═══════════════════════════════════════════════════════════════

def step3_data_cleaning(image_paths, image_labels):
    print("\n" + "=" * 70)
    print("  STEP 3: DATA CLEANING")
    print("=" * 70)

    clean_paths, clean_labels = [], []
    corrupted, duplicates = 0, 0
    hash_set = set()

    for path, label in tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="Cleaning"):
        try:
            img = cv2.imread(path)
            if img is None:
                corrupted += 1
                continue
        except Exception:
            corrupted += 1
            continue

        img_hash = hashlib.md5(cv2.resize(img, (64, 64)).tobytes()).hexdigest()
        if img_hash in hash_set:
            duplicates += 1
            continue

        hash_set.add(img_hash)
        clean_paths.append(path)
        clean_labels.append(label)

    print(f"\n  Original: {len(image_paths)} | Corrupted: {corrupted} | Duplicates: {duplicates} | Clean: {len(clean_paths)}")
    return clean_paths, clean_labels


# ═══════════════════════════════════════════════════════════════
# STEP 4: TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════

def step4_split(clean_paths, clean_labels, class_names):
    print("\n" + "=" * 70)
    print("  STEP 4: TRAIN/VAL/TEST SPLIT (70/15/15)")
    print("=" * 70)

    label_to_idx = {name: i for i, name in enumerate(class_names)}
    encoded = np.array([label_to_idx[l] for l in clean_labels])
    paths = np.array(clean_paths)

    X_train_p, X_temp_p, y_train, y_temp = train_test_split(
        paths, encoded, test_size=0.30, stratify=encoded, random_state=42)
    X_val_p, X_test_p, y_val, y_test = train_test_split(
        X_temp_p, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    print(f"  Train: {len(X_train_p)} | Val: {len(X_val_p)} | Test: {len(X_test_p)}")
    return X_train_p, X_val_p, X_test_p, y_train, y_val, y_test


# ═══════════════════════════════════════════════════════════════
# STEP 5: LOAD & BALANCE
# ═══════════════════════════════════════════════════════════════

def load_images(paths, labels, desc="Loading"):
    images, valid_labels = [], []
    for path, label in tqdm(zip(paths, labels), total=len(paths), desc=desc):
        try:
            img = cv2.imread(path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            valid_labels.append(label)
        except Exception:
            continue
    return np.array(images, dtype='float32') / 255.0, np.array(valid_labels)


def step5_balance(X_train_p, X_val_p, X_test_p, y_train, y_val, y_test, class_names):
    print("\n" + "=" * 70)
    print("  STEP 5: LOADING IMAGES & DATA BALANCING")
    print("=" * 70)

    X_train, y_train_l = load_images(X_train_p, y_train, "Train")
    X_val, y_val_l = load_images(X_val_p, y_val, "Val")
    X_test, y_test_l = load_images(X_test_p, y_test, "Test")

    num_classes = len(class_names)
    y_train_cat = to_categorical(y_train_l, num_classes)
    y_val_cat = to_categorical(y_val_l, num_classes)
    y_test_cat = to_categorical(y_test_l, num_classes)

    weights_arr = compute_class_weight('balanced', classes=np.unique(y_train_l), y=y_train_l)
    class_weights = dict(enumerate(weights_arr))
    print("Class weights:", {class_names[k]: round(v, 3) for k, v in class_weights.items()})

    # Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20, horizontal_flip=True,
        zoom_range=0.15, width_shift_range=0.1, height_shift_range=0.1)
    train_gen = train_datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE)
    val_gen = ImageDataGenerator().flow(X_val, y_val_cat, batch_size=BATCH_SIZE)

    # Plot class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2ecc71', '#e67e22', '#e74c3c', '#f39c12']
    train_counts = Counter(y_train_l)
    classes = [class_names[i] for i in range(num_classes)]
    counts = [train_counts.get(i, 0) for i in range(num_classes)]

    axes[0].bar(classes, counts, color=colors)
    axes[0].set_title('Class Distribution (Training)')
    axes[1].bar(classes, [class_weights[i] for i in range(num_classes)], color=colors)
    axes[1].set_title('Class Weights')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'), dpi=150)
    plt.close()

    return (X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat,
            class_weights, train_gen, val_gen)


# ═══════════════════════════════════════════════════════════════
# STEP 6: TRAINING
# ═══════════════════════════════════════════════════════════════

def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]

def plot_history(history, key):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train', color='#4dabf7')
    axes[0].plot(history.history['val_accuracy'], label='Val', color='#ff6b6b')
    axes[0].set_title(f'{key} — Accuracy')
    axes[0].legend()
    axes[1].plot(history.history['loss'], label='Train', color='#4dabf7')
    axes[1].plot(history.history['val_loss'], label='Val', color='#ff6b6b')
    axes[1].set_title(f'{key} — Loss')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{key}_accuracy.png'), dpi=150)
    plt.close()

def plot_cm(y_true, y_pred, key, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{key} — Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{key}_confusion_matrix.png'), dpi=150)
    plt.close()

def train_model(model, name, key, train_gen, val_gen, X_train, X_val, X_test,
                y_test_cat, class_weights, class_names):
    print(f"\n{'='*60}\n  Training: {name}\n{'='*60}")

    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Parameters: {model.count_params():,}")

    history = model.fit(
        train_gen, steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS, validation_data=val_gen,
        validation_steps=len(X_val) // BATCH_SIZE,
        class_weight=class_weights, callbacks=get_callbacks(), verbose=1)

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    print(classification_report(y_true, y_pred, target_names=class_names))
    f1 = f1_score(y_true, y_pred, average='weighted')

    plot_history(history, key)
    plot_cm(y_true, y_pred, key, class_names)

    model.save(os.path.join(MODELS_DIR, f'{key}_model.h5'))
    print(f"✓ Saved: models/{key}_model.h5")

    return {
        'Model': name, 'Key': key,
        'Train Acc': round(max(history.history['accuracy']), 4),
        'Val Acc': round(max(history.history['val_accuracy']), 4),
        'Test Acc': round(test_acc, 4),
        'F1-Score': round(f1, 4),
        'Params': model.count_params()
    }


def build_all_models(num_classes):
    models = []

    # 1. CNN
    cnn = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE,IMG_SIZE,3)),
        MaxPooling2D((2,2)), BatchNormalization(), Dropout(0.25),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)), BatchNormalization(), Dropout(0.25),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)), BatchNormalization(), Dropout(0.25),
        Flatten(), Dense(256, activation='relu'), BatchNormalization(),
        Dropout(0.5), Dense(num_classes, activation='softmax')
    ], name='CNN')
    models.append((cnn, 'Custom CNN', 'cnn'))

    # 2. MLP
    mlp = Sequential([
        Flatten(input_shape=(IMG_SIZE,IMG_SIZE,3)),
        Dense(512, activation='relu'), BatchNormalization(), Dropout(0.3),
        Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3),
        Dense(128, activation='relu'), BatchNormalization(), Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ], name='MLP')
    models.append((mlp, 'MLP', 'mlp'))

    # 3. VGG16
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    vgg_base.trainable = False
    vgg = Sequential([vgg_base, GlobalAveragePooling2D(), Dense(256, activation='relu'),
                       BatchNormalization(), Dropout(0.5), Dense(num_classes, activation='softmax')], name='VGG16')
    models.append((vgg, 'VGG16', 'vgg16'))

    # 4. ResNet50
    res_base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    res_base.trainable = False
    resnet = Sequential([res_base, GlobalAveragePooling2D(), Dense(256, activation='relu'),
                          BatchNormalization(), Dropout(0.5), Dense(num_classes, activation='softmax')], name='ResNet50')
    models.append((resnet, 'ResNet50', 'resnet50'))

    # 5. EfficientNetB0
    eff_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    eff_base.trainable = False
    effnet = Sequential([eff_base, GlobalAveragePooling2D(), Dense(256, activation='relu'),
                          BatchNormalization(), Dropout(0.5), Dense(num_classes, activation='softmax')], name='EfficientNetB0')
    models.append((effnet, 'EfficientNetB0', 'efficientnet'))

    # 6. MobileNetV2
    mob_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    mob_base.trainable = False
    mobilenet = Sequential([mob_base, GlobalAveragePooling2D(), Dense(256, activation='relu'),
                             BatchNormalization(), Dropout(0.5), Dense(num_classes, activation='softmax')], name='MobileNetV2')
    models.append((mobilenet, 'MobileNetV2', 'mobilenet'))

    return models


# ═══════════════════════════════════════════════════════════════
# STEP 7: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════

def step7_compare(all_results, class_names):
    print("\n" + "=" * 90)
    print("  STEP 7: MODEL COMPARISON")
    print("=" * 90)

    df = pd.DataFrame(all_results)
    df_display = df.copy()
    df_display['Params'] = df_display['Params'].apply(lambda x: f"{x:,}")
    print(df_display[['Model', 'Train Acc', 'Val Acc', 'Test Acc', 'F1-Score', 'Params']].to_string(index=False))

    best_idx = df['Test Acc'].idxmax()
    best = df.iloc[best_idx]
    print(f"\n🏆 Best: {best['Model']} (Test Acc: {best['Test Acc']}, F1: {best['F1-Score']})")

    shutil.copy2(os.path.join(MODELS_DIR, f"{best['Key']}_model.h5"),
                 os.path.join(MODELS_DIR, 'best_model.h5'))
    print("✓ Saved: models/best_model.h5")

    # Comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df['Test Acc'], w, label='Test Acc', color='#4dabf7')
    ax.bar(x, df['Val Acc'], w, label='Val Acc', color='#9775fa')
    ax.bar(x + w, df['Train Acc'], w, label='Train Acc', color='#ff6b6b')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=15)
    ax.set_title('Model Comparison')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"\nTensorFlow {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}\n")

    step1_problem_analysis()

    image_paths, image_labels = step2_data_collection()
    clean_paths, clean_labels = step3_data_cleaning(image_paths, image_labels)

    class_names = sorted(list(set(clean_labels)))
    X_train_p, X_val_p, X_test_p, y_train, y_val, y_test = step4_split(
        clean_paths, clean_labels, class_names)

    (X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat,
     class_weights, train_gen, val_gen) = step5_balance(
        X_train_p, X_val_p, X_test_p, y_train, y_val, y_test, class_names)

    all_models = build_all_models(len(class_names))
    all_results = []
    for model, name, key in all_models:
        result = train_model(model, name, key, train_gen, val_gen,
                            X_train, X_val, X_test, y_test_cat,
                            class_weights, class_names)
        all_results.append(result)

    step7_compare(all_results, class_names)
    print("\n✓ Pipeline complete!")
