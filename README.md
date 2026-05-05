# 🧠 Alzheimer's Disease Detection System

**AI-powered Alzheimer's disease classification from brain MRI scans using deep learning.**

Built with TensorFlow, Keras, Flask, and the OASIS Brain MRI Dataset.

---

## 📋 Overview

This project implements an end-to-end machine learning pipeline for detecting Alzheimer's disease severity from MRI brain scans. It includes:

- **6 trained deep learning models** (CNN, MLP, VGG16, ResNet50, EfficientNetB0, MobileNetV2)
- **Flask web application** with user authentication and real-time prediction
- **4 classification classes**: Non Demented, Very Mild Demented, Mild Demented, Moderate Demented

## 🏗️ Project Structure

```
alzheimer_project/
├── alzheimer_env/           ← Virtual environment
├── dataset/                 ← OASIS images organized by class
├── models/                  ← Saved .h5 model files
├── plots/                   ← Accuracy/loss/confusion matrix PNGs
├── static/
│   ├── css/style.css        ← Dark-blue medical theme
│   ├── js/script.js         ← UI interactions
│   └── uploads/             ← Uploaded MRI images
├── templates/               ← HTML templates (8 files)
├── ml_pipeline.ipynb        ← Colab notebook (training)
├── ml_pipeline.py           ← Training script (local)
├── create_notebook.py       ← Generates the .ipynb file
├── app.py                   ← Flask application
├── database.py              ← SQLite DB setup
├── requirements.txt
└── README.md
```

## 🚀 Setup Instructions

### 1. Create Virtual Environment

```bash
cd alzheimer_project
python -m venv alzheimer_env

# Windows
alzheimer_env\Scripts\activate

# macOS/Linux
source alzheimer_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Models (Google Colab)

Since training 6 deep learning models requires a GPU:

1. Open `ml_pipeline.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set Runtime → Change runtime type → **T4 GPU**
3. Upload your `Data.zip` when prompted
4. Run all cells (training takes ~2-4 hours on T4)
5. Download the generated `alzheimer_artifacts.zip` and `alzheimer_plots.zip`
6. Extract:
   - `.h5` model files → `alzheimer_project/models/`
   - `.png` plot files → `alzheimer_project/plots/`

### 4. Run Flask Application

```bash
python app.py
```

### 5. Open in Browser

```
http://localhost:5050
```

## 🌐 Web Application Pages

| Route | Description |
|---|---|
| `/` or `/home` | Landing page with hero section and statistics |
| `/about` | Disease info, dataset details, tech stack |
| `/register` | User registration |
| `/login` | User login |
| `/methodology` | ML pipeline walkthrough with training plots |
| `/predict` | Upload MRI scan for AI analysis (login required) |
| `/logout` | Logout and clear session |

## 🧪 Models Trained

| Model | Type | Description |
|---|---|---|
| CNN | Custom | 3-block Conv2D architecture |
| MLP | Custom | Multi-layer perceptron |
| VGG16 | Transfer Learning | ImageNet pretrained |
| ResNet50 | Transfer Learning | ImageNet pretrained |
| EfficientNetB0 | Transfer Learning | ImageNet pretrained |
| MobileNetV2 | Transfer Learning | ImageNet pretrained |

## 📊 Dataset

- **Source**: OASIS (Open Access Series of Imaging Studies)
- **Total Images**: ~86,437
- **Classes**: 4 (Non Demented, Very Mild, Mild, Moderate)
- **Image Size**: 224 × 224 pixels (RGB)

## 🛠️ Technologies

- **ML/DL**: TensorFlow, Keras, Scikit-learn
- **Web**: Flask, Bootstrap 5, SQLite
- **Image Processing**: OpenCV, Pillow
- **Visualization**: Matplotlib, Seaborn

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. Results should not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare professional.

---

*Built with ❤️ using Deep Learning and Flask*
