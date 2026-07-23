# 🛡️ AI-Powered Android Adware Screening System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Screen Android adware with 99.32% accuracy using LightGBM and explainable AI with multi-LLM support.**

---

## ✨ Features

- 🎯 **99.32% Accuracy** - LightGBM gradient boosting classifier
- 📊 **SHAP Explanations** - Understand feature contributions for each prediction
- 🤖 **Multi-LLM Support** - AI-powered insights via multiple providers
- 📈 **Interactive Visualizations** - Radar charts, feature importance, SHAP waterfall plots
- 🔄 **Counterfactual Analysis** - Understand what would change the classification
- 📦 **Batch Processing** - Analyze multiple apps at once
- 📥 **Multiple Input Methods** - CSV upload, sample data, or manual entry
- 💾 **Export Results** - Download reports as CSV

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open your browser to `http://localhost:8501`

A hosted demo is available upon request.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.32% |
| **F1-Score** | 99.42% |
| **ROC-AUC** | 99.90% |
| **MCC** | 0.9860 |
| **Misclassifications** | 33 / 4,839 |

Trained on **24,192 Android apps** (14,149 adware, 10,043 benign) using 22 optimized features.

---

## 🎯 How to Use

### 1️⃣ Screening Tab
- Upload CSV, load sample data, or enter features manually
- Click "Run Adware Screening"
- View classification result with confidence and risk level

### 2️⃣ Explainability Tab
- View SHAP waterfall plots showing feature contributions
- Explore feature profiles with radar charts
- See counterfactual suggestions

### 3️⃣ AI Chat Tab
- Select your LLM provider
- Enter your API key
- Get AI-powered security advisories and technical analysis

### 4️⃣ Batch Analysis Tab
- View summary statistics for multiple samples
- Download results as CSV

### 5️⃣ Report Tab
- Generate formatted screening reports

---

## 📁 Input Format

Your CSV file should contain these 22 features:
