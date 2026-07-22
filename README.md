# 🛡️ AI-Powered Android Adware Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cybersecurity-adware-detection-dmhvvdlyvbta7cxwm8ifcm.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Detect Android adware with 99.32% accuracy using LightGBM and explainable AI with multi-LLM support.**

🌐 **[Try it live!](https://cybersecurity-adware-detection-dmhvvdlyvbta7cxwm8ifcm.streamlit.app/)** ← Click to use the app online!

---

## ✨ Features

- 🎯 **99.32% Accuracy** - LightGBM gradient boosting classifier
- 📊 **SHAP Explanations** - Understand feature contributions for each prediction
- 🤖 **Multi-LLM Support** - AI-powered insights via Claude, GPT-4, or Gemini
- 📈 **Interactive Visualizations** - Radar charts, feature importance, SHAP waterfall plots
- 🔄 **Counterfactual Analysis** - Understand what would change the classification
- 📦 **Batch Processing** - Analyze multiple apps at once
- 📥 **Multiple Input Methods** - CSV upload, sample data, or manual entry
- 💾 **Export Results** - Download reports as CSV
- 🌐 **Cloud Deployed** - No installation needed, use in browser

---

## 🚀 Quick Start

### Option 1: Use Online (Recommended)

Visit: **[https://cybersecurity-adware-detection-dmhvvdlyvbta7cxwm8ifcm.streamlit.app/](https://cybersecurity-adware-detection-dmhvvdlyvbta7cxwm8ifcm.streamlit.app/)**

No installation required!

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/parisahjb/Cybersecurity-Adware-Detection.git
cd Cybersecurity-Adware-Detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open your browser to `http://localhost:8501`

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

### 1️⃣ Detection Tab
- Upload CSV, load sample data, or enter features manually
- Click "Run Adware Detection"
- View classification result with confidence and risk level

### 2️⃣ Explainability Tab
- View SHAP waterfall plots showing feature contributions
- Explore feature profiles with radar charts
- See counterfactual suggestions

### 3️⃣ AI Chat Tab
- Select your LLM provider (Claude, GPT-4, or Gemini)
- Enter your API key
- Get AI-powered security advisories and technical analysis

### 4️⃣ Batch Analysis Tab
- View summary statistics for multiple samples
- Download results as CSV

### 5️⃣ Report Tab
- Generate formatted detection reports

---

## 📁 Input Format

Your CSV file should contain these 22 features:
```
network_op, sqllite_op, fileio_op, no_action, noc, dit, lcom, cbo, 
ppiv, apd, start_activities, start_activity, start_service, 
start_action_mode, start_activity_result, start_activity_from_frag, 
start_activity_needed, start_intent_for_result, start_search, 
methods_per_class, bytecode_per_method, cyclomatic
```

Features can be extracted using the [Android Quality Metrics Tool](https://github.com/sealuzh/user_quality).

---

## 🛠️ Technology Stack

- **Model**: LightGBM (Gradient Boosting)
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Frontend**: Streamlit
- **Visualization**: Plotly
- **LLM Integration**: Anthropic Claude, OpenAI GPT-4, Google Gemini
- **Deployment**: Streamlit Cloud

---

## 📂 Repository Structure
```
├── app.py                          # Main Streamlit application
├── adware_model_22features.joblib  # Trained LightGBM model
├── sample_test_data.csv            # Sample data for testing
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---



**Built with ❤️ using Python, Streamlit, LightGBM, and SHAP**

⭐ **Star this repo if you find it useful!** ⭐

</div>
