# 🛡️ AI-Powered Android Adware Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Detect Android adware with 99.63% F1-Score accuracy using advanced machine learning and AI-powered explanations.**

🌐 **[Try it live!](https://your-app-url.streamlit.app)** ← Click to use the app online!

---

## ✨ Features

- 🧠 **99.63% F1-Score** - State-of-the-art neural network classifier
- 🤖 **AI Explanations** - Understand every detection with natural language
- ⚡ **Real-time Analysis** - Instant results (<100ms)
- 📊 **Batch Processing** - Analyze multiple apps at once
- 📈 **Feature Analysis** - Interactive visualizations
- 💾 **Export Results** - Download reports as CSV/TXT
- 🌐 **Web-Based** - No installation needed, use in browser

---

## 🚀 Quick Start

### Option 1: Use Online (Easiest)

Just visit: **[https://your-app-url.streamlit.app](https://your-app-url.streamlit.app)**

No installation, no setup, works immediately!

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/adware-detector-web.git
cd adware-detector-web

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## 📊 Performance

| Metric | Value | Status |
|--------|-------|--------|
| **F1-Score** | 99.63% | ⭐⭐⭐⭐⭐ |
| **Accuracy** | 99.57% | ⭐⭐⭐⭐⭐ |
| **Precision** | 99.69% | ⭐⭐⭐⭐⭐ |
| **Recall** | 99.57% | ⭐⭐⭐⭐⭐ |
| **ROC-AUC** | 99.96% | ⭐⭐⭐⭐⭐ |

Trained on **24,192 Android apps** (14,149 adware, 10,043 benign)

---

## 🎯 How to Use

### 1️⃣ Single App Detection

1. Go to **"📱 Single App"** tab
2. Choose input method:
   - **Upload CSV**: Upload app feature data
   - **Manual Input**: Enter features manually
   - **Use Sample**: Test with pre-loaded samples
3. Click **"Analyze"**
4. View results with AI explanation

### 2️⃣ Batch Processing

1. Go to **"📊 Batch Processing"** tab
2. Upload CSV with multiple apps
3. Click **"Process Batch"**
4. Download results and reports

### 3️⃣ Feature Analysis

1. Go to **"📈 Feature Analysis"** tab
2. Explore feature importance
3. Compare model performance
4. Understand what matters most

---

## 📁 Input Format

Your CSV file should contain these features:

```csv
file_name,network_op,http_clients,show_method,cyclomatic,methods,...
app1.apk,45,12,85,15000,7026,...
app2.apk,2,1,25,25000,40000,...
```

**Top Important Features:**
- `network_op` - Network operations count
- `http_clients` - HTTP client instances
- `show_method` - UI display method calls
- `cyclomatic` - Code complexity
- `methods` - Total methods count

[Download sample CSV](sample_test_app.csv)

---

## 🧠 Model Architecture

```
Input Layer (50 features)
      ↓
Hidden Layer 1 (100 neurons, ReLU)
      ↓
Hidden Layer 2 (50 neurons, ReLU)
      ↓
Output Layer (2 classes, Softmax)
```

**Training Details:**
- Model: Neural Network (MLP Classifier)
- Features: 50 selected from 77 engineered features
- Preprocessing: StandardScaler + Yeo-Johnson transformation
- Validation: 5-fold stratified cross-validation
- Optimizer: Adam
- Framework: scikit-learn

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: scikit-learn
- **Visualization**: Plotly
- **Data Processing**: pandas, numpy
- **Deployment**: Streamlit Cloud / Hugging Face

---

## 📦 Deployment

### Deploy to Streamlit Cloud (Free & Easy)

1. **Fork this repository** to your GitHub

2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**

3. **Click "New app"**

4. **Select your repository**
   - Repository: `yourusername/adware-detector-web`
   - Branch: `main`
   - Main file: `app.py`

5. **Click "Deploy"**

6. **Wait 2-3 minutes** ✨

7. **Your app is live!** Share the URL

### Deploy to Hugging Face Spaces

1. Create account on [Hugging Face](https://huggingface.co)
2. Create new Space (Streamlit type)
3. Upload all files from this repo
4. Your app will be live at `huggingface.co/spaces/username/space-name`

### Deploy to Heroku

```bash
# Install Heroku CLI
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main

# Open
heroku open
```

---

## 🎓 Citation

If you use this system in your research or project:

```bibtex
@software{ai_adware_detector_2025,
  title={AI-Powered Android Adware Detection System},
  author={Your Name},
  year={2025},
  note={Neural Network achieving 99.63\% F1-Score},
  url={https://github.com/yourusername/adware-detector-web}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: Android malware samples from research dataset
- Model: Neural Network trained on 24,192 apps
- Framework: Built with Streamlit and scikit-learn
- Inspired by the need for explainable AI in cybersecurity

---

## 📞 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Email**: your@email.com
- **Website**: [yourwebsite.com](https://yourwebsite.com)

---

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/adware-detector-web&type=Date)](https://star-history.com/#yourusername/adware-detector-web)

---

## 📸 Screenshots

### Main Detection Interface
![Detection](https://via.placeholder.com/800x400?text=Add+Your+Screenshot)

### AI-Powered Explanation
![Explanation](https://via.placeholder.com/800x400?text=Add+Your+Screenshot)

### Batch Processing
![Batch](https://via.placeholder.com/800x400?text=Add+Your+Screenshot)

---

## 🔄 Version History

- **v1.0.0** (2025-01-XX) - Initial release
  - Neural Network model (99.63% F1)
  - AI-powered explanations
  - Web-based interface
  - Batch processing
  - Feature analysis

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. Always verify results with additional security tools before making critical decisions. The developers are not responsible for any misuse or damage caused by this software.

---

## 💡 Future Enhancements

- [ ] Real LLM API integration (Claude/GPT)
- [ ] Mobile-responsive design
- [ ] REST API endpoint
- [ ] User authentication
- [ ] Analysis history tracking
- [ ] Custom model training
- [ ] Real-time APK file scanning
- [ ] Integration with VirusTotal

---

<div align="center">

**Built with ❤️ using Python, Streamlit, and AI**

⭐ **Star this repo if you find it useful!** ⭐

</div>
