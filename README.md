# AI-Powered-Phishing-Detection
AI-Powered Phishing Detection System is a machine learning-based web application that detects phishing websites in real time. It uses Python, Streamlit, and scikit-learn to analyze website URLs and content, classifying them as legitimate or phishing to help users stay safe online.

---

# 🧠 AI-Powered Phishing Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

The *AI-Powered Phishing Detection System* is a machine learning-based web application that identifies *phishing websites* in real time.

It leverages *AI, Natural Language Processing (NLP), and web scraping* techniques to analyze website URLs and content, predicting whether a site is *legitimate or phishing*.

Built using *Streamlit*, this tool provides a simple and interactive interface for safe browsing and cybersecurity awareness.

---

## 🚀 Features

- ✅ Detects phishing vs. legitimate websites in real-time  
- 🧩 Uses NLP and ML for intelligent classification  
- 📊 Displays confidence scores for each prediction  
- 💾 Saves and loads trained models using joblib  
- 🧠 Built using popular Python libraries like pandas, scikit-learn, nltk, and beautifulsoup4

---

## 🧰 Tech Stack & Libraries

| Category | Technology |
|-----------|-------------|
| Programming Language | Python |
| Data Handling | pandas |
| Machine Learning | scikit-learn |
| NLP Processing | nltk |
| Web Scraping | BeautifulSoup4, requests |
| Web App Framework | Streamlit |
| Model Storage | joblib |

---

## ⚙ How It Works

1. *Data Collection:*  
   - Gathers website content and metadata using requests and BeautifulSoup4.

2. *Feature Extraction:*  
   - Extracts textual and structural features using nltk and pandas.

3. *Model Training:*  
   - Trains a classifier (e.g., Logistic Regression or Random Forest) using scikit-learn.

4. *Model Saving:*  
   - Saves the trained model using joblib for reuse.

5. *Web App Interface:*  
   - Streamlit app allows users to input URLs and get instant phishing detection results.

---

## 🧪 Installation & Usage

### 🔹 Clone the repository
```bash
git clone https://github.com/yourusername/ai-phishing-detection.git
cd ai-phishing-detection
```

### 🔹 Install dependencies
```bash
pip install -r requirements.txt
```

### 🔹 Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📈 Example Output

- 🔍 URL: http://phishy-example.com
- ⚠ Prediction: Phishing Website
- 📊 Confidence: 94.7%

---

## 📚 Future Enhancements

✉ Add email phishing detection

🌐 Browser extension for real-time alerts

🧮 Integrate deep learning models (LSTM/CNN)

🔍 Connect API for live URL scanning

---

## 🤝 Contributing

- Contributions, issues, and feature requests are welcome!
- Feel free to open an issue or submit a pull request to improve the project.


---
