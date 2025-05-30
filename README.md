# Financial Document Fraud Analyzer

🔍 **A comprehensive end-to-end system for detecting potential fraud in financial documents using machine learning, anomaly detection, rule-based heuristics, explainability tools, and an interactive Streamlit UI dashboard.**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Document Parsing](#document-parsing)
  - [Fraud Detection](#fraud-detection)
  - [Explainability & Reporting](#explainability--reporting)
  - [Streamlit UI](#streamlit-ui)
  - [Slack Alerts](#slack-alerts)
- [Model Training](#model-training)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Project Overview

This project builds an automated fraud detection pipeline for financial documents (invoices, loan applications, etc.) by combining:

- **Document parsing** from PDFs and images to extract key fields
- **Machine learning-based anomaly detection** using Isolation Forest
- **Rule-based heuristics** for quick fraud spotting
- **Explainability** through surrogate models and SHAP for transparent insights
- **Natural language report generation** with GPT-4 integration
- **Interactive Streamlit dashboard** for single/batch analysis and visualizations
- **Slack integration** for real-time fraud alerts

Designed for ease of use by fraud analysts, auditors, and finance teams.

---

## Features

- PDF/Image parsing to extract invoice ID, vendor, amount, date, and category
- Isolation Forest-based anomaly detection tuned for expected fraud rates
- Customizable rule-based fraud detection rules
- Feature importance via variance and SHAP explanations
- GPT-powered professional fraud analysis reports
- Intuitive Streamlit UI with multi-tab support:
  - Single document analysis with confidence visualization
  - Batch document upload and summary reporting
  - Fraud dashboard with risk distribution and scatter plots
  - Analytics on fraud trends and amounts
- Slack bot for instant fraud alert notifications

---

## Project Architecture

```text
flowchart TD
    A[Upload Document] --> B[DocumentParser]
    B --> C[Feature Engineering]
    C --> D{FraudDetector}
    D -->|IsolationForest| E[Anomaly Scores & Predictions]
    D -->|RuleBasedDetector| F[Rule Scores]
    E & F --> G[Fraud Decision]
    G --> H[Explainability Module (SHAP)]
    H --> I[ReportGenerator (GPT-4)]
    I --> J[Streamlit UI Dashboard]
    G --> K[SlackNotifier Alerts]
```

---

## 📋 Complete Project Structure

```text
financial_fraud_analyzer/
├── requirements.txt
├── synthetic_data_generator.py
├── document_parser.py
├── fraud_detector.py
├── explainer.py
├── streamlit_app.py
├── slack_bot.py
├── auto_responder.py
├── config.py
├── utils.py
├── data/
│   ├── synthetic_invoices.csv
│   └── sample_documents/
├── models/
│   └── fraud_model.pkl
├── tests/
│   ├── test_parser.py
│   ├── test_detector.py
│   └── test_integration.py
└── README.md
```
---

## ⚡ Quick Wins & Time-Saving Tips

### 1. Pre-trained Models to Use
- **EasyOCR**: Faster than training LayoutLM
- **Isolation Forest**: No labeled data needed
- **OpenAI GPT-4**: For explanations and reports
- **Streamlit**: Rapid UI development

### 2. MVP Feature Priority
1. **Document upload + basic OCR** (2 hours)
2. **Simple rule-based fraud detection** (1 hour)
3. **Basic Streamlit UI** (2 hours)
4. **Results visualization** (1 hour)

---
### Deployment Options
```bash
# Local Streamlit
streamlit run streamlit_app.py

# Docker deployment
docker build -t fraud-analyzer .
docker run -p 8501:8501 fraud-analyzer

# Streamlit Cloud (for demo)
# Push to GitHub and deploy via share.streamlit.io
```
---

## 📚 Additional Resources

### Helpful Libraries Documentation
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [SHAP Documentation](https://shap.readthedocs.io/)

### Sample Datasets
- [Kaggle Invoice Dataset](https://www.kaggle.com/datasets)
- [Synthetic Financial Data Generators](https://github.com/sdv-dev/SDV)

### Model Alternatives
If time is tight, consider these quick alternatives:
- **Z-score based detection**: Flag amounts > 2 standard deviations
- **Rule-based systems**: Hardcode suspicious patterns
- **Statistical outlier detection**: Use percentiles and IQR
