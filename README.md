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

```mermaid
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

---

## 📋 Complete Project Structure

```
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

### Configuration File

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
    SLACK_CHANNEL = os.getenv('SLACK_CHANNEL', '#fraud-alerts')

    # Model Parameters
    FRAUD_THRESHOLD = 0.5
    CONTAMINATION_RATE = 0.1

    # File Paths
    DATA_PATH = 'data/'
    MODEL_PATH = 'models/'
    TEMP_PATH = 'temp/'

    # Email Configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    EMAIL_USER = os.getenv('EMAIL_USER')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
```

### Utility Functions

```python
# utils.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fraud_analyzer.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_invoice_data(data: Dict) -> bool:
    """Validate extracted invoice data"""
    required_fields = ['invoice_id', 'amount']
    return all(field in data and data[field] is not None for field in required_fields)

def calculate_risk_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate risk metrics for dashboard"""
    total_invoices = len(df)
    total_amount = df['amount'].sum()
    high_risk_count = sum(df['fraud_risk'] > 0.7)

    return {
        'total_invoices': total_invoices,
        'total_amount': total_amount,
        'high_risk_count': high_risk_count,
        'high_risk_percentage': (high_risk_count / total_invoices) * 100 if total_invoices > 0 else 0,
        'average_amount': total_amount / total_invoices if total_invoices > 0 else 0
    }

def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"

def create_fraud_summary(fraud_cases: List[Dict]) -> str:
    """Create summary of fraud cases"""
    if not fraud_cases:
        return "No fraud cases detected."

    summary = f"Detected {len(fraud_cases)} potential fraud cases:\n"
    for i, case in enumerate(fraud_cases[:5], 1):  # Show top 5
        summary += f"{i}. {case['invoice_id']}: {format_currency(case['amount'])} (Risk: {case['risk']:.1%})\n"

    if len(fraud_cases) > 5:
        summary += f"... and {len(fraud_cases) - 5} more cases"

    return summary
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

### 3. Data Strategy
```python
# Quick synthetic data generation
def quick_fraud_data():
    return pd.DataFrame({
        'invoice_id': [f'INV-{i:04d}' for i in range(100)],
        'amount': np.random.lognormal(7, 1, 100),
        'vendor': np.random.choice(['Vendor A', 'Vendor B', 'Vendor C'], 100),
        'is_fraud': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    })
```

### 4. Demo Script
```python
# demo_script.py - For presentation
def run_demo():
    st.title("🎯 Fraud Detection Demo")

    # Pre-loaded suspicious invoice
    demo_invoice = {
        'invoice_id': 'INV-DEMO-001',
        'vendor_name': 'Suspicious Vendor LLC',
        'amount': 25000.00,  # Unusually high
        'invoice_date': '2024-12-25'  # Christmas day - suspicious
    }

    st.json(demo_invoice)

    # Simulate fraud detection
    if st.button("Analyze Demo Invoice"):
        st.error("🚨 FRAUD DETECTED!")
        st.write("**Risk Factors:**")
        st.write("- Amount 300% higher than vendor average")
        st.write("- Invoice dated on holiday")
        st.write("- New vendor with limited history")

        # Show fake SHAP plot
        st.image("demo_shap_plot.png")  # Pre-generated plot
```

---

## 🐛 Common Pitfalls & Debugging Tips

### 1. OCR Issues
```python
# Preprocessing for better OCR
def preprocess_image(image_path):
    import cv2
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.medianBlur(enhanced, 3)

    return denoised
```

### 2. Model Performance Issues
```python
# Quick model validation
def validate_model(detector, test_data):
    predictions = detector.predict(test_data)

    # Check for common issues
    if all(predictions['fraud_predictions'] == 0):
        print("⚠️ Model not detecting any fraud - check threshold")

    if all(predictions['fraud_predictions'] == 1):
        print("⚠️ Model flagging everything as fraud - check features")

    print(f"Fraud rate: {np.mean(predictions['fraud_predictions']):.1%}")
```

### 3. Streamlit Performance
```python
# Cache expensive operations
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

# Use session state for persistence
if 'results' not in st.session_state:
    st.session_state.results = []
```

### 4. Memory Management
```python
# Clean up temporary files
import tempfile
import os

def process_with_cleanup(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.getbuffer())
        tmp_path = tmp.name

    try:
        result = process_document(tmp_path)
        return result
    finally:
        os.unlink(tmp_path)  # Always clean up
```

---

## 🏆 Final Checklist for Hackathon

### Must-Have Features (MVP)
- [ ] Document upload interface
- [ ] Basic text extraction (OCR)
- [ ] Simple fraud detection (rule-based is OK)
- [ ] Results display with confidence scores
- [ ] Basic explanation of why flagged

### Nice-to-Have Features
- [ ] Batch processing
- [ ] Advanced ML models
- [ ] SHAP explanations
- [ ] Slack integration
- [ ] Auto-generated reports

### Demo Preparation
- [ ] Prepare 3-5 test documents (mix of clean and suspicious)
- [ ] Create compelling demo script
- [ ] Prepare backup slides in case of technical issues
- [ ] Test all features before presentation
- [ ] Have sample outputs ready to show

### Technical Checklist
- [ ] All dependencies in requirements.txt
- [ ] Error handling for file uploads
- [ ] Responsive UI design
- [ ] Fast processing (< 10 seconds per document)
- [ ] Clear user feedback and progress indicators

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
