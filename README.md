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

```text

fraud-detection-system/
│
├── document_parser.py          # Document parsing logic
├── fraud_detector.py           # Fraud detection model (Isolation Forest & rules)
├── explainer.py                # SHAP explainability and GPT-4 report generator
├── streamlit_app.py            # Streamlit UI for analysis and visualization
├── slack_bot.py                # Slack alert integration
├── synthetic_invoices.csv      # Sample data for training/testing
├── fraud_model.pkl             # Saved model artifact
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── docs/                      # Documentation, logos, diagrams, etc.

```
