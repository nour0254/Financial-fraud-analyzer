import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
import os

def setup_logging():
    log_path = os.path.join("/tmp", "fraud_analyzer.log")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
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
