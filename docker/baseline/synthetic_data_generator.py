# Sample Data Sources
# synthetic_data_generator.py
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

def generate_invoice_data(n_invoices=1000, fraud_rate=0.1):
    """Generate synthetic invoice data with fraud cases"""

    invoices = []
    vendors = [fake.company() for _ in range(50)]  # 50 unique vendors

    for i in range(n_invoices):
        vendor = random.choice(vendors)
        base_amount = random.uniform(100, 10000)

        # Determine if this is a fraud case
        is_fraud = random.random() < fraud_rate

        if is_fraud:
            # Create fraud patterns
            fraud_type = random.choice(['inflated_amount', 'duplicate', 'fake_vendor'])

            if fraud_type == 'inflated_amount':
                amount = base_amount * random.uniform(2, 5)  # 2-5x inflation
            elif fraud_type == 'duplicate':
                amount = base_amount
            else:  # fake_vendor
                vendor = fake.company() + " (FAKE)"
                amount = base_amount
        else:
            amount = base_amount

        invoice = {
            'invoice_id': f'INV-{i+1:04d}',
            'vendor_name': vendor,
            'amount': round(amount, 2),
            'invoice_date': fake.date_between(start_date='-1y', end_date='today'),
            'due_date': fake.date_between(start_date='today', end_date='+30d'),
            'payment_terms': random.choice(['Net 30', 'Net 15', 'Due on Receipt']),
            'category': random.choice(['Office Supplies', 'Marketing', 'IT Services', 'Consulting']),
            'is_fraud': is_fraud
        }
        invoices.append(invoice)

    return pd.DataFrame(invoices)

# Generate and save data
df = generate_invoice_data(1000, 0.1)
df.to_csv('synthetic_invoices.csv', index=False)
print(f"Generated {len(df)} invoices with {df['is_fraud'].sum()} fraud cases")
