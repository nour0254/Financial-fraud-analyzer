# Phase 3: Fraud Detection Model
# Anomaly Detection Implementation
# fraud_detector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class FraudDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expected fraud rate
            random_state=42,
            n_estimators=100
        )
        self.feature_names = []

    def create_features(self, df):
        """Create features for fraud detection"""
        features = df.copy()

        # Vendor-based features
        vendor_stats = df.groupby('vendor_name')['amount'].agg(['mean', 'std', 'count']).reset_index()
        vendor_stats.columns = ['vendor_name', 'vendor_avg_amount', 'vendor_std_amount', 'vendor_frequency']
        features = features.merge(vendor_stats, on='vendor_name', how='left')

        # Amount-based features
        features['amount_zscore'] = (features['amount'] - features['amount'].mean()) / features['amount'].std()
        features['amount_deviation_from_vendor_avg'] = features['amount'] - features['vendor_avg_amount']
        features['amount_deviation_ratio'] = features['amount'] / features['vendor_avg_amount']

        # Time-based features
        features['invoice_date'] = pd.to_datetime(features['invoice_date'])
        features['day_of_week'] = features['invoice_date'].dt.dayofweek
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)

        # Category encoding
        features['category_encoded'] = pd.Categorical(features['category']).codes

        # Select numerical features for model
        feature_columns = [
            'amount', 'amount_zscore', 'amount_deviation_from_vendor_avg',
            'amount_deviation_ratio', 'vendor_avg_amount', 'vendor_std_amount',
            'vendor_frequency', 'day_of_week', 'is_weekend', 'category_encoded'
        ]

        # Handle NaN values
        features[feature_columns] = features[feature_columns].fillna(0)

        self.feature_names = feature_columns
        return features[feature_columns]

    def train(self, df):
        """Train the fraud detection model"""
        # Create features
        X = self.create_features(df)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train isolation forest
        self.isolation_forest.fit(X_scaled)

        # Get predictions for evaluation
        predictions = self.isolation_forest.predict(X_scaled)
        anomaly_scores = self.isolation_forest.score_samples(X_scaled)

        # Convert predictions (-1 for anomaly, 1 for normal)
        fraud_predictions = (predictions == -1).astype(int)

        return {
            'predictions': fraud_predictions,
            'anomaly_scores': anomaly_scores,
            'feature_importance': self._get_feature_importance(X)
        }

    def predict(self, df):
        """Predict fraud for new data"""
        X = self.create_features(df)
        X_scaled = self.scaler.transform(X)

        predictions = self.isolation_forest.predict(X_scaled)
        anomaly_scores = self.isolation_forest.score_samples(X_scaled)

        fraud_predictions = (predictions == -1).astype(int)
        confidence_scores = 1 - ((anomaly_scores - anomaly_scores.min()) /
                                (anomaly_scores.max() - anomaly_scores.min()))

        return {
            'fraud_predictions': fraud_predictions,
            'confidence_scores': confidence_scores,
            'anomaly_scores': anomaly_scores
        }

    def _get_feature_importance(self, X):
        """Calculate feature importance based on variance"""
        return dict(zip(self.feature_names, X.var().values))

    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump({
            'scaler': self.scaler,
            'model': self.isolation_forest,
            'feature_names': self.feature_names
        }, filepath)

    def load_model(self, filepath):
        """Load trained model"""
        saved_objects = joblib.load(filepath)
        self.scaler = saved_objects['scaler']
        self.isolation_forest = saved_objects['model']
        self.feature_names = saved_objects['feature_names']

# Usage example
detector = FraudDetector()
df = pd.read_csv('data/synthetic_invoices.csv')
results = detector.train(df)
print(f"Detected {sum(results['predictions'])} potential fraud cases")
