# Phase 4: Explainability & Reporting
# SHAP Integration
# explainer.py
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

class FraudExplainer:
    def __init__(self, fraud_detector):
        self.fraud_detector = fraud_detector
        self.explainer = None

    def create_surrogate_model(self, df):
        """Create a surrogate model for SHAP explanations"""
        X = self.fraud_detector.create_features(df)
        X_scaled = self.fraud_detector.scaler.transform(X)

        # Get fraud predictions from isolation forest
        predictions = self.fraud_detector.isolation_forest.predict(X_scaled)
        y = (predictions == -1).astype(int)

        # Train surrogate random forest
        self.surrogate_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.surrogate_model.fit(X_scaled, y)

        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.surrogate_model)

        return X_scaled, y

    def explain_prediction(self, df, index=0):
        """Explain a single prediction"""
        X = self.fraud_detector.create_features(df)
        X_scaled = self.fraud_detector.scaler.transform(X)

        if self.explainer is None:
            self.create_surrogate_model(df)

        # Get SHAP values for specific instance
        shap_values = self.explainer.shap_values(X_scaled[index:index+1])

        # Create explanation
        explanation = {
            'shap_values': shap_values[1][0],  # For fraud class
            'feature_names': self.fraud_detector.feature_names,
            'feature_values': X_scaled[index],
            'base_value': self.explainer.expected_value[1]
        }

        return explanation

    def generate_explanation_text(self, explanation, original_data):
        """Generate human-readable explanation"""
        feature_impacts = list(zip(
            explanation['feature_names'],
            explanation['shap_values'],
            explanation['feature_values']
        ))

        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        explanations = []
        for feature, impact, value in feature_impacts[:5]:  # Top 5 features
            if abs(impact) > 0.01:  # Only significant impacts
                direction = "increases" if impact > 0 else "decreases"
                explanations.append(f"• {feature} (value: {value:.2f}) {direction} fraud likelihood by {abs(impact):.3f}")

        return explanations

# GPT-4 Integration for Natural Language Reports
class ReportGenerator:
    def __init__(self, api_key):
        import openai
        self.client = openai.OpenAI(api_key=api_key)

    def generate_fraud_report(self, invoice_data, fraud_prediction, explanation):
        """Generate natural language fraud report"""

        prompt = f"""
        Generate a professional fraud analysis report for the following invoice:

        Invoice ID: {invoice_data.get('invoice_id', 'Unknown')}
        Vendor: {invoice_data.get('vendor_name', 'Unknown')}
        Amount: ${invoice_data.get('amount', 0):,.2f}
        Date: {invoice_data.get('invoice_date', 'Unknown')}

        Fraud Risk Level: {'HIGH' if fraud_prediction else 'LOW'}

        Key Risk Factors:
        {chr(10).join(explanation)}

        Please provide:
        1. Executive summary
        2. Risk assessment
        3. Recommended actions
        4. Investigation priority level

        Keep it professional and actionable.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Report generation failed: {str(e)}"
