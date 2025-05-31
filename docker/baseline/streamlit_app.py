# Enhanced Streamlit Application with Full Integration
# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from PIL import Image
import tempfile
import json
from datetime import datetime
from pdf2image import convert_from_bytes
import platform

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
try:
    from document_parser import DocumentParser
    from fraud_detector import FraudDetector
    from explainer import FraudExplainer, ReportGenerator
    from config import Config
    from utils import setup_logging, validate_invoice_data, calculate_risk_metrics, format_currency, create_fraud_summary
    from auto_responder import AutoResponder, add_auto_response_tab
    from rule_based_detector import RuleBasedFraudDetector
    from layoutlm_parser import AdvancedDocumentParser
    from slack_alert import send_slack_alert
    import numpy as np
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Financial Document Fraud Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Setup logging
try:
    logger = setup_logging()
except NameError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Fallback logging initialized due to missing setup_logging function")

# Initialize session state
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'config' not in st.session_state:
    st.session_state.config = Config()

# Load models (cache for performance)
@st.cache_resource
def load_models():
    try:
        parser = DocumentParser()
        detector = FraudDetector()
        rule_detector = RuleBasedFraudDetector()

        # Setup rule-based detector with common fraud rules
        rule_detector.add_rule('high_amount', lambda x: x['amount'] > 10000, weight=0.4)
        rule_detector.add_rule('weekend_invoice', lambda x: pd.to_datetime(x['invoice_date']).weekday() > 4, weight=0.2)
        rule_detector.add_rule('round_number', lambda x: x['amount'] % 100 == 0 and x['amount'] > 1000, weight=0.3)
        rule_detector.add_rule('duplicate_vendor_same_day', lambda x: check_duplicate_vendor_same_day(x), weight=0.5)

        # Initialize auto responder if API key is available
        auto_responder = None
        if st.session_state.config.OPENAI_API_KEY:
            auto_responder = AutoResponder(st.session_state.config.OPENAI_API_KEY)

        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Check if synthetic data exists
        synthetic_data_path = 'data/synthetic_invoices.csv'
        if not os.path.exists(synthetic_data_path):
            # Generate synthetic data
            st.info("Generating synthetic data for demo...")
            from synthetic_data_generator import generate_invoice_data
            df = generate_invoice_data(1000, 0.1)
            df.to_csv(synthetic_data_path, index=False)

        # Load pre-trained model if exists
        model_path = 'models/fraud_model.pkl'
        if os.path.exists(model_path):
            detector.load_model(model_path)
        else:
            # Train on synthetic data for demo
            if os.path.exists(synthetic_data_path):
                df = pd.read_csv(synthetic_data_path)
                detector.train(df)
                detector.save_model(model_path)
            else:
                st.error("Could not load or create synthetic data")
                return None, None, None, None

        return parser, detector, rule_detector, auto_responder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        logger.error(f"Error loading models: {e}")
        return None, None, None, None

def check_duplicate_vendor_same_day(row):
    """Helper function to check for duplicate vendors on same day"""
    # This would need access to the full dataset
    # Simplified implementation for demo
    return False

def combine_results(ml_fraud_result, rule_fraud_result, ml_weight=0.6, rule_weight=0.4, fraud_threshold=0.5):
    """
    Combine ML and rule-based fraud detection results into a final decision and score.

    Args:
        ml_fraud_result (dict): {'is_fraud': bool, 'confidence': float}
        rule_fraud_result (dict): {'is_fraud': bool, 'score': float}
        ml_weight (float): weight for ML model confidence (default 0.6)
        rule_weight (float): weight for rule score (default 0.4)
        fraud_threshold (float): threshold above which combined result is fraud (default 0.5)

    Returns:
        dict: {
            'is_fraud': bool,
            'score': float  # combined fraud risk score between 0 and 1
        }
    """

    # Normalize rule score if necessary (assumed between 0 and 1)
    rule_score = rule_fraud_result.get('score', 0)

    # Combine weighted scores
    combined_score = ml_weight * ml_fraud_result.get('confidence', 0) + rule_weight * rule_score

    # Decide if fraud based on threshold
    is_fraud = combined_score >= fraud_threshold

    return {
        'is_fraud': is_fraud,
        'score': combined_score
    }

def analyze_document_comprehensive(parser, detector, rule_detector, uploaded_file, use_advanced_parser):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name

        # Convert PDF to image
        if uploaded_file.type == 'application/pdf':
            images = convert_from_bytes(uploaded_file.getbuffer())
            image = images[0]
        elif uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
        else:
            return {'error': f'Unsupported file type: {uploaded_file.type}'}

        # Parse document
        if use_advanced_parser:
            try:
                advanced_parser = AdvancedDocumentParser()
                layout_result = advanced_parser.parse_with_layoutlm(temp_path)
                parsing_result = {
                    'invoice_id': 'LAYOUTLM_PARSED',
                    'amount': 0,
                    'date': '2024-01-01',
                    'extraction_confidence': 0.8,
                    'raw_text': ' '.join(layout_result['tokens'])
                }
            except Exception as e:
                st.warning(f"Advanced parser failed, falling back to standard parser: {e}")
                parsing_result = parser.parse(image)
        else:
            parsing_result = parser.parse(image)

        if 'error' not in parsing_result:
            if not validate_invoice_data(parsing_result):
                st.warning("Extracted data validation failed - some required fields may be missing")

            # Construct DataFrame
            doc_df = pd.DataFrame([{
                'invoice_id': parsing_result['invoice_id'],
                'vendor_name': 'Unknown',
                'amount': parsing_result['amount'],
                'invoice_date': parsing_result['date'] or '2024-01-01',
                'category': 'Unknown'
            }])

            # ML fraud detection
            ml_fraud_result = detector.predict(doc_df)
            ml_is_fraud = ml_fraud_result['fraud_predictions'][0]
            ml_confidence = ml_fraud_result['confidence_scores'][0]

            # Rule-based fraud detection
            rule_fraud_result = rule_detector.detect_fraud(doc_df)
            rule_is_fraud = rule_fraud_result['is_fraud'][0]
            rule_score = rule_fraud_result['fraud_scores'][0]
            triggered_rules = rule_fraud_result['triggered_rules'][0]

            # Combined score
            combined_score = (ml_confidence * 0.7) + (rule_score * 0.3)
            combined_is_fraud = combined_score > st.session_state.config.FRAUD_THRESHOLD

            results = {
                'parsing_result': parsing_result,
                'ml_fraud': {
                    'is_fraud': ml_is_fraud,
                    'confidence': ml_confidence
                },
                'rule_fraud': {
                    'is_fraud': rule_is_fraud,
                    'score': rule_score,
                    'triggered_rules': triggered_rules
                },
                'combined': {
                    'is_fraud': combined_is_fraud,
                    'score': combined_score
                },
                'doc_df': doc_df
            }

            if combined_is_fraud and st.session_state.config.SLACK_BOT_TOKEN:
                alert_message = f"""
🚨 FRAUD ALERT 🚨
Document: {uploaded_file.name}
Invoice ID: {parsing_result['invoice_id']}
Amount: {format_currency(parsing_result['amount'])}
Combined Risk Score: {combined_score:.2%}
Triggered Rules: {', '.join(triggered_rules) if triggered_rules else 'None'}
"""
                send_slack_alert(alert_message)
        else:
            results = {'error': parsing_result.get('error', 'Unknown parsing error')}

        os.unlink(temp_path)
        return results

    except Exception as e:
        logger.error(f"Error in document analysis: {e}")
        return {'error': str(e)}


def main():
    st.title("🔍 Financial Document Fraud Analyzer")
    st.markdown("Advanced fraud detection with ML, rules, and automated responses")

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")

    # Fraud threshold slider
    fraud_threshold = st.sidebar.slider(
        "Fraud Detection Threshold",
        0.1, 0.9,
        st.session_state.config.FRAUD_THRESHOLD,
        help="Threshold for combined fraud score"
    )
    st.session_state.config.FRAUD_THRESHOLD = fraud_threshold

    # Parser selection
    use_advanced_parser = st.sidebar.checkbox(
        "Use Advanced Parser (LayoutLM)",
        help="Use LayoutLM for better document understanding"
    )

    # API Configuration status
    st.sidebar.subheader("🔗 Integration Status")
    st.sidebar.write("OpenAI API:", "✅" if st.session_state.config.OPENAI_API_KEY else "⚠️ Missing")
    st.sidebar.write("Slack Integration:", "✅" if st.session_state.config.SLACK_BOT_TOKEN else "⚠️ Missing")

    # Load models
    parser, detector, rule_detector, auto_responder = load_models()
    if not all([parser, detector, rule_detector]):
        st.error("Failed to load essential models.")
        return

    # Tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Document Upload",
        "📊 Batch Analysis",
        "🔍 Fraud Dashboard",
        "📈 Analytics",
        "🤖 Auto Response",
        "⚙️ Rule Management"
    ])

    with tab1:
        st.header("Single Document Analysis")

        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'jpg', 'jpeg', 'png'],
            help="Upload invoice or loan application"
        )

        if uploaded_file is not None:
            if uploaded_file.type == 'application/pdf':
                st.info("PDF uploaded - preview not available, processing will convert PDF pages to images internally.")
            elif uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Document", use_column_width=True)

            if st.button("🔍 Analyze Document", type="primary"):
                with st.spinner("Processing document with comprehensive analysis..."):
                    results = analyze_document_comprehensive(
                        parser, detector, rule_detector, uploaded_file, use_advanced_parser
                    )

                    if results.get('error') is None:
                        # Display results in organized columns
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("📋 Extracted Information")
                            st.json(results['parsing_result'])

                            st.subheader("🧠 ML Analysis")
                            ml_result = results['ml_fraud']
                            if ml_result.get('is_fraud', False):
                                st.error(f"🚨 ML FRAUD DETECTED (Confidence: {ml_result.get('confidence', 0):.2f})")
                            else:
                                st.success(f"✅ ML: Document appears legitimate (Confidence: {1-float(ml_result.get('confidence', 0)):.2f})")

                        with col2:
                            st.subheader("📏 Rule-Based Analysis")
                            rule_result = results['rule_fraud']

                            if rule_result.get('is_fraud', False):
                                st.error(f"🚨 RULES TRIGGERED (Score: {rule_result.get('score', 0):.2f})")
                                if rule_result.get('triggered_rules', []):
                                    st.write("**Triggered Rules:**")
                                    for rule in rule_result.get('triggered_rules', []):
                                        st.write(f"• {rule}")
                            else:
                                st.success(f"✅ RULES: No fraud indicators (Score: {rule_result.get('score', 0):.2f})")

                        # Combined analysis
                        st.subheader("🎯 Combined Analysis")
                        combined = results['combined']

                        col3, col4 = st.columns([1, 2])

                        with col3:
                            if combined.get('is_fraud', False):
                                st.error(f"🚨 **FRAUD DETECTED**")
                                st.write(f"**Combined Score:** {combined.get('score', 0):.2%}")
                            else:
                                st.success(f"✅ **Document Appears Legitimate**")
                                st.write(f"**Combined Score:** {combined.get('score', 0):.2%}")

                        with col4:
                            # Display confidence meter
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = combined.get('score', 0) * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Combined Fraud Risk %"},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkred" if combined.get('is_fraud', False) else "darkgreen"},
                                    'steps': [
                                        {'range': [0, 25], 'color': "lightgreen"},
                                        {'range': [25, 50], 'color': "yellow"},
                                        {'range': [50, 75], 'color': "orange"},
                                        {'range': [75, 100], 'color': "red"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': fraud_threshold * 100
                                    }
                                }
                            ))
                            st.plotly_chart(fig, use_container_width=True)

                        # Store result for later use
                        result_data = {
                            'document': uploaded_file.name,
                            'parsed_data': results['parsing_result'],
                            'fraud_detected': combined.get('is_fraud', False),
                            'confidence': combined.get('score', 0),
                            'ml_result': results['ml_fraud'],
                            'rule_result': results['rule_fraud'],
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state.processed_documents.append(result_data)

                        logger.info(f"Processed document {uploaded_file.name} - Fraud: {combined.get('is_fraud', False)}, Score: {combined.get('score', 0)}")

                    else:
                        st.error(f"❌ Error processing document: {results['error']}")
                        logger.error(f"Document processing error: {results['error']}")

    with tab2:
        st.header("📊 Batch Document Analysis")

        uploaded_files = st.file_uploader(
            "Choose multiple documents",
            type=['pdf', 'jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple documents for batch processing"
        )

        if uploaded_files and st.button("🔍 Analyze All Documents", type="primary"):
            if "progress_bar" not in st.session_state:
                st.session_state.progress_bar = st.progress(0)
            if "status_text" not in st.session_state:
                st.session_state.status_text = st.empty()
            results = []

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")

                try:
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))

                    analysis_results = analyze_document_comprehensive(
                        parser, detector, rule_detector, file, use_advanced_parser
                    )

                    if 'error' not in analysis_results:
                        results.append({
                            'Document': file.name,
                            'Invoice_ID': analysis_results['parsing_result'].get('invoice_id', ''),
                            'Amount': analysis_results['parsing_result'].get('amount', 0),
                            'ML_Fraud': analysis_results['ml_fraud'].get('is_fraud', False),
                            'ML_Confidence': analysis_results['ml_fraud'].get('confidence', 0),
                            'Rule_Fraud': analysis_results['rule_fraud'].get('is_fraud', False),
                            'Rule_Score': analysis_results['rule_fraud'].get('score', 0),
                            'Combined_Fraud': analysis_results['combined'].get('is_fraud', False),
                            'Combined_Score': analysis_results['combined'].get('score', 0),
                            'Triggered_Rules': ', '.join(analysis_results['rule_fraud'].get('triggered_rules', []) or [])
                        })
                    else:
                        st.error(f"Error processing {file.name}: {analysis_results['error']}")

                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
                    logger.error(f"Batch processing error for {file.name}: {e}")

            status_text.text("Processing complete!")

            # Display batch results
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Documents", len(results))
                with col2:
                    st.metric("ML Fraud Cases", sum(df_results['ML_Fraud']))
                with col3:
                    st.metric("Rule Fraud Cases", sum(df_results['Rule_Fraud']))
                with col4:
                    st.metric("Combined Fraud Cases", sum(df_results['Combined_Fraud']))

                # Risk metrics
                if 'calculate_risk_metrics' in globals() and callable(globals()['calculate_risk_metrics']):
                    risk_metrics = globals()['calculate_risk_metrics'](df_results.rename(columns={'Combined_Score': 'fraud_risk'}))
                    st.subheader("📊 Risk Metrics")
                    st.json(risk_metrics)

                # Create fraud summary
                fraud_cases = df_results[df_results['Combined_Fraud']].to_dict('records')
                if fraud_cases and 'create_fraud_summary' in globals() and callable(globals()['create_fraud_summary']):
                    fraud_summary = globals()['create_fraud_summary']([
                        {
                            'invoice_id': case['Invoice_ID'],
                            'amount': case['Amount'],
                            'risk': case['Combined_Score']
                        } for case in fraud_cases
                    ])
                    st.text_area("📋 Fraud Summary", fraud_summary, height=200)

    with tab3:
        st.header("🔍 Fraud Detection Dashboard")

        if st.session_state.processed_documents:
            # Convert to DataFrame
            dashboard_data = []
            for doc in st.session_state.processed_documents:
                dashboard_data.append({
                    'Document': doc['document'],
                    'Amount': doc['parsed_data'].get('amount', 0),
                    'Combined_Risk': doc['confidence'],
                    'ML_Risk': doc.get('ml_result', {}).get('confidence', 0),
                    'Rule_Score': doc.get('rule_result', {}).get('score', 0),
                    'Status': 'FRAUD' if doc['fraud_detected'] else 'CLEAN',
                    'Timestamp': doc.get('timestamp', 'Unknown')
                })

            df_dashboard = pd.DataFrame(dashboard_data)

            # Combined fraud risk distribution
            fig_hist = px.histogram(
                df_dashboard,
                x='Combined_Risk',
                title='Combined Fraud Risk Distribution',
                color='Status',
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # ML vs Rule comparison
            fig_scatter = px.scatter(
                df_dashboard,
                x='ML_Risk',
                y='Rule_Score',
                range_x=[0,1], range_y=[0,1],
                color='Status',
                size='Amount',
                title='ML Risk vs Rule Score Comparison',
                hover_data=['Document']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Amount vs Risk scatter
            fig_amount = px.scatter(
                df_dashboard,
                x='Amount',
                y='Combined_Risk',
                color='Status',
                title='Amount vs Combined Fraud Risk',
                hover_data=['Document']
            )
            st.plotly_chart(fig_amount, use_container_width=True)

        else:
            st.info("📄 No documents processed yet. Upload documents in the Document Upload tab.")

    with tab4:
        st.header("📈 Analytics & Insights")

        # Load sample data for demo
        synthetic_data_path = 'data/synthetic_invoices.csv'
        if os.path.exists(synthetic_data_path):
            try:
                df_sample = pd.read_csv(synthetic_data_path)

                # Add tabs for different analytics
                analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(["Category Analysis", "Trend Analysis", "Model Performance"])

                with analytics_tab1:
                    # Fraud by category
                    fraud_by_category = df_sample.groupby('category')['is_fraud'].agg(['count', 'sum']).reset_index()
                    fraud_by_category['fraud_rate'] = fraud_by_category['sum'] / fraud_by_category['count']

                    fig_category = px.bar(
                        fraud_by_category,
                        x='category',
                        y='fraud_rate',
                        title='Fraud Rate by Category'
                    )
                    st.plotly_chart(fig_category, use_container_width=True)

                    # Amount distribution
                    fig_amount = px.box(
                        df_sample,
                        x='is_fraud',
                        y='amount',
                        title='Amount Distribution: Fraud vs Legitimate'
                    )
                    st.plotly_chart(fig_amount, use_container_width=True)

                with analytics_tab2:
                    # Time-based analysis
                    df_sample['invoice_date'] = pd.to_datetime(df_sample['invoice_date'], errors='coerce')
                    df_sample = df_sample.dropna(subset=['invoice_date'])
                    df_sample['month'] = df_sample['invoice_date'].dt.to_period('M')

                    monthly_fraud = df_sample.groupby('month')['is_fraud'].agg(['count', 'sum']).reset_index()
                    monthly_fraud['fraud_rate'] = monthly_fraud['sum'] / monthly_fraud['count']
                    monthly_fraud['month'] = monthly_fraud['month'].astype(str)

                    fig_trend = px.line(
                        monthly_fraud,
                        x='month',
                        y='fraud_rate',
                        title='Fraud Rate Trend Over Time'
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

                with analytics_tab3:
                    # Model performance metrics (simulated)
                    st.subheader("Model Performance Comparison")

                    performance_data = {
                        'Model': ['ML Model', 'Rule-Based', 'Combined'],
                        'Precision': [0.85, 0.78, 0.88],
                        'Recall': [0.82, 0.71, 0.85],
                        'F1-Score': [0.83, 0.74, 0.86]
                    }

                    df_performance = pd.DataFrame(performance_data)

                    fig_performance = px.bar(
                        df_performance.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                        x='Model',
                        y='Score',
                        color='Metric',
                        barmode='group',
                        title='Model Performance Comparison'
                    )
                    st.plotly_chart(fig_performance, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading analytics data: {e}")
                logger.error(f"Analytics error: {e}")
        else:
            st.info("📊 No synthetic data available for analytics.")

    with tab5:
        if auto_responder:
            if 'add_auto_response_tab' in globals() and callable(globals()['add_auto_response_tab']):
                globals()['add_auto_response_tab'](st, auto_responder)
            else:
                st.error("Auto-response tab function not available")
        else:
            st.header("🤖 Auto-Response System")
            st.warning("⚠️ Auto-response system requires OpenAI API key. Please configure in your environment variables.")
            st.info("Set OPENAI_API_KEY in your .env file to enable automated email generation.")

    with tab6:
        st.header("⚙️ Rule Management")

        st.subheader("Current Active Rules")
        if rule_detector and hasattr(rule_detector, 'rules') and rule_detector.rules:
            rules_data = []
            for rule in rule_detector.rules:
                rules_data.append({
                    'Rule Name': rule['name'],
                    'Weight': rule['weight'],
                    'Status': 'Active'
                })

            df_rules = pd.DataFrame(rules_data)
            st.dataframe(df_rules, use_container_width=True)
        else:
            st.info("No rules configured")

        st.subheader("Add New Rule")
        with st.form("add_rule_form"):
            rule_name = st.text_input("Rule Name")
            rule_description = st.text_area("Rule Description")
            rule_weight = st.slider("Rule Weight", 0.1, 1.0, 0.5)

            # Predefined rule templates
            rule_template = st.selectbox("Rule Template", [
                "Custom",
                "High Amount Threshold",
                "Weekend/Holiday Invoice",
                "Round Number Amount",
                "Frequent Vendor Same Day"
            ])

            if st.form_submit_button("Add Rule"):
                if rule_detector and hasattr(rule_detector, 'rules'):
                    new_rule = {
                        'name': rule_name,
                        'description': rule_description,
                        'weight': rule_weight
                    }
                    rule_detector.rules.append(new_rule)
                    st.success(f"Rule '{rule_name}' added successfully.")
                else:
                    st.error("Rule engine is not initialized or cannot add rules.")

if __name__ == "__main__":
    main()
