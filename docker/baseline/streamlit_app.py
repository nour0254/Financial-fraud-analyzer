import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from PIL import Image
import tempfile
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
try:
    from document_parser import DocumentParser
    from layoutlm_parser import AdvancedDocumentParser
    from fraud_detector import FraudDetector
    from rule_based_detector import RuleBasedFraudDetector
    from explainer import FraudExplainer, ReportGenerator
    from auto_responder import AutoResponder, add_auto_response_tab
    from config import Config
    from utils import setup_logging, validate_invoice_data, calculate_risk_metrics, create_fraud_summary
    from slack_alert import send_slack_alert
    from synthetic_data_generator import generate_invoice_data
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Setup logging
logger = setup_logging()

# Page config
st.set_page_config(
    page_title="Advanced Financial Document Fraud Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'detection_method' not in st.session_state:
    st.session_state.detection_method = 'isolation_forest'
if 'parser_method' not in st.session_state:
    st.session_state.parser_method = 'easyocr'

# Configuration sidebar
def render_sidebar():
    st.sidebar.header("⚙️ Configuration")

    # Detection method selection
    detection_method = st.sidebar.selectbox(
        "Detection Method",
        ['isolation_forest', 'rule_based', 'hybrid'],
        help="Choose fraud detection approach"
    )
    st.session_state.detection_method = detection_method

    # Parser method selection
    parser_method = st.sidebar.selectbox(
        "Document Parser",
        ['easyocr', 'layoutlm'],
        help="Choose document parsing method"
    )
    st.session_state.parser_method = parser_method

    # Fraud threshold
    fraud_threshold = st.sidebar.slider("Fraud Detection Threshold", 0.1, 0.9, Config.FRAUD_THRESHOLD)

    # API Configuration
    st.sidebar.subheader("🔧 API Configuration")
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password",
                                      value=Config.OPENAI_API_KEY or "")
    slack_token = st.sidebar.text_input("Slack Bot Token", type="password",
                                       value=Config.SLACK_BOT_TOKEN or "")

    # Slack integration toggle
    use_slack = st.sidebar.checkbox("Enable Slack Alerts", value=bool(Config.SLACK_BOT_TOKEN))

    return {
        'detection_method': detection_method,
        'parser_method': parser_method,
        'fraud_threshold': fraud_threshold,
        'openai_key': openai_key,
        'slack_token': slack_token,
        'use_slack': use_slack
    }

# Load models with caching
@st.cache_resource
def load_models(detection_method, parser_method):
    try:
        # Initialize parsers
        if parser_method == 'easyocr':
            parser = DocumentParser()
        else:
            parser = AdvancedDocumentParser()

        # Initialize detectors
        if detection_method == 'isolation_forest':
            detector = FraudDetector()
        elif detection_method == 'rule_based':
            detector = RuleBasedFraudDetector()
            # Add common fraud rules
            detector.add_rule('high_amount', lambda x: x['amount'] > 10000, weight=0.4)
            detector.add_rule('weekend_invoice', lambda x: pd.to_datetime(x['invoice_date']).weekday() > 4, weight=0.2)
            detector.add_rule('round_number', lambda x: x['amount'] % 100 == 0 and x['amount'] > 1000, weight=0.3)
        else:  # hybrid
            isolation_detector = FraudDetector()
            rule_detector = RuleBasedFraudDetector()
            # Setup rules for hybrid approach
            rule_detector.add_rule('high_amount', lambda x: x['amount'] > 10000, weight=0.4)
            rule_detector.add_rule('weekend_invoice', lambda x: pd.to_datetime(x['invoice_date']).weekday() > 4, weight=0.2)
            rule_detector.add_rule('round_number', lambda x: x['amount'] % 100 == 0 and x['amount'] > 1000, weight=0.3)
            detector = {'isolation': isolation_detector, 'rule': rule_detector}

        # Setup data directories
        os.makedirs(Config.DATA_PATH, exist_ok=True)
        os.makedirs(Config.MODEL_PATH, exist_ok=True)
        os.makedirs(Config.TEMP_PATH, exist_ok=True)

        # Load or generate synthetic data
        synthetic_data_path = os.path.join(Config.DATA_PATH, 'synthetic_invoices.csv')
        if not os.path.exists(synthetic_data_path):
            st.info("Generating synthetic data for demo...")
            df = generate_invoice_data(1000, Config.CONTAMINATION_RATE)
            df.to_csv(synthetic_data_path, index=False)

        # Train models if needed
        if detection_method == 'isolation_forest':
            model_path = os.path.join(Config.MODEL_PATH, 'fraud_model.pkl')
            if os.path.exists(model_path):
                detector.load_model(model_path)
            else:
                if os.path.exists(synthetic_data_path):
                    df = pd.read_csv(synthetic_data_path)
                    detector.train(df)
                    detector.save_model(model_path)
        elif detection_method == 'hybrid':
            model_path = os.path.join(Config.MODEL_PATH, 'fraud_model.pkl')
            if os.path.exists(model_path):
                detector['isolation'].load_model(model_path)
            else:
                if os.path.exists(synthetic_data_path):
                    df = pd.read_csv(synthetic_data_path)
                    detector['isolation'].train(df)
                    detector['isolation'].save_model(model_path)

        return parser, detector
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {e}")
        return None, None

def process_single_document(uploaded_file, parser, detector, config):
    """Process a single document with enhanced functionality"""
    try:
        # Display uploaded image
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Document", use_column_width=True)

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name

        # Parse document
        if config['parser_method'] == 'layoutlm':
            parsing_result = parser.parse_with_layoutlm(temp_path)
            # Convert LayoutLM result to standard format
            parsing_result = {
                'invoice_id': 'LAYOUTLM_' + uploaded_file.name.split('.')[0],
                'amount': 1000.0,  # Placeholder - would need proper extraction
                'date': datetime.now().strftime('%Y-%m-%d'),
                'extraction_confidence': 0.8
            }
        else:
            parsing_result = parser.parse_document(temp_path)

        if 'error' not in parsing_result and validate_invoice_data(parsing_result):
            # Create DataFrame for fraud detection
            doc_df = pd.DataFrame([{
                'invoice_id': parsing_result['invoice_id'],
                'vendor_name': 'Unknown',  # Would need vendor detection
                'amount': parsing_result['amount'],
                'invoice_date': parsing_result.get('date', datetime.now().strftime('%Y-%m-%d')),
                'category': 'Unknown'
            }])

            # Detect fraud based on method
            if config['detection_method'] == 'rule_based':
                fraud_result = detector.detect_fraud(doc_df)
                is_fraud = fraud_result['is_fraud'][0]
                confidence = fraud_result['fraud_scores'][0]
                triggered_rules = fraud_result['triggered_rules'][0]
            elif config['detection_method'] == 'hybrid':
                # Combine both methods
                isolation_result = detector['isolation'].predict(doc_df)
                rule_result = detector['rule'].detect_fraud(doc_df)

                isolation_fraud = isolation_result['fraud_predictions'][0]
                rule_fraud = rule_result['is_fraud'][0]

                # Weighted combination
                is_fraud = isolation_fraud or rule_fraud
                confidence = (isolation_result['confidence_scores'][0] + rule_result['fraud_scores'][0]) / 2
                triggered_rules = rule_result['triggered_rules'][0]
            else:  # isolation_forest
                fraud_result = detector.predict(doc_df)
                is_fraud = fraud_result['fraud_predictions'][0]
                confidence = fraud_result['confidence_scores'][0]
                triggered_rules = []

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📄 Extracted Information")
                st.json(parsing_result)

                if triggered_rules:
                    st.subheader("⚠️ Triggered Rules")
                    for rule in triggered_rules:
                        st.warning(f"• {rule}")

            with col2:
                st.subheader("🔍 Fraud Analysis")
                if is_fraud:
                    st.error(f"🚨 FRAUD DETECTED (Confidence: {confidence:.2f})")

                    # Send Slack alert if enabled
                    if config['use_slack'] and config['slack_token']:
                        alert_message = f"🚨 Fraud Alert: {parsing_result['invoice_id']} - Amount: ${parsing_result['amount']:,.2f} - Risk: {confidence:.2%}"
                        send_slack_alert(alert_message)
                        st.info("Slack alert sent!")

                else:
                    st.success(f"✅ Document appears legitimate (Confidence: {1-confidence:.2f})")

                # Display confidence meter
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Risk %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if is_fraud else "darkgreen"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            # Generate AI report if OpenAI key is provided
            if config['openai_key'] and is_fraud:
                with st.expander("📋 AI-Generated Fraud Report"):
                    try:
                        report_generator = ReportGenerator(config['openai_key'])
                        report = report_generator.generate_fraud_report(
                            doc_df.iloc[0].to_dict(),
                            is_fraud,
                            triggered_rules or ["High anomaly score detected"]
                        )
                        st.markdown(report)
                    except Exception as e:
                        st.error(f"Error generating report: {e}")

            # Store result
            result_data = {
                'document': uploaded_file.name,
                'parsed_data': parsing_result,
                'fraud_detected': is_fraud,
                'confidence': confidence,
                'triggered_rules': triggered_rules if 'triggered_rules' in locals() else [],
                'timestamp': datetime.now()
            }
            st.session_state.processed_documents.append(result_data)

            return True
        else:
            st.error(f"Error parsing document: {parsing_result.get('error', 'Invalid data extracted')}")
            return False

    except Exception as e:
        logger.error(f"Error processing document {uploaded_file.name}: {e}")
        st.error(f"Error processing document: {e}")
        return False
    finally:
        # Clean up temp file
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except:
            pass

def main():
    st.title("🔍 Advanced Financial Document Fraud Analyzer")
    st.markdown("Upload invoices and loan applications to detect potential fraud using multiple AI techniques")

    # Render sidebar and get configuration
    config = render_sidebar()

    # Load models
    models = load_models(config['detection_method'], config['parser_method'])
    if models[0] is None or models[1] is None:
        st.error("Failed to load models. Please check the logs.")
        return

    parser, detector = models

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Document Upload",
        "📊 Batch Analysis",
        "🔍 Fraud Dashboard",
        "📈 Analytics",
        "🤖 Auto Response",
        "📋 Reports"
    ])

    with tab1:
        st.header("Single Document Analysis")

        # Method info
        st.info(f"Using {config['parser_method'].upper()} for parsing and {config['detection_method'].upper()} for fraud detection")

        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'jpg', 'jpeg', 'png'],
            help="Upload invoice or loan application"
        )

        if uploaded_file is not None:
            if st.button("🔍 Analyze Document", type="primary"):
                with st.spinner("Processing document..."):
                    process_single_document(uploaded_file, parser, detector, config)

    with tab2:
        st.header("Batch Document Analysis")

        uploaded_files = st.file_uploader(
            "Choose multiple documents",
            type=['pdf', 'jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("🔍 Analyze All Documents", type="primary"):
            progress_bar = st.progress(0)
            results = []

            for i, file in enumerate(uploaded_files):
                progress_bar.progress((i + 1) / len(uploaded_files))
                success = process_single_document(file, parser, detector, config)
                if success:
                    results.append(file.name)

            st.success(f"Successfully processed {len(results)} out of {len(uploaded_files)} documents")

    with tab3:
        st.header("Fraud Detection Dashboard")

        if st.session_state.processed_documents:
            # Real-time metrics
            metrics = calculate_risk_metrics(pd.DataFrame([
                {
                    'amount': doc['parsed_data'].get('amount', 0),
                    'fraud_risk': doc['confidence']
                } for doc in st.session_state.processed_documents
            ]))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Documents", int(metrics['total_invoices']))
            with col2:
                st.metric("High Risk Cases", int(metrics['high_risk_count']))
            with col3:
                st.metric("Risk Rate", f"{metrics['high_risk_percentage']:.1f}%")
            with col4:
                st.metric("Avg Amount", f"${metrics['average_amount']:,.2f}")

            # Recent fraud cases
            fraud_cases = [doc for doc in st.session_state.processed_documents if doc['fraud_detected']]
            if fraud_cases:
                st.subheader("🚨 Recent Fraud Cases")
                fraud_summary = create_fraud_summary([
                    {
                        'invoice_id': case['parsed_data']['invoice_id'],
                        'amount': case['parsed_data']['amount'],
                        'risk': case['confidence']
                    } for case in fraud_cases
                ])
                st.text(fraud_summary)

            # Visualizations
            dashboard_data = []
            for doc in st.session_state.processed_documents:
                dashboard_data.append({
                    'Document': doc['document'],
                    'Amount': doc['parsed_data'].get('amount', 0),
                    'Fraud_Risk': doc['confidence'],
                    'Status': 'FRAUD' if doc['fraud_detected'] else 'CLEAN',
                    'Timestamp': doc.get('timestamp', datetime.now())
                })

            df_dashboard = pd.DataFrame(dashboard_data)

            # Time series of detections
            df_dashboard['Date'] = pd.to_datetime(df_dashboard['Timestamp']).dt.date
            daily_stats = df_dashboard.groupby(['Date', 'Status']).size().unstack(fill_value=0)

            if not daily_stats.empty:
                fig_timeline = px.line(
                    daily_stats.reset_index(),
                    x='Date',
                    y=['FRAUD', 'CLEAN'] if 'FRAUD' in daily_stats.columns else ['CLEAN'],
                    title='Daily Fraud Detection Timeline'
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

        else:
            st.info("No documents processed yet. Upload documents in the first tab.")

    with tab4:
        st.header("Analytics & Insights")

        # Load sample data for comprehensive analytics
        synthetic_data_path = os.path.join(Config.DATA_PATH, 'synthetic_invoices.csv')
        if os.path.exists(synthetic_data_path):
            try:
                df_sample = pd.read_csv(synthetic_data_path)

                col1, col2 = st.columns(2)

                with col1:
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

                with col2:
                    # Amount distribution
                    fig_amount = px.box(
                        df_sample,
                        x='is_fraud',
                        y='amount',
                        title='Amount Distribution: Fraud vs Legitimate'
                    )
                    st.plotly_chart(fig_amount, use_container_width=True)

                # Vendor analysis
                st.subheader("Vendor Risk Analysis")
                vendor_stats = df_sample.groupby('vendor_name').agg({
                    'is_fraud': ['count', 'sum'],
                    'amount': ['mean', 'sum']
                }).round(2)
                vendor_stats.columns = ['Total_Invoices', 'Fraud_Cases', 'Avg_Amount', 'Total_Amount']
                vendor_stats['Fraud_Rate'] = (vendor_stats['Fraud_Cases'] / vendor_stats['Total_Invoices']).round(3)

                # Show top risky vendors
                risky_vendors = vendor_stats[vendor_stats['Fraud_Rate'] > 0].sort_values('Fraud_Rate', ascending=False).head(10)
                if not risky_vendors.empty:
                    st.dataframe(risky_vendors)

            except Exception as e:
                st.error(f"Error loading analytics data: {e}")

    with tab5:
        # Auto Response System
        if config['openai_key']:
            auto_responder = AutoResponder(config['openai_key'])
            add_auto_response_tab(st, auto_responder)
        else:
            st.warning("Please provide OpenAI API key in the sidebar to use auto-response features.")

    with tab6:
        st.header("📋 Comprehensive Reports")

        if st.session_state.processed_documents:
            # Generate comprehensive report
            if st.button("Generate Full Report"):
                with st.spinner("Generating comprehensive report..."):

                    # Summary statistics
                    total_docs = len(st.session_state.processed_documents)
                    fraud_docs = sum(1 for doc in st.session_state.processed_documents if doc['fraud_detected'])
                    total_amount = sum(doc['parsed_data'].get('amount', 0) for doc in st.session_state.processed_documents)
                    fraud_amount = sum(doc['parsed_data'].get('amount', 0) for doc in st.session_state.processed_documents if doc['fraud_detected'])

                    st.markdown(f"""
                    ## 📊 Executive Summary

                    **Analysis Period:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

                    **Document Statistics:**
                    - Total Documents Processed: {total_docs}
                    - Fraud Cases Detected: {fraud_docs}
                    - Fraud Detection Rate: {(fraud_docs/total_docs*100):.1f}%

                    **Financial Impact:**
                    - Total Amount Analyzed: ${total_amount:,.2f}
                    - Fraudulent Amount: ${fraud_amount:,.2f}
                    - Potential Savings: ${fraud_amount:,.2f}

                    **Detection Method:** {config['detection_method'].upper()}
                    **Parser Method:** {config['parser_method'].upper()}
                    """)

                    # Detailed fraud cases
                    if fraud_docs > 0:
                        st.markdown("## 🚨 Fraud Cases Detail")
                        fraud_details = []
                        for doc in st.session_state.processed_documents:
                            if doc['fraud_detected']:
                                fraud_details.append({
                                    'Document': doc['document'],
                                    'Invoice ID': doc['parsed_data'].get('invoice_id', 'Unknown'),
                                    'Amount': f"${doc['parsed_data'].get('amount', 0):,.2f}",
                                    'Risk Score': f"{doc['confidence']:.2%}",
                                    'Triggered Rules': ', '.join(doc.get('triggered_rules', [])) or 'Anomaly Detection'
                                })

                        df_fraud = pd.DataFrame(fraud_details)
                        st.dataframe(df_fraud, use_container_width=True)
        else:
            st.info("No processed documents available for reporting.")

if __name__ == "__main__":
    main()
