# Phase 5: UI & Integration
# Streamlit Application
# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from PIL import Image
import tempfile

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
try:
    from document_parser import DocumentParser
    from fraud_detector import FraudDetector
    from explainer import FraudExplainer, ReportGenerator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Financial Document Fraud Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []

# Load models (cache for performance)
@st.cache_resource
def load_models():
    try:
        parser = DocumentParser()
        detector = FraudDetector()

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
                return None, None

        return parser, detector
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def main():
    st.title("🔍 Financial Document Fraud Analyzer")
    st.markdown("Upload invoices and loan applications to detect potential fraud")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    fraud_threshold = st.sidebar.slider("Fraud Detection Threshold", 0.1, 0.9, 0.5)

    # Load models
    models = load_models()
    if models[0] is None or models[1] is None:
        st.error("Failed to load models. Please check the logs.")
        return

    parser, detector = models

    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Upload", "📊 Batch Analysis", "🔍 Fraud Dashboard", "📈 Analytics"])

    with tab1:
        st.header("Single Document Analysis")

        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'jpg', 'jpeg', 'png'],
            help="Upload invoice or loan application"
        )

        if uploaded_file is not None:
            # Display uploaded image
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Document", use_column_width=True)

            if st.button("Analyze Document"):
                with st.spinner("Processing document..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            temp_path = tmp_file.name

                        # Parse document
                        parsing_result = parser.parse_document(temp_path)

                        if 'error' not in parsing_result:
                            # Create DataFrame for fraud detection
                            doc_df = pd.DataFrame([{
                                'invoice_id': parsing_result['invoice_id'],
                                'vendor_name': 'Unknown',  # Would need vendor detection
                                'amount': parsing_result['amount'],
                                'invoice_date': parsing_result['date'] or '2024-01-01',
                                'category': 'Unknown'
                            }])

                            # Detect fraud
                            fraud_result = detector.predict(doc_df)
                            is_fraud = fraud_result['fraud_predictions'][0]
                            confidence = fraud_result['confidence_scores'][0]

                            # Display results
                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("Extracted Information")
                                st.json(parsing_result)

                            with col2:
                                st.subheader("Fraud Analysis")
                                if is_fraud:
                                    st.error(f"🚨 FRAUD DETECTED (Confidence: {confidence:.2f})")
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

                            # Store result
                            result_data = {
                                'document': uploaded_file.name,
                                'parsed_data': parsing_result,
                                'fraud_detected': is_fraud,
                                'confidence': confidence
                            }
                            st.session_state.processed_documents.append(result_data)
                        else:
                            st.error(f"Error parsing document: {parsing_result.get('error', 'Unknown error')}")

                        # Clean up temp file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass

                    except Exception as e:
                        st.error(f"Error processing document: {e}")

    with tab2:
        st.header("Batch Document Analysis")

        uploaded_files = st.file_uploader(
            "Choose multiple documents",
            type=['pdf', 'jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Analyze All Documents"):
            progress_bar = st.progress(0)
            results = []

            for i, file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))

                    # Process each file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp_file:
                        tmp_file.write(file.getbuffer())
                        temp_path = tmp_file.name

                    parsing_result = parser.parse_document(temp_path)

                    if 'error' not in parsing_result:
                        doc_df = pd.DataFrame([{
                            'invoice_id': parsing_result['invoice_id'],
                            'vendor_name': 'Unknown',
                            'amount': parsing_result['amount'],
                            'invoice_date': parsing_result['date'] or '2024-01-01',
                            'category': 'Unknown'
                        }])

                        fraud_result = detector.predict(doc_df)

                        results.append({
                            'Document': file.name,
                            'Invoice_ID': parsing_result['invoice_id'],
                            'Amount': parsing_result['amount'],
                            'Fraud_Detected': fraud_result['fraud_predictions'][0],
                            'Confidence': fraud_result['confidence_scores'][0]
                        })

                    try:
                        os.unlink(temp_path)
                    except:
                        pass

                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

            # Display batch results
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Documents", len(results))
                with col2:
                    st.metric("Fraud Cases", sum(df_results['Fraud_Detected']))
                with col3:
                    st.metric("Fraud Rate", f"{sum(df_results['Fraud_Detected'])/len(results)*100:.1f}%")

    with tab3:
        st.header("Fraud Detection Dashboard")

        if st.session_state.processed_documents:
            # Convert to DataFrame
            dashboard_data = []
            for doc in st.session_state.processed_documents:
                dashboard_data.append({
                    'Document': doc['document'],
                    'Amount': doc['parsed_data'].get('amount', 0),
                    'Fraud_Risk': doc['confidence'],
                    'Status': 'FRAUD' if doc['fraud_detected'] else 'CLEAN'
                })

            df_dashboard = pd.DataFrame(dashboard_data)

            # Fraud risk distribution
            fig_hist = px.histogram(
                df_dashboard,
                x='Fraud_Risk',
                title='Fraud Risk Distribution',
                color='Status',
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Amount vs Risk scatter
            fig_scatter = px.scatter(
                df_dashboard,
                x='Amount',
                y='Fraud_Risk',
                color='Status',
                title='Amount vs Fraud Risk',
                hover_data=['Document']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No documents processed yet. Upload documents in the first tab.")

    with tab4:
        st.header("Analytics & Insights")

        # Load sample data for demo
        synthetic_data_path = 'data/synthetic_invoices.csv'
        if os.path.exists(synthetic_data_path):
            try:
                df_sample = pd.read_csv(synthetic_data_path)

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

            except Exception as e:
                st.error(f"Error loading analytics data: {e}")
        else:
            st.info("No synthetic data available for analytics.")

if __name__ == "__main__":
    main()
