import openai
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

class AutoResponder:
    def __init__(self, openai_api_key):
        self.client = openai.OpenAI(api_key=openai_api_key)

    def generate_vendor_email(self, invoice_data, fraud_details):
        """Generate email to vendor about suspicious invoice"""

        prompt = f"""
        Generate a professional email to a vendor regarding a potentially fraudulent invoice.

        Invoice Details:
        - Invoice ID: {invoice_data.get('invoice_id')}
        - Amount: ${invoice_data.get('amount', 0):,.2f}
        - Date: {invoice_data.get('invoice_date')}

        Fraud Indicators:
        {fraud_details}

        The email should:
        1. Be professional and diplomatic
        2. Request clarification without directly accusing
        3. Ask for supporting documentation
        4. Provide a deadline for response

        Subject line and body needed.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Email generation failed: {str(e)}"

    def generate_internal_alert(self, invoice_data, fraud_score):
        """Generate internal alert for finance team"""

        prompt = f"""
        Create an internal alert email for the finance team about a high-risk invoice.

        Invoice: {invoice_data.get('invoice_id')}
        Vendor: {invoice_data.get('vendor_name')}
        Amount: ${invoice_data.get('amount', 0):,.2f}
        Risk Score: {fraud_score:.1%}

        Include:
        1. Urgent action required notice
        2. Investigation steps
        3. Hold payment instruction
        4. Contact information for questions
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Alert generation failed: {str(e)}"

# Integration with Streamlit app
def add_auto_response_tab(st, auto_responder):
    """Add auto-response functionality to Streamlit app"""

    st.header("🤖 Auto-Response System")

    # Select processed document
    if st.session_state.processed_documents:
        doc_options = [doc['document'] for doc in st.session_state.processed_documents]
        selected_doc = st.selectbox("Select document for auto-response:", doc_options)

        if selected_doc:
            doc_data = next(doc for doc in st.session_state.processed_documents
                           if doc['document'] == selected_doc)

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Generate Vendor Email"):
                    with st.spinner("Generating email..."):
                        email_content = auto_responder.generate_vendor_email(
                            doc_data['parsed_data'],
                            f"Fraud score: {doc_data['confidence']:.1%}"
                        )
                        st.text_area("Generated Email:", email_content, height=300)

            with col2:
                if st.button("Generate Internal Alert"):
                    with st.spinner("Generating alert..."):
                        alert_content = auto_responder.generate_internal_alert(
                            doc_data['parsed_data'],
                            doc_data['confidence']
                        )
                        st.text_area("Internal Alert:", alert_content, height=300)
    else:
        st.info("No processed documents available for auto-response.")
