import streamlit as st
from slack_alert import send_slack_alert

st.title("📄 Financial Document Analyzer")

uploaded_file = st.file_uploader("Upload a financial document (PDF/Image):", type=["pdf", "png", "jpg"])

if uploaded_file:
    st.success("Document uploaded successfully!")

    # Placeholder: parsing + fraud detection logic here
    # Replace below with actual model inference code
    st.info("Running fraud detection model...")
    is_fraudulent = True  # Mock result

    if is_fraudulent:
        st.error("🚨 Fraud Detected!")
        explanation = "Invoice amount is 50% higher than vendor average."
        st.write("Explanation:", explanation)

        if st.button("Send Alert to Slack"):
            send_slack_alert(f"🚨 Fraudulent document detected. Reason: {explanation}")
    else:
        st.success("✅ No fraud detected in the document.")
