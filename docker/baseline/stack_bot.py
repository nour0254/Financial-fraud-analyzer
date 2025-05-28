# Slack Integration
# slack_bot.py
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os
from datetime import datetime

class SlackNotifier:
    def __init__(self, token):
        self.client = WebClient(token=token)

    def send_fraud_alert(self, channel, invoice_data, fraud_score):
        """Send fraud detection alert to Slack"""

        color = "danger" if fraud_score >
