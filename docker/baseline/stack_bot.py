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

        color = "danger" if fraud_score > 0.5 else "warning"
        emoji = "🚨" if fraud_score > 0.5 else "⚠️"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Fraud Detection Alert"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Invoice ID:*\n{invoice_data.get('invoice_id', 'Unknown')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Vendor:*\n{invoice_data.get('vendor_name', 'Unknown')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Amount:*\n${invoice_data.get('amount', 0):,.2f}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Risk Score:*\n{fraud_score:.2%}"
                    }
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Investigate"
                        },
                        "style": "primary",
                        "action_id": "investigate_fraud"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Mark as Safe"
                        },
                        "action_id": "mark_safe"
                    }
                ]
            }
        ]

        try:
            response = self.client.chat_postMessage(
                channel=channel,
                blocks=blocks,
                text=f"Fraud Alert: {invoice_data.get('invoice_id', 'Unknown')}"
            )
            return response
        except SlackApiError as e:
            print(f"Error sending message: {e}")
            return None

# Flask app for Slack webhooks
from flask import Flask, request, jsonify

app = Flask(__name__)
slack_client = SlackNotifier(os.getenv('SLACK_BOT_TOKEN'))

@app.route('/slack/events', methods=['POST'])
def slack_events():
    """Handle Slack events"""
    data = request.json

    if 'challenge' in data:
        return jsonify({'challenge': data['challenge']})

    # Handle button clicks
    if 'payload' in request.form:
        import json
        payload = json.loads(request.form['payload'])

        if payload['actions'][0]['action_id'] == 'investigate_fraud':
            # Trigger investigation workflow
            return jsonify({'text': 'Investigation initiated...'})
        elif payload['actions'][0]['action_id'] == 'mark_safe':
            # Mark as false positive
            return jsonify({'text': 'Marked as safe.'})
