import os
import requests

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#alerts")

def send_slack_alert(message):
    if not SLACK_BOT_TOKEN:
        print("[WARN] SLACK_BOT_TOKEN not set. Skipping Slack alert.")
        return

    payload = {
        "channel": SLACK_CHANNEL,
        "text": message
    }
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json"
    }
    
    response = requests.post("https://slack.com/api/chat.postMessage", json=payload, headers=headers)

    if not response.ok or not response.json().get("ok"):
        print(f"[ERROR] Slack API error: {response.text}")
    else:
        print(f"[INFO] Slack message sent to {SLACK_CHANNEL}.")
