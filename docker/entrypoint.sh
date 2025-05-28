#!/bin/bash

set -e

# Activate virtual environment
source /opt/venv/bin/activate

# Start Jupyter Lab in background
jupyter-lab --ip=0.0.0.0 --port=${JUPYTER_PORT} --no-browser &
JUPYTER_PID=$!

# Optionally start Streamlit UI
if [ "$STREAMLIT_PORT" != "" ]; then
    streamlit run ${JUPYTER_WORKSPACE_DIR}/app.py --server.port=${STREAMLIT_PORT} --server.headless=true &
fi

# Optionally run Slack notifier if enabled and token provided
if [ "$USE_SLACK" = "true" ] && [ "$SLACK_BOT_TOKEN" != "" ]; then
    python ${JUPYTER_WORKSPACE_DIR}/slack_notifier.py &
fi

wait $JUPYTER_PID
