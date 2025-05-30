#!/bin/bash

set -e

# Activate virtual environment
source /opt/venv/bin/activate

# Set PYTHONPATH to current directory
export PYTHONPATH=${JUPYTER_WORKSPACE_DIR}:$PYTHONPATH

# Create necessary directories if they don't exist
mkdir -p ${JUPYTER_WORKSPACE_DIR}/data
mkdir -p ${JUPYTER_WORKSPACE_DIR}/models
mkdir -p ${JUPYTER_WORKSPACE_DIR}/temp

## Generate synthetic data if it doesn't exist
#if [ ! -f "${JUPYTER_WORKSPACE_DIR}/data/synthetic_invoices.csv" ]; then
#    echo "Generating synthetic data..."
#    cd ${JUPYTER_WORKSPACE_DIR}
#    python synthetic_data_generator.py
#    mv synthetic_invoices.csv data/synthetic_invoices.csv
#fi

# Start Jupyter Lab in background
echo "Starting Jupyter Lab on port ${JUPYTER_PORT}..."
jupyter-lab --ip=0.0.0.0 --port=${JUPYTER_PORT} --no-browser --allow-root --notebook-dir=${JUPYTER_WORKSPACE_DIR} &
JUPYTER_PID=$!

# Wait a moment for Jupyter to start
sleep 5

# Optionally start Streamlit UI
if [ "$STREAMLIT_PORT" != "" ]; then
    echo "Starting Streamlit on port ${STREAMLIT_PORT}..."
    streamlit run ${JUPYTER_WORKSPACE_DIR}/streamlit_app.py --server.port=${STREAMLIT_PORT} --server.address=0.0.0.0 --server.headless=true &
    STREAMLIT_PID=$!
fi

## Optionally run Slack notifier if enabled and token provided
#if [ "$USE_SLACK" = "true" ] && [ "$SLACK_BOT_TOKEN" != "" ]; then
#    echo "Starting Slack notifier..."
#    python ${JUPYTER_WORKSPACE_DIR}/slack_bot.py &
#fi

echo "Services started successfully!"
echo "Jupyter Lab: http://localhost:${JUPYTER_PORT}"
if [ "$STREAMLIT_PORT" != "" ]; then
    echo "Streamlit: http://localhost:${STREAMLIT_PORT}"
fi

# Wait for Jupyter Lab
wait $JUPYTER_PID
