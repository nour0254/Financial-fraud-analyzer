########################################
# Stage 1: builder
########################################
ARG PYTHON_IMAGE=python:3.10-slim
FROM ${PYTHON_IMAGE} AS builder

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG VENV_PATH=/opt/venv
ARG REQUIREMENTS_FILE=requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    tini \
    jq \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

RUN addgroup --gid ${GROUP_ID} jupytergroup \
    && useradd -ms /bin/bash -u ${USER_ID} -g jupytergroup jupyteruser

WORKDIR ${VENV_PATH}
RUN python3 -m venv . \
    && . bin/activate \
    && pip install --upgrade pip

COPY ${REQUIREMENTS_FILE} /tmp/
RUN . ${VENV_PATH}/bin/activate \
    && pip install --no-cache-dir --use-deprecated=legacy-resolver -r /tmp/${REQUIREMENTS_FILE}

########################################
# Stage 2: runtime
########################################
ARG PYTHON_IMAGE=python:3.9-slim
FROM ${PYTHON_IMAGE} AS runtime

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG VENV_PATH=/opt/venv
ARG JUPYTER_WORKSPACE_DIR=/home/jupyteruser/work
ARG JUPYTER_PORT=8888
ARG STREAMLIT_PORT=8501
ARG USE_SLACK=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    tini \
    jq \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

COPY --from=builder ${VENV_PATH} ${VENV_PATH}

RUN addgroup --gid ${GROUP_ID} jupytergroup \
    && useradd -ms /bin/bash -u ${USER_ID} -g jupytergroup jupyteruser

RUN mkdir -p /etc/sudoers.d \
    && usermod -aG sudo jupyteruser \
    && echo "jupyteruser ALL=(ALL) NOPASSWD: /usr/bin/su" > /etc/sudoers.d/jupyteruser-su \
    && chmod 0440 /etc/sudoers.d/jupyteruser-su

USER jupyteruser
WORKDIR ${JUPYTER_WORKSPACE_DIR}

ENV PATH="${VENV_PATH}/bin:$PATH:${JAVA_HOME}/bin" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JUPYTER_PORT=${JUPYTER_PORT} \
    JUPYTER_WORKSPACE_DIR=${JUPYTER_WORKSPACE_DIR} \
    STREAMLIT_PORT=${STREAMLIT_PORT} \
    USE_SLACK=${USE_SLACK}

COPY --chown=jupyteruser:jupytergroup entrypoint.sh /entrypoint.sh
COPY --chown=jupyteruser:jupytergroup baseline/ ${JUPYTER_WORKSPACE_DIR}/

RUN chmod +x /entrypoint.sh

EXPOSE ${JUPYTER_PORT}
EXPOSE ${STREAMLIT_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${JUPYTER_PORT}/api || exit 1

ENTRYPOINT ["tini", "--", "/entrypoint.sh"]
CMD jupyter-lab --ip=0.0.0.0 --port=${JUPYTER_PORT} --no-browser
