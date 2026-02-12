FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMPASS_UI_HOST=0.0.0.0 \
    COMPASS_UI_PORT=5005

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.ui.txt /tmp/requirements.ui.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /tmp/requirements.ui.txt

COPY . /workspace/multi_agent_system
WORKDIR /workspace/multi_agent_system

RUN chmod +x /workspace/multi_agent_system/docker/entrypoint.sh

EXPOSE 5005

ENTRYPOINT ["/workspace/multi_agent_system/docker/entrypoint.sh"]
CMD []
