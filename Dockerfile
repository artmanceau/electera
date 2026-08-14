FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY config ./config
COPY assets ./assets

RUN uv sync --frozen

ENV PYTHONPATH=/app/src:/app

CMD ["uv", "run", "python", "-m", "electera.pipeline.election_backtester"]
