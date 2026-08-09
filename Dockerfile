FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY src ./src
COPY config ./config

ENV PYTHONPATH=/app/src

CMD ["uv", "run", "python", "-m", "electera.pipeline.election_backtester"]
CMD ["uv", "run", "python", "-m", "electera.pipeline.explain_model"]
