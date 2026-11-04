FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY app ./app
COPY assets ./assets
COPY config ./config

RUN uv sync --frozen --no-dev

ENV PYTHONPATH=/app/src:/app

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
