FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/anarkali \
    HF_HOME=/home/anarkali/.cache/huggingface \
    ANARKALI_HOST=0.0.0.0 \
    ANARKALI_PORT=8000 \
    ANARKALI_MODEL=toufiqqureshi651/anarkali

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir '.[serve]' \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin anarkali \
    && mkdir -p /home/anarkali/.cache/huggingface \
    && chown -R 10001:10001 /home/anarkali

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["sh", "-c", "exec anarkali serve --model \"$ANARKALI_MODEL\""]