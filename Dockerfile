# Multi-stage build: a wheel is built in a throwaway stage, the runtime image only gets the
# installed package (no compilers, no source tree, no test dependencies) and runs as non-root.

FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build==1.2.2
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AGENT_REPLAY_LOG_LEVEL=INFO
RUN groupadd --system app && useradd --system --gid app --home-dir /app --create-home app
COPY --from=build /dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl
WORKDIR /app
USER app
EXPOSE 4318
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4318/healthz', timeout=2)"]
ENTRYPOINT ["agent-replay"]
CMD ["serve", "--host", "0.0.0.0", "--port", "4318"]
