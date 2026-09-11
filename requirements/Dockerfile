# CPU reference container. GPU environments are qualified separately on PRISM.
FROM python:3.13-slim
WORKDIR /opt/terragpu
COPY pyproject.toml setup.py README.md LICENSE ./
COPY terragpu ./terragpu
RUN python -m pip install --no-cache-dir '.[parallel]'
CMD ["python", "-m", "terragpu.benchmark", "--size", "512"]
