FROM python:3.11.10-slim-bookworm

# Fully offline build after the base image: test dependencies come from the vendored, hash-pinned
# wheels in vendor/wheels (no PyPI, no apt). The container also runs without network:
#   docker build -t auditenv:1.0.0 .
#   docker run --rm --network none auditenv:1.0.0
WORKDIR /app
COPY requirements.txt .
COPY vendor/wheels/ vendor/wheels/
RUN pip install --no-cache-dir --no-index --find-links vendor/wheels --require-hashes -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home auditor && chown -R auditor /app
USER auditor
CMD ["python", "scripts/check.py"]
