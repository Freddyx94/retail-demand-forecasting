# Image for the forecast API (api/main.py).
#
# Build:  docker build -t retail-forecast-api .
# Run:    docker run --rm -p 8000:8000 -v "${PWD}/data:/app/data:ro" retail-forecast-api
#         then open http://127.0.0.1:8000/docs
#
# The database is NOT copied into the image: the Kaggle data may not be
# redistributed, and forecasts change with every `python -m forecasting.predict`.
# It is mounted read-only at run time instead (-v ... :ro).

# Small official Python image; same Python version the project was tested on
FROM python:3.10-slim

# No .pyc files, and logs appear immediately in `docker logs`
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FORECAST_DB=/app/data/forecast.duckdb

WORKDIR /app

# Dependencies first: Docker caches each step, so editing the code later
# does not reinstall the libraries
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ api/

# Run as a normal user, not root: limits the damage if the service is ever compromised
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Docker calls /health every 30 s and marks the container healthy/unhealthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# 0.0.0.0: accept connections from outside the container (the default,
# 127.0.0.1, would only answer requests from inside it)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
