FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY apps ./apps
COPY bazar_core ./bazar_core
COPY bazar_data ./bazar_data
COPY rank_system ./rank_system
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
