FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_CONFIG=production \
    HOST=0.0.0.0 \
    AUTO_MIGRATE_ON_START=1 \
    DATABASE_URL=sqlite:////data/msl.db

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p /data

EXPOSE 8080

CMD ["sh", "-c", "flask --app run.py db upgrade && flask --app run.py seed-vocabulary && exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 120 run:app"]
