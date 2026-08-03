FROM python:3.12-slim

WORKDIR /app

# Installa dipendenze OS necessarie per asyncpg/bcrypt
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non root in produzione
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Produzione: niente --reload; usa la porta fornita dalla piattaforma ($PORT),
# con fallback a 8000 in locale. Forma "shell" per espandere ${PORT}.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
