# Fase 11 — Deployment

Obiettivo: app deployata in produzione con Docker, Nginx, variabili d'ambiente sicure, e health check.

---

## Checklist

### Backend — Dockerfile produzione

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non root in produzione
RUN adduser --disabled-password --gecos '' appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### Frontend — Dockerfile produzione (multi-stage)

```dockerfile
# frontend/Dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json .
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

### Nginx — configurazione (`frontend/nginx.conf`)

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    # SPA: tutte le route non-file → index.html
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API al backend
    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Docker Compose produzione (`docker-compose.prod.yml`)

```yaml
version: "3.9"

services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "${POSTGRES_USER}"]
      interval: 10s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    env_file: .env.prod
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
      - "443:443"  # se usi HTTPS con certbot
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  postgres_data:
```

### Health check endpoint

- [ ] `GET /health` → `{"status": "ok", "db": "ok"}` (controlla connessione DB)
- [ ] Usato da Docker healthcheck e da eventuali monitoring tool

### Variabili d'ambiente produzione (`.env.prod`)

```env
# Database
POSTGRES_USER=totosport
POSTGRES_PASSWORD=<password-sicura>
POSTGRES_DB=totosport
DATABASE_URL=postgresql+asyncpg://totosport:<password>@db:5432/totosport

# Auth (genera con: python -c "import secrets; print(secrets.token_hex(32))")
SECRET_KEY=<64-char-hex>
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# API-Football
API_FOOTBALL_KEY=<chiave-rapidapi>
API_FOOTBALL_HOST=v3.football.api-sports.io

# CORS (origins permessi)
ALLOWED_ORIGINS=https://tuodominio.com
```

**Regole per i secret:**
- Non committare mai `.env.prod`
- Usa Docker secrets o un secret manager (Vault, AWS SSM) per produzione seria
- Ruota `SECRET_KEY` invalidando tutti i token esistenti — avvisa gli utenti

### Migrazioni in produzione

```bash
# Prima di ogni deploy con modifiche al DB:
docker-compose -f docker-compose.prod.yml exec backend alembic upgrade head
```

Automatizza questo come parte del deploy script.

### Deploy workflow

```bash
#!/bin/bash
# deploy.sh

git pull origin main

# Build e avvio
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Migrazioni
docker-compose -f docker-compose.prod.yml exec backend alembic upgrade head

echo "Deploy completato"
```

### HTTPS (opzionale ma consigliato)

Se il server è esposto su internet, usa **Certbot + Let's Encrypt**:

```bash
docker run --rm -v /etc/letsencrypt:/etc/letsencrypt \
  certbot/certbot certonly --standalone -d tuodominio.com
```

Poi aggiorna `nginx.conf` con la config HTTPS e i path ai certificati.

### Backup DB

```bash
# Backup
docker-compose exec db pg_dump -U totosport totosport > backup_$(date +%Y%m%d).sql

# Restore
docker-compose exec -T db psql -U totosport totosport < backup_20260101.sql
```

Automatizza il backup giornaliero con un cron job sull'host.

---

## Test di accettazione fase 11

1. `docker-compose -f docker-compose.prod.yml up -d` senza errori
2. `GET /health` → `{"status": "ok", "db": "ok"}`
3. Frontend accessibile su porta 80, React Router funzionante (F5 su una route interna non dà 404)
4. Login funzionante end-to-end in produzione
5. Migrazione Alembic applicata (`alembic current` mostra la head revision)
6. Backup DB eseguito e restore verificato su DB di test
