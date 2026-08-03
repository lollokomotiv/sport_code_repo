# Fase 11 — Deployment (percorso gratuito)

Obiettivo: TotoSport **online**, con **HTTPS**, a **costo €0**, e con piena
possibilità di fare modifiche in futuro (vedi `docs/OPERATIONS.md`).

## Scelta effettuata: managed gratuito, 3 pezzi separati

| Pezzo | Servizio (free) | Cosa fa | Note |
|---|---|---|---|
| **Frontend** (React statico) | **Vercel** (o Cloudflare Pages) | serve la SPA + HTTPS | sempre attivo, mai cold start |
| **Backend** (FastAPI) | **Render** (free) | espone le API | si addormenta da idle → cold start ~30–60s |
| **Database** (Postgres) | **Neon** (free) | dati persistenti | si risveglia in ~1s |

L'autenticazione è **header-based** (JWT in `localStorage`, non cookie): quindi
frontend e backend su **domini diversi** funzionano senza problemi di cookie.

> ⚠️ **Differenza chiave rispetto al self-host.** Qui frontend e backend stanno su
> **domini diversi**, perciò:
> - il frontend va buildato con `VITE_API_BASE_URL = <URL pubblico del backend>`
>   (NON `/api`, che vale solo nello scenario Nginx self-host);
> - il backend deve autorizzare il **CORS** dell'origine del frontend
>   (`ALLOWED_ORIGINS`).

---

## Checklist di deploy

### Step 0 — Prerequisiti
- [ ] Repo su GitHub (già presente).
- [ ] Account creati: **GitHub**, **Neon**, **Render**, **Vercel** (o Cloudflare).

### Step 1 — Database su Neon
- [ ] Crea un progetto Neon → copia la **connection string**.
- [ ] Ricava due forme della URL:
  - **Runtime (backend, asyncpg):** `postgresql+asyncpg://USER:PASS@HOST/DB`
  - **Migrazioni (Alembic → psycopg2):** `postgresql+psycopg2://USER:PASS@HOST/DB?sslmode=require`
    (`env.py` converte già asyncpg→psycopg2; serve solo che l'SSL sia richiesto).
- [ ] **Nota SSL:** Neon richiede SSL. Se il backend non si connette con asyncpg,
  va aggiunto l'SSL (una riga in `app/database.py` o `?ssl=require` nella URL).
  È un fix di 1 riga: lo sistemiamo al primo tentativo se serve.

### Step 2 — Backend su Render
- [ ] **New → Web Service** → collega il repo.
- [ ] **Root directory:** `totosport_project/backend`
- [ ] **Runtime:** Docker (usa `backend/Dockerfile`, già presente).
- [ ] **Start command** (deve ascoltare sulla porta della piattaforma):
      `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- [ ] **Health check path:** `/health`
- [ ] **Environment variables:**
  - `DATABASE_URL` → Neon, **forma asyncpg** (Step 1)
  - `SECRET_KEY` → genera con `python -c "import secrets; print(secrets.token_hex(32))"`
  - `ACCESS_TOKEN_EXPIRE_MINUTES=15`
  - `REFRESH_TOKEN_EXPIRE_DAYS=7`
  - `ALLOWED_ORIGINS=["https://IL-TUO-FRONTEND.vercel.app"]` (lo riempi dopo lo Step 3)
  - *(API-Football non serve finché la Fase 7 non è fatta)*
- [ ] Deploy → annota l'**URL pubblico** (es. `https://totosport-api.onrender.com`).

### Step 3 — Frontend su Vercel (o Cloudflare Pages)
- [ ] **New Project** → importa il repo.
- [ ] **Root directory:** `totosport_project/frontend`
- [ ] **Build command:** `npm run build` — **Output:** `dist`
- [ ] **Environment variable:** `VITE_API_BASE_URL = https://totosport-api.onrender.com`
      (l'URL del backend dallo Step 2)
- [ ] **SPA rewrite** (tutte le route → `index.html`): serve un file di config
      (`vercel.json` per Vercel, `_redirects` per Cloudflare Pages) — **lo creo io**
      nel repo quando partiamo.
- [ ] Deploy → annota l'**URL** (es. `https://totosport.vercel.app`).

### Step 4 — Collega i due lati (CORS)
- [ ] Su Render, imposta `ALLOWED_ORIGINS = ["https://totosport.vercel.app"]`
      (l'URL del frontend) → **redeploy** del backend.

### Step 5 — Migrazioni + primo admin
- [ ] **Migrazioni** (dal tuo Mac, o dalla Shell di Render):
      `DATABASE_URL="<Neon forma psycopg2 con sslmode=require>" alembic upgrade head`
- [ ] **Primo admin** (il DB parte vuoto): esegui `backend/scripts/create_admin.py`
      puntando al DB Neon (username / email / password a tua scelta).
- [ ] Poi, da interfaccia admin: crei stagione, giocatori, giornate.

### Step 6 — Verifica (test di accettazione)
- [ ] `GET https://<backend>/health` → `{"status": "ok", ...}`
- [ ] Apri il frontend, login con l'admin, crea una giornata di prova.
- [ ] F5 su una route interna (es. `/player/tabellone`) → **niente 404** (SPA rewrite ok).
- [ ] Su iPhone (Safari): **"Aggiungi a Home"** → PWA installata (HTTPS presente).

---

## Note sul tier gratuito

- **Cold start**: il backend Render free si addormenta da idle → il primo accesso
  dopo la pausa attende ~30–60s. Per eliminarlo: piano Render a pagamento
  (~**7 $/mese**). Non cambia nulla d'altro.
- **Neon free**: dati persistenti, storage generoso per ~20 amici; si risveglia in ~1s.
- **Modifiche dopo il go-live** (codice, schema, dati, backup): vedi
  **`docs/OPERATIONS.md`**.

---

## Appendice — Alternativa: self-host VPS + Docker

Se in futuro preferissi un VPS (~5 €/mese, controllo pieno, un solo box con
frontend+backend+DB e Nginx che fa da reverse proxy), qui restano i template.
In questo scenario frontend e backend sono **stesso origine** → `VITE_API_BASE_URL=/api`
e niente CORS.

### `frontend/Dockerfile` (multi-stage)
```dockerfile
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

### `frontend/nginx.conf`
```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;
    location / { try_files $uri $uri/ /index.html; }   # SPA
    location /api/ {                                     # proxy al backend
        proxy_pass http://backend:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### `docker-compose.prod.yml` (estratto)
```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes: [postgres_data:/var/lib/postgresql/data]
    restart: unless-stopped
  backend:
    build: { context: ./backend }
    env_file: .env.prod
    depends_on: [db]
    restart: unless-stopped
  frontend:
    build: { context: ./frontend }
    ports: ["80:80", "443:443"]
    depends_on: [backend]
    restart: unless-stopped
volumes: { postgres_data: {} }
```

- HTTPS con **Certbot/Let's Encrypt**; migrazioni con
  `docker compose -f docker-compose.prod.yml exec backend alembic upgrade head`.
- Confronto completo dei provider: [hosting_options.md](hosting_options.md).
