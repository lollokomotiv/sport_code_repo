# Fase 1 — Foundation

Obiettivo: repo funzionante con DB avviabile, struttura cartelle definita, nessun codice applicativo ancora.

> **Stato: ✅ Backend completato. Setup frontend chiuso in Fase 8.**
> Backend, Docker (db + backend), Alembic e qualità Python a posto fin da subito.
> Il setup frontend, inizialmente incompleto, è stato completato nella **Fase 8**:
> Tailwind v3 installato + `postcss.config.js`, `tsconfig` in `strict`, Prettier,
> boilerplate Vite rimosso. Resta rimandato alla **Fase 11** solo il servizio
> `frontend` in `docker-compose.yml` (deploy).
> Nota versioni: lo stack è su React 19 / TS 6 / Vite 8 (più recenti di CLAUDE.md §3,
> React 18 / TS 5 / Vite 5) — scelta confermata in Fase 8, sono compatibili.

---

## Checklist

### Struttura cartelle
- [x] Crea `backend/` con struttura descritta in `CLAUDE.md` §9
- [x] Crea `frontend/` con Vite (`npm create vite@latest frontend -- --template react-ts`)
- [x] Crea `.env.example` con tutte le variabili (vedi `CLAUDE.md` §10)
- [x] Aggiungi `.gitignore` adeguato (`.env`, `__pycache__`, `node_modules`, `.venv`, `dist`)

### Docker Compose (sviluppo)
- [~] `docker-compose.yml` con servizi: `db` (postgres:16), `backend`, ~~`frontend`~~ (frontend rimandato a Fase 11)
- [x] Il servizio `db` monta un volume named per la persistenza
- [x] `backend` dipende da `db` con healthcheck (`pg_isready`)
- [x] Variabili d'ambiente iniettate dal file `.env`
- [x] `docker-compose up` avvia tutto, `docker-compose up db` avvia solo il DB per lo sviluppo locale

### Backend — setup base
- [x] `requirements.txt` con versioni pinned (usa `pip-compile` o specifica `>=x.y`)
- [x] `app/config.py` con pydantic-settings che legge `.env`
- [x] `app/database.py` con `AsyncEngine`, `AsyncSessionLocal`, `get_db` dependency
- [x] `app/main.py` con app factory, CORS, lifespan (startup/shutdown), router placeholder
- [x] Alembic inizializzato: `alembic init migrations`
- [x] `alembic.ini` e `migrations/env.py` configurati per leggere `DATABASE_URL` da env
- [x] `alembic upgrade head` eseguito senza errori (migrazione iniziale vuota)

### Frontend — setup base  (✅ completato in Fase 8)
- [x] TypeScript `strict: true` in `tsconfig.app.json`
- [x] Tailwind CSS installato e configurato (`tailwind.config.ts`, `postcss.config.js`)
- [x] `src/api/client.ts` — Axios instance con `baseURL` da `import.meta.env.VITE_API_BASE_URL`
- [x] Rimosso tutto il boilerplate di default di Vite (`App.tsx`/`App.css`/assets, `index.css` con direttive Tailwind)
- [x] `npm run dev` / `npm run build` → app TotoSport senza errori

### Qualità del codice
- [x] `black` + `isort` + `mypy` configurati per il backend (`pyproject.toml` o `setup.cfg`)
- [x] ESLint + Prettier configurati per il frontend
- [ ] Pre-commit hook (opzionale ma consigliato): `pre-commit` con black, isort, eslint

---

## File chiave da creare

### `docker-compose.yml`
```yaml
version: "3.9"

services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-postgres}
      POSTGRES_DB: ${POSTGRES_DB:-totosport}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "postgres"]
      interval: 5s
      retries: 5

  backend:
    build: ./backend
    env_file: .env
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    volumes:
      - ./backend:/app

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    command: npm run dev -- --host

volumes:
  postgres_data:
```

### `backend/app/config.py`
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    api_football_key: str = ""
    api_football_host: str = "v3.football.api-sports.io"

    class Config:
        env_file = ".env"

settings = Settings()
```

### `backend/app/database.py`
```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

---

## Test di accettazione fase 1

1. `docker-compose up db` → postgres accessibile su localhost:5432
2. `alembic upgrade head` → nessun errore
3. `uvicorn app.main:app --reload` → `GET /` ritorna `{"status": "ok"}`
4. `npm run dev` → pagina vuota senza errori in console
