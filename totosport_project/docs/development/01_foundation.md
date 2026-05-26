# Fase 1 — Foundation

Obiettivo: repo funzionante con DB avviabile, struttura cartelle definita, nessun codice applicativo ancora.

---

## Checklist

### Struttura cartelle
- [ ] Crea `backend/` con struttura descritta in `CLAUDE.md` §9
- [ ] Crea `frontend/` con Vite (`npm create vite@latest frontend -- --template react-ts`)
- [ ] Crea `.env.example` con tutte le variabili (vedi `CLAUDE.md` §10)
- [ ] Aggiungi `.gitignore` adeguato (`.env`, `__pycache__`, `node_modules`, `.venv`, `dist`)

### Docker Compose (sviluppo)
- [ ] `docker-compose.yml` con servizi: `db` (postgres:16), `backend`, `frontend`
- [ ] Il servizio `db` monta un volume named per la persistenza
- [ ] `backend` dipende da `db` con healthcheck (`pg_isready`)
- [ ] Variabili d'ambiente iniettate dal file `.env`
- [ ] `docker-compose up` avvia tutto, `docker-compose up db` avvia solo il DB per lo sviluppo locale

### Backend — setup base
- [ ] `requirements.txt` con versioni pinned (usa `pip-compile` o specifica `>=x.y`)
- [ ] `app/config.py` con pydantic-settings che legge `.env`
- [ ] `app/database.py` con `AsyncEngine`, `AsyncSessionLocal`, `get_db` dependency
- [ ] `app/main.py` con app factory, CORS, lifespan (startup/shutdown), router placeholder
- [ ] Alembic inizializzato: `alembic init migrations`
- [ ] `alembic.ini` e `migrations/env.py` configurati per leggere `DATABASE_URL` da env
- [ ] `alembic upgrade head` eseguito senza errori (migrazione iniziale vuota)

### Frontend — setup base
- [ ] TypeScript `strict: true` in `tsconfig.json`
- [ ] Tailwind CSS installato e configurato (`tailwind.config.js`, `postcss.config.js`)
- [ ] `src/api/client.ts` — Axios instance con `baseURL` da `import.meta.env.VITE_API_BASE_URL`
- [ ] Rimosso tutto il boilerplate di default di Vite (`App.tsx` pulito, `index.css` con solo Tailwind directives)
- [ ] `npm run dev` → app vuota senza errori

### Qualità del codice
- [ ] `black` + `isort` + `mypy` configurati per il backend (`pyproject.toml` o `setup.cfg`)
- [ ] ESLint + Prettier configurati per il frontend
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
