from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import settings
from app.database import engine
from app.routers import (
    admin_seasons,
    admin_tabellone,
    admin_users,
    auth,
    matches,
    rounds,
    tabellone,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verifica connessione al DB
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    print("✅ Database connesso")
    yield
    # Shutdown
    await engine.dispose()
    print("🔌 Database disconnesso")


app = FastAPI(
    title="TotoSport API",
    version="0.1.0",
    description="Backend per il gioco di pronostici calcistici TotoSport",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth.router)
app.include_router(admin_users.router)
app.include_router(tabellone.router)
app.include_router(admin_seasons.router)
app.include_router(admin_tabellone.router)
app.include_router(rounds.router)
app.include_router(matches.router)


@app.get("/")
async def root():
    return {"status": "ok", "app": "TotoSport API"}


@app.get("/health")
async def health():
    """Health check — usato da Docker e monitoring."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"
    return {"status": "ok" if db_status == "ok" else "degraded", "db": db_status}
