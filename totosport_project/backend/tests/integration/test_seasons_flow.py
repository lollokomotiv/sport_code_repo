"""
Integration test dell'endpoint player-facing GET /seasons/current.
Serve al frontend (pagina Tabellone) per conoscere stato e deadline stagione.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.models.season import Season, SeasonStatus
from app.models.user import User, UserRole
from app.services.auth import create_access_token, hash_password

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def player(db_session) -> User:
    u = User(
        username="giocatore",
        email="giocatore@example.com",
        password_hash=hash_password("test1234"),
        role=UserRole.player,
    )
    db_session.add(u)
    await db_session.commit()
    return u


def _auth(user: User) -> dict:
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return {"Authorization": f"Bearer {token}"}


async def test_player_reads_current_season(client: AsyncClient, db_session, player):
    s = Season(name="2025-26", status=SeasonStatus.mercato)
    db_session.add(s)
    await db_session.commit()

    r = await client.get("/seasons/current", headers=_auth(player))
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "2025-26"
    assert body["status"] == "mercato"


async def test_current_season_requires_auth(client: AsyncClient):
    r = await client.get("/seasons/current")
    assert r.status_code == 401


async def test_current_season_404_when_none(client: AsyncClient, player):
    r = await client.get("/seasons/current", headers=_auth(player))
    assert r.status_code == 404
