"""Integration test del flusso Rounds & Matches (criteri di accettazione Fase 4)."""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.models.season import Season, SeasonStatus
from app.models.staged_fixture import StagedFixture
from app.models.round import Competition
from app.models.user import User, UserRole
from app.services.auth import create_access_token, hash_password

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def admin(db_session) -> User:
    u = User(username="admin", email="admin@example.com",
             password_hash=hash_password("x"), role=UserRole.admin)
    db_session.add(u)
    await db_session.commit()
    return u


@pytest_asyncio.fixture
async def player(db_session) -> User:
    u = User(username="player", email="player@example.com",
             password_hash=hash_password("x"), role=UserRole.player)
    db_session.add(u)
    await db_session.commit()
    return u


@pytest_asyncio.fixture
async def season(db_session) -> Season:
    s = Season(name="2025-26", status=SeasonStatus.active)
    db_session.add(s)
    await db_session.commit()
    return s


def _auth(user: User) -> dict:
    return {"Authorization": f"Bearer {create_access_token({'sub': str(user.id), 'role': user.role})}"}


def _future() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()


async def _create_round(client, admin, **overrides) -> str:
    body = {"name": "Giornata 1", "competition": "serie_a", "deadline": _future()}
    body.update(overrides)
    r = await client.post("/rounds", headers=_auth(admin), json=body)
    assert r.status_code == 201
    return r.json()["id"]


async def _add_match(client, admin, round_id, home, away) -> str:
    r = await client.post(
        f"/rounds/{round_id}/matches",
        headers=_auth(admin),
        json={"home_team": home, "away_team": away},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ─── 1-3: creazione, partite, apertura, visibilità ────────────────────────────


async def test_create_add_matches_and_open(client: AsyncClient, admin, player, season):
    rid = await _create_round(client, admin)
    assert (await client.get(f"/rounds/{rid}", headers=_auth(admin))).json()["status"] == "draft"

    for home, away in [("Inter", "Milan"), ("Roma", "Lazio"), ("Napoli", "Juventus")]:
        await _add_match(client, admin, rid, home, away)

    # Apertura
    r = await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "open"})
    assert r.status_code == 200
    assert r.json()["status"] == "open"

    # Il giocatore lo vede tra le giornate
    lst = await client.get("/rounds", headers=_auth(player))
    assert rid in [r_["id"] for r_ in lst.json()]


# ─── precondizioni apertura ───────────────────────────────────────────────────


async def test_cannot_open_without_matches(client: AsyncClient, admin, season):
    rid = await _create_round(client, admin)
    r = await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "open"})
    assert r.status_code == 409


async def test_cannot_open_with_past_deadline(client: AsyncClient, admin, season):
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    rid = await _create_round(client, admin, deadline=past)
    await _add_match(client, admin, rid, "Inter", "Milan")
    r = await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "open"})
    assert r.status_code == 409


# ─── 4: aggiungere partita a round aperto → 409 ───────────────────────────────


async def test_cannot_add_match_to_open_round(client: AsyncClient, admin, season):
    rid = await _create_round(client, admin)
    await _add_match(client, admin, rid, "Inter", "Milan")
    await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "open"})
    r = await client.post(
        f"/rounds/{rid}/matches", headers=_auth(admin), json={"home_team": "Roma", "away_team": "Lazio"}
    )
    assert r.status_code == 409


async def test_duplicate_match_rejected(client: AsyncClient, admin, season):
    rid = await _create_round(client, admin)
    await _add_match(client, admin, rid, "Inter", "Milan")
    r = await client.post(
        f"/rounds/{rid}/matches", headers=_auth(admin), json={"home_team": "Inter", "away_team": "Milan"}
    )
    assert r.status_code == 409


# ─── 5-6: chiusura e invisibilità al player ───────────────────────────────────


async def test_close_hides_from_player(client: AsyncClient, admin, player, season):
    rid = await _create_round(client, admin)
    await _add_match(client, admin, rid, "Inter", "Milan")
    await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "open"})
    await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "closed"})

    lst = await client.get("/rounds", headers=_auth(player))
    assert rid not in [r_["id"] for r_ in lst.json()]
    # il dettaglio per il player diventa 404
    assert (await client.get(f"/rounds/{rid}", headers=_auth(player))).status_code == 404


# ─── 7: risultato + precondizioni di stato ────────────────────────────────────


async def test_result_only_when_closed(client: AsyncClient, admin, season):
    rid = await _create_round(client, admin)
    mid = await _add_match(client, admin, rid, "Inter", "Milan")

    # draft → 409
    r = await client.patch(f"/matches/{mid}/result", headers=_auth(admin), json={"home_goals": 2, "away_goals": 1})
    assert r.status_code == 409

    await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "open"})
    # open → 409
    r = await client.patch(f"/matches/{mid}/result", headers=_auth(admin), json={"home_goals": 2, "away_goals": 1})
    assert r.status_code == 409

    await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "closed"})
    # closed → 200
    r = await client.patch(f"/matches/{mid}/result", headers=_auth(admin), json={"home_goals": 2, "away_goals": 1})
    assert r.status_code == 200
    assert r.json()["actual_home_goals"] == 2


async def test_complete_requires_all_results(client: AsyncClient, admin, season):
    rid = await _create_round(client, admin)
    m1 = await _add_match(client, admin, rid, "Inter", "Milan")
    await _add_match(client, admin, rid, "Roma", "Lazio")
    await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "open"})
    await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "closed"})
    await client.patch(f"/matches/{m1}/result", headers=_auth(admin), json={"home_goals": 1, "away_goals": 0})

    # Manca il risultato della 2ª partita → 409
    r = await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "completed"})
    assert r.status_code == 409


# ─── transizioni illegali e delete ────────────────────────────────────────────


async def test_illegal_transition_rejected(client: AsyncClient, admin, season):
    rid = await _create_round(client, admin)
    await _add_match(client, admin, rid, "Inter", "Milan")
    r = await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "completed"})
    assert r.status_code == 409  # draft → completed non ammessa


async def test_delete_only_in_draft(client: AsyncClient, admin, season):
    rid = await _create_round(client, admin)
    await _add_match(client, admin, rid, "Inter", "Milan")
    assert (await client.delete(f"/rounds/{rid}", headers=_auth(admin))).status_code == 204

    rid2 = await _create_round(client, admin)
    await _add_match(client, admin, rid2, "Inter", "Milan")
    await client.patch(f"/rounds/{rid2}/status", headers=_auth(admin), json={"status": "open"})
    assert (await client.delete(f"/rounds/{rid2}", headers=_auth(admin))).status_code == 409


# ─── permessi e staged fixture ────────────────────────────────────────────────


async def test_player_cannot_create_round(client: AsyncClient, player, season):
    r = await client.post(
        "/rounds", headers=_auth(player),
        json={"name": "X", "competition": "serie_a", "deadline": _future()},
    )
    assert r.status_code == 403


async def test_add_match_from_staged_fixture(client: AsyncClient, admin, season, db_session):
    fixture = StagedFixture(
        api_fixture_id=12345, competition=Competition.serie_a,
        home_team="Atalanta", away_team="Torino",
    )
    db_session.add(fixture)
    await db_session.commit()

    rid = await _create_round(client, admin)
    r = await client.post(
        f"/rounds/{rid}/matches", headers=_auth(admin), json={"staged_fixture_id": str(fixture.id)}
    )
    assert r.status_code == 201
    assert r.json()["home_team"] == "Atalanta"
    assert r.json()["api_fixture_id"] == 12345
