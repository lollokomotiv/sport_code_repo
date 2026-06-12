"""
Integration test delle previsioni settimanali e dello scoring (Fase 5),
inclusi il totale gol PER LEGA e il bonus weekend.
"""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select

from app.models.match import Match
from app.models.round import Competition, Round, RoundStatus
from app.models.round_score import RoundScore
from app.models.season import Season, SeasonStatus
from app.models.user import User, UserRole
from app.services.auth import create_access_token, hash_password

pytestmark = pytest.mark.asyncio


def _future() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()


async def _mk_user(db, username, role=UserRole.player) -> User:
    u = User(username=username, email=f"{username}@ex.com", password_hash=hash_password("x"), role=role)
    db.add(u)
    await db.commit()
    return u


@pytest_asyncio.fixture
async def admin(db_session):
    return await _mk_user(db_session, "admin", UserRole.admin)


@pytest_asyncio.fixture
async def p1(db_session):
    return await _mk_user(db_session, "p1")


@pytest_asyncio.fixture
async def p2(db_session):
    return await _mk_user(db_session, "p2")


@pytest_asyncio.fixture
async def season(db_session):
    s = Season(name="2025-26", status=SeasonStatus.active)
    db_session.add(s)
    await db_session.commit()
    return s


def _auth(u: User) -> dict:
    return {"Authorization": f"Bearer {create_access_token({'sub': str(u.id), 'role': u.role})}"}


async def _open_round_with_matches(client, admin, matches, deadline=None):
    """Crea una giornata draft, aggiunge le partite, la apre. Ritorna (round_id, [match_ids])."""
    r = await client.post(
        "/rounds",
        headers=_auth(admin),
        json={"name": "G1", "competition": "mixed", "deadline": deadline or _future()},
    )
    rid = r.json()["id"]
    match_ids = []
    for home, away, comp in matches:
        mr = await client.post(
            f"/rounds/{rid}/matches",
            headers=_auth(admin),
            json={"home_team": home, "away_team": away, "competition": comp},
        )
        match_ids.append(mr.json()["id"])
    await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "open"})
    return rid, match_ids


# ─── Submit + lettura ────────────────────────────────────────────────────────


async def test_submit_match_and_view_me(client: AsyncClient, admin, p1, season):
    rid, [m1] = await _open_round_with_matches(client, admin, [("Inter", "Milan", "serie_a")])

    r = await client.post(
        "/predictions/match",
        headers=_auth(p1),
        json={"match_id": m1, "predicted_home_goals": 2, "predicted_away_goals": 1},
    )
    assert r.status_code == 200
    assert r.json()["derived_sign"] == "1"

    me = await client.get(f"/predictions/me?round_id={rid}", headers=_auth(p1))
    assert me.status_code == 200
    assert len(me.json()["match_predictions"]) == 1


async def test_match_prediction_is_upsert(client: AsyncClient, admin, p1, season):
    rid, [m1] = await _open_round_with_matches(client, admin, [("Inter", "Milan", "serie_a")])
    await client.post("/predictions/match", headers=_auth(p1),
                      json={"match_id": m1, "predicted_home_goals": 2, "predicted_away_goals": 1})
    r = await client.post("/predictions/match", headers=_auth(p1),
                          json={"match_id": m1, "predicted_home_goals": 0, "predicted_away_goals": 0})
    assert r.status_code == 200
    assert r.json()["derived_sign"] == "X"
    me = await client.get(f"/predictions/me?round_id={rid}", headers=_auth(p1))
    assert len(me.json()["match_predictions"]) == 1  # aggiornata, non duplicata


async def test_round_goals_competition_not_in_round(client: AsyncClient, admin, p1, season):
    rid, _ = await _open_round_with_matches(client, admin, [("Inter", "Milan", "serie_a")])
    # Nessuna partita di Serie B nella giornata
    r = await client.post(
        "/predictions/round-goals",
        headers=_auth(p1),
        json={"round_id": rid, "competition": "serie_b", "total_goals_guess": 3},
    )
    assert r.status_code == 409


# ─── Deadline / stato ────────────────────────────────────────────────────────


async def test_submit_after_deadline_forbidden(client: AsyncClient, admin, p1, season, db_session):
    # Giornata aperta ma con deadline già passata (seed diretto)
    rnd = Round(season_id=season.id, name="G", competition=Competition.serie_a,
                status=RoundStatus.open, deadline=datetime.now(timezone.utc) - timedelta(hours=1))
    db_session.add(rnd)
    await db_session.flush()
    m = Match(round_id=rnd.id, competition=Competition.serie_a, home_team="A", away_team="B")
    db_session.add(m)
    await db_session.commit()

    r = await client.post("/predictions/match", headers=_auth(p1),
                          json={"match_id": str(m.id), "predicted_home_goals": 1, "predicted_away_goals": 0})
    assert r.status_code == 403


async def test_submit_when_not_open_forbidden(client: AsyncClient, admin, p1, season):
    # Giornata in draft (non aperta)
    r = await client.post(
        "/rounds", headers=_auth(admin),
        json={"name": "G", "competition": "serie_a", "deadline": _future()},
    )
    rid = r.json()["id"]
    mr = await client.post(f"/rounds/{rid}/matches", headers=_auth(admin),
                           json={"home_team": "A", "away_team": "B", "competition": "serie_a"})
    r = await client.post("/predictions/match", headers=_auth(p1),
                          json={"match_id": mr.json()["id"], "predicted_home_goals": 1, "predicted_away_goals": 0})
    assert r.status_code == 403


# ─── Scoring completo con totale gol per lega + bonus weekend ──────────────────


async def test_full_round_scoring(client: AsyncClient, admin, p1, p2, season, db_session):
    rid, [m_sa1, m_sa2, m_sb] = await _open_round_with_matches(
        client, admin,
        [("Inter", "Milan", "serie_a"), ("Roma", "Lazio", "serie_a"), ("Parma", "Como", "serie_b")],
    )

    # P1: SA1 2-1 (esatto casa), SA2 1-1 (esatto pari), SB 1-0 (segno sbagliato)
    for mid, h, a in [(m_sa1, 2, 1), (m_sa2, 1, 1), (m_sb, 1, 0)]:
        await client.post("/predictions/match", headers=_auth(p1),
                          json={"match_id": mid, "predicted_home_goals": h, "predicted_away_goals": a})
    # totale gol: SA reale = 3+2=5, SB reale = 0
    await client.post("/predictions/round-goals", headers=_auth(p1),
                      json={"round_id": rid, "competition": "serie_a", "total_goals_guess": 5})
    await client.post("/predictions/round-goals", headers=_auth(p1),
                      json={"round_id": rid, "competition": "serie_b", "total_goals_guess": 0})

    # P2: SA1 1-0 (solo segno), SA2 2-0 (segno sbagliato), SB 0-0 (esatto pari)
    for mid, h, a in [(m_sa1, 1, 0), (m_sa2, 2, 0), (m_sb, 0, 0)]:
        await client.post("/predictions/match", headers=_auth(p2),
                          json={"match_id": mid, "predicted_home_goals": h, "predicted_away_goals": a})
    await client.post("/predictions/round-goals", headers=_auth(p2),
                      json={"round_id": rid, "competition": "serie_a", "total_goals_guess": 4})  # sbagliato
    await client.post("/predictions/round-goals", headers=_auth(p2),
                      json={"round_id": rid, "competition": "serie_b", "total_goals_guess": 0})  # giusto

    # Chiudi, inserisci risultati, completa
    await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "closed"})
    for mid, h, a in [(m_sa1, 2, 1), (m_sa2, 1, 1), (m_sb, 0, 0)]:
        rr = await client.patch(f"/matches/{mid}/result", headers=_auth(admin),
                                json={"home_goals": h, "away_goals": a})
        assert rr.status_code == 200
    await client.patch(f"/rounds/{rid}/status", headers=_auth(admin), json={"status": "completed"})

    # Verifica RoundScore
    res = await db_session.execute(select(RoundScore).where(RoundScore.round_id == rid))
    scores = {rs.player_id: rs for rs in res.scalars().all()}
    s1, s2 = scores[p1.id], scores[p2.id]

    # P1: segno 1+1+0=2; esatto 5+7+0=12; totgol 3+3=6; base 20; bonus 1°=6 → 26
    assert (s1.sign_points, s1.exact_points, s1.total_goals_points) == (2, 12, 6)
    assert s1.weekend_bonus == 6
    assert s1.total_round_points == 26
    # P2: segno 1+0+1=2; esatto 0+0+7=7; totgol 0+3=3; base 12; bonus 2°=4 → 16
    assert (s2.sign_points, s2.exact_points, s2.total_goals_points) == (2, 7, 3)
    assert s2.weekend_bonus == 4
    assert s2.total_round_points == 16


# ─── Admin: previsioni visibili solo dopo deadline ────────────────────────────


async def test_admin_predictions_gated_by_deadline(client: AsyncClient, admin, p1, season):
    rid, [m1] = await _open_round_with_matches(client, admin, [("Inter", "Milan", "serie_a")])
    # deadline nel futuro → admin non può ancora vedere
    r = await client.get(f"/admin/predictions/{rid}", headers=_auth(admin))
    assert r.status_code == 403
