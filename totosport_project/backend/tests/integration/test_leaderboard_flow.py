"""
Integration test della classifica e dei bonus di fine stagione (Fase 6).
Seeda direttamente RoundScore + TablePrediction (lo scoring è già coperto in Fase 5).
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select

from app.models.round import Competition, Round, RoundStatus
from app.models.round_score import RoundScore
from app.models.season import Season, SeasonStatus
from app.models.table_prediction import TablePrediction
from app.models.user import User, UserRole
from app.services.auth import create_access_token, hash_password

pytestmark = pytest.mark.asyncio


def _auth(u: User) -> dict:
    return {"Authorization": f"Bearer {create_access_token({'sub': str(u.id), 'role': u.role})}"}


@pytest_asyncio.fixture
async def setup(db_session):
    """Stagione + admin + 3 player + 2 giornate completate con RoundScore e TablePrediction."""
    admin = User(username="admin", email="a@ex.com", password_hash=hash_password("x"), role=UserRole.admin)
    p1 = User(username="p1", email="p1@ex.com", password_hash=hash_password("x"), role=UserRole.player)
    p2 = User(username="p2", email="p2@ex.com", password_hash=hash_password("x"), role=UserRole.player)
    p3 = User(username="p3", email="p3@ex.com", password_hash=hash_password("x"), role=UserRole.player)
    season = Season(name="2025-26", status=SeasonStatus.active)
    db_session.add_all([admin, p1, p2, p3, season])
    await db_session.flush()

    r1 = Round(season_id=season.id, name="G1", competition=Competition.serie_a, status=RoundStatus.completed)
    r2 = Round(season_id=season.id, name="G2", competition=Competition.serie_a, status=RoundStatus.completed)
    db_session.add_all([r1, r2])
    await db_session.flush()

    def rs(player, rnd, sign, exact, tg, wb):
        return RoundScore(
            player_id=player.id, round_id=rnd.id, sign_points=sign, exact_points=exact,
            total_goals_points=tg, weekend_bonus=wb, total_round_points=sign + exact + tg + wb,
        )

    db_session.add_all([
        rs(p1, r1, 5, 10, 3, 6), rs(p1, r2, 3, 0, 0, 4),   # round_total 31
        rs(p2, r1, 3, 5, 0, 4), rs(p2, r2, 5, 10, 3, 6),   # round_total 36
        rs(p3, r1, 1, 0, 0, 0), rs(p3, r2, 1, 0, 0, 2),    # round_total 4
    ])
    db_session.add_all([
        TablePrediction(player_id=p1.id, season_id=season.id, total_points=20),
        TablePrediction(player_id=p2.id, season_id=season.id, total_points=10),
        TablePrediction(player_id=p3.id, season_id=season.id, total_points=20),
    ])
    await db_session.commit()
    return {"admin": admin, "p1": p1, "p2": p2, "p3": p3, "season": season}


def _by_user(rows: list[dict]) -> dict:
    return {r["username"]: r for r in rows}


# ─── Classifica generale ─────────────────────────────────────────────────────


async def test_general_leaderboard(client: AsyncClient, setup):
    r = await client.get("/leaderboard", headers=_auth(setup["admin"]))
    assert r.status_code == 200
    rows = _by_user(r.json())

    # p1: round 31 + tabellone 20 = 51 ; p2: 36+10=46 ; p3: 4+20=24
    assert rows["p1"]["total_points"] == 51
    assert rows["p2"]["total_points"] == 46
    assert rows["p3"]["total_points"] == 24
    assert rows["p1"]["rank"] == 1
    assert rows["p2"]["rank"] == 2
    assert rows["p3"]["rank"] == 3
    # Breakdown p1
    assert rows["p1"]["sign_points"] == 8
    assert rows["p1"]["exact_points"] == 10
    assert rows["p1"]["total_goals_points"] == 3
    assert rows["p1"]["weekend_bonus_total"] == 10
    assert rows["p1"]["tabellone_points"] == 20


async def test_metric_leaderboards(client: AsyncClient, setup):
    headers = _auth(setup["admin"])
    signs = _by_user((await client.get("/leaderboard/signs", headers=headers)).json())
    exacts = _by_user((await client.get("/leaderboard/exacts", headers=headers)).json())
    tabellone = _by_user((await client.get("/leaderboard/tabellone", headers=headers)).json())

    # segni: p1=8, p2=8, p3=2 → parità in testa
    assert signs["p1"]["points"] == 8 and signs["p2"]["points"] == 8
    assert signs["p1"]["rank"] == 1 and signs["p2"]["rank"] == 1
    # pieni+gol: p2 = (5+10)+(0+3)=18 in testa
    assert exacts["p2"]["points"] == 18 and exacts["p2"]["rank"] == 1
    # tabellone: p1=20, p3=20 → parità
    assert tabellone["p1"]["points"] == 20 and tabellone["p3"]["points"] == 20


# ─── Finalizzazione ──────────────────────────────────────────────────────────


async def test_finalize_assigns_bonuses_and_closes(client: AsyncClient, setup):
    sid = str(setup["season"].id)
    r = await client.post(f"/admin/seasons/{sid}/finalize", headers=_auth(setup["admin"]))
    assert r.status_code == 200
    res = r.json()
    # segni: p1,p2 ; pieni+gol: p2 ; tabellone: p1,p3
    assert set(res["bonus_signs"]) == {str(setup["p1"].id), str(setup["p2"].id)}
    assert res["bonus_exacts"] == [str(setup["p2"].id)]
    assert set(res["bonus_tabellone"]) == {str(setup["p1"].id), str(setup["p3"].id)}

    # Stagione chiusa
    seasons = (await client.get("/admin/seasons", headers=_auth(setup["admin"]))).json()
    assert next(s for s in seasons if s["id"] == sid)["status"] == "closed"

    # Classifica dopo i bonus: p1 51+20=71, p2 46+20=66, p3 24+10=34
    rows = _by_user((await client.get("/leaderboard", headers=_auth(setup["admin"]))).json())
    assert rows["p1"]["season_bonus_total"] == 20 and rows["p1"]["total_points"] == 71
    assert rows["p2"]["season_bonus_total"] == 20 and rows["p2"]["total_points"] == 66
    assert rows["p3"]["season_bonus_total"] == 10 and rows["p3"]["total_points"] == 34


# ─── Round leaderboard + penalità tabellone riflessa ──────────────────────────


async def test_round_leaderboard(client: AsyncClient, setup, db_session):
    res = await db_session.execute(select(Round).where(Round.name == "G2"))
    r2 = res.scalar_one()
    rows = _by_user((await client.get(f"/leaderboard/rounds/{r2.id}", headers=_auth(setup["admin"]))).json())
    # G2: p2 total 24 (1°), p1 7 (2°), p3 3 (3°)
    assert rows["p2"]["round_points"] == 24 and rows["p2"]["rank"] == 1
    assert rows["p1"]["round_points"] == 7
    assert rows["p3"]["round_points"] == 3


async def test_tabellone_penalty_reflected(client: AsyncClient, db_session):
    # Un giocatore con penalità mercato: total_points già al netto (es. 20 - 5 = 15)
    season = Season(name="2099-00", status=SeasonStatus.active)
    pl = User(username="penal", email="pen@ex.com", password_hash=hash_password("x"), role=UserRole.player)
    db_session.add_all([season, pl])
    await db_session.flush()
    db_session.add(TablePrediction(player_id=pl.id, season_id=season.id, mercato_penalty=-5, total_points=15))
    await db_session.commit()

    rows = _by_user((await client.get("/leaderboard", headers=_auth(pl))).json())
    assert rows["penal"]["tabellone_points"] == 15
    assert rows["penal"]["total_points"] == 15
