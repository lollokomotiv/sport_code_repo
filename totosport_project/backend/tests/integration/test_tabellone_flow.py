"""
Integration test del flusso Tabellone (criteri di accettazione Fase 3).
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select

from app.models.season import Season, SeasonStatus
from app.models.table_modification import TablePredictionModification
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


@pytest_asyncio.fixture
async def admin(db_session) -> User:
    u = User(
        username="admin",
        email="admin@example.com",
        password_hash=hash_password("test1234"),
        role=UserRole.admin,
    )
    db_session.add(u)
    await db_session.commit()
    return u


@pytest_asyncio.fixture
async def season(db_session) -> Season:
    s = Season(name="2025-26", status=SeasonStatus.setup)
    db_session.add(s)
    await db_session.commit()
    return s


def _auth(user: User) -> dict:
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return {"Authorization": f"Bearer {token}"}


# ─── 1: compilazione ────────────────────────────────────────────────────────────


async def test_player_submits_tabellone(client: AsyncClient, player, season):
    r = await client.post(
        "/tabellone",
        headers=_auth(player),
        json={"scudetto_team": "Inter", "scudetto_points_guess": 90, "champions_winner": "Real Madrid"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["scudetto_team"] == "Inter"
    assert body["player_id"] == str(player.id)
    assert body["is_modifiable"] is False  # stagione in 'setup'


async def test_submit_is_upsert(client: AsyncClient, player, season):
    await client.post("/tabellone", headers=_auth(player), json={"scudetto_team": "Inter"})
    r = await client.post("/tabellone", headers=_auth(player), json={"scudetto_team": "Milan"})
    assert r.status_code == 200
    assert r.json()["scudetto_team"] == "Milan"
    # GET conferma il valore aggiornato
    me = await client.get("/tabellone/me", headers=_auth(player))
    assert me.json()["scudetto_team"] == "Milan"


# ─── 2: modifica fuori finestra → 403 ─────────────────────────────────────────────


async def test_modify_outside_mercato_forbidden(client: AsyncClient, player, season):
    await client.post("/tabellone", headers=_auth(player), json={"scudetto_team": "Inter"})
    r = await client.patch("/tabellone/me", headers=_auth(player), json={"scudetto_team": "Milan"})
    assert r.status_code == 403


# ─── 3 & 4: mercato + modifica con penalità ───────────────────────────────────────


async def test_modify_in_mercato_applies_penalty(client: AsyncClient, player, season, admin, db_session):
    await client.post(
        "/tabellone",
        headers=_auth(player),
        json={"scudetto_team": "Inter", "coppa_italia_winner": "Juventus"},
    )
    # Admin porta la stagione a 'mercato'
    r = await client.patch(
        f"/admin/seasons/{season.id}/status", headers=_auth(admin), json={"status": "mercato"}
    )
    assert r.status_code == 200

    # Player modifica 2 voci
    r = await client.patch(
        "/tabellone/me",
        headers=_auth(player),
        json={"scudetto_team": "Milan", "coppa_italia_winner": "Atalanta"},
    )
    assert r.status_code == 200
    assert r.json()["mercato_penalty"] == -10
    assert r.json()["is_modifiable"] is True

    # 2 righe di modifica registrate
    mods = await db_session.execute(select(TablePredictionModification.field_name))
    voices = set(mods.scalars().all())
    assert voices == {"scudetto", "coppa_italia_winner"}


async def test_modify_same_value_not_charged(client: AsyncClient, player, season, admin):
    await client.post("/tabellone", headers=_auth(player), json={"scudetto_team": "Inter"})
    await client.patch(
        f"/admin/seasons/{season.id}/status", headers=_auth(admin), json={"status": "mercato"}
    )
    # Invia lo stesso valore → nessun addebito
    r = await client.patch("/tabellone/me", headers=_auth(player), json={"scudetto_team": "Inter"})
    assert r.status_code == 200
    assert r.json()["mercato_penalty"] == 0


async def test_remodifying_same_voice_charged_once(client: AsyncClient, player, season, admin):
    """Modificare due volte la stessa voce (in salvataggi separati) resta -5."""
    await client.post("/tabellone", headers=_auth(player), json={"scudetto_team": "Inter"})
    await client.patch(
        f"/admin/seasons/{season.id}/status", headers=_auth(admin), json={"status": "mercato"}
    )
    r1 = await client.patch("/tabellone/me", headers=_auth(player), json={"scudetto_team": "Milan"})
    assert r1.json()["mercato_penalty"] == -5
    # Seconda modifica della STESSA voce → ancora -5, non -10
    r2 = await client.patch("/tabellone/me", headers=_auth(player), json={"scudetto_team": "Juventus"})
    assert r2.json()["mercato_penalty"] == -5


async def test_reverting_to_original_resets_penalty(
    client: AsyncClient, player, season, admin, db_session
):
    """Tornare al valore originale azzera la penalità e rimuove la modifica."""
    await client.post("/tabellone", headers=_auth(player), json={"scudetto_team": "Inter"})
    await client.patch(
        f"/admin/seasons/{season.id}/status", headers=_auth(admin), json={"status": "mercato"}
    )
    r1 = await client.patch("/tabellone/me", headers=_auth(player), json={"scudetto_team": "Milan"})
    assert r1.json()["mercato_penalty"] == -5
    # Rimette l'originale → penalità azzerata
    r2 = await client.patch("/tabellone/me", headers=_auth(player), json={"scudetto_team": "Inter"})
    assert r2.json()["mercato_penalty"] == 0

    mods = await db_session.execute(select(TablePredictionModification.field_name))
    assert list(mods.scalars().all()) == []


# ─── Compilazione tardiva (mercato, nessun tabellone) ─────────────────────────────


async def test_late_compile_in_mercato_applies_flat_penalty(client: AsyncClient, player, season, admin):
    # Il giocatore NON compila in setup; l'admin apre il mercato
    await client.patch(
        f"/admin/seasons/{season.id}/status", headers=_auth(admin), json={"status": "mercato"}
    )
    # Compila in ritardo → penalità fissa -30
    r = await client.post(
        "/tabellone",
        headers=_auth(player),
        json={"scudetto_team": "Inter", "champions_winner": "Real Madrid"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["scudetto_team"] == "Inter"
    assert body["mercato_penalty"] == -30
    assert body["is_modifiable"] is True


async def test_late_compile_then_modify_adds_voice_penalty(client: AsyncClient, player, season, admin):
    await client.patch(
        f"/admin/seasons/{season.id}/status", headers=_auth(admin), json={"status": "mercato"}
    )
    await client.post("/tabellone", headers=_auth(player), json={"scudetto_team": "Inter"})  # -30
    # Una modifica successiva di una voce aggiunge -5 → -35
    r = await client.patch("/tabellone/me", headers=_auth(player), json={"scudetto_team": "Milan"})
    assert r.json()["mercato_penalty"] == -35


async def test_submit_in_mercato_when_already_compiled_conflicts(client: AsyncClient, player, season, admin):
    # Compila in tempo (setup) → nessuna penalità
    await client.post("/tabellone", headers=_auth(player), json={"scudetto_team": "Inter"})
    await client.patch(
        f"/admin/seasons/{season.id}/status", headers=_auth(admin), json={"status": "mercato"}
    )
    # Ricompilare (POST) con tabellone già esistente in mercato → 409 (va usata la modifica)
    r = await client.post("/tabellone", headers=_auth(player), json={"scudetto_team": "Milan"})
    assert r.status_code == 409


# ─── 5 & 6: outcome + scoring con cap ─────────────────────────────────────────────


async def test_full_scoring_with_modification_cap(client: AsyncClient, player, season, admin):
    # Compila
    await client.post(
        "/tabellone",
        headers=_auth(player),
        json={"scudetto_team": "Inter", "champions_winner": "Real Madrid"},
    )
    # Mercato → modifica lo scudetto (voce 'scudetto' → cap 50%)
    await client.patch(
        f"/admin/seasons/{season.id}/status", headers=_auth(admin), json={"status": "mercato"}
    )
    await client.patch("/tabellone/me", headers=_auth(player), json={"scudetto_team": "Napoli"})

    # Admin inserisce i risultati reali: Napoli campione, Real Madrid in Champions
    await client.post(
        "/admin/season-outcome",
        headers=_auth(admin),
        json={"scudetto_team": "Napoli", "champions_winner": "Real Madrid"},
    )
    # Scoring
    r = await client.post("/admin/tabellone/score", headers=_auth(admin))
    assert r.status_code == 200
    scored = r.json()[0]

    # Scudetto modificato e corretto: 25 → cap 12. Champions corretta non modificata: 25.
    # Penalità immediata: -5. Totale = 12 + 25 - 5 = 32.
    assert scored["points_breakdown"]["scudetto"] == 12
    assert scored["points_breakdown"]["champions_winner"] == 25
    assert scored["total_points"] == 32


async def test_scoring_without_outcome_returns_400(client: AsyncClient, player, season, admin):
    await client.post("/tabellone", headers=_auth(player), json={"scudetto_team": "Inter"})
    r = await client.post("/admin/tabellone/score", headers=_auth(admin))
    assert r.status_code == 400
