"""
Integration test del flusso auth — copre i criteri di accettazione della Fase 2
(vedi docs/development/02_auth_and_users.md).
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.models.user import User, UserRole
from app.services.auth import create_access_token, hash_password

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def admin_user(db_session) -> User:
    user = User(
        username="admin",
        email="admin@example.com",
        password_hash=hash_password("test1234"),
        role=UserRole.admin,
    )
    db_session.add(user)
    await db_session.commit()
    return user


@pytest_asyncio.fixture
async def admin_token(admin_user) -> str:
    return create_access_token({"sub": str(admin_user.id), "role": admin_user.role})


# ─── 1 & 2: login ──────────────────────────────────────────────────────────────


async def test_login_success(client: AsyncClient, admin_user):
    r = await client.post("/auth/login", json={"username": "admin", "password": "test1234"})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["refresh_token"]


async def test_login_wrong_password(client: AsyncClient, admin_user):
    r = await client.post("/auth/login", json={"username": "admin", "password": "sbagliata"})
    assert r.status_code == 401


async def test_login_unknown_user(client: AsyncClient):
    r = await client.post("/auth/login", json={"username": "nessuno", "password": "x"})
    assert r.status_code == 401


# ─── 3 & 4: /auth/me ───────────────────────────────────────────────────────────


async def test_me_with_valid_token(client: AsyncClient, admin_token):
    r = await client.get("/auth/me", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "admin"
    assert body["role"] == "admin"
    assert "password_hash" not in body  # non deve mai trapelare


async def test_me_without_token(client: AsyncClient):
    r = await client.get("/auth/me")
    assert r.status_code == 401  # HTTPBearer → 401 se manca l'header


async def test_me_with_invalid_token(client: AsyncClient):
    r = await client.get("/auth/me", headers={"Authorization": "Bearer non-valido"})
    assert r.status_code == 401


# ─── 5: refresh ─────────────────────────────────────────────────────────────────


async def test_refresh_returns_new_tokens(client: AsyncClient, admin_user):
    login = await client.post("/auth/login", json={"username": "admin", "password": "test1234"})
    refresh_token = login.json()["refresh_token"]

    r = await client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert r.status_code == 200
    assert r.json()["access_token"]


async def test_refresh_rejects_access_token(client: AsyncClient, admin_token):
    # Un access token NON deve essere accettato come refresh
    r = await client.post("/auth/refresh", json={"refresh_token": admin_token})
    assert r.status_code == 401


# ─── 6 & 7: register ─────────────────────────────────────────────────────────────


async def test_register_as_admin_creates_player(client: AsyncClient, admin_token):
    r = await client.post(
        "/auth/register",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"username": "mario", "email": "mario@example.com", "password": "secret99"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["username"] == "mario"
    assert body["role"] == "player"


async def test_register_duplicate_username_conflict(client: AsyncClient, admin_token):
    payload = {"username": "dup", "email": "dup@example.com", "password": "secret99"}
    first = await client.post(
        "/auth/register", headers={"Authorization": f"Bearer {admin_token}"}, json=payload
    )
    assert first.status_code == 201
    second = await client.post(
        "/auth/register", headers={"Authorization": f"Bearer {admin_token}"}, json=payload
    )
    assert second.status_code == 409


async def test_register_as_player_forbidden(client: AsyncClient, db_session):
    # Crea un player e usa il suo token → register deve dare 403
    player = User(
        username="player1",
        email="player1@example.com",
        password_hash=hash_password("test1234"),
        role=UserRole.player,
    )
    db_session.add(player)
    await db_session.commit()
    player_token = create_access_token({"sub": str(player.id), "role": player.role})

    r = await client.post(
        "/auth/register",
        headers={"Authorization": f"Bearer {player_token}"},
        json={"username": "x", "email": "x@example.com", "password": "secret99"},
    )
    assert r.status_code == 403


# ─── 8: admin users ──────────────────────────────────────────────────────────────


async def test_list_users_as_admin(client: AsyncClient, admin_token):
    r = await client.get("/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == 200
    usernames = [u["username"] for u in r.json()]
    assert "admin" in usernames


async def test_list_users_requires_auth(client: AsyncClient):
    r = await client.get("/admin/users")
    assert r.status_code == 401  # nessun header → HTTPBearer 401


# ─── 8b: admin gestisce gli account (modifica/attiva-disattiva/elimina) ─────────────


@pytest_asyncio.fixture
async def player_user(db_session) -> User:
    u = User(
        username="giocatore",
        email="g@example.com",
        password_hash=hash_password("old12345"),
        role=UserRole.player,
    )
    db_session.add(u)
    await db_session.commit()
    return u


def _hdr(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def test_admin_updates_email_and_password(client, admin_token, player_user):
    r = await client.patch(
        f"/admin/users/{player_user.id}",
        headers=_hdr(admin_token),
        json={"email": "nuova@example.com", "password": "new12345"},
    )
    assert r.status_code == 200
    assert r.json()["email"] == "nuova@example.com"
    # la nuova password funziona per il login
    login = await client.post(
        "/auth/login", json={"username": "giocatore", "password": "new12345"}
    )
    assert login.status_code == 200


async def test_admin_toggles_active(client, admin_token, player_user):
    off = await client.patch(
        f"/admin/users/{player_user.id}", headers=_hdr(admin_token), json={"is_active": False}
    )
    assert off.status_code == 200 and off.json()["is_active"] is False
    on = await client.patch(
        f"/admin/users/{player_user.id}", headers=_hdr(admin_token), json={"is_active": True}
    )
    assert on.json()["is_active"] is True


async def test_admin_deletes_user(client, admin_token, player_user):
    r = await client.delete(f"/admin/users/{player_user.id}", headers=_hdr(admin_token))
    assert r.status_code == 204
    users = await client.get("/admin/users", headers=_hdr(admin_token))
    assert all(u["id"] != str(player_user.id) for u in users.json())


async def test_admin_cannot_delete_self(client, admin_user, admin_token):
    r = await client.delete(f"/admin/users/{admin_user.id}", headers=_hdr(admin_token))
    assert r.status_code == 400


async def test_update_email_conflict(client, admin_token, player_user, db_session):
    other = User(
        username="altro",
        email="altro@example.com",
        password_hash=hash_password("x1234567"),
        role=UserRole.player,
    )
    db_session.add(other)
    await db_session.commit()
    r = await client.patch(
        f"/admin/users/{player_user.id}",
        headers=_hdr(admin_token),
        json={"email": "altro@example.com"},
    )
    assert r.status_code == 409
