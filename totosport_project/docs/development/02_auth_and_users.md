# Fase 2 — Auth & Users

Obiettivo: sistema di autenticazione JWT funzionante, con ruoli `admin` e `player`. Modello `Season` creato.

---

## Checklist

### Modelli ORM
- [ ] `app/models/user.py` — modello `User` (vedi schema in `CLAUDE.md` §7)
- [ ] `app/models/season.py` — modello `Season` con status enum
- [ ] Migrazione Alembic: `alembic revision --autogenerate -m "add user season"`
- [ ] `alembic upgrade head` senza errori

### Schemas Pydantic
- [ ] `app/schemas/user.py`: `UserCreate`, `UserOut`, `UserLogin`
- [ ] `app/schemas/season.py`: `SeasonCreate`, `SeasonOut`
- [ ] `app/schemas/auth.py`: `TokenPair` (`access_token`, `refresh_token`, `token_type`)

### Servizio Auth
- [ ] `app/services/auth.py`:
  - `hash_password(plain: str) -> str` — bcrypt
  - `verify_password(plain: str, hashed: str) -> bool`
  - `create_access_token(data: dict) -> str` — JWT, scade in `ACCESS_TOKEN_EXPIRE_MINUTES`
  - `create_refresh_token(data: dict) -> str` — JWT, scade in `REFRESH_TOKEN_EXPIRE_DAYS`
  - `decode_token(token: str) -> dict` — lancia `HTTPException 401` se invalido/scaduto

### Dependencies FastAPI
- [ ] `app/dependencies/auth.py`:
  - `get_current_user(token, db) -> User` — legge Bearer token, ritorna utente
  - `require_admin(user) -> User` — lancia 403 se `user.role != "admin"`

### Router Auth (`/auth`)
- [ ] `POST /auth/login` — body: `{username, password}`, risposta: `TokenPair`
- [ ] `POST /auth/refresh` — body: `{refresh_token}`, risposta: nuovo `TokenPair`
- [ ] `POST /auth/register` — solo admin, crea nuovo player; body: `UserCreate`
- [ ] `GET /auth/me` — utente corrente (player o admin)

### Router Admin Users (`/admin/users`)
- [ ] `GET /admin/users` — lista tutti i giocatori (solo admin)
- [ ] `PATCH /admin/users/{id}` — modifica username/email (solo admin)
- [ ] `DELETE /admin/users/{id}` — disabilita account (soft delete, solo admin)

### Seed iniziale
- [ ] Script `backend/scripts/create_admin.py` per creare il primo utente admin da CLI
  ```
  python scripts/create_admin.py --username admin --email admin@example.com --password <pwd>
  ```

---

## Note tecniche

### JWT payload

```python
# Access token payload
{
    "sub": str(user.id),   # UUID come stringa
    "role": user.role,
    "type": "access",
    "exp": ...
}

# Refresh token payload
{
    "sub": str(user.id),
    "type": "refresh",
    "exp": ...
}
```

Controlla sempre `payload["type"]` per evitare che un access token venga usato come refresh e viceversa.

### Season status machine

```
setup → active → mercato → active → closed
```

- `setup`: stagione creata, tabellone non ancora aperto
- `active`: stagione in corso, giornate settimanali
- `mercato`: finestra di modifica tabellone aperta (dopo mercato invernale)
- `closed`: stagione terminata, bonus finali assegnati

Solo l'admin può fare le transizioni di stato.

### Nota su password

Non usare MD5 o SHA. Solo `bcrypt` via `passlib`:
```python
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
```

---

## Test di accettazione fase 2

1. `POST /auth/login` con credenziali admin corrette → ritorna `TokenPair`
2. `POST /auth/login` con password sbagliata → 401
3. `GET /auth/me` con access token valido → dati utente
4. `GET /auth/me` con token scaduto → 401
5. `POST /auth/refresh` → nuovo access token
6. `POST /auth/register` (con token admin) → crea player
7. `POST /auth/register` (con token player) → 403
8. `GET /admin/users` con token admin → lista utenti
