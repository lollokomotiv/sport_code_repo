# Fase 7 — Integrazione API-Football

Obiettivo: l'admin può recuperare le fixture da API-Football, vederle come "staged" e aggiungerle a una giornata con un click.

> API: [API-Football](https://www.api-football.com) via RapidAPI. Free tier: 100 req/giorno (sufficiente per uso settimanale).

---

## Checklist

### Servizio API-Football (`app/services/fixtures.py`)

- [ ] Client `httpx.AsyncClient` configurato con headers RapidAPI:
  ```python
  headers = {
      "X-RapidAPI-Key": settings.api_football_key,
      "X-RapidAPI-Host": settings.api_football_host,
  }
  BASE_URL = f"https://{settings.api_football_host}"
  ```
- [ ] `fetch_fixtures(competition: str, matchday: int, season: int) -> list[dict]`
  - Mappa `competition` → `league_id` (Serie A=135, Serie B=136, Champions=2)
  - Chiama `GET /fixtures?league={id}&season={season}&round=Regular Season - {matchday}`
  - Ritorna lista di fixture normalizzate
- [ ] `upsert_staged_fixtures(fixtures: list[dict], db) -> int` — salva su DB, ritorna numero di nuove fixture
- [ ] Gestione errori: 429 (rate limit), 403 (chiave non valida), timeout

### Router Admin Fixtures (`/admin/fixtures`)

- [ ] `POST /admin/fixtures/fetch` — body: `{competition, matchday, season}`
  - Chiama `fetch_fixtures` + `upsert_staged_fixtures`
  - Risposta: `{fetched: int, new: int}`
- [ ] `GET /admin/fixtures/staged` — lista fixture in cache non ancora aggiunte a nessun round
- [ ] `POST /admin/fixtures/{fixture_id}/add-to-round` — body: `{round_id}`
  - Crea un `Match` dal `StagedFixture`
  - Marca `StagedFixture.added_to_round = true`

### Mapping competizioni API-Football

```python
COMPETITION_LEAGUE_IDS = {
    "serie_a": 135,
    "serie_b": 136,
    "champions_league": 2,
}

ROUND_FORMAT = {
    "serie_a": "Regular Season - {matchday}",
    "serie_b": "Regular Season - {matchday}",
    "champions_league": "League Stage - {matchday}",  # nuovo formato UCL dal 2024-25
}
```

> **Nota Champions League**: dal 2024-25 la Champions ha sostituito i gironi con un "League Stage" da 8 partite. Il formato dell'API round string va verificato per la stagione corrente.

### Auto-sync risultati (punto aperto C)

Questa feature è **opzionale** nella fase attuale. Se si decide di implementarla:

- Background task lanciato dal lifespan dell'app (APScheduler o asyncio loop)
- Ogni ora, per le partite con `kickoff < now - 2h` e risultato `NULL`, chiama `GET /fixtures?id={api_fixture_id}`
- Se lo status API è `FT` (Full Time), aggiorna `actual_home_goals` / `actual_away_goals` e scatena lo scoring
- **Non implementare in questa fase** — lascia l'inserimento manuale e torna qui se l'admin lo richiede

### Normalizzazione risposta API-Football

La risposta di API-Football è verbosa. Estrai solo:
```python
{
    "api_fixture_id": response["fixture"]["id"],
    "home_team": response["teams"]["home"]["name"],
    "away_team": response["teams"]["away"]["name"],
    "kickoff": response["fixture"]["date"],  # ISO 8601, converti a datetime
    "matchday": response["league"]["round"],  # stringa es. "Regular Season - 34"
    "status": response["fixture"]["status"]["short"],  # "NS", "FT", "HT", ...
}
```

---

## Test di accettazione fase 7

1. `POST /admin/fixtures/fetch` con chiave API valida → fixture salvate in `staged_fixtures`
2. Stessa chiamata ripetuta → niente duplicati (upsert per `api_fixture_id`)
3. `GET /admin/fixtures/staged` → fixture visibili
4. `POST /admin/fixtures/{id}/add-to-round` → `Match` creato nel round, `StagedFixture.added_to_round = true`
5. Test con `api_football_key=""` → risposta chiara (`API_FOOTBALL_KEY non configurata`)
6. Mock del client httpx nei test (non fare chiamate reali in CI)
