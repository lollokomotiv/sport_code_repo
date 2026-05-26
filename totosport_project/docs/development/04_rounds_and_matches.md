# Fase 4 — Rounds & Matches

Obiettivo: l'admin può creare giornate, aggiungere partite manualmente o dalla cache API-Football, e gestire il ciclo di vita della giornata.

---

## Checklist

### Modelli ORM
- [ ] `app/models/round.py` — modello `Round`
- [ ] `app/models/match.py` — modello `Match`
- [ ] `app/models/staged_fixture.py` — modello `StagedFixture`
- [ ] Migrazione Alembic

### Schemas Pydantic
- [ ] `RoundCreate`, `RoundOut`, `RoundStatusUpdate`
- [ ] `MatchCreate` (manuale o da staged fixture), `MatchOut`, `MatchResultUpdate`
- [ ] `StagedFixtureOut`

### Router Rounds (`/rounds`)
- [ ] `GET /rounds` — lista giornate visibili al player (status `open` o `completed`), paginate
- [ ] `GET /rounds/{id}` — dettaglio giornata con lista partite
- [ ] `POST /rounds` — crea nuova giornata (admin)
- [ ] `PATCH /rounds/{id}/status` — transizione stato (admin)
- [ ] `DELETE /rounds/{id}` — elimina (solo se in stato `draft`, admin)

### Router Matches (`/matches`)
- [ ] `GET /rounds/{round_id}/matches` — lista partite della giornata
- [ ] `POST /rounds/{round_id}/matches` — aggiunge partita alla giornata (admin, solo se `draft`)
  - Body: manuale (`home_team`, `away_team`, `kickoff`) oppure `staged_fixture_id`
- [ ] `PATCH /matches/{id}/result` — imposta risultato finale e scatena scoring (admin)
  - Richiama `scoring_service.score_match_and_persist(match_id, db)` (implementato in Fase 5)
- [ ] `DELETE /matches/{id}` — rimuovi partita (admin, solo se round in `draft`)

### Status machine del Round

```
draft ──► open ──► closed ──► completed
```

| Transizione | Chi | Precondizioni |
|---|---|---|
| `draft → open` | admin | Almeno 1 partita nel round; `deadline` impostata e nel futuro |
| `open → closed` | admin (o job) | Deadline superata |
| `closed → completed` | admin | Tutti i risultati inseriti, scoring eseguito |
| `completed → closed` | ❌ | Non ammessa: il round è immutabile |

Nota su **punto aperto B** (`CLAUDE.md` §11): per ora implementa la transizione `open → closed` manuale. Il job automatico può essere aggiunto in Fase 7.

### Validazioni importanti
- [ ] Non si può aggiungere una partita a un round non in stato `draft`
- [ ] Non si può impostare un risultato se il round è in stato `draft` o `open`
- [ ] La deadline deve essere nel futuro al momento della transizione `draft → open`
- [ ] Non si possono aggiungere due partite con le stesse squadre nella stessa giornata

---

## Note sul modello Match

Il campo `api_fixture_id` (nullable) collega la partita alla fixture di API-Football. Serve per l'auto-sync dei risultati (Fase 7). Se la partita è inserita manualmente, è `NULL`.

**Non esiste `prediction_type` sul Match.** Tutte le partite richiedono sempre un risultato esatto. (Vedi `CLAUDE.md` §6.1)

---

## Test di accettazione fase 4

1. Admin crea round in stato `draft`
2. Admin aggiunge 3 partite manualmente
3. Admin porta round a `open` → giocatori lo vedono in `GET /rounds`
4. Tenta di aggiungere partita a round `open` → 422/403
5. Admin porta round a `closed`
6. `GET /rounds` player non mostra più il round come "aperto"
7. Admin imposta risultato su partita → risposta 200, scoring verrà chiamato (testato in Fase 5)
