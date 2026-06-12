# Fase 6 — Leaderboard & Bonus di Fine Stagione

Obiettivo: classifica generale in tempo reale, classifiche separate per i bonus finali, assegnazione bonus di fine stagione.

> Fonte di verità: `docs/rules/REGOLAMENTO.md` §5 e §6.

> **Stato: ✅ Completata.** 10 test (5 unit + 5 integration), 91 totali verdi.
> Scelte di implementazione:
> - `PlayerSeasonProfile` tenuto MINIMALE: solo i 3 bonus di fine stagione. I punti
>   tabellone e la penalità mercato si leggono da `TablePrediction.total_points`
>   (già al netto della penalità, Fase 3) — niente denormalizzazione/drift.
> - Aggiunte le classifiche per metrica (`/leaderboard/signs|exacts|tabellone`) con uno
>   schema `MetricLeaderboardEntry`; `LeaderboardEntry` ha in più `season_bonus_total`.
> - La leaderboard usa `get_latest_season` (qualsiasi stato) così resta consultabile
>   anche dopo la chiusura della stagione.
> - Finalizzazione su `POST /admin/seasons/{id}/finalize` (plurale, coerente col resto).
> - `assign_season_bonus`: nessun bonus se il punteggio migliore è ≤ 0.
> - Bonus parità: integrale a tutti i pari merito (REGOLAMENTO §6).

---

## Checklist

### Schemas Pydantic
- [x] `LeaderboardEntry`: `{rank, player_id, username, total_points, sign_points, exact_points, total_goals_points, tabellone_points, weekend_bonus_total}`
- [x] `RoundLeaderboardEntry`: `{rank, player_id, username, round_points, sign_points, exact_points, total_goals_points, weekend_bonus}`

### Router Leaderboard (`/leaderboard`)

**Player & Admin:**
- [x] `GET /leaderboard` — classifica generale della stagione corrente
  - Aggrega tutti i `RoundScore` + penalità tabellone + punti tabellone (se già calcolati)
  - Ordinato per `total_points` DESC
- [x] `GET /leaderboard/rounds/{round_id}` — classifica di una singola giornata
- [x] `GET /leaderboard/signs` — classifica per soli segni (per bonus fine stagione)
- [x] `GET /leaderboard/exacts` — classifica pieni+gol (per bonus fine stagione)
- [x] `GET /leaderboard/tabellone` — classifica tabellone (per bonus fine stagione)

### Calcolo classifica generale

La classifica generale è la somma di:
- `sum(RoundScore.total_round_points)` per tutti i round della stagione
- `TablePrediction.total_points` (calcolato e salvato dopo `POST /admin/tabellone/score`)
- `sum(TablePredictionModification.penalty_points)` (già inclusi come negativi nel round score? Decidi una strategia coerente)

> **Strategia raccomandata**: le penalità delle modifiche tabellone (-5pt per modifica) vengono salvate come `tabellone_penalty_total` sul profilo del giocatore per la stagione (campo su `User` o tabella separata `PlayerSeasonProfile`). Così la classifica generale è:
> `round_points_total + tabellone_points - tabellone_penalties`

### Admin — Bonus di Fine Stagione (`/admin/season/{id}/finalize`)

- [x] `POST /admin/season/{id}/finalize` — calcola e assegna i 3 bonus da 10pt
  1. Recupera classifica segni → trova il/i migliore/i → assegna +10pt a ciascuno
  2. Recupera classifica pieni+gol → stessa logica
  3. Recupera classifica tabellone → stessa logica
  4. Salva i bonus in una tabella `SeasonBonus` o come campo su `PlayerSeasonProfile`
  5. Porta la stagione a stato `closed`

```python
# Logica assegnazione bonus (stessa per tutte e 3 le classifiche)
def assign_season_bonus(
    ranking: list[tuple[str, int]],  # (player_id, score)
    bonus_points: int = 10
) -> dict[str, int]:
    """In caso di parità, il bonus è assegnato integralmente a tutti i pari merito."""
    if not ranking:
        return {}
    best_score = max(pts for _, pts in ranking)
    return {pid: bonus_points for pid, pts in ranking if pts == best_score}
```

### Modello `PlayerSeasonProfile` (opzionale ma consigliato)

```
PlayerSeasonProfile
  id UUID PK
  player_id UUID FK → User
  season_id UUID FK → Season
  tabellone_penalty_total SMALLINT DEFAULT 0   -- somma di tutti i -5pt delle modifiche
  tabellone_points SMALLINT DEFAULT 0          -- punti tabellone finali
  season_bonus_signs SMALLINT DEFAULT 0        -- 0 o 10
  season_bonus_exacts SMALLINT DEFAULT 0       -- 0 o 10
  season_bonus_tabellone SMALLINT DEFAULT 0    -- 0 o 10
  UNIQUE (player_id, season_id)
```

---

## Test di accettazione fase 6

1. `GET /leaderboard` con 3 giocatori e 2 giornate completate → ranking corretto
2. Parità in classifica → stessa posizione, entrambi ricevono il bonus fine stagione
3. `POST /admin/season/{id}/finalize` → bonus assegnati, stagione `closed`
4. Dopo finalizzazione → `GET /leaderboard` mostra i punti finali con bonus inclusi
5. Test: giocatore con modifiche tabellone vede le penalità riflesse in classifica
