# Fase 5 — Predictions & Scoring Engine

Obiettivo: i giocatori inseriscono le previsioni prima della deadline; quando l'admin inserisce un risultato, lo scoring è calcolato e persistito automaticamente.

> Fonte di verità: `docs/rules/REGOLAMENTO.md` §3, §4, §5. Pseudocodice in `CLAUDE.md` §8.

---

## Checklist

### Modelli ORM
- [ ] `app/models/match_prediction.py` — `MatchPrediction` (vedi `CLAUDE.md` §7)
- [ ] `app/models/round_prediction.py` — `RoundPrediction` (totale gol giornata)
- [ ] `app/models/round_score.py` — `RoundScore` (score aggregato per player/round)
- [ ] Migrazione Alembic

### Schemas Pydantic
- [ ] `MatchPredictionCreate`: `{match_id, predicted_home_goals, predicted_away_goals}`
- [ ] `MatchPredictionOut`: include `points_earned`, `derived_sign`
- [ ] `RoundPredictionCreate`: `{round_id, total_goals_guess}`
- [ ] `RoundScoreOut`: tutti i campi del `RoundScore`

### Servizio Scoring (`app/services/scoring.py`)

Il cuore del sistema. Deve essere **testabile in isolamento** (nessuna dipendenza da FastAPI, DB opzionale nei test unitari).

```python
# Costanti — NON modificare senza aggiornare il REGOLAMENTO
SIGN_POINT = 1
EXACT_BONUS = {"1": 5, "X": 7, "2": 9}
HIGH_SCORING_BONUS = 2       # partita con 5+ gol totali
HIGH_SCORING_THRESHOLD = 5
TOTAL_GOALS_POINTS = 3
WEEKEND_BONUS = {1: 6, 2: 4, 3: 2}

def derive_sign(home: int, away: int) -> Literal["1", "X", "2"]:
    if home > away: return "1"
    if home == away: return "X"
    return "2"

def score_match(
    pred_home: int, pred_away: int,
    actual_home: int, actual_away: int
) -> tuple[int, int]:
    """
    Restituisce (sign_points, exact_points).
    sign_points: 1 se segno indovinato, 0 altrimenti
    exact_points: 5/7/9 (+2 se 5+ gol) se risultato esatto indovinato, 0 altrimenti
    Totale punti = sign_points + exact_points
    """
    pred_sign = derive_sign(pred_home, pred_away)
    actual_sign = derive_sign(actual_home, actual_away)

    sign_pts = SIGN_POINT if pred_sign == actual_sign else 0
    exact_pts = 0

    if pred_home == actual_home and pred_away == actual_away:
        exact_pts = EXACT_BONUS[actual_sign]
        if actual_home + actual_away >= HIGH_SCORING_THRESHOLD:
            exact_pts += HIGH_SCORING_BONUS

    return sign_pts, exact_pts

def score_total_goals(guess: int, matches: list[tuple[int, int]]) -> int:
    """
    matches: lista di (actual_home, actual_away) per le partite con risultato noto.
    """
    actual_total = sum(h + a for h, a in matches)
    return TOTAL_GOALS_POINTS if guess == actual_total else 0

def assign_weekend_bonus(
    scores: list[tuple[str, int]]  # (player_id, total_round_points)
) -> dict[str, int]:
    """
    Restituisce {player_id: bonus_points}.
    In caso di parità, il bonus della posizione è assegnato integralmente a tutti i pari merito.
    """
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    result = {}
    rank = 1
    i = 0
    while i < len(sorted_scores) and rank <= 3:
        group_score = sorted_scores[i][1]
        group = [pid for pid, pts in sorted_scores if pts == group_score]
        bonus = WEEKEND_BONUS.get(rank, 0)
        for pid in group:
            result[pid] = bonus
        rank += len(group)
        i += len(group)
    return result
```

### Funzione di scoring persistente

```python
# app/services/scoring.py
async def score_match_and_persist(match_id: UUID, db: AsyncSession) -> None:
    """
    Chiamata dopo che l'admin ha impostato il risultato di una partita.
    1. Calcola i punti per ogni MatchPrediction della partita
    2. Aggiorna MatchPrediction.points_earned
    3. Se tutte le partite del round hanno risultato → calcola score totale giornata,
       aggiorna RoundScore per ogni giocatore, assegna weekend bonus
    """
    ...

async def finalize_round(round_id: UUID, db: AsyncSession) -> None:
    """
    Da chiamare quando l'admin porta il round a 'completed'.
    Calcola/aggiorna RoundScore per ogni player (incluso total_goals_points)
    e assegna il weekend bonus.
    """
    ...
```

### Router Predictions (`/predictions`)

**Player:**
- [ ] `POST /predictions/match` — inserisce/aggiorna previsione per una partita (upsert)
  - Controlla: round `open`, deadline non superata, partita nel round
  - Body: `MatchPredictionCreate`
- [ ] `POST /predictions/round-goals` — inserisce/aggiorna previsione totale gol (upsert)
  - Stessi controlli
- [ ] `GET /predictions/me?round_id=...` — le proprie previsioni per una giornata
- [ ] `GET /predictions/me/history` — tutte le previsioni con punti, paginate per round

**Admin:**
- [ ] `GET /admin/predictions/{round_id}` — tutte le previsioni di tutti i giocatori per una giornata (solo dopo deadline)
- [ ] `GET /admin/predictions/{round_id}/{match_id}` — previsioni per una singola partita

---

## Importante: separazione dei punti per le classifiche separate

In `RoundScore`, tenere separati:
- `sign_points`: solo i punti da segno (1pt per segno giusto)
- `exact_points`: solo i punti da esatto (bonus 5/7/9 + high-scoring bonus)
- `total_goals_points`: 0 o 3 da totale gol giornata
- `weekend_bonus`: 0, 2, 4, o 6
- `total_round_points`: sign_points + exact_points + total_goals_points + weekend_bonus

Questo permette di calcolare le 3 classifiche separate di fine stagione:
1. Classifica Segni → somma `sign_points` di tutte le giornate
2. Classifica Pieni+Gol → somma `exact_points + total_goals_points`
3. Classifica Tabellone → punti da `TablePrediction` (Fase 3)

---

## Test di accettazione fase 5

### Test unitari (niente DB, solo funzioni pure)
- [ ] `score_match(2, 1, 2, 1)` → `(1, 5)` — casa esatto
- [ ] `score_match(1, 1, 1, 1)` → `(1, 7)` — pareggio esatto
- [ ] `score_match(0, 2, 0, 2)` → `(1, 9)` — trasferta esatto
- [ ] `score_match(3, 3, 3, 3)` → `(1, 7+2)` = `(1, 9)` — pareggio esatto con 6+ gol (6 gol totali ≥ 5)
- [ ] `score_match(2, 1, 1, 0)` → `(1, 0)` — segno giusto (casa), esatto sbagliato
- [ ] `score_match(0, 1, 1, 0)` → `(0, 0)` — segno sbagliato
- [ ] `score_total_goals(5, [(2,1),(1,1)])` → `3` — totale giusto
- [ ] `score_total_goals(6, [(2,1),(1,1)])` → `0` — totale sbagliato
- [ ] `assign_weekend_bonus([("A",10),("B",8),("C",8),("D",5)])` → `{"A":6,"B":2,"C":2}` — parità 2°/3°

### Test di integrazione
- [ ] Player inserisce previsione per tutte le partite + totale gol
- [ ] Player tenta di inserire dopo deadline → 403
- [ ] Admin imposta risultato → `MatchPrediction.points_earned` aggiornato
- [ ] Dopo tutti i risultati → `RoundScore` creato con tutti i campi corretti
- [ ] Weekend bonus assegnato correttamente, incluso caso di parità
