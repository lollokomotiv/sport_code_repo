# Fase 3 — Tabellone (Previsioni Annuali)

Obiettivo: un giocatore può compilare il tabellone a inizio stagione; l'admin può aprire/chiudere la finestra, inserire i risultati reali e assegnare i punti. Supporto completo alle modifiche post-mercato.

> Fonte di verità per tutte le regole: `docs/rules/REGOLAMENTO.md` §2 e §7.

---

## Checklist

### Modelli ORM
- [ ] `app/models/table_prediction.py` — modello `TablePrediction` (tutti i campi, vedi `CLAUDE.md` §7)
- [ ] `app/models/table_modification.py` — modello `TablePredictionModification`
- [ ] `app/models/season_outcome.py` — modello `SeasonOutcome` (risultati reali della stagione inseriti dall'admin)
- [ ] Migrazione Alembic

### Modello `SeasonOutcome`

Contiene i valori reali della stagione che l'admin inserisce a fine anno (o man mano che le competizioni si concludono). Specchia i campi di `TablePrediction`:

```python
# app/models/season_outcome.py
class SeasonOutcome(Base):
    __tablename__ = "season_outcomes"
    id: UUID PK
    season_id: UUID FK → Season (UNIQUE — una per stagione)
    # Serie A
    scudetto_team: str | None
    scudetto_points: int | None
    relegated_a_1/2/3: str | None
    top_scorer_a: str | None
    top_scorer_a_goals: int | None
    # Serie B
    promoted_b_direct_1/2: str | None
    promoted_b_first_points: int | None
    promoted_b_via_playoff: bool | None  # True = promossa via playoff
    promoted_b_playoff: str | None
    playoffs_held: bool | None           # False → si usano le regole "no playoff"
    relegated_b_c_direct_1/2/3: str | None
    playout_b_1/2: str | None
    relegated_b_c_via_playout: bool | None
    playout_held: bool | None
    relegated_b_c_playout: str | None
    top_scorer_b: str | None
    top_scorer_b_goals: int | None
    # Coppe
    coppa_italia_winner: str | None
    champions_winner: str | None
    europa_winner: str | None
    conference_winner: str | None
```

### Schemas Pydantic
- [ ] `TablePredictionCreate` — tutti i campi del tabellone, richiesti
- [ ] `TablePredictionOut` — con flag `is_modifiable` (True se stagione in stato `mercato`)
- [ ] `TablePredictionModify` — solo i campi che si possono modificare (stessi di Create)
- [ ] `SeasonOutcomeCreate` / `SeasonOutcomeOut`

### Servizio Tabellone (`app/services/tabellone.py`)

- [ ] `submit_tabellone(player_id, season_id, data, db)` — crea o aggiorna (se stagione in `setup` con deadline non superata)
- [ ] `modify_tabellone(player_id, season_id, changes, db)` — modifica post-mercato, applica penalità, registra modifiche
- [ ] `score_tabellone(season_id, db)` — calcola i punti per tutti i giocatori basandosi su `SeasonOutcome`

### Logica di scoring del Tabellone

Implementa in `app/services/tabellone.py` la funzione `score_single_prediction(pred, outcome)` che restituisce il numero di punti per ogni campo. Usa le costanti da `REGOLAMENTO.md` §2:

```python
TABELLONE_POINTS = {
    "scudetto_team": 25,
    "scudetto_points": 8,            # bonus, solo se scudetto_team corretto
    "relegated_a_1": 25,
    "relegated_a_2": 25,
    "relegated_a_3": 25,
    "top_scorer_a": 15,
    "top_scorer_a_goals": 8,         # bonus, solo se top_scorer_a corretto
    "promoted_b_direct_1": 30,       # diventa 15 se la squadra è poi promossa via playoff
    "promoted_b_direct_2": 30,
    "promoted_b_first_points": 8,    # bonus, solo se promoted_b_direct_1 corretto
    "playoff_b_1": 20,               # .. fino a playoff_b_6
    "promoted_b_playoff": 30,        # diventa 15 se la squadra è promossa direttamente
    "no_playoffs": 40,               # alternativa se playoffs_held == False
    "relegated_b_c_direct_1": 30,
    "relegated_b_c_direct_2": 30,
    "relegated_b_c_direct_3": 30,
    "playout_b_1": 20,
    "playout_b_2": 20,
    "relegated_b_c_playout": 30,
    "no_playouts": 35,               # alternativa se playout_held == False
    "top_scorer_b": 15,
    "top_scorer_b_goals": 8,
    "coppa_italia_winner": 25,
    "champions_winner": 25,
    "europa_winner": 25,
    "conference_winner": 25,
}
```

Casi speciali da gestire:
- **Playoff/Playout non disputati**: se `outcome.playoffs_held == False`, i punti per le 6 squadre ai playoff e la promossa con playoff sono sostituiti da un unico punteggio `no_playoffs = 40pt` → assegnato se il giocatore aveva previsto correttamente **almeno una** delle promosse dirette? *(da chiarire con l'admin)*
- **Promozione via metodo sbagliato**: se prevedi una squadra come "promossa diretta" ma sale via playoff → 50% (15pt). Viceversa anche.
- **Parità capocannoniere**: se ci sono due capocannonieri a pari gol, i 15pt vengono divisi per 2 (arrotondati per difetto)

### Router Tabellone

**Player**
- [ ] `POST /tabellone` — compila o aggiorna tabellone (solo se stagione in stato `setup` o `active` prima della deadline)
- [ ] `GET /tabellone/me` — visualizza il proprio tabellone con punti (se già calcolati)
- [ ] `PATCH /tabellone/me` — modifica post-mercato (solo se stagione in stato `mercato`, applica penalità)

**Admin**
- [ ] `GET /admin/tabellone` — tutti i tabelloni dei giocatori per la stagione corrente
- [ ] `POST /admin/season-outcome` — inserisce/aggiorna i risultati reali della stagione
- [ ] `GET /admin/season-outcome` — vedi risultati inseriti
- [ ] `POST /admin/tabellone/score` — calcola e salva i punti per tutti i giocatori
- [ ] `PATCH /admin/season/{id}/status` — transizione di stato stagione

---

## Gestione modifiche post-mercato

Quando la stagione è in stato `mercato`, `PATCH /tabellone/me` accetta un body con i soli campi che si vogliono modificare. Per ogni campo modificato:

1. Controlla che il campo sia diverso dal valore attuale (non addebitare se rimane uguale)
2. Crea una riga in `TablePredictionModification`
3. Applica immediatamente `-5pt` in classifica (salvato come `tabellone_penalty` su `User` per la stagione, oppure come riga speciale in `RoundScore`)
4. Marca il campo come "modificato" — il cap al 50% verrà applicato durante lo scoring finale

Il cap al 50% si applica **campo per campo**: se hai modificato solo lo Scudetto, solo lo Scudetto è cappato al 50%. Gli altri campi mantengono i punti pieni.

---

## Test di accettazione fase 3

1. Player compila tabellone a inizio stagione → salvato correttamente
2. Player tenta di modificare tabellone fuori dalla finestra → 403
3. Admin porta stagione a stato `mercato`
4. Player modifica 2 campi → `-10pt` immediati, 2 righe in `TablePredictionModification`
5. Admin inserisce `SeasonOutcome` con i risultati reali
6. Admin lancia scoring → punti assegnati correttamente (inclusi cap 50% per campi modificati)
7. Test unitario per ogni caso speciale dello scoring (playoff non disputati, parità capocannoniere, promozione via metodo sbagliato)
