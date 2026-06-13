# CLAUDE.md — TotoSport Development Guide

Guida operativa per sviluppare TotoSport. Leggila interamente prima di toccare il codice.

---

## 1. Che cos'è il progetto

TotoSport è un gioco di pronostici calcistici per un gruppo di amici (circa 20 giocatori). Non è un prodotto SaaS: è un'applicazione **privata e closed**, con pochi utenti, gestita da un admin umano.

Questo vincola alcune scelte architetturali:
- La semplicità batte la scalabilità orizzontale
- Un singolo admin umano gestisce le stagioni manualmente
- Non servono pipeline real-time: il polling periodico è sufficiente
- La user experience su mobile è prioritaria (i giocatori usano lo smartphone)

---

## 2. Gerarchia delle fonti — qual documento ha la precedenza

| Priorità | Documento | Usato per |
|---|---|---|
| 1 (massima) | `docs/rules/REGOLAMENTO.md` | Tutte le regole di business, scoring, punti |
| 2 | `ARCHITECTURE.md` | Struttura tecnica del sistema |
| 3 | `README.md` | Overview di alto livello |
| 4 | `docs/development/*.md` | Piano di sviluppo passo per passo |

> **ATTENZIONE:** `ARCHITECTURE.md` contiene alcune incoerenze con il REGOLAMENTO (vedi §6).
> In caso di conflitto, il REGOLAMENTO vince sempre.

---

## 3. Stack tecnico — versioni esatte

### Backend
| Componente | Versione | Note |
|---|---|---|
| Python | 3.11 | f-strings, `match` statement, `tomllib` |
| FastAPI | 0.111+ | async-first, Pydantic v2 |
| SQLAlchemy | 2.0 | stile `async` con `asyncpg` |
| Alembic | 1.13+ | migrazioni DB |
| asyncpg | 0.29+ | driver PostgreSQL async |
| Pydantic | 2.x | modelli request/response |
| python-jose | 3.x | JWT (HS256) |
| passlib[bcrypt] | 1.7 | hashing password |
| httpx | 0.27+ | client HTTP async per API-Football |
| pytest + pytest-asyncio | latest | test |

### Frontend
| Componente | Versione | Note |
|---|---|---|
| Node.js | 20 LTS | |
| React | 18 | concurrent features |
| TypeScript | 5.x | strict mode abilitato |
| Vite | 5.x | bundler |
| Tailwind CSS | 3.x | utility-first |
| React Router | 6.x | client-side routing |
| Axios | 1.x | HTTP client con interceptors |
| Zustand | 4.x | state management leggero (alternativa a Redux) |
| React Query (TanStack) | 5.x | server state, caching, refetch |
| Vitest + Testing Library | latest | test |

### Infrastruttura
| Componente | Versione | Note |
|---|---|---|
| PostgreSQL | 16 | |
| Docker + Docker Compose | latest stable | ambienti dev e prod |
| Nginx | alpine | reverse proxy + serve frontend build |

---

## 4. Conoscenza del dominio — Calcio italiano

Questa sezione è fondamentale. Senza capire il funzionamento del calcio italiano, l'implementazione del Tabellone sarà sbagliata.

### 4.1 Struttura del campionato

**Serie A**
- 20 squadre, 38 giornate
- Le ultime 3 in classifica retrocedono direttamente in Serie B
- Il campionato inizia a settembre e finisce a maggio
- Il vincitore si chiama **Scudetto**
- Il primo marcatore assoluto è il **Capocannoniere**

**Serie B**
- 20 squadre, 38 giornate
- **Promozione diretta**: le prime 2 in classifica salgono in A senza playoff
- **Playoff di promozione**: le squadre dalla 3ª all'8ª si affrontano. La vincente sale in A
- **Playout di retrocessione**: la 17ª e la 18ª si affrontano. La perdente scende in C
- **Retrocessione diretta**: le ultime 2 (19ª e 20ª) scendono in C senza playout
- In alcune stagioni playoff o playout non si disputano (es. per distacco di punti)
- La prima classificata della Serie B è il **Capocannoniere B**

**Coppe**
- **Coppa Italia**: torneo a eliminazione diretta, tutte le squadre italiane
- **Champions League**: massimo torneo europeo per club, formato league phase + knockout
- **Europa League** e **Conference League**: tornei europei di secondo e terzo livello

### 4.2 Il mercato di trasferimento

- **Mercato estivo**: giugno–settembre (prima dell'inizio del campionato)
- **Mercato invernale**: gennaio, si chiude di solito il 31 gennaio
- La finestra di modifica del Tabellone apre alla chiusura del mercato invernale

### 4.3 Cos'è una "giornata"

Una **giornata** (matchday) è il turno settimanale del campionato. In Serie A le partite si giocano tipicamente venerdì, sabato, domenica e lunedì. L'admin crea una giornata TotoSport aggregando partite da una o più competizioni nello stesso weekend.

### 4.4 Il Tabellone

È la schedina annuale compilata a inizio stagione. Contiene pronostici su eventi che si risolvono solo a fine stagione (chi vince lo Scudetto, chi retrocede, ecc.). I punti vengono assegnati a fine stagione dall'admin. Vedi `docs/rules/REGOLAMENTO.md` §2 per tutti i campi.

---

## 5. Architettura — decisioni chiave

### 5.1 Una singola app React (SPA)
Il frontend è una Single Page Application. Il routing si divide in:
- `/login` — pubblica
- `/player/*` — giocatori autenticati
- `/admin/*` — solo admin

### 5.2 Backend API-first
Il backend espone solo JSON API. Non esiste server-side rendering. Il frontend consuma le API tramite Axios.

### 5.3 JWT stateless
- **Access token**: 15 minuti
- **Refresh token**: 7 giorni (ruotato ad ogni uso)
- Entrambi trasportati via header `Authorization: Bearer` e cookie HttpOnly rispettivamente
- Il frontend gestisce l'auto-refresh in Axios interceptors

### 5.4 Scoring asincrono
Lo scoring di una partita è scatenato quando l'admin salva il risultato finale (`PATCH /matches/{id}/result`). Non è un job in background: è sincrono sulla request. Con ~20 giocatori è assolutamente gestibile.

### 5.5 API-Football
Usata solo per recuperare fixture (squadre, orari, giornate). I risultati finali **possono** essere auto-sincronizzati o inseriti manualmente dall'admin. La strategia è ancora aperta (vedi §10).

---

## 6. Correzioni critiche rispetto ad ARCHITECTURE.md

`ARCHITECTURE.md` è stato scritto prima di finalizzare il REGOLAMENTO. Queste sezioni sono **superate**:

### 6.1 Segno scelto su ogni partita; risultato esatto solo su alcune

> ⚠️ **Aggiornato in Fase 9 su indicazione del proprietario del gioco** — supera sia
> `ARCHITECTURE.md` sia la versione precedente di questa stessa sezione.

Regola corretta:
- Su **ogni** partita il giocatore **sceglie il segno** (1/X/2). Il segno **NON è derivato**:
  è una scelta esplicita (campo `predicted_sign` su `MatchPrediction`, menù a tendina nel form).
- Solo sulle **3-4 partite scelte dall'admin per giornata** si pronostica **anche il risultato
  esatto** (campo `Match.requires_exact_score = True`). Su queste si guadagna il bonus esatto;
  sulle altre il massimo è 1 pt (solo segno).

Quindi un flag per-partita **esiste** (`requires_exact_score`), reintrodotto rispetto alla
precedente nota "niente prediction_type". `derive_sign` resta in uso solo per calcolare il
segno del **risultato reale** (per confrontarlo col segno scelto):

```python
def derive_sign(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals: return "1"
    if home_goals == away_goals: return "X"
    return "2"
```

### 6.2 Punteggi dello scoring sono diversi

`ARCHITECTURE.md` e il vecchio `README.md` indicano: segno corretto = 2pt, risultato esatto = 5pt. **Entrambi sbagliati.**

Il sistema reale (da REGOLAMENTO §4):
```
Segno corretto (solo):         1 pt
Risultato esatto — casa:       6 pt totali  (1 segno + 5 bonus)
Risultato esatto — pareggio:   8 pt totali  (1 segno + 7 bonus)
Risultato esatto — trasferta: 10 pt totali  (1 segno + 9 bonus)
Bonus 5+ gol (solo se esatto): +2 pt extra
```

### 6.3 Schema `MatchPrediction` (aggiornato in Fase 9)

`MatchPrediction` ha `predicted_sign` come campo **esplicito e obbligatorio** (scelto dal
giocatore). `predicted_home_goals`/`predicted_away_goals` sono **opzionali** e si compilano
solo per le partite con `requires_exact_score = True`. (Questo supera la precedente nota che
diceva di rimuovere `predicted_sign`: era basata sull'assunzione errata del "solo risultato esatto".)

### 6.4 Mancano le entità del Tabellone

`ARCHITECTURE.md` non modella affatto il Tabellone. È una componente di primo piano del gioco. Vedi REGOLAMENTO §9 per lo schema completo.

### 6.5 Mancano le classifiche separate

Il REGOLAMENTO prevede 3 classifiche distinte per i bonus di fine stagione:
1. **Classifica Segni** — solo punti da segno corretto
2. **Classifica Pieni+Gol** — punti da risultato esatto + totale gol giornata
3. **Classifica Tabellone** — punti dal tabellone annuale

Il `Score` per ogni `(player, round)` deve tracciare queste componenti separatamente.

---

## 7. Modello dati corretto (supersede ARCHITECTURE.md)

```
User
  id UUID PK
  username VARCHAR(50) UNIQUE NOT NULL
  email VARCHAR(100) UNIQUE NOT NULL
  password_hash TEXT NOT NULL
  role ENUM('admin', 'player') NOT NULL
  created_at TIMESTAMP

Season
  id UUID PK
  name VARCHAR(10)  -- es. "2025-26"
  status ENUM('setup', 'active', 'mercato', 'closed')
  -- 'mercato': finestra di modifica tabellone aperta
  tabellone_deadline TIMESTAMP  -- scadenza compilazione tabellone inizio anno
  modification_deadline TIMESTAMP  -- scadenza modifiche post-mercato
  created_at TIMESTAMP

TablePrediction  (una per giocatore per stagione)
  id UUID PK
  player_id UUID FK → User
  season_id UUID FK → Season
  -- UNIQUE (player_id, season_id)
  submitted_at TIMESTAMP
  -- Serie A
  scudetto_team VARCHAR(80)
  scudetto_points_guess SMALLINT
  relegated_a_1 VARCHAR(80)
  relegated_a_2 VARCHAR(80)
  relegated_a_3 VARCHAR(80)
  top_scorer_a VARCHAR(80)
  top_scorer_a_goals SMALLINT
  -- Serie B
  promoted_b_direct_1 VARCHAR(80)
  promoted_b_direct_2 VARCHAR(80)
  promoted_b_first_points SMALLINT  -- punti della 1ª classificata B
  playoff_b_1..6 VARCHAR(80)        -- 6 squadre (colonne separate o JSONB)
  promoted_b_playoff VARCHAR(80)
  relegated_b_c_direct_1..3 VARCHAR(80)
  playout_b_1 VARCHAR(80)
  playout_b_2 VARCHAR(80)
  relegated_b_c_playout VARCHAR(80)
  top_scorer_b VARCHAR(80)
  top_scorer_b_goals SMALLINT
  -- Coppe
  coppa_italia_winner VARCHAR(80)
  champions_winner VARCHAR(80)
  europa_winner VARCHAR(80)
  conference_winner VARCHAR(80)

TablePredictionModification  (log delle modifiche post-mercato)
  id UUID PK
  prediction_id UUID FK → TablePrediction
  field_name VARCHAR(80)       -- nome del campo modificato
  old_value TEXT
  new_value TEXT
  penalty_points SMALLINT      -- -5 applicati
  max_points_cap SMALLINT      -- cap al 50% per questo campo
  modified_at TIMESTAMP

Round  (Giornata)
  id UUID PK
  season_id UUID FK → Season
  name VARCHAR(100)            -- es. "Giornata 34 — Serie A"
  competition ENUM('serie_a', 'serie_b', 'champions_league', 'mixed')
  matchday SMALLINT
  deadline TIMESTAMP
  status ENUM('draft', 'open', 'closed', 'completed')
  created_at TIMESTAMP

Match
  id UUID PK
  round_id UUID FK → Round
  home_team VARCHAR(80) NOT NULL
  away_team VARCHAR(80) NOT NULL
  kickoff TIMESTAMP
  actual_home_goals SMALLINT   -- NULL fino a quando non giocata
  actual_away_goals SMALLINT   -- NULL fino a quando non giocata
  api_fixture_id INTEGER       -- da API-Football, nullable

MatchPrediction  (una per giocatore per partita)
  id UUID PK
  player_id UUID FK → User
  match_id UUID FK → Match
  predicted_home_goals SMALLINT NOT NULL
  predicted_away_goals SMALLINT NOT NULL
  -- sign derivato, NON salvato: derive_sign(predicted_home_goals, predicted_away_goals)
  points_earned SMALLINT DEFAULT 0  -- calcolato dopo il risultato
  submitted_at TIMESTAMP
  -- UNIQUE (player_id, match_id)

RoundPrediction  (totale gol giornata, una per giocatore per giornata)
  id UUID PK
  player_id UUID FK → User
  round_id UUID FK → Round
  total_goals_guess SMALLINT NOT NULL
  points_earned SMALLINT DEFAULT 0
  submitted_at TIMESTAMP
  -- UNIQUE (player_id, round_id)

RoundScore  (score aggregato per (giocatore, giornata), calcolato dopo scoring)
  id UUID PK
  player_id UUID FK → User
  round_id UUID FK → Round
  sign_points SMALLINT DEFAULT 0        -- per classifica segni
  exact_points SMALLINT DEFAULT 0       -- punti da risultati esatti (incluso bonus 5+gol)
  total_goals_points SMALLINT DEFAULT 0 -- per classifica pieni+gol
  weekend_bonus SMALLINT DEFAULT 0      -- 0, 2, 4, o 6
  total_round_points SMALLINT DEFAULT 0 -- somma di tutto
  -- UNIQUE (player_id, round_id)

StagedFixture  (cache fixtures da API-Football)
  id UUID PK
  api_fixture_id INTEGER UNIQUE NOT NULL
  competition ENUM
  season VARCHAR(10)
  matchday SMALLINT
  home_team VARCHAR(80)
  away_team VARCHAR(80)
  kickoff TIMESTAMP
  fetched_at TIMESTAMP
  added_to_round BOOLEAN DEFAULT false
```

---

## 8. Regole di business critiche — non sbagliare queste

### 8.1 Scoring partita (backend/app/services/scoring.py)

```python
EXACT_BONUS = {"1": 5, "X": 7, "2": 9}

def score_match(pred_home: int, pred_away: int,
                actual_home: int, actual_away: int) -> tuple[int, int]:
    """
    Returns (sign_points, exact_points).
    sign_points: 0 o 1
    exact_points: 0, oppure bonus (5/7/9) + eventuale +2 per 5+ gol
    """
    pred_sign = derive_sign(pred_home, pred_away)
    actual_sign = derive_sign(actual_home, actual_away)

    sign_pts = 1 if pred_sign == actual_sign else 0
    exact_pts = 0

    if pred_home == actual_home and pred_away == actual_away:
        exact_pts = EXACT_BONUS[actual_sign]
        if actual_home + actual_away >= 5:
            exact_pts += 2  # bonus partite con 5+ gol

    return sign_pts, exact_pts
    # Nota: exact_pts include già il segno quando esatto è corretto?
    # NO. Il totale finale è sign_pts + exact_pts.
    # Se esatto corretto: sign_pts=1, exact_pts=5/7/9 (+2) → totale 6/8/10 (+2)
```

### 8.2 Scoring totale gol giornata

```python
def score_total_goals(guess: int, round_matches: list[Match]) -> int:
    actual = sum(m.actual_home_goals + m.actual_away_goals for m in round_matches
                 if m.actual_home_goals is not None)
    return 3 if guess == actual else 0
```

### 8.3 Bonus weekend

Il bonus va assegnato **dopo** che TUTTE le partite della giornata sono state inserite e lo scoring è completo. Algoritmo:

```python
def assign_weekend_bonus(round_scores: list[RoundScore]) -> None:
    # Totale per classifica: sign + exact + total_goals (NON include weekend_bonus)
    totals = [(s.sign_points + s.exact_points + s.total_goals_points, s)
              for s in round_scores]
    totals.sort(key=lambda x: x[0], reverse=True)
    
    bonus_map = {1: 6, 2: 4, 3: 2}
    rank = 1
    i = 0
    while i < len(totals) and rank <= 3:
        # trova tutti i giocatori con lo stesso punteggio
        group_score = totals[i][0]
        group = [s for score, s in totals if score == group_score]
        for s in group:
            s.weekend_bonus = bonus_map[rank]  # assegnato a TUTTI nel gruppo
        rank += len(group)
        i += len(group)
```

### 8.4 Deadline predictions

Un giocatore **non può** inserire o modificare previsioni dopo la `deadline` della giornata. Il backend deve controllarlo server-side, non solo client-side.

```python
if round.deadline < datetime.utcnow():
    raise HTTPException(403, "Deadline superata")
```

### 8.5 Modifiche al Tabellone (mercato invernale)

- La stagione deve essere in stato `mercato` per permettere modifiche
- Ogni campo modificato registra una riga in `TablePredictionModification`
- La penalità (-5pt immediati) va applicata al `RoundScore` corrente del giocatore (o in un campo separato `tabellone_penalty` su `User` per la stagione)
- Il cap al 50% viene applicato **al momento dello scoring finale del tabellone**, controllando `TablePredictionModification` per vedere se un campo era stato modificato
- Arrotondamento: `floor(punti_normali * 0.5)`
- Vedi `docs/rules/REGOLAMENTO.md` §7 per la tabella completa dei cap

### 8.6 Classifiche separate (fine stagione)

Alla fine della stagione esistono 3 classifiche indipendenti:
1. **Classifica Segni**: somma di tutti i `sign_points` delle giornate
2. **Classifica Pieni+Gol**: somma di `exact_points + total_goals_points` delle giornate
3. **Classifica Tabellone**: punti dal `TablePrediction` (calcolati dall'admin)

Per ciascuna, il/i migliore/i ricevono +10pt bonus in classifica generale.

---

## 9. Convenzioni di sviluppo

### Python / Backend

```
backend/
├── app/
│   ├── main.py            # FastAPI app factory, lifespan, middleware
│   ├── config.py          # pydantic-settings, legge da .env
│   ├── database.py        # async engine, get_db dependency
│   ├── models/            # SQLAlchemy ORM models (uno per file)
│   ├── schemas/           # Pydantic v2 schemas (request/response)
│   ├── routers/           # FastAPI APIRouter (uno per dominio)
│   ├── services/          # Business logic pura (scoring, fixtures, auth)
│   ├── dependencies/      # FastAPI dependencies (get_current_user, require_admin)
│   └── migrations/        # Alembic
├── tests/
│   ├── unit/              # test unitari (scoring puro, niente DB)
│   └── integration/       # test con DB in-memory (SQLite) o test container
└── requirements.txt
```

- Usa `async def` ovunque nelle route
- I service non dipendono da FastAPI (no `Request`, no `HTTPException`) — solo da modelli e DB
- Gli `HTTPException` vanno nei router, non nei service
- Naming: `snake_case` per tutto Python
- Ogni modello ORM ha il suo file in `models/`
- Ogni router ha il suo prefix: `/auth`, `/rounds`, `/matches`, `/predictions`, `/leaderboard`, `/admin/...`

### TypeScript / Frontend

```
frontend/src/
├── api/
│   ├── client.ts          # Axios instance con interceptors (auto-refresh)
│   └── endpoints/         # funzioni tipizzate per ogni endpoint
├── store/                 # Zustand stores
├── components/            # componenti riutilizzabili
├── pages/
│   ├── Login.tsx
│   ├── player/
│   └── admin/
├── hooks/                 # custom React hooks
├── types/                 # TypeScript interfaces (rispecchiano i Pydantic schemas)
└── utils/                 # helpers puri (es. formatDate, deriveSign)
```

- `strict: true` in tsconfig
- Niente `any` — usa `unknown` e type guard se necessario
- I tipi in `types/` rispecchiano 1:1 i Pydantic response schemas del backend
- Usa React Query per tutte le chiamate API (niente `useEffect + fetch`)
- Zustand solo per stato globale client (auth user, tema)

### Convenzioni DB

- UUID come PK ovunque (`uuid_generate_v4()` o `gen_random_uuid()`)
- `TIMESTAMP WITH TIME ZONE` per tutte le date
- Migrazioni Alembic con `--autogenerate`, mai modificare migrazioni già applicate
- Nomi tabelle: `snake_case` plurali (`users`, `rounds`, `match_predictions`)
- FK naming: `{table}_{column}_fkey`

---

## 10. Ambiente di sviluppo — setup iniziale

```bash
# Copia il template env
cp .env.example .env
# Modifica DATABASE_URL, SECRET_KEY, API_FOOTBALL_KEY

# Avvia il DB
docker-compose up -d db

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

### Variabili d'ambiente richieste

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/totosport

# Auth
SECRET_KEY=<stringa random 64 chars>
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# API-Football (RapidAPI)
API_FOOTBALL_KEY=<chiave RapidAPI>
API_FOOTBALL_HOST=v3.football.api-sports.io

# Frontend (Vite)
VITE_API_BASE_URL=http://localhost:8000
```

---

## 11. Punti aperti — decidere prima di implementare

| # | Questione | Impatto |
|---|---|---|
| **A** | Un giocatore può cambiare la propria previsione dopo averla inviata, purché prima della deadline? | Schema `MatchPrediction` (upsert vs insert-only) |
| **B** | Auto-chiusura delle giornate alla deadline (job schedulato) o manuale dall'admin? | Serve APScheduler o Celery se auto |
| **C** | Auto-sync risultati da API-Football dopo il calcio d'inizio, o inserimento manuale? | Serve background task se auto |
| **D** | Limite al numero di modifiche al tabellone per giocatore? | Schema `TablePredictionModification` |
| **E** | Mobile: PWA responsive o app nativa con Capacitor? | Build pipeline frontend |
| **F** | Il totale gol giornata è inserito prima o dopo aver visto le partite in programma? (impatta UX del form) | UX del form giornata |

---

## 12. Sequenza di sviluppo raccomandata

Vedi `docs/development/README.md` per il piano dettagliato con checklist.

Ordine delle fasi:
1. Foundation (Docker, DB, Alembic, struttura cartelle)
2. Auth & Users
3. Tabellone (modello + CRUD + scoring manuale)
4. Rounds & Matches
5. Predictions & Scoring engine
6. Leaderboard & Bonus
7. API-Football integration
8. Frontend foundation (auth, routing, layout)
9. Frontend player
10. Frontend admin
11. Testing & deployment
