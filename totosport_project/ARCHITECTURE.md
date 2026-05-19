# TotoSport — Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────┐
│                    API-Football (RapidAPI)                 │
│     fixtures · results · teams · competitions             │
└──────────────────────┬───────────────────────────────────┘
                       │  HTTP (scheduled fetch / on-demand)
┌──────────────────────▼───────────────────────────────────┐
│                        Browser / Mobile                   │
│              React SPA  (Tailwind CSS · TypeScript)       │
│         /admin/*  ──────────  /player/*                   │
└──────────────────────┬────────────────────────────────────┘
                       │  HTTPS  (JWT in Authorization header)
┌──────────────────────▼───────────────────────────────────┐
│                  FastAPI Backend                           │
│   /auth   /rounds   /matches   /predictions   /leaderboard│
│   /admin/fixtures  (fetch & stage from API-Football)      │
└──────────────────────┬────────────────────────────────────┘
                       │  SQLAlchemy (async)
┌──────────────────────▼───────────────────────────────────┐
│                  PostgreSQL                               │
│  users · rounds · matches · predictions · staged_fixtures │
└───────────────────────────────────────────────────────────┘
```

---

## Data Models

### `User`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| username | VARCHAR(50) | unique |
| email | VARCHAR(100) | unique |
| password_hash | TEXT | bcrypt |
| role | ENUM | `admin` \| `player` |
| created_at | TIMESTAMP | |

### `Round` (Giornata)
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| name | VARCHAR(100) | e.g. "Giornata 34 — Serie A" |
| competition | ENUM | `serie_a` \| `serie_b` \| `champions_league` |
| season | VARCHAR(10) | e.g. "2025-26" |
| deadline | TIMESTAMP | predictions locked after this |
| status | ENUM | `draft` → `open` → `closed` → `completed` |
| created_at | TIMESTAMP | |

### `Match`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| round_id | UUID FK → Round | |
| home_team | VARCHAR(80) | |
| away_team | VARCHAR(80) | |
| kickoff | TIMESTAMP | |
| prediction_type | ENUM | `sign` \| `exact_score` |
| actual_home_goals | SMALLINT | nullable, set after match |
| actual_away_goals | SMALLINT | nullable, set after match |

### `Prediction`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| user_id | UUID FK → User | |
| match_id | UUID FK → Match | |
| predicted_sign | ENUM | `1` \| `X` \| `2` |
| predicted_home_goals | SMALLINT | nullable (only for exact_score) |
| predicted_away_goals | SMALLINT | nullable (only for exact_score) |
| points_earned | SMALLINT | computed after result, default 0 |
| submitted_at | TIMESTAMP | |

**Unique constraint:** `(user_id, match_id)` — one prediction per player per match.

### `StagedFixture`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| api_fixture_id | INTEGER | ID from API-Football, unique |
| competition | ENUM | `serie_a` \| `serie_b` \| `champions_league` |
| season | VARCHAR(10) | |
| matchday | SMALLINT | |
| home_team | VARCHAR(80) | |
| away_team | VARCHAR(80) | |
| kickoff | TIMESTAMP | |
| fetched_at | TIMESTAMP | |
| added_to_round | BOOLEAN | false until moved into a Round |

---

## API Routes

### Auth
| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/auth/login` | public | Returns access + refresh JWT |
| POST | `/auth/refresh` | public | Rotate refresh token |
| POST | `/auth/register` | admin | Create a new player account |

### Rounds
| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/rounds/` | player | List open & completed rounds |
| POST | `/rounds/` | admin | Create a new round |
| PATCH | `/rounds/{id}/status` | admin | Transition round status |

### Matches
| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/rounds/{id}/matches` | player | List matches in a round |
| POST | `/rounds/{id}/matches` | admin | Add a match to a round (from staged or manual) |
| PATCH | `/matches/{id}/result` | admin | Set final score → triggers scoring |
| DELETE | `/matches/{id}` | admin | Remove match (only if round is draft) |

### Fixtures (API-Football integration)
| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/admin/fixtures/fetch` | admin | Fetch & stage fixtures from API-Football |
| GET | `/admin/fixtures/staged` | admin | List staged fixtures available to add to a round |
| POST | `/admin/fixtures/{fixture_id}/add-to-round` | admin | Move a staged fixture into a round as a Match |

### Predictions
| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/predictions/` | player | Submit / update a prediction (before deadline) |
| GET | `/predictions/me` | player | Own predictions with points |
| GET | `/predictions/{match_id}` | admin | All predictions for a match (post-deadline) |

### Leaderboard
| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/leaderboard/` | player | All-time cumulative ranking |
| GET | `/leaderboard/rounds/{id}` | player | Per-round ranking |

---

## Scoring Logic

Computed in `backend/app/services/scoring.py` and triggered when an admin sets a match result via `PATCH /matches/{id}/result`.

```
For each Prediction linked to the Match:
  if prediction_type == "sign":
      actual_sign = derive_sign(actual_home_goals, actual_away_goals)  # 1 / X / 2
      points = 2 if predicted_sign == actual_sign else 0

  if prediction_type == "exact_score":
      if predicted_home_goals == actual_home_goals AND predicted_away_goals == actual_away_goals:
          points = 5
      elif derive_sign(predicted_*) == actual_sign:
          points = 2   # right sign, wrong exact score → partial credit (TBD)
      else:
          points = 0
```

> The partial credit rule for exact-score matches is still under discussion.

---

## Auth Flow

```
Login → POST /auth/login
  └─► {access_token (15 min), refresh_token (7 days)}

All protected calls → Authorization: Bearer <access_token>

Token expired → POST /auth/refresh → new access_token

Role guard:
  - admin routes: role == "admin" required
  - player routes: any authenticated user
```

---

## Frontend Structure

```
src/
├── context/
│   └── AuthContext.tsx       # JWT storage, user role, login/logout
├── utils/
│   └── api.ts                # Axios instance with interceptors (auto-refresh)
├── components/
│   ├── MatchCard.tsx          # Match + prediction input (sign or exact score)
│   ├── Leaderboard.tsx
│   └── RoundStatusBadge.tsx
├── pages/
│   ├── Login.tsx
│   ├── player/
│   │   ├── Home.tsx           # Open rounds
│   │   ├── RoundDetail.tsx    # Matches + prediction form
│   │   ├── MyPredictions.tsx
│   │   └── Leaderboard.tsx
│   └── admin/
│       ├── Dashboard.tsx
│       ├── RoundManager.tsx   # Create / edit rounds
│       ├── MatchManager.tsx   # Add matches, set results
│       └── PlayerList.tsx
└── hooks/
    ├── useRound.ts
    └── usePredictions.ts
```

---

## API-Football Integration

**Provider**: [API-Football](https://www.api-football.com) via RapidAPI  
**Free tier**: 100 requests/day — sufficient for weekly usage (a single fetch per matchday costs 1 request).

### Competition IDs (API-Football)
| Competition | League ID | Season format |
|---|---|---|
| Serie A | 135 | 2025 (start year) |
| Serie B | 136 | 2025 |
| UEFA Champions League | 2 | 2025 |

### Fetch flow

```
Admin → POST /admin/fixtures/fetch
          { competition: "serie_a", matchday: 34 }
              │
              ▼
  GET https://v3.football.api-sports.io/fixtures
      ?league=135&season=2025&round=Regular Season - 34
              │
              ▼
  Parse response → upsert into staged_fixtures
  (skip if api_fixture_id already exists)
              │
              ▼
  Admin sees list in GET /admin/fixtures/staged
  and picks which matches to add to the round
```

### Result auto-sync (planned)
After all matches in a round have kicked off, a background task can poll `GET /fixtures?id={api_fixture_id}` to retrieve live/final scores and fill `actual_home_goals` / `actual_away_goals` automatically, skipping manual entry. Status: **open decision** (see below).

### Implementation
- `backend/app/services/fixtures.py` — API-Football client (httpx, async)
- `backend/app/routers/fixtures.py` — admin endpoints
- API key stored in env var `API_FOOTBALL_KEY`, never committed

---

## Round Status Machine

```
draft ──► open ──► closed ──► completed
  │                              ▲
  └── admin can edit matches     │
                                 └── all results entered, scoring done
```

- `draft`: admin is building the round; players cannot see it yet
- `open`: visible to players, predictions accepted until `deadline`
- `closed`: deadline passed, results not yet entered (or partially entered)
- `completed`: all results in, points computed, round frozen

---

## Deployment (planned)

```yaml
# docker-compose.yml (skeleton)
services:
  db:        # postgres:16
  backend:   # uvicorn, port 8000
  frontend:  # nginx serving React build, port 80
```

Environment variables managed via `.env` (not committed):
- `DATABASE_URL`
- `SECRET_KEY` (JWT signing)
- `ACCESS_TOKEN_EXPIRE_MINUTES`
- `REFRESH_TOKEN_EXPIRE_DAYS`
- `API_FOOTBALL_KEY` (RapidAPI key for API-Football)
- `API_FOOTBALL_HOST` (default: `v3.football.api-sports.io`)

---

## Open Decisions

- [ ] Partial credit on exact-score matches (sign correct but score wrong)?
- [ ] Can a player change a prediction after submitting (before deadline)?
- [ ] Auto-close rounds at deadline via scheduled job, or manual admin action?
- [ ] Mobile: stay with responsive PWA, or eventually wrap with Capacitor for app stores?
- [x] Match data source: **API-Football** (RapidAPI) — free tier (100 req/day), covers Serie A, Serie B, Champions League. Admin fetches fixtures on demand; they land in `staged_fixtures` and are added to rounds manually.
- [ ] Auto-sync results: poll API-Football after match kickoff to auto-fill scores, or keep it manual?
