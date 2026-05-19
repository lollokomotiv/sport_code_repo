# TotoSport

A weekly football prediction game for friends. Every week an admin publishes a set of matches from **Serie A**, **Serie B**, or the **Champions League**, and players submit their predictions before the deadline. Points are awarded based on prediction accuracy.

---

## How it works

1. **Admin** creates a weekly round and selects matches from the chosen competition.
2. For each match the admin decides the prediction type:
   - **Sign** — predict the outcome: home win (1), draw (X), or away win (2).
   - **Exact score** — predict the precise final scoreline (awarded on selected matches only).
3. Players submit predictions before the round deadline.
4. Once matches are played the admin enters the results and the system computes points automatically.
5. A live leaderboard tracks cumulative scores across all rounds.

### Scoring

| Prediction type | Correct | Wrong |
|---|---|---|
| Sign (1/X/2) | 2 pts | 0 pts |
| Exact score | 5 pts | 0 pts |

---

## Features

### Player
- Browse open rounds and view selected matches
- Submit predictions (sign or exact score) before the deadline
- View personal prediction history and points per round
- Live leaderboard

### Admin
- Create and manage rounds (Serie A / Serie B / Champions League)
- Add matches and choose prediction type per match
- Set round deadline and open/close submissions
- Enter final results and trigger automatic scoring
- Manage registered players

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11 · FastAPI · SQLAlchemy |
| Database | PostgreSQL |
| Frontend | React 18 · TypeScript · Tailwind CSS |
| Auth | JWT (access + refresh tokens) |
| Deployment | Docker · docker-compose |

---

## Project Structure

```
totosport_project/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI entry point
│   │   ├── config.py        # Settings (env vars)
│   │   ├── database.py      # SQLAlchemy session
│   │   ├── models/          # ORM models
│   │   ├── routers/         # API route handlers
│   │   ├── schemas/         # Pydantic schemas
│   │   └── services/        # Business logic (auth, scoring)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/      # Shared UI components
│   │   ├── pages/
│   │   │   ├── admin/       # Admin-only pages
│   │   │   └── player/      # Player pages
│   │   ├── hooks/           # Custom React hooks
│   │   ├── context/         # Auth context
│   │   └── utils/           # API client, helpers
│   └── public/
├── docker-compose.yml
├── README.md
└── ARCHITECTURE.md
```

---

## Quick Start

```bash
cd totosport_project

# Start backend + database
docker-compose up -d db
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload

# Start frontend
cd frontend && npm install && npm run dev
```

> Full Docker setup (single command) coming once the stack is finalised.
