# Sports Analytics Portfolio

A collection of personal projects applying **Data Science, Data Engineering, and AI** to the world of sports. The goal is to build models and analysis pipelines on public datasets, with a focus on football and other sports.

> Profile: Data Engineer & Project Manager in the data space — background in IT/healthcare consulting, applied here to the sports domain.

---

## Projects

### [xGoals Project](./xgoals_project/)

**Domain:** Football | **Tech:** Python, LightGBM, StatsBomb Open Data, Jupyter

**Expected Goals (xG)** and **Expected Assists (xA)** models trained on StatsBomb open-source data, with 360° freeze frame data to measure defensive pressure and goalkeeper positioning at the moment of the shot.

Includes a complete inference pipeline (`score_match.py`) to evaluate xG shot-by-shot on any StatsBomb match, with aggregated output per team.

- Two LightGBM models: baseline (geometry) + 360-enhanced (player positioning)
- Feature engineering on event and freeze frame data
- Match-level train/validation/holdout split (no data leakage)
- CLI for inference on a single match or holdout set

---

### [Snooker Project](./snooker_project/)

**Domain:** Snooker | **Tech:** Python, pandas, Jupyter

Exploratory analysis on snooker match data. Early-stage project, focused on exploring available datasets and initial descriptive statistics.

---

### [TotoSport](./totosport_project/)

**Domain:** Football | **Tech:** Python, FastAPI, React, PostgreSQL, Docker

A weekly football prediction game webapp for friends. An admin publishes rounds of matches from **Serie A**, **Serie B**, or the **Champions League**; players submit predictions before the deadline and earn points for correct results or exact scores.

- Full-stack web app: FastAPI backend + React/TypeScript/Tailwind frontend
- JWT authentication (access + refresh tokens)
- Admin panel: create rounds, add matches, set prediction types, enter results, trigger automatic scoring
- Player panel: submit predictions, view history, live leaderboard
- Containerised with Docker and docker-compose

---

## Coming Soon

- Data engineering pipelines for real-time ingestion and transformation of sports data
- Interactive dashboards on advanced metrics

---

## Setup

```bash
# xGoals project
cd xgoals_project
pip install -r requirements.txt

# TotoSport
cd totosport_project
docker-compose up -d db
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload
# In a separate terminal:
cd frontend && npm install && npm run dev
```

StatsBomb data: [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)
