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

## Coming Soon

- Predictive models for match outcomes (football/other sports)
- Data engineering pipelines for real-time ingestion and transformation of sports data
- Interactive dashboards on advanced metrics

---

## Setup

```bash
# For the xGoals project
cd xgoals_project
pip install -r requirements.txt
```

StatsBomb data: [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)
