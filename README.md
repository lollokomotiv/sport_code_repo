# Sports Analytics Portfolio

Raccolta di progetti personali che applicano **Data Science, Data Engineering e AI** al mondo dello sport. L'obiettivo è costruire modelli e pipeline di analisi su dataset pubblici, con focus su football e altri sport.

> Profilo: Data Engineer & Project Manager nel mondo dei dati — background in consulenza IT/healthcare, applicato qui al dominio sportivo.

---

## Progetti

### [xGoals Project](./xgoals_project/)

**Dominio:** Football | **Tech:** Python, LightGBM, StatsBomb Open Data, Jupyter

Modelli di **Expected Goals (xG)** e **Expected Assists (xA)** addestrati sui dati open-source di StatsBomb, con dati di freeze frame a 360° per misurare la pressione difensiva e il posizionamento del portiere al momento del tiro.

Include una pipeline di inferenza completa (`score_match.py`) per valutare xG tiro per tiro su qualsiasi partita StatsBomb, con output aggregato per squadra.

- Due modelli LightGBM: baseline (geometria) + 360-enhanced (posizionamento giocatori)
- Feature engineering su dati evento e freeze frame
- Split train/validation/holdout a livello di partita (no data leakage)
- CLI per inferenza su singola partita o holdout set

---

### [Snooker Project](./snooker_project/)

**Dominio:** Snooker | **Tech:** Python, pandas, Jupyter

Analisi esplorativa su dati di partite di snooker. Progetto in fase iniziale, focalizzato sull'esplorazione dei dataset disponibili e sulle prime analisi statistiche descrittive.

---

## In arrivo

- Modelli predittivi su risultati di partite (football/altri sport)
- Pipeline di data engineering per ingestione e trasformazione di dati sportivi in real-time
- Dashboard interattive su metriche avanzate

---

## Setup

```bash
# Per il progetto xGoals
cd xgoals_project
pip install -r requirements.txt
```

Dati StatsBomb: [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)
