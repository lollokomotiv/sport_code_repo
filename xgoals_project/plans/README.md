# Plans

Piani di lavoro del progetto xGoals. Un file per filone, numerato per ordine di priorità.

## Convenzione

- `NN-nome-filone.md` — un piano per filone di lavoro
- Ogni piano ha: **Obiettivo**, **Stato**, **Step**, **Fatto / Aperto**
- Gli step sono checkbox: `- [ ]` da fare, `- [x]` fatto
- Quando un filone è chiuso, si sposta in `plans/done/`

## Indice

| Piano | Stato | Sintesi |
|---|---|---|
| [00 — Modello xG](00-xg-calibrazione.md) | 🟡 Calibrato, poco discriminante | Centrato sul benchmark, ma AUC 0.79 su test e valore dei dati 360 non dimostrato. Serve più dati |
| [01 — Completamento xA](01-xa-completamento.md) | 🔴 Prioritario | Bug nella feature engineering dei tiri (angolo invertito): da chiudere prima delle viz |
| [02 — Metriche avanzate](02-metriche-avanzate.md) | ⚪ Da iniziare | Nuovi modelli su dati esistenti (possession value, ecc.) |
| [03 — Webapp](03-webapp.md) | ⚪ Da iniziare | Interfaccia per generare le visualizzazioni da dati StatsBomb |
