# Letteratura

Schede dei paper in [`docs/`](../../docs/), una per lavoro. Prodotte con la skill
`scheda-paper`.

**Non sono riassunti.** Ogni scheda risponde alla domanda che serve qui: *questo
metodo sopravvive a 20 partite di broadcast tracking con il 41% delle posizioni
estrapolate?* Il verdetto viene prima della spiegazione del metodo.

## Schede

Ordinate per verdetto, non alfabeticamente.

| Scheda | Cosa misura | Dati richiesti | Verdetto |
|---|---|---|---|
| [pressione-tempo-arrivo](pressione-tempo-arrivo.md) | pressione sul portatore = tempo minimo di arrivo dell'avversario più vicino | tracking + eventi, 25 fps, 306 partite | **Replicabile con riserve** — dipende dai giocatori *vicini* alla palla, dove i nostri dati sono migliori |
| [pressing-exPressV2](pressing-exPressV2.md) | probabilità che un pressing recuperi palla, e merito individuale | tracking + eventi, 25 Hz, 36 partite | **Adattabile** — impostazione e etichette trasferibili, la GNN no: batte la regressione logistica di 0,013 di AUC |

## Da leggere

| File | pp. | Priorità |
|---|---|---|
| `Pressing_Intensity-An_Intuitive_Measure_for_Pressing_in_Soccer.pdf` | 8 | **prossimo** — citato da entrambe le schede; exPressV2 lo usa per definire cosa conta come pressing. Implementato in `unravelsports` |
| `exPressV2_Contextual_Evaluation_Pressing.pdf` ✅ | 13 | fatto — aggiunto dopo, non era fra i paper originali |
| `Blame_is_easier_than_praise.pdf` | 27 | alta — valutazione difensiva senza palla |
| `Quantifying...` ✅ | 11 | fatto |
| `Better_Prevent_than_Tackle.pdf` | 28 | |
| `Evaluation_of_soccer_team_defense_based_on_prediction_models...pdf` | 15 | |
| `Prediction-based_evaluation_of_back-four_defense_with_spatial_control.pdf` | 22 | |
| `Off_Ball_Defensive_Role_and_Performance_Evaluation_in_Football.pdf` | 40 | |
| `EFPI_using_Template_Matching_and_Linear_Assignment.pdf` | 11 | template matching — famiglia interpretabile |
| `Revisiting_Expected_Possession_Value_in_Football.pdf` | 16 | |
| `defensivepressurearticle_researchgate.pdf` | 19 | |
| `NFL_Ghosts.pdf` | 34 | bassa — ghosting, poco praticabile con 20 partite |
| `Spatio-Temporal_Analysis_of_Team_Sports.pdf` | 42 | survey, da consultare non da leggere in blocco |

**Versione superata, rimossa.** `docs/` conteneva lo stesso lavoro di Bekkers in
due versioni: la v1 di gennaio 2025 e la v2 arXiv del 30 giugno 2025. Tenuta la
v2 (ha DOI, arXiv ID e licenza nei metadati), eliminata la v1. Se dovesse
servire: [arXiv:2501.04712v1](https://arxiv.org/abs/2501.04712v1).

**I PDF non sono versionati.** `docs/` è in `.gitignore`: sono ~24 MB di paper
sotto copyright dell'editore. Quello che vale sono le schede in questa cartella.

## Direzioni di ricerca

`docs/Research_Directions_AnalyticsCup.pages` elenca sei ambiti poco esplorati.
Il testo è estraibile con lo script della skill; in sintesi:

1. **Processo decisionale del difensore** — quando esce e quando tiene la posizione (via On-Ball Engagements)
2. **Coordinazione temporale** — il lag di reazione dei singoli difensori ai trigger
3. **Passaggi di consegne in marcatura** nel gioco aperto (l'HMM di Groom copre i corner)
4. **Trigger e trappole di pressing** — cosa innesca il pressing, non quanto è intenso
5. **Qualità dei dati broadcast** — quanto le metriche difensive sono robuste all'estrapolazione
6. **Adattamento all'avversario** — la forma difensiva in funzione della struttura di costruzione avversaria

> **Correzione al documento:** dice «circa 10 partite, controlla il numero
> esatto». Sono **20**: il dataset è cresciuto dopo l'edizione 2026. Raddoppia il
> campione rispetto al repo citato. Sul metodo la conclusione va sfumata: il
> ghosting resta poco praticabile, una GNN si addestra (exPressV2, 36 partite)
> ma guadagna poco su un modello semplice.
