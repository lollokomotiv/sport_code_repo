# Esplorazioni

Notebook esplorativi, numerati, **una domanda ciascuno**. Possono essere sporchi —
è il loro scopo. Ma ognuno deve avere in cima una cella markdown che dice quale
domanda sta ponendo, perché fra due mesi non te lo ricordi.

Quello che matura qui viene poi portato in `submission/src/`, riscrivendo il
caricamento dati (vedi [`../plans/02-costruire-la-submission.md`](../plans/02-costruire-la-submission.md)).

| # | Domanda | Risposta breve |
|---|---|---|
| [00](00-quanto-e-osservato.ipynb) | Quanto di questo tracking è davvero osservato e non estrapolato? | **59%** su una partita intera. 87% vicino alla palla, 18% oltre i 40 m, 15% per il portiere. |
| [01](01-struttura-dei-file-match.ipynb) | Cosa contengono i quattro file di una partita e come si collegano? | `frame` è la chiave universale. Documentato file per file in [`notes/dati/`](../notes/dati/). |
| [02](02-body-pose.ipynb) | Quanto del body pose è davvero utilizzabile? | **45,7%** dei giocatori in campo (non 31%: conta il denominatore). 77% vicino alla palla, 10% oltre i 40 m. Busto preciso a 6 cm, estremità a 21. |

## Come eseguirli

```bash
source ~/Documents/Projects/sport_venvs/analytics_cup_project/bin/activate
jupyter lab
```

I notebook aggiungono `..` a `sys.path` per importare `lib/`.
