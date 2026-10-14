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
| [03](03-calcolo-tau-opp.ipynb) | Come si calcola τ_opp (Narizuka et al.) sui dati SkillCorner, passo per passo? | Modello, primo attraversamento, velocità per derivata centrale, un possesso spiegato. Coincide con `scripts/calcola_tau.py` e con `reports/tau_opp.csv`. §10: confronto con `time_to_impact`. |
| [04](04-pressing-intensity-unravel.ipynb) | La Pressing Intensity di unravelsports gira sui nostri dati, e sul portatore dice la stessa cosa di τ_opp? | Gira via kloppy. Su una partita il TTI minimo sul portatore concorda poco con τ_opp (ρ = 0,65) e con `time_to_impact` (−0,67, contro −0,86 di τ_opp). Soglia di 0,7 s; nel 16% dei possessi la squadra in possesso non coincide con quella SkillCorner. |
| [05](05-valore-concesso-verifica.ipynb) | La regola di coinvolgimento di *Blame* regge sui nostri dati come valore concesso per difensore? | Geometria sì: 97–99% dei difensori coinvolti osservati, coinvolgimento grezzo stabile fra metà delle partite (r = 0,69). Classifica per valore no (r = 0,03), per le code pesanti del valore e non per l'EPV chiuso: con Δ distanza r = 0,36. |
| [06](06-esce-o-tiene-verifica.ipynb) | Il gruppo di trattamento di *esce o tiene* è stabile, e il gruppo "tiene" si costruisce dal tracking? | Sì come insieme sotto soglia al rilascio (S90 = 2,51 s): azzerando le velocità resta l'87% delle coppie, il difensore più rapido invece cambia nel 22%. "Tiene": 25.085 coppie, `other` ne copre il 4%; il 71% è copertura con un compagno che esce. Il 62% delle uscite parte durante il volo del pallone. |

## Come eseguirli

```bash
source .venv/bin/activate          # dalla cartella analytics_cup_project/
jupyter lab
```

I notebook aggiungono `..` a `sys.path` per importare `lib/`.
