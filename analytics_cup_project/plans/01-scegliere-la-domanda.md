# 01 — Scegliere track e domanda

**Bloccato da [00](00-capire-i-dati.md).** Scegliere la domanda prima di sapere
cosa i dati reggono è il modo più comune di sprecare mesi.

## La scelta del track

| | Research | Analyst |
|---|---|---|
| Premia | metodo, domanda ben posta | tool usabile da uno staff tecnico |
| Deliverable 2026 | ~10 file, `submission.ipynb` + `src/` | pipeline ELT + app (43 file) |
| Rischio | domanda già esplorata da altri | tanta ingegneria, poca idea |
| Copre | ciò che `xgoals_project/` già mostra | il lato Data Engineering, meno rappresentato |

Nota per la 2027: è annunciata con *"due sport, due competizioni regionali, due
finali"*. La struttura dei track potrebbe cambiare — verificare all'annuncio.

## Il vincolo che stringe di più

**Max 2 figure** (o 2 tabelle, o 1+1) e **500 parole** di abstract.

Questo non è un vincolo di impaginazione, è un vincolo sul *tipo* di risultato.
Una domanda che ha bisogno di sei pannelli per essere capita non è presentabile.
Va usato come filtro fin dall'inizio: **se non riesci a immaginare la figura
finale, la domanda non è pronta.**

## Criteri per una buona domanda

1. **Regge su 20 partite?** Se la risposta richiede potenza statistica che non
   hai, la domanda è sbagliata a prescindere da quanto è interessante.
2. **Sfrutta il tracking?** Se si risponde con dati evento, il tracking non
   serviva — e i giudici lo noteranno.
3. **Sopravvive all'estrapolazione?** Vedi 00. Una domanda che dipende da
   traiettorie precise lontano dalla palla poggia su dati inventati.
4. **Sta in una figura?**
5. **È già stata presentata?** Leggere gli abstract dei finalisti 2026 e delle
   submission dell'edizione 1.0 prima di innamorarsi di un'idea.

## Cosa è già stato fatto (edizione 2026)

Da non rifare, e da usare come calibrazione del livello atteso:

- **Simulated annealing per il posizionamento difensivo** (Amar Shah, vincitore) —
  ottimizzazione di superfici arbitrarie senza dati di training
- **Positional scouting** (Hadi Sotudeh) — griglia 5×5 relativa ai compagni,
  position map per fase, nearest neighbour con distanza di Hellinger
- **Off-ball run decision making** (Zach Cochran)
- **Worst-case scenario running demands** (Emaly Vatne)
- **Dynamic Skills Finder** (Oscar Bartolome Pato, Analyst)
- **SkPy Analytics Platform** (Antoine Verdon, Analyst)

## Piste da valutare

Da riempire dopo il 00. Per ognuna: quale dato serve, quale limite la minaccia,
che figura produce.
