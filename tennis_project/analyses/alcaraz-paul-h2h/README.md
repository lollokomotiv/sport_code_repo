# Alcaraz — Paul: il confronto diretto nei dati

## Domanda

**Dove si è deciso il confronto tra Carlos Alcaraz e Tommy Paul**: nei momenti
chiave o nei punti ordinari?

## Dati

Due livelli, due fonti, nessun dato di mercato.

**Livello 1 — tutti gli 8 incontri** (TennisMyLife, statistiche ufficiali):
servizio, risposta, palle break. 1.559 punti.

**Livello 2 — i 5 incontri annotati colpo per colpo** (Match Charting Project):
lunghezza degli scambi, dritto e rovescio, gioco a rete. 962 punti.

| Data | Torneo | Sup. | Vincitore | Annotato |
|---|---|---|---|---|
| 2022-08-08 | Canada Masters R32 | Hard | Paul | ✅ |
| 2023-03-20 | Miami Masters R16 | Hard | Alcaraz | ✅ |
| 2023-08-07 | Canada Masters QF | Hard | Paul | ✅ |
| 2023-08-14 | Cincinnati Masters R16 | Hard | Alcaraz | ❌ |
| 2024-07-01 | Wimbledon QF | Grass | Alcaraz | ✅ |
| 2024-07-29 | Paris Olympics QF | Clay | Alcaraz | ❌ |
| 2025-05-26 | Roland Garros QF | Clay | Alcaraz | ✅ |
| 2026-01-25 | Australian Open R16 | Hard | Alcaraz | ❌ |

L'elenco non è scritto a mano: lo script lo ricava da TennisMyLife e marca da
solo quali incontri il MCP ha annotato, collegando le due fonti per coppia di
giocatori, turno e prossimità di data.

## Metodo

Le statistiche ufficiali passano da `tml_to_long()`, che le porta allo stesso
schema delle righe `Total` del MCP: così `add_serve_metrics()` calcola le stesse
percentuali su entrambe le fonti, e i due livelli sono confrontabili.

Per il dettaglio colpo per colpo si aggregano i file MCP filtrando sempre la
colonna `row` al livello giusto (`BP`/`BPO` per le palle break, `1-3`…`10` per
la lunghezza degli scambi, `F`/`B` per i colpi a rimbalzo).

## Risultato

**Sul confronto completo, Paul gioca le palle break meglio di Alcaraz — e perde
6-2.**

| | Alcaraz | Paul |
|---|---|---|
| Palle break salvate | 66,0% (35/53) | **69,9% (72/103)** |
| Palle break convertite | 30,1% (31/103) | **34,0% (18/53)** |

In entrambe le direzioni è Paul ad avere la percentuale migliore. Quello che
cambia è **quante volte ciascuno ci arriva**: Alcaraz si è procurato 103 palle
break, Paul 53, in un numero di game al servizio praticamente identico
(113 contro 112). Il divario non nasce nei momenti importanti — nasce prima.

A monte, i tre scarti che generano quel volume:

- **servizio**: Alcaraz vince il 67,3% dei punti al servizio, Paul il 59,5%;
- **risposta**: 40,5% contro 32,7%;
- **il dritto** (sui 5 match annotati): a parità di colpi giocati, 698 contro
  721, Alcaraz produce **10,5 vincenti ogni 100 dritti, Paul 3,3**. Sui rovesci
  sono identici (3,0 contro 3,1). L'intero scarto tecnico sta in un colpo solo.

E Alcaraz chiude prima: vince il 35,2% dei propri punti al servizio entro tre
colpi, Paul il 25,6%.

## Limiti

- **Il sottoinsieme annotato non è rappresentativo**: 3-2 Alcaraz contro il 6-2
  reale. Entrambe le vittorie di Paul sono annotate, solo 3 delle 6 di Alcaraz.
  Le sezioni del livello 2 vanno lette sapendo che il campione pende dalla parte
  di Paul — quindi **sottostimano** il divario. Lo script stampa lo scarto
  prima di ogni media.
- **5 match, 962 punti** al livello 2: le differenze per singolo match sono
  rumore. Roland Garros 2025 (dominance ratio 3,88) è un valore anomalo che
  sposta ogni aggregato.
- **Superfici e formati mescolati**: 5 hard, 2 terra, 1 erba; 4 al meglio dei 3
  e 4 al meglio dei 5. Le due vittorie di Paul sono entrambe hard, al meglio dei
  3, in Canada.
- Il MCP è annotato a mano: vincente o gratuito è un giudizio umano. Il
  controllo incrociato con le statistiche ufficiali sugli stessi match dà però
  scarti massimi di 1 unità (vedi `docs/fonti-dati.md`).
- Niente di tutto questo è predittivo. Descrive partite giocate.

## Come si riproduce

```bash
python3 -m lib.download tml --tour atp
python3 -m lib.download mcp --gender m
python3 analyses/alcaraz-paul-h2h/run.py
```
