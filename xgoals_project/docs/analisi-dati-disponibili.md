# Analisi: cosa contengono i dati e cosa non stiamo usando

**Data:** 2026-08-20
**Metodo:** inventario empirico su 30 match con dati 360 (eventi e freeze frame) e su un campione casuale di 40 match per le statistiche sui tiri.

---

## Inventario dei campi inutilizzati

| Campo / evento | Disponibilità | Stato |
|---|---|---|
| `shot.end_location` (3D: x, y, **z**) | 100% dei tiri; z presente sul 100% dei tiri in porta | mai usato |
| `Carry` (conduzioni) | 24.431 eventi in 30 match | mai usato |
| `Pressure` | 9.426 eventi | mai usato |
| `pass.recipient` | 96,2% dei passaggi | mai usato |
| `pass.length`, `pass.angle` | 100% | `length` ricalcolata a mano, `angle` mai usata |
| `possession` (id della catena) | 100%, mediana 14 eventi per possesso | mai usato |
| `shot.aerial_won` | 10,2% dei tiri | mai usato |
| `shot.deflected` / `open_goal` / `redirect` | 1,4% / 0,9% / 0,1% | mai usato |
| `duration` dell'evento | 100% | mai usato |
| `Goal Keeper` (eventi tipizzati) | 828 in 30 match | mai usato |
| `position` nel freeze frame del tiro | 100% | mai usato |

### Volumi di riferimento

| Tipo di evento (30 match) | Conteggio |
|---|---|
| Pass | 30.646 |
| Ball Receipt | 29.494 |
| Carry | 24.431 |
| Pressure | 9.426 |
| Duel | 1.906 |
| Dribble | 847 |
| Goal Keeper | 828 |
| Shot | 705 |

### Esiti dei tiri (campione di 40 match, rigori esclusi)

| Esito | Quota |
|---|---|
| Off T | 32,4% |
| Saved | 25,6% |
| Blocked | 22,5% |
| Goal | 9,7% |
| Wayward | 6,7% |
| Post | 2,1% |
| altri | <1% |

**In porta** (Goal, Saved, Saved to Post): **35,5%** dei tiri. Stima su tutti i 3.464 match: **~29.900 tiri in porta**.
Altezza del pallone in porta (`z`): min 0,00 — mediana 0,90 — max 2,60 m.

---

## Le quattro direzioni con più valore

### 1. Post-shot xG — priorità

L'xG stima la qualità della *situazione*. Il post-shot xG stima P(gol) *dato dove il pallone è finito*, usando `end_location` in tre dimensioni.

**Valore calcistico** — due sottrazioni, due metriche standard del settore:
- `PSxG − xG` = qualità di **finalizzazione** dell'attaccante
- `PSxG − gol subiti` = rendimento del **portiere** (la metrica con cui si valutano oggi)

**Valore statistico** — il problema è molto meglio condizionato di quello dell'xG:

| | Righe | Eventi (gol) |
|---|---|---|
| Modello xG attuale (training) | 5.065 | ~486 |
| PSxG su tutti i match | **~29.900** | **~8.100** |

Non richiede dati 360, quindi usa tutti i 3.464 match: **16x gli eventi** del modello xG.

**Vincolo:** `end_location` è informazione posteriore al tiro. Non deve mai entrare come feature nel modello xG — sarebbe leakage. Sono due modelli distinti con scopi distinti.

**Nicchia che apre:** analisi dei portieri, area sottoservita in cui i club assumono.

### 2. Catene di possesso e conduzioni

`possession` identifica la catena (mediana 14 eventi); 24.431 conduzioni sono inutilizzate. Insieme danno il contesto oggi assente: **come nasce l'occasione**.

Feature costruibili per xG e xA: passaggi nella catena, durata, distanza progredita, recupero alto, tiro dopo conduzione. È anche il prerequisito per il possession value / EPV del [piano 02](../plans/02-metriche-avanzate.md).

Costo medio: richiede di ricostruire le sequenze — lavoro di data engineering.

### 3. Reti di passaggio

`pass.recipient` (96,2%) permette il grafo dei passaggi: chi serve chi, quante volte, in che zona.
Calcisticamente: struttura di squadra, giocatori-perno, lati preferiti. Statisticamente: metriche di rete su grafo pesato.
Costo basso, ed è la cosa più immediatamente **guardabile** dell'elenco.

### 4. Metriche di pressing

9.426 eventi `Pressure` permettono PPDA, altezza del pressing, intensità per zona. Descrittive, nessun modello da validare. Costo bassissimo.

---

## Vittorie minori, quasi gratis

- `pass.length` e `pass.angle` sono già forniti da StatsBomb: feature gratuite per il P_shot
- `shot.aerial_won` (10,2%) distingue i colpi di testa in elevazione contestata
- `position` nel freeze frame dà il **ruolo** del difensore più vicino
- `pass.type` (16,3%) marca corner, punizioni e rimesse — contesto da palla inattiva oggi assente

## Cosa lasciare perdere

- **Pitch control** e metriche che richiedono traiettorie: i freeze frame sono istantanee, non tracking continuo
- **Valutazioni per singolo giocatore**: con 326 match su competizioni diverse i minuti per giocatore sono troppo pochi; servirebbe come minimo uno shrinkage bayesiano e resterebbe debole
