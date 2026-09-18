# `{id}_phases_of_play.csv` — le fasi di gioco

**Cos'è:** ~450 righe × 44 colonne per partita (~116 KB). Segmenta la partita in
fasi: intervalli di frame con un'identità tattica.

È il file più piccolo e il più sottovalutato: dà **contesto** a tutto il resto. Un
passaggio in `build_up` e lo stesso passaggio in `finish` non sono la stessa cosa,
e questo file è quello che permette di distinguerli.

## Una riga descrive entrambe le squadre

Non c'è una riga per squadra: ogni riga copre un intervallo di frame e dice
contemporaneamente cosa fa chi ha la palla e cosa fa chi non ce l'ha.

```
frame_start  frame_end  duration  team_in_possession  in_possession  out_of_possession
         28         89       6.1           Newcastle         create       medium_block
         89        124       3.5           Newcastle         direct   defending_direct
        189        232       4.3         Auckland FC       build_up         high_block
```

## Le due tassonomie sono speculari

A ogni fase offensiva corrisponde **esattamente una** fase difensiva. Non sono due
etichette indipendenti: sono due nomi dello stesso momento visto dalle due parti.

| in possesso | id | fuori possesso | id | fasi (es. `1886347`) |
|---|---|---|---|---|
| `build_up` | 0 | `high_block` | 10 | 66 |
| `create` | 1 | `medium_block` | 9 | 156 |
| `finish` | 2 | `low_block` | 8 | 94 |
| `quick_break` | 3 | `defending_quick_break` | 12 | 4 |
| `transition` | 4 | `defending_transition` | 11 | 1 |
| `chaotic` | 5 | `chaotic` | 5 | 82 |
| `direct` | 6 | `defending_direct` | 15 | 25 |
| `set_play` | 7 | `defending_set_play` | 13 | 21 |

Conseguenza pratica: **non serve raggruppare per squadra** per sapere cosa faceva
l'avversario. Una sola colonna basta, l'altra è ridondante per costruzione.

`chaotic` è la categoria che raccoglie i momenti senza struttura riconoscibile —
82 fasi su 449, non è un residuo trascurabile.

## Colonne

| Famiglia | Colonne |
|---|---|
| **Tempo** | `frame_start`, `frame_end`, `time_start`, `time_end`, `minute_start`, `second_start`, `duration`, `period`, `index` |
| **Squadre** | `team_in_possession_id`, `team_in_possession_shortname`, `attacking_side_id`, `attacking_side` |
| **Fase** | `team_in_possession_phase_type(_id)`, `team_out_of_possession_phase_type(_id)` |
| **Esito** | `n_player_possessions_in_phase`, `team_possession_loss_in_phase`, `team_possession_lead_to_shot`, `team_possession_lead_to_goal` |
| **Posizione** | `x_start`/`y_start`, `x_end`/`y_end`, `channel_*`, `third_*`, `penalty_area_*` |
| **Forma delle squadre** | `team_in_possession_width_{start,end}`, `team_in_possession_length_{start,end}`, e le stesse per `team_out_of_possession` |

## La compattezza è già calcolata

`width` e `length` di **entrambe** le squadre, a inizio e fine fase. Sulla partita
`1886347` la larghezza della squadra in possesso varia da 12,6 a 66,4 m.

È una metrica di reparto disponibile senza doverla derivare dal tracking — e,
soprattutto, senza esporsi al problema dell'estrapolazione, visto che è SkillCorner
a calcolarla. (Con l'avvertenza di sempre: è comunque il loro calcolo, non il tuo.)

## Copertura

Durata mediana di una fase: **5,2 s**; massima 46,9 s. Il totale coperto dalle fasi
è ~53 minuti, contro i ~72 minuti di gioco osservabile nel tracking.

Le fasi non coprono tutta la partita: i momenti di gioco fermo non ne hanno una.
Non usare la somma delle durate come denominatore di una percentuale "sulla
partita".

## Vocabolario delle zone

Condiviso con `dynamic_events`:

- **`channel`**: `wide_left`, `half_space_left`, `center`, `half_space_right`, `wide_right`
- **`third`**: `defensive_third`, `middle_third`, `attacking_third`

## Come leggerlo

```python
from lib import data

pop = data.load_phases_of_play(1886347)

# Tutte le fasi di rifinitura contro blocco basso
finish = pop[pop.team_in_possession_phase_type == "finish"]

# Da qui all'animazione: frame_start / frame_end filtrano il tracking
fase = finish.iloc[0]
frames = range(int(fase.frame_start), int(fase.frame_end) + 1)
```

Il collegamento con `dynamic_events` passa da `phase_index`, che in quel file
compare come colonna su ogni evento.

## Riferimenti

- Tutorial ufficiale: `notebooks/tutorials/02_.../Part2_Data_Aggregating_Phases_of_Play_Tutorial.ipynb`
- `src/features/PhasesOfPlayAggregator.py` nel repo dei dati
- [Glossari SkillCorner](https://skillcorner.crunch.help/en/glossaries)
