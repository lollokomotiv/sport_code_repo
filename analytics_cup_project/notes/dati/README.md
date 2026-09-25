# I dati SkillCorner, file per file

Documentazione del dataset: cosa contiene ogni file, con che granularità, e cosa
va saputo prima di usarlo.

Ricavata leggendo i dati, non la documentazione ufficiale — il codice che ha
prodotto queste conclusioni sta in
[`explorations/01-struttura-dei-file-match.ipynb`](../../explorations/01-struttura-dei-file-match.ipynb).

## Dentro ogni partita

`data/matches/{match_id}/` contiene quattro file.

| File | Righe | Granularità | Documento |
|---|---|---|---|
| `{id}_match.json` | — | partita | [match-json.md](match-json.md) |
| `{id}_tracking_extrapolated.jsonl` | ~59.000 | **frame** (10 fps) | [tracking-extrapolated.md](tracking-extrapolated.md) |
| `{id}_dynamic_events.csv` | ~5.100 × 322 col. | evento | [dynamic-events.md](dynamic-events.md) |
| `{id}_phases_of_play.csv` | ~450 × 44 col. | fase di gioco | [phases-of-play.md](phases-of-play.md) |

I numeri si riferiscono alla partita `1886347` (Auckland FC – Newcastle) e sono
rappresentativi.

## Come si collegano

**`frame` è la chiave universale.** Tutto si aggancia lì.

```
match.json ──player_id──> tracking.jsonl <──frame── dynamic_events.csv
     │                          ▲                         │
     └──────player_id, team_id──┘                         │ phase_index
                                │                         ▼
                                └─────frame───── phases_of_play.csv
```

| Da | A | Chiave |
|---|---|---|
| dynamic_events | tracking | `frame_start` / `frame_end` → `frame` |
| phases_of_play | tracking | `frame_start` / `frame_end` → `frame` |
| dynamic_events | phases_of_play | `phase_index` |
| tracking | match.json | `player_id` → `players[].id` |

È anche il motivo per cui animare un'azione è immediato: prendi `frame_start` e
`frame_end` da un evento o da una fase, e filtri il tracking su quell'intervallo.

## Fuori dalle partite

| Percorso | Contenuto |
|---|---|
| `data/matches.json` | indice delle 20 partite: id, data, squadre |
| `data/aggregates/aus1league_physicalaggregates_20242025.csv` | metriche fisiche per giocatore-stagione (406 righe × 65 col.), solo prestazioni sopra i 60 minuti |
| `data/aggregates/aus1league_obraggregates_20242025.csv` | off-ball runs aggregate |
| `data/aggregates/aus1league_passingaggregates_20242025.csv` | passaggi aggregati |
| `data/bodypose/` | 3D body pose, 29 giunti, 25 fps — **solo 2 partite** → [bodypose.md](bodypose.md) |

Gli aggregati stagionali e `matches.json` **non sono ancora documentati**.

## Le tre cose da sapere prima di usare qualsiasi cosa

**1. È tracking da broadcast.** Le posizioni fuori inquadratura sono stimate, non
osservate: complessivamente **59% osservato**, 87% vicino alla palla, 18% oltre i
40 m, 15% per il portiere. Vedi
[`explorations/00-quanto-e-osservato.ipynb`](../../explorations/00-quanto-e-osservato.ipynb).

**2. Gli eventi sono derivati, non annotati.** `dynamic_events.csv` è prodotto dai
modelli di SkillCorner, e contiene anche loro stime proprietarie (`xthreat`,
`xpass_completion`, EPV, pressione, spazio). Costruirci sopra è legittimo, farlo
senza dichiararlo no.

**3. Le durate nominali non sono il gioco osservabile.** 98 minuti dichiarati, ~72
minuti con giocatori nel tracking, ~53 minuti coperti dalle fasi di gioco. Tre
denominatori diversi: sbagliarli significa sbagliare ogni percentuale.

## Glossari ufficiali

- [Glossari SkillCorner](https://skillcorner.crunch.help/en/glossaries)
- [Physical Data Glossary](https://skillcorner.crunch.help/en/glossaries/physical-data-glossary)
