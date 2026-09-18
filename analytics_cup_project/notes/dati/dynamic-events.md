# `{id}_dynamic_events.csv` — gli eventi derivati

**Cos'è:** ~5.100 righe × **322 colonne** per partita (~4,7 MB). È il file più
denso di informazione del dataset.

**Non sono eventi annotati a mano.** A differenza di StatsBomb o Opta, qui non c'è
un operatore che guarda la partita e tagga: gli eventi sono **derivati dal tracking
dai modelli di SkillCorner**. Questo cambia due cose — la copertura è completa e
coerente, ma la definizione di ogni evento è quella del loro modello, non uno
standard di settore.

## I quattro tipi

| `event_type` | righe | `event_subtype` |
|---|---|---|
| `passing_option` | 2.573 | — |
| `player_possession` | 990 | — |
| `on_ball_engagement` | 957 | `pressing`, `pressure`, `counter_press`, `recovery_press`, `other` |
| `off_ball_run` | 595 | `behind`, `coming_short`, `cross_receiver`, `dropping_off`, `overlap`, `underlap`, `pulling_wide`, `pulling_half_space`, `run_ahead_of_the_ball`, `support` |

Cosa significano:

- **`player_possession`** — un giocatore ha la palla. È l'unità base: conduzione,
  tocco, passaggio.
- **`passing_option`** — un compagno *era* un'opzione di passaggio in quel momento,
  che il passaggio sia stato tentato o no. È la categoria più numerosa e la più
  caratteristica di SkillCorner: descrive **ciò che sarebbe stato possibile**, non
  solo ciò che è successo.
- **`off_ball_run`** — una corsa senza palla, classificata per tipo.
- **`on_ball_engagement`** — un difensore che ingaggia il portatore.

Durata mediana: da 0,8 s (`passing_option`) a 2,2 s (`off_ball_run`).

## Le 322 colonne non valgono per tutti

Ogni tipo popola un sottoinsieme:

| tipo | colonne popolate |
|---|---|
| `player_possession` | 205 |
| `passing_option` | 143 |
| `on_ball_engagement` | 131 |
| `off_ball_run` | 115 |

**Filtra per `event_type` prima di guardare le colonne**, altrimenti lavori su
tabelle per l'80% vuote.

## Famiglie di colonne

**Identificazione e tempo** — `event_id`, `match_id`, `frame_start`, `frame_end`,
`time_start`, `time_end`, `duration`, `period`.

**Attori** — `player_id`/`player_name`/`player_position`,
`player_in_possession_*`, `player_targeted_*`, `team_id`, `team_shortname`.

**Posizione** — `x_start`/`y_start`, `x_end`/`y_end`, più le versioni discretizzate
`channel_*` e `third_*` (vedi il vocabolario in basso) e `penalty_area_*`.

**Contesto di partita** — `game_state`, `team_score`, `opponent_team_score`,
`phase_index`, `team_in_possession_phase_type`, `game_interruption_before/after`.

**Passaggi** — `pass_distance`, `pass_angle`, `pass_direction`, `pass_range`,
`pass_outcome`, `high_pass`, `one_touch`, `quick_pass`, `carry`, `is_header`,
`lead_to_shot`, `lead_to_goal`.

**Linee difensive** — `last_defensive_line_x_*`, `delta_to_last_defensive_line_*`,
`inside_defensive_shape_*`, `n_defensive_lines`, `defensive_structure`,
`first_line_break`, `second_last_line_break`, `last_line_break`,
`n_opponents_bypassed`. È la parte più ricca e la più difficile da ricostruire da
soli.

**Conteggi di contesto** — una quarantina di colonne `n_*`:
`n_passing_options`, `n_simultaneous_runs`, `n_teammates_ahead_*`,
`n_opponents_within_5m_*`, `n_passing_options_line_break`…

**Pressing** — `pressing_chain`, `pressing_chain_length`, `pressing_chain_end_type`,
`consecutive_on_ball_engagements`, `angle_of_engagement`, `interplayer_distance_*`.

## Le metriche modellate

Qui il file smette di descrivere e comincia a **stimare**:

| Colonna | Cosa stima |
|---|---|
| `xthreat` | minaccia generata |
| `xpass_completion` | probabilità di riuscita del passaggio |
| `xshot_player_possession_{start,end,max}` | probabilità di tiro |
| `xloss_player_possession_{start,end,max}` | probabilità di perdere palla |
| `possession_epv_*`, `pass_epv_*` | expected possession value, per e contro |
| `reception_difficulty_start` | difficoltà del controllo |
| `space_constraint_start` | spazio disponibile |
| `overall_pressure_{start,end}` | pressione subita |
| `time_to_impact_{start,end}` | tempo prima dell'intervento avversario |
| `passing_option_ease_{start,end}` | facilità dell'opzione di passaggio |
| `separation_{start,end,gain}` | distacco dal marcatore |

Sono una scorciatoia potente e sono anche un rischio: **output di modelli chiusi di
terze parti**. Un risultato costruito sopra eredita assunzioni che non puoi
ispezionare né difendere in sede di giudizio. Usarle va benissimo, dichiararlo è
obbligatorio.

## Il file dichiara la propria qualità

| Colonna | Significato | Sulla partita `1886347` |
|---|---|---|
| `fully_extrapolated` | evento costruito solo su posizioni stimate | **13,9% dei `passing_option`**, 0,4% dei `player_possession` |
| `is_player_possession_start_matched` | inizio allineato al tracking | 5.075 / 5.115 |
| `is_player_possession_end_matched` | fine allineata | 5.074 / 5.115 |
| `is_previous_pass_matched` | passaggio precedente allineato | 1.923 / 1.946 |
| `is_pass_reception_matched` | ricezione allineata | 2.428 / 2.448 |

`fully_extrapolated` è il collegamento diretto con il limite descritto in
[`tracking-extrapolated.md`](tracking-extrapolated.md): SkillCorner segnala loro
stessi quali eventi poggiano interamente su posizioni non osservate. **Filtrarli, o
almeno riportarne la quota, è il minimo.**

Nota: la colonna è valorizzata solo per `passing_option` e `player_possession`; per
gli altri due tipi è `NaN`, il che non significa "non estrapolato".

## Vocabolario delle zone

Le stesse etichette compaiono anche in `phases_of_play`:

- **`channel`** (fasce verticali): `wide_left`, `half_space_left`, `center`,
  `half_space_right`, `wide_right`
- **`third`**: `defensive_third`, `middle_third`, `attacking_third`

## Come leggerlo

```python
from lib import data

de = data.load_dynamic_events(1886347)
corse = de[de.event_type == "off_ball_run"].dropna(axis=1, how="all")
```

Per animare un evento bastano `frame_start` e `frame_end`, da usare come filtro sul
tracking.

## Riferimenti

- Tutorial ufficiali: `notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/` nel repo dei dati (7 notebook, fra cui l'animazione e un "build your own metric")
- `src/features/DynamicEventsAggregator.py` nel repo dei dati, già pronto
- [Glossari SkillCorner](https://skillcorner.crunch.help/en/glossaries)
