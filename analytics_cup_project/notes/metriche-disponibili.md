# Metriche disponibili: SkillCorner e librerie consigliate

Inventario di cosa esiste già, prima di calcolare qualcosa da zero: le metriche
nei file open data, il codice che SkillCorner distribuisce con i dati, e le
librerie indicate dalla challenge (vedi [`risorse.md`](risorse.md)).

Per ogni metrica: **cosa stima, in quali righe e istanti esiste, quanto è
documentata**. Il "dove esiste" conta quanto il "cosa": una metrica presente solo
a inizio e fine possesso non serve a chi la vuole frame per frame.

Stato al 03/10/2026. Clone `SkillCorner/opendata` al commit `4340d27`; versioni
delle librerie nelle tabelle.

## Fonti

| Sigla | Fonte |
|---|---|
| **[DE p.N]** | *Dynamic Events CSV Specifications*, 16/02/2025, 89 pagine ([PDF](https://26560301.fs1.hubspotusercontent-eu1.net/hubfs/26560301/Guides/Dynamic%20Events/20250216%20-%20Dynamic%20Events%20CSV%20Specifications.pdf)) |
| **[PoP]** | *Phases of Play CSV Specifications*, 16/02/2025, 8 pagine ([PDF](https://26560301.fs1.hubspotusercontent-eu1.net/hubfs/26560301/Guides/Phases%20of%20Play/20250216%20-%20Phases%20of%20Play%20CSV%20Specifications.pdf)) |
| **[README]** | `README.md` del clone `SkillCorner/opendata` |
| **[Glossario]** | [Physical Data Glossary](https://skillcorner.crunch.help/en/glossaries/physical-data-glossary) |
| **[codice: …]** | sorgente della libreria, alla versione indicata |
| **[verificato]** | controllato sui nostri dati in questa sessione; come rifarlo è scritto accanto |

Quando una riga non ha fonte, è perché la fonte non c'è: lo si dice.

## In breve

- **I modelli SkillCorner sono tutti chiusi**, ma non tutti ugualmente
  documentati. `xthreat`, `xpass_completion`, `xloss`, `xshot` ed EPV hanno almeno
  una definizione e il tipo di modello [DE p.10–11]. Le 8 metriche di pressione e
  difficoltà hanno una riga di README e nient'altro.
- **Ogni metrica modellata vive su un solo tipo di evento e in pochi istanti.**
  EPV solo sui `player_possession`, `xloss`/`xshot` solo sugli
  `on_ball_engagement`, `xthreat` sulle opzioni di passaggio. Nessuna è frame per
  frame.
- **Il tracking non ha metriche**, nemmeno la velocità: va derivata.
- **I tutorial ufficiali non usano le metriche modellate** nel codice. Usano gli
  aggregati, gli attributi geometrici e le somme di `xthreat`.
- **Delle librerie, solo databallpy aggiunge metriche sui nostri dati** passando
  da kloppy: pitch control, pressione sul giocatore, velocità. Lo stesso vale per
  unravelsports (Pressing Intensity), provato su una partita. floodlight non
  legge il formato attuale. socceraction richiede dati evento che non abbiamo.

---

## 1. SkillCorner: i file open data

### 1.1 Tracking

Posizioni a 10 fps, `is_detected`, palla con `z`. **Nessuna metrica.** Le colonne
di velocità che kloppy espone sono vuote su questi dati, e il README avverte:
*"Some speed or acceleration smoothing and control should be applied to the raw
data"* [README]. Dettagli in [`dati/tracking-extrapolated.md`](dati/tracking-extrapolated.md).

### 1.2 Dynamic events: le metriche modellate

322 colonne, identiche in tutte e 20 le partite [verificato]. Sulle 20 partite:
48.308 `passing_option`, 18.710 `player_possession`, 17.445 `on_ball_engagement`,
9.898 `off_ball_run`.

La copertura è la quota di righe di quel tipo con valore non nullo, sulle 20
partite [verificato]. PP = `player_possession`, PO = `passing_option`, OBR =
`off_ball_run`, OBE = `on_ball_engagement`.

| Metrica | Cosa stima | Modello | Dove esiste | Istante | Documentazione |
|---|---|---|---|---|---|
| `xthreat` | probabilità di gol entro 10 s se il giocatore riceve un passaggio riuscito | xThreat model [DE p.10] | PO 100%, OBR 100% | al passaggio se servito, altrimenti al *passing moment* teorico [DE p.10, p.68] | definizione |
| `player_targeted_xthreat` | lo stesso, per il destinatario del passaggio | idem | PP 81% | al passaggio | definizione |
| `xpass_completion` | probabilità di completare il passaggio verso il giocatore | GNN [DE p.11] | PO 100%, OBR 100%; PP 81% come `player_targeted_xpass_completion` | come sopra | definizione |
| `passing_option_score` | probabilità che il giocatore riceva il prossimo passaggio | Receiver model [DE p.10] | PO 100%, OBR 100% | come sopra | definizione; ≥ 0,6 ⇒ `predicted_passing_option` [DE p.69] |
| `dangerous` / `difficult_pass_target` | soglie: `xthreat` > 0,02 / `xpass_completion` < 0,65 | derivate | PO, OBR | come sopra | [DE p.67–68] |
| `xloss_player_possession_{start,end,max}` | probabilità che il portatore perda palla nel possesso (passaggio sbagliato compreso, tiri esclusi) | Progression model, GNN-LSTM [DE p.11] | **solo OBE 97%**, PP 0% | inizio, fine, massimo del possesso ingaggiato | definizione [DE p.52]; il PDF indica "TRUE / FALSE" come valori di un float |
| `xshot_player_possession_{start,end,max}` | probabilità che il possesso finisca con un tiro | idem | **solo OBE 97%** | idem | [DE p.52] |
| `possession_epv_*`, `pass_epv_*`, `pass_reception_epv_*` (12 colonne) | valore della posizione per e contro la squadra in possesso, e le variazioni | EPV, GNN-LSTM: probabilità di segnare entro 90 s o prima che la palla esca [DE p.11] | **solo PP**: 96–97% `possession_*`, 80–82% `pass_*` | inizio e fine possesso, ricezione, passaggio | le 12 colonne sono descritte solo nel [README]; il PDF di febbraio 2025 descrive il modello ma non queste colonne |
| `possession_danger`, `beaten_by_possession`, `beaten_by_movement`, `stop_possession_danger`, `reduce_possession_danger` | flag difensivi costruiti sull'EPV (es. EPV > 3% almeno in un frame) | derivati da EPV | solo OBE 100% | finestra ingaggio → fine possesso | [DE p.44–46], con soglie in parte non numeriche ("significantly increase") |
| `force_backward` | il portatore è costretto al passaggio all'indietro (+30 punti di probabilità di regressione) | Progression model | solo OBE 100% | fine possesso | [DE p.46] |
| `overall_pressure_{start,end}`, `time_to_impact_{start,end}`, `passing_option_ease_{start,end}`, `space_constraint_start`, `reception_difficulty_start` | pressione e difficoltà, in 5 classi | **non dichiarato** | solo PP 87–89% | inizio e/o fine possesso | **una riga nel [README]**, assenti dal PDF. `time_to_impact` è analizzato in [`explorations/03`](../explorations/03-calcolo-tau-opp.ipynb) §10 |

Conseguenze pratiche:

- **`xloss` e `xshot` vanno letti dagli ingaggi difensivi**, non dai possessi:
  descrivono il possesso che il difensore sta ingaggiando. Un possesso senza
  ingaggio non ha `xloss`.
- **L'EPV c'è solo dove c'è un possesso**: niente valore frame per frame, niente
  valore per l'alternativa non giocata.
- **`xthreat` sommato non è un valore atteso del giocatore.** È una somma di
  probabilità calcolate su opzioni che si sovrappongono nel tempo
  (`n_simultaneous_passing_options`). L'aggregatore ufficiale lo somma lo stesso
  (§2.1).

### 1.3 Dynamic events: attributi geometrici e regole

Non sono modelli ma regole sul tracking, quindi sono ricostruibili. Le definizioni
che servono più spesso:

| Attributo | Definizione | Fonte |
|---|---|---|
| `separation_{start,end,gain}` | distanza dall'avversario più vicino (m) | [DE p.64] |
| `received_in_space` | ricevuto con l'avversario più vicino ad almeno 3 m | [DE p.44] |
| `speed_avg`, `speed_avg_band` | velocità media nell'evento; fasce: jogging < 15, running 15–20, hsr 20–25, sprinting > 25 km/h | [DE p.23, p.49] |
| `distance_covered`, `trajectory_angle`, `trajectory_direction` | spostamento nell'evento | [DE] |
| `carry` / `quick_pass` / `one_touch` | ≥ 2 m percorsi / possesso < 1 s chiuso da passaggio, non di prima / di prima | [DE p.76] |
| `forward_momentum` | primo controllo che avvia un'azione progressiva, dopo un passaggio da dietro o di lato | [DE p.25] |
| `last_defensive_line_{x,height}_*`, `delta_to_last_defensive_line_*` | posizione e altezza dell'ultimo difendente, distanza lungo x | [DE p.63] |
| `inside_defensive_shape_*` | dentro il convex hull della squadra avversaria | [DE p.63] |
| `organised_defense`, `defensive_structure`, `n_defensive_lines` | difesa schierata (es. `442`); solo se schierata si calcolano le rotture di linea | [DE p.70] |
| `first_line_break`, `second_last_line_break`, `last_line_break` (+ `_type` through/around) | rottura della linea; su un passaggio sbagliato è **predetta** come se fosse riuscito | [DE p.71] |
| `n_opponents_bypassed` | avversari davanti al passatore meno avversari davanti al ricevitore | [DE p.75] |
| `goal_side_*`, `angle_of_engagement`, `interplayer_distance_*` | geometria dell'ingaggio difensivo | [DE p.60–62] |
| `pressing_chain`, `pressing_chain_length`, `pressing_chain_end_type` | catena di pressing, esito `regain` / `disruption` | [DE p.86] |
| `push_defensive_line`, `break_defensive_line`, `intended_run_behind` | solo corse `behind`: linea arretrata di 10 m, superata di 1 m, intenzione | [DE p.82–83] |
| `lead_to_shot`, `lead_to_goal` | tiro o gol entro 10 s dalla fine dell'evento | [DE p.44] |
| `fully_extrapolated` | giocatore mai visto dalla telecamera durante l'evento | [DE p.89] |
| `n_*` (una quarantina) | conteggi di compagni e avversari davanti, entro 5 m, opzioni di passaggio | [DE] |

Le coordinate degli eventi non sono riscalate a un campo standard [README]. Il
resto delle colonne è in [`dati/dynamic-events.md`](dati/dynamic-events.md).

### 1.4 Phases of play

Otto coppie di fasi speculari (in possesso / fuori possesso), definite a regole su
eventi e tracking [PoP p.3–5]:

| In possesso | Fuori possesso | Regola, in breve |
|---|---|---|
| `build_up` | `high_block` | palla nel proprio terzo, portatore pressato o avversari alti |
| `create` | `medium_block` | fase di default, tipicamente nel terzo centrale |
| `finish` | `low_block` | ultimo o terzo centrale, linea difensiva vicino all'area, possesso stabile da ≥ 1 s |
| `direct` | `defending_direct` | lancio di 32+ m lungo x dalla propria metà, fino alla ricezione |
| `quick_break` | `defending_quick_break` | recupero nella metà avversaria e progressione rapida |
| `transition` | `defending_transition` | recupero nella propria metà e progressione rapida |
| `set_play` | `defending_set_play` | corner, punizioni, rimesse lunghe in area; finisce entro 20 s |
| `chaotic` | `chaotic` | possesso conteso: meno di 3 passaggi e meno di 5 s |

Le metriche della fase: larghezza e lunghezza di **entrambe** le squadre a inizio
e fine (`team_*_width_*`, `team_*_length_*`), possessi nella fase, perdita, tiro,
gol. Le fasi esistono solo per partite con indice di qualità > 4 [PoP p.3].
Dettagli in [`dati/phases-of-play.md`](dati/phases-of-play.md).

### 1.5 Aggregati stagionali

Tre file per giocatore-stagione, A-League 2024/25: `physical` (406 × 65),
`obr` (407 × 130), `passing` (407 × 48). Solo prestazioni oltre i 60 minuti
[README].

**Come si leggono i nomi delle colonne**, verificato sui file perché il README non
lo dice:

| Pezzo del nome | Significato | Fonte |
|---|---|---|
| `minutes*`, `*_full_all`, `*_full_tip` in `physical` | **media per partita**, non totale (distanza mediana 10,15 km, 89 minuti) | [verificato] |
| `*_total` in `obr` e `passing` | totale di stagione | [verificato] |
| `tip` / `otip` | squadra in possesso / avversario in possesso | [Glossario] |
| `bip` | palla in gioco = tip + otip | [codice: `skillcornerviz.utils.skillcorner_physical_utils.add_p60_bip`] |
| `all` | tutta la partita, palla ferma compresa: tip + otip è il 56% dei minuti `all` (mediana) | [verificato] |
| `p30tip` | per 30 minuti di possesso della squadra: `totale / (minutes_tip × performance_included_count) × 30`, rapporto 1,000 su tutte le righe | [verificato] |

Metriche fisiche [Glossario], soglie per il calcio maschile:

| Metrica | Definizione |
|---|---|
| running / HSR / sprint | 15–20 / 20–25 / > 25 km/h; HI = HSR + sprint |
| medium / high accel, decel | 1,5–3 / > 3 m/s², e i simmetrici negativi |
| `explacceltohsr`, `explacceltosprint` | accelerazione alta partita sotto i 9 km/h che raggiunge HSR / sprint |
| `timetohsr_top3`, `timetosprint_top3` | media dei 3 tempi migliori da 9 km/h alla soglia |
| `psv99`, `psv99_top5` | 99° percentile della velocità di punta, proxy della velocità massima; media delle 5 migliori prestazioni. Qui fra 24,3 e 32,7 km/h [verificato] |
| `total_metersperminute` | distanza / minuti (coerente sui file, scarto ≤ 0,01) [verificato] |

`obr`: 11 tipi di corsa × {conteggio, con tiro entro 10 s, con gol entro 10 s,
cercato, ricevuto, sopra HSR, pericoloso, pericoloso cercato, pericoloso ricevuto,
distanza media}, quasi tutte per 30' tip.
`passing`: occasioni e tentativi di passaggio, completati, `avgxpass`, lunghi, di
prima, rapidi, che rompono linee, verso corse, pericolosi, difficili.

### 1.6 Body pose

29 giunti 3D a 25 fps, 2 partite, errore stimato per giunto (`p90_mae_cm`).
Nessuna metrica pronta; l'unico esempio ufficiale è l'orientamento delle spalle
(§2.1). Dettagli in [`dati/bodypose.md`](dati/bodypose.md).

---

## 2. SkillCorner: il codice

### 2.1 `src/` del repo opendata

Licenza MIT [codice: `LICENSE`]: si può copiare nella submission, citando la
fonte. Non è un pacchetto installabile: si importa aggiungendo la radice del clone
a `sys.path`.

**`features/DynamicEventsAggregator.py`.** Aggrega per giocatore, o per
qualunque `group_by`, contando eventi in **contesti** (filtri) e applicando a
ciascun contesto un gruppo di **metriche**. Sulla partita 1886347
[verificato, `generate_aggregates(["player_id"], tipo)`]:

| Tipo di aggregato | Contesti | Metriche | Colonne |
|---|---|---|---|
| `off_ball_runs` | 39 (tipo di corsa × fase) | 21: conteggi cercato/ricevuto/pericoloso/difficile, somme di `xthreat` e `xpass_completion`, velocità, HSR, sprint, canale | 819 |
| `line_breaking_options` | 28 (linea × through/around × fase) | 17 | 476 |
| `passes_to_off_ball_runs` | 39 | 16: occasioni, tentativi, completati, con somme di `xthreat` | 624 |
| `line_breaking_passes` | 28 | 16 | 448 |
| `possessions` | 6 | 8 | 48 |
| `on_ball_engagements` | 9 | 22: esiti, goal-side, beaten, danger, force backward, catene | 198 |
| `pressing_` / `pressure_` / `counter_press_` / `recovery_press_engagements` | 5–7 ciascuno | 19 | 95–133 |

Totale 3.031 colonne. Il tutorial `02_Part1` cita 821 colonne: con ogni
probabilità è il solo `off_ball_runs` (819) più due colonne di raggruppamento, ma
non l'ho verificato rieseguendo il tutorial.

Alcune metriche sono **definite dentro l'aggregatore**, non nel CSV:

| Metrica | Regola | Fonte |
|---|---|---|
| `received_in_tight_space` / `received_in_open_space` | `separation_start` ≤ 2 m / ≥ 6 m | [codice: riga 1235] |
| `8m_carry`, `8m_carry_at_speed` | `carry` e `distance_covered` ≥ 8 m (+ velocità) | [codice: riga 1241] |
| `count_got_close` | `interplayer_distance` da ≥ 3 m a ≤ 1,5 m | [codice: riga 1275] |
| `count_got_goal_side` | `goal_side` da falso a vero | [codice: riga 1272] |
| `isolated_engagement` | nessun compagno e nessun avversario entro 5 m dal bersaglio | [codice: riga 888] |
| `possession_retentions` | possesso chiuso da passaggio riuscito | [codice: riga 860] |

Nei nomi dei contesti ci sono refusi (`around_first_line_create` accanto a
`around_first_line_in_build_up`, `overlap_in_finish` accanto a
`overlap_runs_in_finish`): chi filtra per nome se ne accorge.

**`features/PhasesOfPlayAggregator.py`.** Per squadra e partita, in possesso e
fuori possesso, 187 colonne [verificato]: numero e durata delle fasi, possessi,
perdite, tiri, gol, larghezza e lunghezza medie, e la matrice delle transizioni
fra fasi (`count_into_X_from_Y`).

**`features/pose_orientation.py`.** Orientamento delle spalle dal body pose, con
cinque filtri dichiarati: pose presente, entrambe le spalle, spalle non
coincidenti, larghezza plausibile, errore sotto soglia.

**`data/`.** `pose_loading.py` scarica e legge il body pose da Hugging Face.
`basic_loading.py` non è una libreria: è uno script che all'import carica una
partita intera.

**`visualization/`.** `head2head_viz.plot_head2head` (confronto fra due squadre o
giocatori) e `sectioned_summary_table_viz.ranking_plot` (percentili per sezioni).

### 2.2 `skillcornerviz` 1.2.2

**Solo visualizzazione e normalizzazione: non calcola metriche.**
- **Grafici:** `bar_plot`, `scatter_plot`, `swarm_violin_plot`, `radar_plot`,
  `summary_table`, `table_grid`, `zscore_dotplot`, `narrative_ranking_plot`.
- **Utilità:** p90, p30 tip / otip, p60 bip, percentili, età, gruppi di corse e
  passaggi [codice: `skillcornerviz.utils`].

`plot_radar` legge i percentili da colonne `<metrica>_pct`. Il tutorial
`01_Part1` usa un argomento (`physical_df=`) che nessuna versione pubblicata
accetta. La copia corretta è in
[`explorations/sc-tutorial-01-visualization.ipynb`](../explorations/sc-tutorial-01-visualization.ipynb).

### 2.3 I tutorial

| Tutorial | Cosa usa | Metriche modellate nel codice |
|---|---|---|
| `01` Part 1–3 | aggregati + `skillcornerviz`: grafici, z-score, archetipi di attaccanti | nessuna |
| `02` Part 1–2 | `DynamicEventsAggregator`, `PhasesOfPlayAggregator` | somme di `xthreat` e `xpass_completion` dentro l'aggregatore |
| `02` Part 3–5, 7 | corse su campo, merge eventi-tracking, animazione, forma di squadra per fase | nessuna |
| `02` Part 6 | *Build your own metric*: i cutback, da zone e direzione del passaggio | nessuna |
| `03` | basi del tracking, kloppy | nessuna |
| `04` | radar e tabelle a sezioni sugli aggregati | nessuna |
| `05` | body pose, orientamento delle spalle | nessuna |

Nessun tutorial usa nel codice EPV, `xloss`, `xshot` o le metriche di pressione
[verificato: ricerca nelle celle di codice]. Sette tutorial importano
`skillcorner.client`, il client dell'API a pagamento, ma poi leggono i CSV
locali.

### 2.4 Il client `skillcorner`

Serve per l'API commerciale, con credenziali. Non aggiunge nulla ai dati open.

---

## 3. Le librerie consigliate

| Libreria | Versione | Legge i nostri dati? | Cosa aggiunge | Provato qui |
|---|---|---|---|---|
| kloppy | 3.19.0 | sì, solo tracking | caricamento, orientamento, `to_df`; nessuna metrica sul tracking | sì |
| databallpy | 0.8.1 | **sì, via `get_game_from_kloppy`** | velocità, accelerazione, pitch control, pressione sul giocatore, possesso individuale, distanze per fascia | sì, su 3.000 frame |
| floodlight | 1.2.0 | **no**: il lettore SkillCorner è per il dataset del 2021 | modelli geometrici e cinematici su array XY | lettore: fallisce |
| mplsoccer | 1.8.1 | non legge dati | Voronoi, convex hull, binning, angoli; `pitch_type="skillcorner"` | in uso nel workbench |
| socceraction | 1.5.3 | **no**: servono dati evento | SPADL, VAEP, xT | no, non installabile nel venv |
| unravelsports | 1.2.1 | **sì, via kloppy** | Pressing Intensity, formazioni (EFPI), grafi per GNN | Pressing Intensity sì, su una partita |
| Soccer Analytics Handbook | update feb. 2023 | non è una libreria | esempi su StatsBomb e Metrica | letto |

### 3.1 kloppy

- **Caricamento:** `skillcorner.load(meta_data, raw_data, …)` e
  `load_open_data(match_id, …)`, con `sample_rate`, `limit`, `only_alive`
  [codice: `kloppy.skillcorner`].
- **Cosa perde:** `is_detected`. Le colonne di velocità restano vuote.
- **Trasformazioni:** orientamento e sistema di coordinate (`transform`),
  `filter`, `to_df` in pandas o polars.
- **Cosa non serve qui:** gli state builder (`score`, `lineup`, `sequence`,
  `formation`) e l'aggregatore `minutes_played` lavorano sui dataset di eventi
  [codice: `kloppy.domain.services`]. Per SkillCorner kloppy carica solo il
  tracking, quindi non si applicano.

### 3.2 databallpy

`get_game` non supporta SkillCorner: i provider di tracking sono tracab, metrica e
inmotio [codice: docstring di `get_game`]. **`get_game_from_kloppy` invece
funziona**: su 3.000 frame della partita 1886347 crea un `Game` con
`frame_rate = 10` [verificato].

| Metodo di `TrackingData` | Cosa calcola | Base | Provato |
|---|---|---|---|
| `add_velocity`, `add_acceleration`, `filter_tracking_data` | derivate, con filtro Savitzky–Golay opzionale e tetto di velocità | — | `add_velocity` ok |
| `get_covered_distance` | distanza per fascia di velocità e accelerazione | — | no |
| `get_pressure_on_player` | pressione su un giocatore in un frame | Herold et al. 2022 (`d_front`), Andrienko et al. 2016 [codice: `features/pressure.py`] | ok, restituisce uno scalare |
| `get_pitch_control` | superficie di controllo per griglia | Fernández & Bornn 2018, *Wide Open Spaces* [codice: `features/pitch_control.py`] | ok, griglia 53 × 34 |
| `get_approximate_voronoi` | Voronoi su griglia | — | **fallisce** (`ValueError` sulla forma dei dati), non indagato |
| `add_individual_player_possession` | chi ha la palla, da distanza e velocità della palla | — | no |
| `add_team_possession` | possesso di squadra | richiede dati evento | non applicabile |
| `add_dangerous_accessible_space` | spazio accessibile pericoloso (DAS) | — | no |
| `synchronise_tracking_and_event_data` | sincronizzazione evento-tracking | richiede un provider di eventi | non applicabile |

Il `Game` ha 3.125 righe contro 2.999 frame caricati: databallpy riempie i buchi
di frame [verificato]. Va tenuto presente quando si confrontano conteggi.

### 3.3 floodlight

`io.skillcorner.read_position_data_json` legge lo *SkillCorner Open Dataset* in
JSON del 2021 (9 partite) [codice: docstring]. Sul formato JSONL attuale fallisce
con `JSONDecodeError` [verificato].

I modelli lavorano su oggetti `XY` che si possono costruire a mano dagli array del
tracking (non provato):
- **geometria:** `CentroidModel`, `ConvexHullModel`, `NearestMateModel`,
  `NearestOpponentModel`;
- **cinematica:** `DistanceModel`, `VelocityModel`, `AccelerationModel`;
- **cinetica:** `MetabolicPowerModel`;
- **spazio:** `DiscreteVoronoiModel`;
- **metriche:** `entropy`, `trajectory_clustering`, `zone_aggregation`.

### 3.4 mplsoccer

Visualizzazione, con alcune utilità analitiche: `Pitch.voronoi`, `convexhull`,
`bin_statistic` (anche per zone e posizionale), `goal_angle`,
`calculate_angle_and_distance`, formazioni [codice: `mplsoccer.Pitch`]. Ha un
`pitch_type="skillcorner"` già usato nel workbench.

### 3.5 socceraction

SPADL, VAEP (anche atomico) e xT [codice: `socceraction/`]. **Lavora su dati
evento**: convertitori per StatsBomb, Opta, Wyscout e `EventDataset` di kloppy.
SkillCorner open data non ha un dataset evento caricabile da kloppy, quindi per
usarlo bisognerebbe costruire le azioni SPADL a mano da `dynamic_events`.

Nel venv non si installa: richiede `lxml<5`, `numpy<2` e `pandas<3`
[codice: `METADATA`], mentre il venv ha lxml 5.4, numpy 2.2.6 e pandas 2.3.3. Il
commento in `requirements.txt` cita solo il conflitto su lxml.

### 3.6 unravelsports

Versione 1.2.1, installata nel venv. Le dipendenze vere sono solo kloppy,
polars ≥ 1.35 e scipy [codice: `METADATA`]: l'installazione ha aggiunto solo il
pacchetto [verificato]. Tensorflow, torch e spektral stanno nell'extra `test` e
servono solo per addestrare le GNN.

- **`KloppyPolarsDataset`**: da un dataset kloppy calcola velocità e
  accelerazioni (Savitzky–Golay di default), con tetti a 12 m/s e 6 m/s², e
  inferisce il portatore di palla.
- **`PressingIntensity`** (Bekkers 2025): per ogni frame, la matrice dei tempi di
  intercetto fra difensori e attaccanti e la probabilità di intercetto (una
  sigmoide del tempo). Parametri di default: reazione 0,7 s, soglia 1,5 s,
  σ = 0,45 [codice: `soccer/models/pressing_intensity.py`]. È della stessa
  famiglia di τ_opp, ma frame per frame e per ogni coppia di giocatori.
  **Provata sulla partita 1886347** [verificato,
  [`explorations/04`](../explorations/04-pressing-intensity-unravel.ipynb)]:
  - gira su tutti i 43.458 frame in circa 35 s;
  - il tempo di intercetto ha una soglia minima di 0,7 s (la reazione) e punta
    alla posizione dell'attaccante fra 1 s;
  - il minimo sul portatore a inizio possesso concorda con τ_opp a ρ = 0,65 e con
    `time_to_impact` a −0,67;
  - nel 16% dei possessi, al frame d'inizio, la squadra in possesso di kloppy non
    è quella di SkillCorner.

  `KloppyPolarsDataset` scrive `provider="secondspectrum"` per qualunque fonte: è
  un'etichetta.
- **`EFPI`**: riconosce la formazione con l'assegnazione ungherese su modelli di
  riferimento (mplsoccer o Shaw–Glickman), per frame o per possesso.
- **`SoccerGraphConverter`** e `classifiers`: trasformano il tracking in grafi per
  le GNN.

### 3.7 Soccer Analytics Handbook

Un solo notebook, aggiornato a febbraio 2023, su dati StatsBomb e Metrica
[README del repo]:
- visualizzazione;
- clustering (K-Means, GMM);
- difficoltà del passaggio con XGBoost;
- tracking: traiettorie, corse ad alta intensità, curve di Bézier, regioni di
  confidenza, **time to intercept**.

**Il pitch control non c'è** nella versione attuale; [`risorse.md`](risorse.md) lo
cita come punto di partenza per il pitch control.

---

## 4. Cosa usare per cosa

| Serve | Pronto in SkillCorner | Pronto in una libreria | Limite |
|---|---|---|---|
| pressione sul portatore | 8 classi su PP, inizio e fine | databallpy `get_pressure_on_player`; unravelsports `PressingIntensity` | SkillCorner: non documentato, 5 classi, solo 2 istanti. Librerie: richiedono velocità derivate; il TTI di unravelsports non scende sotto 0,7 s |
| valore di un'azione o di una posizione | EPV su PP; `xthreat` sulle opzioni | socceraction VAEP / xT, solo dopo una conversione SPADL fatta a mano | EPV senza controfattuale e non frame per frame |
| probabilità di perdere palla o di tirare | `xloss`, `xshot` | — | solo sui possessi ingaggiati da un difensore |
| spazio e controllo del campo | larghezza e lunghezza per fase; `separation` | databallpy pitch control; floodlight e mplsoccer Voronoi | estrapolazione lontano dalla palla (59% osservato) |
| velocità e accelerazioni | `speed_avg` per evento; aggregati fisici | databallpy, unravelsports, `lib/pressione.velocita` | tracking a 10 fps, picchi fino a 13,9 m/s |
| formazione | `defensive_structure` sui passaggi a difesa schierata | unravelsports EFPI | — |
| profili fisici per giocatore | aggregati `physical` | `skillcornerviz` per i grafici | medie per partita sopra i 60 minuti |

---

## 5. Discrepanze trovate

Corrette il 03/10/2026 nei file dove stavano:

- [`dati/dynamic-events.md`](dati/dynamic-events.md): `xloss` e `xshot` erano
  elencati senza dire che esistono solo sugli `on_ball_engagement`; `separation`
  era fra le metriche modellate come "distacco dal marcatore", ma è la distanza
  dall'avversario più vicino [DE p.64].
- [`dati/README.md`](dati/README.md): gli aggregati stagionali risultavano "non
  ancora documentati".
- [`../requirements.txt`](../requirements.txt): socceraction confligge anche su
  numpy, non solo su lxml. Il commento su unravelsports ("pesante, tensorflow")
  valeva solo per la parte GNN, e ora il pacchetto è installato. Il commento su
  databallpy ne indicava come punto di forza la sincronizzazione, che qui non si
  applica.
- [`risorse.md`](risorse.md): la sincronizzazione di databallpy (come sopra);
  l'Handbook che "parte dal pitch control", che non contiene più; floodlight
  presentato come lettore utilizzabile.
- [`../CLAUDE.md`](../CLAUDE.md) §1: "Restano da coprire gli aggregati stagionali
  e il body pose".

Fuori dal nostro controllo:

- [DE p.52]: `xloss_*` è un float ma la colonna "Values" dice "TRUE / FALSE".
- Tutorial `01_Part1`: `physical_df=` non esiste in `skillcornerviz` (§2.2).

## Come rifare le verifiche

- **Copertura delle colonne:** concatenare `data.load_dynamic_events(m)` sulle 20
  partite, poi `groupby("event_type").apply(lambda x: x.notna().mean())`.
- **`p30tip`:** `o.offballrun_count_total / (o.minutes_tip * o.performance_included_count) * 30`
  diviso `o.offballrun_count_p30tip`, con `o = data.load_aggregate("obr")`.
- **databallpy:** `skillcorner.load(meta_data=…, raw_data=…, limit=3000)`, poi
  `databallpy.get_game_from_kloppy(tracking_dataset=ds)` e i metodi di
  `game.tracking_data`.
- **floodlight:** `floodlight.io.skillcorner.read_position_data_json(<jsonl>, <match.json>)`.
