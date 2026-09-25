# Quantifying defensive pressure on the ball carrier based on minimum arrival time

**Riferimento:** Narizuka, Sakamoto, Yamamoto, Yamazaki (2026 ca.), preprint.
**File:** `docs/Quantifying_defensive_pressure_on_the_ball_carrier_in_soccer_based_on_minimum_arrival_time.pdf` (11 pagine)

## Domanda e metodo

Misura la pressione difensiva sul portatore di palla come **tempo minimo di
arrivo dell'avversario più vicino** alla posizione del portatore, `τ_opp`.

Il tempo di arrivo si calcola con un modello fisico del moto (Fujimura–Sugihara):
il giocatore è una massa soggetta a una forza motrice costante e a una resistenza
viscosa, quindi l'insieme dei punti raggiungibili dopo un tempo *t* è un cerchio
che dipende da posizione e **velocità** attuali. Parametri usati: α = 1,0 s⁻¹ e
V_max = 10,0 m/s, calibrati in un loro lavoro precedente su sprint reali.

La tesi metodologica è di parsimonia: le misure di pressione esistenti aggiungono
assunzioni sopra il tempo di arrivo (soglie, punteggi compositi, classificatori),
e così perdono interpretabilità. Il tempo di arrivo grezzo è già una misura
sensata, e serve come baseline.

Risultati: la pressione all'inizio del possesso è associata a minore avanzamento
della palla (e regge dentro ciascuna delle tre zone di campo, quindi non è solo
un effetto di posizione); la pressione al rilascio è associata a maggiore
probabilità di perdere palla, e i passaggi giocati sotto pressione finiscono in
zone dove i compagni hanno meno vantaggio temporale sugli avversari.

## Dati richiesti

| | |
|---|---|
| Tipo | tracking **e** dati evento sincronizzati |
| Frequenza | 25 fps |
| Partite usate | **306** (J1 League 2023, DataStadium) |
| Completezza posizionale | tutti i 22, ma il calcolo usa solo l'avversario più vicino |
| Preprocessing | filtro Savitzky–Golay e spline cubica, velocità derivate dalle traiettorie lisciate |
| Annotazioni extra | intervalli di possesso individuale; tipo di passaggio (ordinario / filtrante); esito del possesso |

## Verdetto di fattibilità

**Replicabile con riserve — ed è il candidato migliore fra quelli letti finora.**

Il motivo è strutturale e vale la pena dirlo per esteso, perché è raro: **questo
metodo dipende dai giocatori vicini alla palla, che è esattamente dove i nostri
dati sono migliori.** `τ_opp` si calcola sull'avversario più vicino al portatore;
entro 5 metri dalla palla il 87% delle nostre posizioni è osservato, contro il
18% oltre i 40 metri. Quasi tutta la letteratura difensiva ha il problema
opposto.

Gli intervalli di possesso, che il paper ricava sincronizzando evento e tracking,
noi li abbiamo **già pronti**: `player_possession` in `dynamic_events.csv` ha
`frame_start`, `frame_end`, `end_type`, e il portatore identificato. È il pezzo
di lavoro che il paper descrive come preprocessing e che a noi non serve fare.

Anche l'esclusione dei possessi del portiere, che il paper applica per ragioni
tattiche, ci conviene per ragioni di dati: il portiere è osservato al 15%.

Il campione conta meno del solito, perché **il metodo non addestra nulla**: è
una misura fisica calcolata frame per frame. Venti partite limitano la potenza
statistica delle curve (il paper ne usa 306, con ~548 possessi per partita),
non la validità della misura.

### Le riserve, in ordine di gravità

**1. Le velocità non ci sono e vanno derivate.** Il modello del moto usa la
velocità iniziale `v_p(0)` come ingresso diretto. Kloppy espone colonne `_s`
(speed) e `_d` (distance) ma **per questi dati sono completamente vuote** —
verificato: 0 valori su 10.978. Vanno calcolate per differenze finite dalle
posizioni.

**2. 10 fps contro 25 fps.** Derivare velocità a 10 fps dà stime più rumorose, e
il rumore entra nel modello moltiplicato. Il paper liscia con Savitzky–Golay a
25 fps; a 10 fps serve più lisciatura, che introduce ritardo. È il punto dove la
replica può degradare in modo non ovvio, e va quantificato invece che assunto.

**3. Le velocità derivate ereditano l'estrapolazione.** Una posizione stimata
produce una velocità stimata. Anche restando vicino alla palla, il 13% delle
posizioni non è osservato, e in quei casi `τ_opp` è calcolato su una cinematica
inventata.

**4. α e V_max sono calibrati su sprint di J-League.** Usarli tali e quali
sull'A-League è un'assunzione da dichiarare, non da nascondere.

## Cosa servirebbe per replicarlo

Poco, ed è la ragione principale del verdetto.

| Pezzo | Stato |
|---|---|
| Intervalli di possesso | **già pronti** (`player_possession`) |
| Esito del possesso | **già pronto** (`end_type`, `lead_to_shot`) |
| Posizioni a 10 fps | **già pronte** (`lib/data.py`) |
| Flag di osservazione | **già pronto** (`is_detected`) |
| Velocità dei giocatori | **da scrivere** — differenze finite + lisciatura |
| Risoluzione di `r_p(τ) = ‖x − c_p(τ)‖` | **da scrivere** — una radice numerica scalare per giocatore/frame |
| Avanzamento della palla | **da derivare** — distanza dalla porta a inizio e fine possesso |

Il nucleo numerico è un solo solver scalare applicato a 22 giocatori per frame.
Non è un progetto di ricerca, è una giornata di lavoro più la validazione.

### Un controllo che avremmo gratis

`dynamic_events.csv` contiene già `time_to_impact` e `overall_pressure`, cioè la
versione SkillCorner della stessa idea. Sono modelli chiusi, quindi non
sostituiscono `τ_opp` — ma **confrontare la nostra misura con la loro è una
validazione esterna a costo zero**, dello stesso tipo del benchmark
`statsbomb_xg` in `xgoals_project`.

## Il buco che lascia

Il paper valida su tracking pulito a 25 fps da telecamere fisse. **Non si chiede
cosa succeda alla misura quando le posizioni sono ricostruite**, che è la
condizione di chiunque lavori con dati broadcast — cioè la maggior parte di chi
non è un club.

La domanda concreta che ne esce: *quanto è robusto `τ_opp` all'estrapolazione?*
Ed è misurabile con quello che abbiamo, perché `is_detected` ci dice quali
posizioni sono reali:

- quante volte **l'avversario più vicino** al portatore è estrapolato invece che
  osservato — non basta il tasso medio, conta proprio quel giocatore lì;
- di quanto si sposta `τ_opp` ricalcolandolo escludendo le posizioni stimate;
- se le relazioni del paper (pressione → avanzamento, pressione → perdita)
  sopravvivono restringendosi ai possessi interamente osservati;
- se l'errore è casuale o sistematico — l'estrapolazione tende a sottostimare o
  sovrastimare la pressione?

Corrisponde alla direzione **«Qualità dei dati broadcast»** in
`docs/Research_Directions_AnalyticsCup.pages`, ma con un oggetto preciso invece
che generico: non "le metriche difensive" in astratto, ma **una misura
pubblicata, interpretabile e già validata altrove**, usata come banco di prova.

## Note

- Il paper cita Bekkers 2025 *Pressing Intensity* come lavoro che trasforma il
  tempo di arrivo in punteggi più elaborati. È in `docs/` e va letto subito dopo
  questo: sono la stessa famiglia. Bekkers è l'autore di **`unravelsports`**, una
  delle librerie consigliate da PySport (vedi `notes/risorse.md`).
- Cita anche Spearman 2017 (pass probabilities), Fernández & Bornn 2018 (spazio),
  Gudmundsson & Horton 2017 (survey, in `docs/`) e TacticAI.
- `τ_same`, il tempo di arrivo del compagno più vicino, è calcolato ma usato poco:
  la differenza `τ_opp − τ_same` alla destinazione del passaggio è la quantità
  che predice meglio la perdita di palla. Anche questa è una pista.
