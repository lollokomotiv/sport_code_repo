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

Per ognuna: quale dato serve, quale limite la minaccia, che figura produce.

### A — Robustezza di `τ_opp` all'estrapolazione

**Da:** [`notes/letteratura/pressione-tempo-arrivo.md`](../notes/letteratura/pressione-tempo-arrivo.md)

Narizuka et al. misurano la pressione sul portatore come tempo minimo di arrivo
dell'avversario più vicino, su 306 partite di tracking pulito a 25 fps da
telecamere fisse. Non si chiedono cosa succeda alla misura quando le posizioni
sono ricostruite — che è la condizione di chiunque usi dati broadcast.

*La domanda:* quanto è robusto `τ_opp` all'estrapolazione?

- quante volte l'avversario **più vicino** al portatore è estrapolato invece che
  osservato (il tasso medio non basta: conta proprio quel giocatore lì)
- di quanto si sposta `τ_opp` ricalcolandolo sulle sole posizioni osservate
- se le relazioni del paper (pressione → avanzamento, pressione → perdita di
  palla) sopravvivono restringendosi ai possessi interamente osservati
- se l'errore è casuale o **sistematico**: l'estrapolazione sottostima o
  sovrastima la pressione?

| | |
|---|---|
| **Dato che serve** | tracking + `player_possession`, `is_detected`, velocità derivate |
| **Cosa la minaccia** | velocità a 10 fps invece di 25: stime rumorose, e il rumore entra nel modello del moto |
| **Perché regge** | dipende dai giocatori *vicini* alla palla, dove siamo all'87% osservato |
| **Figura** | una: `τ_opp` osservato contro `τ_opp` estrapolato, o lo scostamento in distribuzione |
| **Validazione gratis** | confronto con `time_to_impact` di SkillCorner |

Corrisponde alla direzione 5 («Qualità dei dati broadcast») delle Research
Directions, ma con un oggetto preciso invece che generico: non "le metriche
difensive" in astratto, ma **una misura pubblicata, interpretabile e già
validata altrove**, usata come banco di prova.

È anche l'unica pista che trasforma il limite principale dei nostri dati da
problema in oggetto di studio. Le altre lo subiscono.


### B — «Esce o tiene?»: la scelta del difensore

**Da:** [`notes/letteratura/`](../notes/letteratura/README.md) e
[`explorations/02-body-pose.ipynb`](../explorations/02-body-pose.ipynb)

Quasi tutta la letteratura difensiva misura lo **stato** (pressione, spazio,
forma) o l'**esito** (valore concesso). La scelta è toccata solo dal ghosting —
Groom et al. sui corner, Yurko et al. nel football americano — che però chiede
*"quanto è diverso da un difensore medio?"*, non *"ha scelto bene fra le opzioni
che aveva?"*.

*La domanda:* quando un difensore esce sul portatore e quando tiene la
posizione, e la scelta era quella giusta? Si confronta l'esito atteso delle due
alternative, cioè il framing di VAEP applicato alla difesa.

#### Cosa esiste già (ricerca del 26/09/2026)

**Il "tipo di difensore" da solo non è una novità.** I provider lo fanno già:

| Chi | Cosa fa | Cosa non fa |
|---|---|---|
| **Hudl StatsBomb — [DefR](https://www.hudl.com/blog/defensive-responsibility-defr-statsbomb)** | per ogni azione avversaria stima quale difensore avrebbe dovuto intervenire; 4 archetipi da proattività × soppressione (Upamecano proattivo ed efficace, Rouault proattivo ma permissivo, Rüdiger passivo e permissivo, Delprato passivo ma efficace) | solo dati evento, livello stagione, nessuna valutazione della singola scelta |
| **SkillCorner** | tassonomia degli ingaggi con *Jockeying / Holding Ground* distinto dalla pressione; profili dei centrali (stepping forward, covering space in behind, recovery runs) | profili come **frequenze in z-score**: dicono quanto spesso esce, non se ha fatto bene |
| **Opta Vision** | Pressure Intensity sui 3 difensori più vicini, corse difensive senza palla | misura la pressione, non la scelta |

**Il lavoro accademico più vicino è exPressV2** (Lee et al., MLSA 2025,
[scheda](../notes/letteratura/pressing-exPressV2.md)): 36 partite, GRU + GAT,
probabilità di recupero palla e merito individuale. Ma **seleziona sull'azione**
— analizza solo i momenti in cui il pressing c'è già — quindi non può dire nulla
sulla scelta di non pressare. È esattamente lo spazio della pista B.

Metodologicamente utile anche *Tackling Causality* (football americano, Sloan):
tratta l'azione del difensore come un **trattamento** e ne stima l'effetto con
stimatori *doubly robust*. Per "esce o tiene" è l'impostazione più solida, e
17.445 eventi sono il campione che le serve.

#### La riformulazione: decisione contro esecuzione

DefR dice che Rouault è *proattivo ma permissivo*, ma non perché. Può uscire nei
momenti sbagliati — problema di **decisione** — oppure uscire nei momenti giusti
e perdere il duello — problema di **esecuzione**. **Separare le due cose è il
contributo che manca**, e nessun provider lo offre.

#### Il disegno: selezionare sulla situazione, non sull'azione

1. **La situazione**: momenti in cui un difensore *avrebbe potuto* uscire, per
   esempio perché il suo tempo di arrivo sul portatore (`τ_opp`, pista A) era
   sotto una soglia.
2. **Il trattamento**: è uscito (`pressure`, `pressing`) oppure ha tenuto.
3. **L'esito**: recupero, interruzione, pericolo ridotto, oppure battuto.
4. **Il confronto**: effetto dell'uscita a parità di situazione, con
   aggiustamento per ciò che rende una situazione più adatta a uscire.

**Il gruppo "tiene" potrebbe essere già etichettato.** Il sottotipo `other` degli
`on_ball_engagement` (3.251 eventi su 20 partite) ha il profilo atteso
dall'*Holding Ground*: è il più lento (velocità mediana 10,3 contro 14,0 di
`pressure`), quello in cui il difensore si muove meno (2,7 m contro 5,7), il più
breve (1,1 s), e parte già più vicino al portatore. **È una compatibilità, non
una conferma**: la documentazione non dice cosa ci sia dentro `other`.

Resta un problema di selezione: quando il difensore non entra in contatto con il
portatore non nasce nessun evento. Il gruppo "tiene" va quindi costruito anche
dal tracking, non solo dagli eventi.

**Due livelli, con campioni diversi.** È la parte che rende la pista praticabile:

| | Livello 1 — la scelta | Livello 2 — l'intenzione |
|---|---|---|
| Dati | `on_ball_engagement` + tracking + esito | + body pose (orientamento del busto) |
| Campione | **17.445 eventi su 20 partite** | **1.961 eventi su 2 partite** |
| Cosa regge | modello statistico, con split per partita | dimostrazione di metodo |

Il primo livello sta in piedi da solo: se il pose si rivelasse inutilizzabile,
non trascina giù il resto.

| | |
|---|---|
| **Dato che serve** | `on_ball_engagement` (sottotipo, `frame_start`/`frame_end`), tracking, esiti (`end_type`, `pressing_chain_end_type`) |
| **Cosa la minaccia** | costruire il gruppo "tiene" senza selezionarlo sull'esito; il 60% delle catene di pressing ha esito vuoto (948 su 1.567), da chiarire prima di usarlo come etichetta |
| **Perché regge** | gli ingaggi sono ravvicinati per definizione, quindi cadono dove tracking (87%) e pose (77%) funzionano meglio |
| **Figura** | una: esito atteso uscita vs mantenimento, o la mappa delle scelte |

**Il body pose è un'assenza verificata nella letteratura**, non solo poco usato:
su 281 pagine e 13 paper, "body pose" compare zero volte e "body orientation"
due, **entrambe come cosa che gli autori dichiarano di non aver usato** —
Bischofberger nei lavori futuri, Narizuka nei limiti. E SkillCorner spedisce già
un esempio di orientamento delle spalle in `src/features/pose_orientation.py`.

### A o B?

**Non sono più alternative.** Il disegno della pista B usa `τ_opp` per definire
le situazioni in cui un difensore poteva uscire: la pista A diventa un pezzo
della B, e la robustezza all'estrapolazione ne diventa la validazione.

L'ordine di lavoro che ne segue:

1. **`τ_opp` sui nostri dati** (replica di Narizuka) — serve a entrambe
2. **Che cosa c'è dentro `other`** — se è *Holding Ground*, il gruppo "tiene" è
   già in parte etichettato
3. **Che cosa significa l'esito vuoto** delle catene di pressing
4. **Il confronto uscita / mantenimento** a parità di situazione
5. **Robustezza**: lo stesso confronto ristretto alle situazioni interamente
   osservate
6. **Il body pose**, solo dopo, come dimostrazione di metodo sulle 2 partite

### Sul metodo: interpretabile per scelta, non per necessità

Finora questo piano diceva che con 20 partite una GNN non è praticabile.
**exPressV2 lo smentisce**: 36 partite, una GRU + GAT, risultati utilizzabili.

La stessa tabella però mostra che la rete guadagna **0,013 di AUC** su una
regressione logistica con le stesse feature (0,731 contro 0,718), senza
intervalli di confidenza e con 6 partite di test. Su questo volume un modello
lineare ottiene quasi tutto il segnale.

Quindi la ragione per gli approcci interpretabili non è che le alternative siano
impossibili: è che **non guadagnano abbastanza da giustificare la perdita di
interpretabilità**, davanti a una giuria che chiederà come è calcolato ogni
numero.
