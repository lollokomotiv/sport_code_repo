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

**Aggiornamento del 04/10/2026: nell'edizione 2.0 i track non ci sono più**, e il
tema per il calcio è *Defensive Positioning*. La tabella qui sopra resta come
traccia. Dettagli e fonte in [`notes/challenge.md`](../notes/challenge.md).

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

#### Dopo la replica (30/09/2026)

`τ_opp` è calcolato su 17.674 possessi delle 20 partite
([`reports/tau_opp.md`](../reports/tau_opp.md),
[`explorations/03`](../explorations/03-calcolo-tau-opp.ipynb)). Come domanda a
sé, la pista A si è ristretta:

- **l'avversario decisivo è estrapolato solo nel 3,3% dei possessi** (573), e
  più spesso dove la pressione è bassa: 2,6% nel quartile più pressato, 5,0% nel
  meno pressato. La telecamera segue la pressione. Buona notizia per chi usa la
  misura, poca sostanza per una submission. La scheda si aspettava circa il 13%:
  la differenza è da capire;
- **"ricalcolare sulle sole posizioni osservate" è mal posto**: togliere un
  avversario può solo alzare τ, quindi lo scostamento è positivo per costruzione.
  E la posizione vera dell'estrapolato non la conosciamo. Serve un altro disegno,
  per esempio il salto di posizione quando un giocatore rientra in inquadratura;
- **pressione → perdita è debole e non monotona** già su tutti i dati (ρ = −0,05
  nei possession play). Chiedersi se "sopravvive" sugli osservati non ha senso
  finché non si capisce perché non si replica;
- **la velocità pesa quanto l'estrapolazione, forse di più**: nel possesso
  d'esempio sposta τ di circa 0,1 s e cambia l'avversario decisivo. Le velocità
  vengono da differenze a 10 fps, con picchi fino a 13,9 m/s;
- **il confronto con `time_to_impact` (ρ = −0,86) non è ancora una validazione
  esterna**: se anche il loro modello è un tempo di arrivo, le due misure non
  sono indipendenti. Da verificare nel glossario SkillCorner.

Conclusione: A resta come passo e validazione della B (vedi «A o B?» sotto), non
come domanda autonoma.


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
   esempio perché il **suo** tempo di arrivo sul portatore τ_p era sotto una
   soglia. Non basta `τ_opp`, che è il minimo sugli avversari e dice solo chi
   arriva primo: serve il τ di ogni difensore (`lib.pressione.tempi_arrivo`).
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

### Revisione del 04/10/2026: prima `time_to_impact`, poi il pose

**Da:** [`explorations/03`](../explorations/03-calcolo-tau-opp.ipynb) §10,
[`explorations/04`](../explorations/04-pressing-intensity-unravel.ipynb),
[`notes/metriche-disponibili.md`](../notes/metriche-disponibili.md).

Tre misure della stessa famiglia sono ora sul tavolo: τ_opp (nostro, aperto),
`time_to_impact` (SkillCorner, chiuso, 5 classi) e il tempo di intercetto della
Pressing Intensity (unravelsports). Prima di costruire altro sopra τ_opp va
deciso se serve calcolarlo, o se la misura SkillCorner basta.

**Che cosa può sostituire `time_to_impact`, e che cosa no.** Nel piano τ_opp ha
due ruoli:

| Ruolo | Dove | `time_to_impact` lo sostituisce? |
|---|---|---|
| pressione sul portatore a inizio e fine possesso | replica di Narizuka | **da verificare**: è il test qui sotto |
| tempo di ogni difensore sul portatore (τ_p) | pista B, passo 2 | **no, per costruzione**: una classe per possesso, senza il difensore |

Per questo **la pista B è sospesa**, non abbandonata: richiede τ_p, che solo il
calcolo nostro fornisce. Si riprende, o si chiude, dopo il test e la scelta della
tesi.

#### Il test di equivalenza (protocollo fissato prima dei risultati)

*"Equivalente"* vuol dire: `time_to_impact` può sostituire τ_opp **come pressione
sul portatore a inizio e fine possesso**. Le definizioni sono quelle di
[`scripts/report_tau.py`](../scripts/report_tau.py), così il confronto è alla
pari: progressione, zone, perdita di palla, esclusione dei portieri. Tutti i
confronti usano **gli stessi possessi**, cioè quelli dove esistono entrambe le
misure.

1. **Pressione → avanzamento** (paper, Fig. 6): Spearman di ciascuna misura con la
   progressione, nel complesso e nelle tre zone di campo.
2. **Pressione → perdita** (paper, Fig. 7a): Spearman di ciascuna misura con la
   perdita, separando possession play e direct play.
3. **Informazione aggiuntiva, nei due sensi**: dentro ogni classe di
   `time_to_impact`, τ_opp è ancora associato all'esito? E dentro ogni quintile di
   τ_opp, lo è ancora `time_to_impact`? Si riporta ρ per strato e la media pesata
   per numerosità.
4. **Copertura**: quanti possessi perde chi usa solo `time_to_impact`, e di che
   tipo.

Criterio di uscita: i quattro confronti sono calcolati e riportati in
`reports/equivalenza_tti.md`, prodotto da `scripts/equivalenza_tti.py`.
Nessuna soglia sul risultato; l'interpretazione si fa insieme.

**Numeri del 04/10/2026** ([`reports/equivalenza_tti.md`](../reports/equivalenza_tti.md)),
interpretazione ancora da fare:

| Esito | ρ τ_opp | ρ `time_to_impact` | τ_opp dentro le classi tti | tti dentro i quintili τ_opp |
|---|---|---|---|---|
| avanzamento (N = 12.804) | 0,277 | −0,154 | 0,294 | **+0,142** |
| perdita, possession play (N = 10.229) | −0,056 | 0,152 | **+0,103** | 0,165 |
| perdita, direct play (N = 2.666) | −0,066 | 0,146 | **+0,072** | 0,155 |

- Nessuna delle due misure domina: τ_opp è più legata all'avanzamento,
  `time_to_impact` alla perdita (dal 4,5% al 18,4% fra la classe 1 e la 5).
- **Due inversioni di segno** negli strati, in grassetto: a parità di τ_opp, più
  pressione SkillCorner va con più avanzamento; a parità di classe SkillCorner, più
  tempo τ_opp va con più perdite. Prima di leggerle come effetti veri va escluso
  un confondente: zona di campo e lunghezza del passaggio sono i primi candidati.
- Usando solo `time_to_impact` si perdono 1.521 possessi all'inizio, il 78% dei
  quali direct play (il 26,9% di tutti i direct play).

#### Dopo il test: una tesi con il pose e le metriche SkillCorner

Vincoli noti in anticipo:

- **Il body pose copre 2 partite**, 45,7% utilizzabile
  ([`explorations/02`](../explorations/02-body-pose.ipynb)). In locale c'è solo la
  `1925299`; la `1996435` va scaricata. Una tesi sul pose è una **dimostrazione
  di metodo**, non un risultato statistico.
- **Le metriche SkillCorner possono fare da esito o da contesto**: pressione,
  `reception_difficulty`, `forward_momentum`, `xloss`, EPV. Il pose aggiunge ciò
  che nessuna di queste contiene: l'orientamento del corpo.

Candidate, da scegliere dopo il test:

- l'orientamento di **chi riceve**, prima della ricezione, rispetto alla
  difficoltà e all'esito del possesso;
- l'orientamento del **difensore** all'ingaggio rispetto all'esito: è il livello 2
  della pista B, che però senza il livello 1 resta senza la parte statistica.

Il risultato del test decide quale misura di pressione entra come contesto.

### A o B?

*L'ordine di lavoro di questa sezione è superato dalla revisione del 04/10/2026
qui sopra. Resta come traccia del ragionamento.*

**Non sono più alternative.** Il disegno della pista B usa `τ_opp` per definire
le situazioni in cui un difensore poteva uscire: la pista A diventa un pezzo
della B, e la robustezza all'estrapolazione ne diventa la validazione.

L'ordine di lavoro che ne segue:

1. ~~**`τ_opp` sui nostri dati** (replica di Narizuka)~~ — fatto il 30/09/2026,
   vedi «Dopo la replica» nella pista A
2. **τ per difensore e sensibilità alle velocità**: quanto cambia τ_p con
   velocità azzerate o disturbate, e quanto spesso cambia *chi* arriva primo. Nel
   possesso d'esempio tre avversari arrivano entro 0,1 s l'uno dall'altro: se
   l'identità del difensore "che poteva uscire" dipende da errori di quell'ordine,
   il gruppo di trattamento è instabile. Va misurato prima di costruirlo
3. **Che cosa c'è dentro `other`** — se è *Holding Ground*, il gruppo "tiene" è
   già in parte etichettato
4. **Che cosa significa l'esito vuoto** delle catene di pressing
5. **Il confronto uscita / mantenimento** a parità di situazione
6. **Robustezza**: lo stesso confronto ristretto alle situazioni interamente
   osservate. Dal 3,3% misurato sulla replica ci si aspetta una verifica breve,
   perché gli ingaggi cadono vicino al portatore
7. **Il body pose**, solo dopo, come dimostrazione di metodo sulle 2 partite

Nota per ogni join fra eventi e tracking: le coordinate degli eventi sono quelle
del tracking ruotate di 180° quando la squadra attacca `right_to_left`
([`notes/dati/dynamic-events.md`](../notes/dati/dynamic-events.md)).

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
