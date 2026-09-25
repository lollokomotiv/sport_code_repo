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
| **Dato che serve** | `on_ball_engagement` (sottotipo, `frame_start`/`frame_end`), tracking, `end_type` del possesso |
| **Cosa la minaccia** | definire l'alternativa controfattuale "tiene la posizione" senza un modello di ghosting |
| **Perché regge** | gli ingaggi sono ravvicinati per definizione, quindi cadono dove tracking (87%) e pose (77%) funzionano meglio |
| **Figura** | una: esito atteso uscita vs mantenimento, o la mappa delle scelte |

**Il body pose è un'assenza verificata nella letteratura**, non solo poco usato:
su 281 pagine e 13 paper, "body pose" compare zero volte e "body orientation"
due, **entrambe come cosa che gli autori dichiarano di non aver usato** —
Bischofberger nei lavori futuri, Narizuka nei limiti. E SkillCorner spedisce già
un esempio di orientamento delle spalle in `src/features/pose_orientation.py`.

### A o B?

Non sono alternative pulite. La robustezza all'estrapolazione (pista A) potrebbe
essere **la validazione** della pista B invece di un lavoro separato: se il
modello della scelta poggia su posizioni ricostruite, va detto quanto.

Da decidere prima di scrivere codice.
