# exPressV2: Contextual Evaluation of Individual Contributions from Pressing Situations in Football

**Riferimento:** Lee, Jo, Hong, Bauer, Ko (2025). MLSA 2025 (workshop ECML-PKDD),
Springer. [PDF del workshop](https://dtai.cs.kuleuven.be/events/MLSA25/papers/MLSA25_paper_248.pdf) ·
[codice](https://github.com/leemingo/express-v2)
**File:** `docs/exPressV2_Contextual_Evaluation_Pressing.pdf` (13 pagine). Non era
fra i paper originali: trovato cercando lavori sulla scelta del difensore.

## Domanda e metodo

Quanto è efficace un pressing, e quanto merito ha ciascun giocatore?

Individua automaticamente le situazioni di pressing con la **Pressing Intensity**
di Bekkers: c'è pressing quando almeno un difensore ha probabilità di
intercettare il portatore sopra 0,9. Per ogni situazione prende i 5 secondi
precedenti (10 frame, tutti i 22 giocatori e la palla) e con una rete
GRU + Graph Attention stima la probabilità che la squadra recuperi palla entro
5 secondi (`xPr`).

Due applicazioni: il merito viene diviso fra i giocatori in proporzione alla
pressione che ciascuno esercita, e un controfattuale sposta un difensore in
posizioni alternative e ricalcola `xPr`.

## Dati richiesti

| | |
|---|---|
| Tipo | tracking + eventi sincronizzati (BEPRO) |
| Frequenza | 30 Hz, riportati a 25 Hz |
| Partite usate | **36** (K League 1 2024): 24 training, 6 validazione, **6 test** |
| Completezza posizionale | tutti i 22 e la palla, per 5 secondi prima del pressing |
| Annotazioni extra | tipo dell'evento successivo; esito da dati evento (passaggio riuscito, tiro, recupero) |
| Campione | 7.800 situazioni di pressing (~217 per partita) |

## Verdetto di fattibilità

**Adattabile.** L'impostazione si trasferisce, la rete neurale no — e il motivo
sta nella tabella dei risultati del paper stesso.

**Il campione è comparabile al nostro.** 36 partite contro 20, e una rete che
funziona su quel volume. Questo smentisce l'affermazione, fatta finora in questo
progetto, che con venti partite una GNN non sia praticabile. Lo è.

**Ma non conviene, e lo dice la loro Tabella 1.** Con le stesse feature:

| modello | AUC | Brier | Log loss |
|---|---|---|---|
| Regressione logistica | 0,718 | 0,212 | 0,616 |
| Random forest | 0,717 | 0,181 | **0,544** |
| XGBoost | 0,723 | 0,216 | 0,623 |
| **exPressV2** (GRU + GAT) | **0,731** | **0,179** | 0,546 |

La rete guadagna **0,013 di AUC sulla regressione logistica**, senza intervalli
di confidenza, con un test set di 6 partite. E il testo afferma che exPressV2 è
il migliore "su tutte le metriche", ma sulla log loss la random forest fa meglio
(0,544 contro 0,546): **l'affermazione è falsa per una delle tre metriche**.

Letta onestamente, la tabella dice che su questo volume un modello lineare
ottiene quasi tutto il segnale disponibile. Per noi, che dobbiamo spiegare il
metodo a una giuria, è l'argomento più forte a favore degli approcci
interpretabili: non "la GNN non si può fare", ma "non guadagna abbastanza da
giustificare la perdita di interpretabilità".

**Dipendenza dai giocatori lontani: parziale.** L'individuazione del pressing
usa il difensore più vicino al portatore, cioè la zona dove i nostri dati sono
buoni (87% osservato entro 5 m). Ma l'input del modello sono **tutti i 22
giocatori** per 5 secondi, compresi quelli oltre i 40 m, osservati solo al 18%.
Sui nostri dati il modello imparerebbe anche dalle posizioni estrapolate.

**Velocità: da derivare.** Il modello usa posizione e velocità, e le colonne di
velocità di kloppy per questi dati sono vuote. A 10 fps invece di 25 le stime
sono più rumorose.

## Cosa servirebbe per replicarlo

Meno di quanto sembri, perché i pezzi esistono quasi tutti.

| Pezzo | Stato |
|---|---|
| Individuazione del pressing (Pressing Intensity) | **pronta** in `unravelsports` (`unravel.soccer.PressingIntensity`), di Bekkers |
| Esito del pressing | **pronto** in `dynamic_events`: `pressing_chain_end_type` (`regain` / `disruption`) e `end_type` del singolo ingaggio |
| Catene di pressing già segmentate | **pronte**: 1.567 catene su 20 partite (~78 per partita), mediana 3 ingaggi |
| Posizioni | **pronte** (`lib/data.py`) |
| Velocità | **da derivare** |
| Il modello GRU + GAT | codice pubblico, ma sconsigliato (vedi verdetto) |

**Attenzione alle due cose che non tornano ancora:**

- **Il 60% delle catene ha esito vuoto** (948 su 1.567). Può voler dire
  "fallita" o "non classificata", e le due letture danno tassi di successo
  molto diversi. Va chiarito prima di usarle come etichetta.
- Le catene di SkillCorner sono **~78 per partita**, i pressing di exPressV2
  **~217**: la definizione è molto diversa, e i numeri non sono confrontabili
  senza allinearla.

`unravelsports` è commentato nel `requirements.txt` perché si porta dietro
TensorFlow: per usare solo la Pressing Intensity conviene valutare se il modulo
si importa da solo.

## Il buco che lascia

Tre, in ordine di importanza per noi.

**1. Seleziona sull'azione, non sulla situazione.** Il paper analizza solo i
momenti in cui il pressing **c'è già** (intensità sopra 0,9). Non può quindi
dire niente sulla scelta di *non* pressare: la popolazione è selezionata proprio
sulla variabile che la pista B vuole valutare. È il limite più importante, ed è
esattamente lo spazio lasciato libero.

Un disegno che lo supera: **selezionare sulla situazione** — i momenti in cui un
difensore *avrebbe potuto* uscire, per esempio perché il suo tempo di arrivo sul
portatore era basso — e poi confrontare chi è uscito (`pressure`, `pressing`)
con chi ha tenuto (`other`, se si conferma che è *Holding Ground*, e i casi senza
ingaggio). Il tempo di arrivo è lo stesso `τ_opp` della
[scheda di Narizuka](pressione-tempo-arrivo.md): la pista A diventa un pezzo
della pista B invece di un'alternativa.

**2. L'attribuzione individuale è una regola, non un risultato del modello.** La
rete stima il successo della squadra; la divisione fra i giocatori è
proporzionale alla pressione che ciascuno esercita, sopra una soglia di 0,5. Chi
pressa molto dentro un pressing riuscito prende merito anche se il suo contributo
causale è nullo. Non distingue decisione da esecuzione.

**3. Il controfattuale è un esempio, non un metodo validato.** Un solo
difensore, spostato su una linea in 4 posizioni, in una sola situazione. Il
modello è addestrato su configurazioni reali, quindi le posizioni ipotetiche
possono cadere fuori da ciò che ha visto, e non c'è una misura di incertezza. La
"validazione esterna" dei giocatori migliori sono tre trasferimenti.

## Note

- Autori fra Saarland e Seoul; Pascal Bauer è coautore anche di *Blame is
  easier than praise*, in `docs/`. È un gruppo attivo sul tema.
- Costruisce su *exPress* (versione 1, su dati evento con 360) e sulla Pressing
  Intensity di Bekkers, che è in `docs/` e va schedata: ora la priorità è ancora
  più alta, perché è il metodo che definisce cosa conta come pressing.
- L'ablation mostra che il tipo di evento aggiunge poco (+0,006 di AUC): la
  differenza di vocabolario fra i loro eventi e i `dynamic_events` di SkillCorner
  pesa meno di quanto sembri.
