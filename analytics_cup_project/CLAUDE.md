# CLAUDE.md — Analytics Cup

Preparazione alla **SkillCorner X PySport Analytics Cup**, su dati di tracking
broadcast open source. A differenza degli altri progetti qui c'è un giudice
esterno, regole formali e una scadenza.

Questo cambia il criterio: altrove "regge se qualcuno del settore lo esamina" è
un'aspirazione, qui qualcuno lo esaminerà davvero con una griglia.

---

## 1. Come si lavora qui

Questo progetto si esplora **insieme all'utente**, non si consegna finito. Il
modo di procedere conta quanto il risultato.

**Iterazioni piccole.** Un obiettivo per volta, quello che l'utente dà di volta
in volta. Poche righe di codice, non script lunghi o pipeline complete in un
colpo solo. Dopo ogni passo fermarsi e dire: cosa hai fatto, cosa hai trovato,
cosa proponi come passo successivo. Non incatenare molti passi senza chiedere
feedback.

**Spiegare prima di eseguire.** Prima di lanciare del codice, due righe sulla
logica dietro. L'utente vuole seguire il ragionamento, non ricevere un risultato
"chiuso".

**Commentare i risultati nel merito.** Dopo l'esecuzione: cosa dicono i dati,
cosa implicano per la struttura del dataset, quali domande nuove aprono. Un
output stampato e non letto non è un passo completato.

**Il codice lo scrivi tu.** L'utente guida con obiettivi in linguaggio naturale,
tu traduci in codice — query, analisi, esplorazioni — e lo esegui.

**Proporre, non assumere.** A fine iterazione suggerisci esplicitamente il passo
successivo possibile e aspetta conferma prima di procedere. In caso di ambiguità,
fai una domanda mirata invece di svolgere una mole di lavoro su un'assunzione non
verificata.

**Eccezione: i compiti con un criterio di uscita deterministico** — test che
passano, file prodotti, conteggi verificabili — si possono affidare a un `/goal`
invece che a passi singoli. Tre condizioni:

- **test e parametri si fissano prima**, con l'utente, e il loop non li modifica.
  Se un test sembra sbagliato ci si ferma e se ne discute, non lo si corregge;
- **il criterio riguarda il processo, non il risultato**: "la correlazione è
  calcolata e riportata", mai "la correlazione supera 0,7". Una soglia sul
  risultato insegna al loop a raggiungerla;
- **alla fine si torna ai turni**: il loop riporta i numeri, non li interpreta.
  L'interpretazione si fa insieme.

### L'obiettivo primario è capire il dataset

La priorità non è arrivare in fretta a un output finale, ma che l'utente
costruisca progressivamente una comprensione di:

- **Struttura dei dati** — quali file esistono, quali colonne e tipi, quale
  granularità, quali chiavi di join. I quattro file di una partita sono già
  documentati in `notes/dati/`: leggi lì prima di riesplorare. Restano da coprire
  gli aggregati stagionali e il body pose.
- **Semantica del dominio** — cosa rappresentano concretamente le metriche
  SkillCorner (distanza percorsa, sprint count, PSV99, off-ball runs, metriche di
  pressing e di spazio) e come si legano al contesto sportivo: ruolo, fase di
  gioco, stato della partita. I glossari stanno in `notes/risorse.md`.
- **Qualità e limiti** — valori mancanti, outlier, copertura diversa fra partite e
  giocatori, bias di campionamento tipici degli open data. Il limite principale è
  già misurato: vedi §4.
- **Convergenza verso una domanda** — le esplorazioni devono restringersi via via
  verso una domanda di ricerca definita, non accumularsi scollegate. Il filo lo
  tiene `plans/01-scegliere-la-domanda.md`.

## 2. Due repo, non due cartelle

Il lavoro vive in **due posti distinti**, e la distinzione non è cosmetica.

| | Dove | Cosa | Git |
|---|---|---|---|
| **Workbench** | `analytics_cup_project/` (questa cartella) | esplorazioni, piani, idee scartate, note | versionato in `sport_code_repo` |
| **Submission** | `analytics_cup_project/submission/` | il fork del template PySport | **repo separato**, in `.gitignore` |

La submission è un **fork di `PySport/analytics_cup_<track>`**, con la sua
history e il suo remote. Sta dentro questa cartella solo per comodità sul disco
(gli import dal workbench funzionano senza acrobazie), ma git tiene le due
storie separate e deve continuare a farlo.

**Non aggiungere `submission/` al git di `sport_code_repo`.** Se lo fai, git lo
tratta come embedded repo e il fork smette di essere pushabile dove serve.

Il workbench non è materiale di scarto: racconta come ci sei arrivato, che è la
parte che il fork non può mostrare. Le idee che non funzionano si annotano, non
si cancellano.

## 3. I dati non si copiano mai nella submission

Il regolamento è esplicito: *"Make sure your GitHub repository does **not**
contain big data files."* Nella submission i dati si caricano da remoto, in una
riga:

```python
from kloppy import skillcorner

dataset = skillcorner.load_open_data(match_id=1886347, coordinates="skillcorner")
```

Nel workbench invece si usa il **clone locale**, perché iterare su 20 partite
riscaricando da GitHub ogni volta è lento:

```
/Users/lorenzoguercio/Documents/Projects/sport_data/skillcorner-opendata/data/
```

Due modalità, due scopi: locale per la velocità, `load_open_data` per la
riproducibilità. Il notebook di submission deve girare su una macchina pulita che
non ha mai visto il tuo filesystem, ed è un requisito scritto, non una buona
pratica.

Quando porti codice dal workbench alla submission il caricamento dati è la cosa
da riscrivere. È l'unico punto in cui le due versioni divergono di proposito,
quindi va isolato in una funzione sola invece di spargere percorsi nel codice.

## 4. I limiti di questi dati

Sono i vincoli che decidono quali domande sono ponibili. Vanno conosciuti prima
di scegliere la domanda, non scoperti dopo.

**È tracking da broadcast, non da telecamere fisse.** `is_detected` distingue le
posizioni osservate da quelle estrapolate: fuori inquadratura la posizione è
stimata. Misurato su una partita intera, **solo il 59% è osservato** — 87% entro
5 m dalla palla, 18% oltre i 40 m, 15% per il portiere (vedi
`explorations/00-quanto-e-osservato.ipynb`). Kloppy non espone `is_detected`:
per misurarlo serve il JSONL grezzo.

Quindi i dati reggono le domande centrate sull'azione, e reggono male quelle
sull'organizzazione collettiva lontano dalla palla.

**Venti partite.** Un campione così piccolo esclude quasi tutto il machine
learning supervisionato. Il vincitore dell'edizione 2026 ha scelto
l'ottimizzazione matematica perché con dieci partite il ML non reggeva, e lo ha
scritto nell'abstract. Un metodo scelto per il vincolo reale vale più di un
modello sovradimensionato.

**A-League australiana 2024/25.** Un campionato solo, una stagione sola. Ogni
conclusione è condizionata a quel contesto.

**10 fps, metri, origine al centro del campo.** L'asse x è il lato lungo. Le
dimensioni variano per partita — 104×68, 105×68 e 106×68 fra le 20 disponibili —
e stanno in `{id}_match.json`: vanno lette, non assunte.

I limiti si dichiarano nei risultati. Con un abstract da 500 parole la tentazione
di tagliarli è forte, ed è il taglio sbagliato.

## 5. I vincoli formali sono stretti

Dal template di submission:

- `submission.ipynb` nella **root** del fork, **max 2000 parole**
- tutto il resto del codice in `src/`, **importato** nel notebook — il notebook
  è narrazione e chiamate, non implementazione
- abstract nel `README.md`, **max 500 parole**, struttura fissa:
  Introduction / Methods / Results / Conclusion
- **max 2 figure, oppure 2 tabelle, oppure 1 figura + 1 tabella**
- consegna su [pretalx.pysport.org](https://pretalx.pysport.org)
- solo strumenti **open source**, partecipazione **individuale**

Violarli *"may result in a point deduction or disqualification"*.

Il limite di 2 figure condiziona **che tipo di risultato** puoi presentare, non
solo l'impaginazione. Tienilo presente quando scegli la domanda: un risultato che
ha bisogno di sei pannelli per essere capito non è presentabile qui.

## 6. Struttura del workbench

```
plans/          filoni di lavoro numerati per priorità, con README.md come indice
explorations/   notebook esplorativi, numerati, una domanda ciascuno
lib/            codice condiviso: dati, modelli, figure (social.py: media per X)
tests/          test scritti prima del codice: criteri di stop dei /goal
scripts/        calcoli su tutte le partite, eseguibili da riga di comando
reports/        output dei calcoli, in markdown, da leggere insieme
notes/          risorse e link
notes/dati/     com'è fatto il dataset, un documento per tipo di file
notes/letteratura/   una scheda per paper, con verdetto di fattibilità
docs/           i paper originali e le Research Directions (non versionati)
figures/        output visivi; figures/post/ per i media dei post
posts/          testi dei post pubblicati, con fonte e commit (skill post-x)
submission/     il fork (gitignored)
```

Venv: `.venv/` in questa cartella (Python 3.12.8, da `requirements.txt`, in `.gitignore`).

Come in `xgoals_project/`: quando emerge un problema che non si chiude subito,
**va annotato nel piano pertinente** invece di restare in una conversazione.

Gli `explorations/` possono essere sporchi — è il loro scopo. Ma ognuno deve
avere in cima una cella markdown che dice **quale domanda sta ponendo**, perché
fra due mesi non te lo ricordi.

## 7. Stack

Le librerie installate e a cosa servono stanno in `requirements.txt`, commentate,
e in `notes/risorse.md` con i riferimenti alle fonti.

`kloppy` è di PySport, cioè dell'organizzazione che co-organizza la gara, ed è lo
standard che il tutorial ufficiale usa. Parsare il `jsonl` a mano quando esiste il
loro deserializer è una scelta che ti verrà chiesta di giustificare.

Coerentemente con il resto del repo: commenti e testo in italiano nel workbench.
**Nella submission tutto in inglese** — README, notebook, docstring, nomi delle
variabili.

## 8. Prima di accettare un numero

Vale la regola del resto del repo, che qui pesa di più perché il pubblico è
tecnico e ostile per mestiere:

- gli ordini di grandezza sono plausibili per il calcio?
- il risultato regge se cambio partita, o è un artefatto di una sola?
- quanta parte del dato che ho usato era estrapolata invece che osservata?

Quando un numero è implausibile, **dirlo esplicitamente** invece di lasciarlo
passare.
