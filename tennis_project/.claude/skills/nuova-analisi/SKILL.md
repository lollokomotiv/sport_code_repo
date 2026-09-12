---
name: nuova-analisi
description: >-
  Avvia e conduce una nuova analisi sui dati tennis in tennis_project. Usare
  quando l'utente propone un confronto tra due giocatori ("Alcaraz contro
  Sinner", "come se la giocano X e Y"), chiede di verificare una tesi sui dati,
  dice "nuova analisi", "analizziamo", "vediamo se", "mi interessa capire",
  oppure nomina due tennisti insieme. Copre il flusso completo — copertura del
  campione, scelta delle analisi già fatte sul matchup da riusare, verifica
  della tesi, script, grafico, README dell'analisi e del matchup.
---

# Nuova analisi

## Regola che vale per tutti i passi: ogni numero nasce da codice committato

**Qualunque cifra o grafico mostrato all'utente deve venire da codice che sta in
`analyses/<matchup>/<analisi>/run.py`.** Mai da un comando lanciato al volo
nella shell e poi perso.

Il motivo è pratico, non formale: un numero calcolato a mano non si può
ricontrollare, non si aggiorna quando i dati cambiano, e nessuno sa più con
quali filtri era stato ottenuto. In un progetto di portfolio è la differenza fra
un risultato difendibile e un aneddoto.

In concreto:

- esplorare nella shell va benissimo — è il modo giusto per capire i dati;
- ma **appena un numero viene comunicato**, il calcolo che lo produce va scritto
  nello script, e lo script va rieseguito per verificare che dia lo stesso valore;
- questo vale anche per i calcoli di contorno: test di significatività,
  intervalli di confidenza, confronti citati di sfuggita;
- `run.py` deve stampare tutto ciò che compare nel README. Se un numero è nel
  README ma non nell'output dello script, manca del codice.

**Nessun numero cablato nei testi dei grafici.** Titoli, sottotitoli ed etichette
si costruiscono con f-string dai dati calcolati. Un valore scritto a mano nel
sottotitolo sopravvive all'aggiornamento dei dati e diventa falso in silenzio —
è già successo qui, con un "+9,6" mentre il calcolo dava +9,8.

## Il flusso, in ordine

Non saltare passi. Il passo 1 esiste perché senza di esso il passo 4 produce
numeri distorti senza che si veda.

### 1. Copertura — prima di qualunque altra cosa

Quando l'utente nomina due giocatori, la **prima** risposta è sempre questa,
non un'analisi:

- **quanti incontri esistono davvero** → TennisMyLife (`loaders.load_tml`)
- **quanti sono annotati colpo per colpo** → Match Charting Project
- **chi ha vinto quelli mancanti**

Se i match non annotati pendono da una parte, il campione dettagliato è
distorto: dillo subito e quantificalo. Esempio reale da `analyses/alcaraz-paul/h2h/`:
H2H vero 6-2, campione annotato 3-2, e i tre mancanti tutti vinti dallo stesso
giocatore.

Il modo più rapido è il comando già pronto, che stampa la tabella, i due H2H,
l'elenco dei match mancanti e l'avviso quando sono tutti vinti dalla stessa
persona:

```bash
python3 -m lib.catalog --player "Carlos Alcaraz" --vs "Tommy Paul"
```

Da codice, per proseguire con l'analisi:

```python
from lib import loaders

d = loaders.h2h("Carlos Alcaraz", "Tommy Paul")   # match ufficiali
d[d.charted]                                       # quelli annotati colpo per colpo
d.attrs["annotati_senza_ufficiale"]                # esibizioni e Challenger fuori dal circuito
```

Presenta il risultato come tabella con una colonna `annotato` sì/no, più le due
righe di riepilogo (H2H completo / H2H annotato).

### 2. Il matchup ha già delle analisi? Chiedi quali usare

Le analisi sono raggruppate per matchup: `analyses/<matchup>/<analisi>/`, con i
cognomi in ordine alfabetico (`alcaraz-zverev`, mai `zverev-alcaraz`). Controlla
se la cartella del matchup esiste già.

**Se esiste**, leggi il suo `README.md`: ha la copertura dell'H2H e la tabella
delle analisi già fatte, con la domanda e il risultato in una riga. Poi,
insieme al quadro di copertura, **chiedi all'utente quali di queste analisi
vuole usare** per il lavoro nuovo. Elencale tutte con la loro riga di
risultato; se sono al massimo quattro usa `AskUserQuestion` con
`multiSelect: true`, altrimenti elencale nel testo e chiedi. Nessuna analisi è
sempre una risposta valida.

Non decidere al posto dell'utente, né in un senso né nell'altro: caricare tutto
riempie la conversazione di numeri che non c'entrano, ignorare tutto porta a
rifare lavoro già validato o a contraddirlo senza accorgersene.

Per ogni analisi scelta:

- **rileggi README e output di `run.py`**, non il ricordo della conversazione
  in cui è nata: i limiti sono lì;
- **riusa le scelte già difese** — riferimento, filtri, trattamento dei
  confondenti — invece di reinventarle: due analisi dello stesso matchup con
  riferimenti diversi producono numeri che il lettore confronterà comunque;
- **se la nuova domanda ha già risposta** in una di esse, dillo prima di
  scrivere codice. Se la estende, decidi con l'utente se è una sezione in più di
  quell'analisi o una sottocartella nuova: la regola resta una domanda per
  cartella;
- **confronta la copertura** del passo 1 con quella scritta nel README del
  matchup. Se nel frattempo si è giocato un altro incontro, le analisi scelte
  vanno rieseguite prima di citarne i numeri.

**Se non esiste**, creala con un `README.md`: la copertura del passo 1 e una
tabella vuota delle analisi (`Cartella | Domanda | Risultato in una riga |
Fonte`). Le analisi su un giocatore solo, contro tutto il circuito, vanno in
`analyses/<cognome>/`.

### 3. La tesi la scrive l'utente

Dopo il quadro di copertura, **fermati e aspetta**. L'utente formula una tesi:
"secondo me Paul perde perché non regge gli scambi lunghi", "Sinner serve meglio
sul veloce". Non anticiparla e non proporne una al suo posto — al massimo
segnala quali tesi i dati disponibili *non* possono verificare.

Se la tesi è ambigua, chiedi cosa la renderebbe vera o falsa **in termini di
numeri** prima di scrivere codice.

### 4. Lo script che la verifica

Piccolo e mirato: la tesi è una domanda sola, lo script risponde a quella.

- vive in `analyses/<matchup>/<analisi>/run.py`, copiato da `analyses/_template/`
  (`cp -r analyses/_template analyses/<matchup>/<analisi>`); il nome
  dell'analisi è il tema, senza ripetere i giocatori (`smorzate`, non
  `alcaraz-zverev-smorzate`)
- carica **solo** da `lib/`, mai download propri, mai percorsi relativi
- stampa il quadro di copertura del passo 1 **prima** dei numeri della tesi
- chiude dichiarando la dimensione del campione

**Prima di confrontare due gruppi, chiediti cosa li rende diversi oltre alla
tesi.** Un confronto grezzo misura quasi sempre il calendario invece
dell'avversario. I confondenti ricorrenti in questo dominio:

| Confondente | Perché morde |
|---|---|
| **superficie** | quasi ogni metrica di gioco cambia fra terra, cemento ed erba: le smorzate vanno da 2,5 a 3,7 ogni 100 colpi, gli ace da 5% a 10% |
| **formato** | al meglio dei 5 i punti sono più numerosi e la gestione cambia |
| **anno** | i giocatori evolvono, e il MCP annota molto di più le stagioni recenti |
| **livello del torneo** | uno Slam e un 250 non hanno lo stesso avversario medio |

Il rimedio standard è confrontare **osservato contro atteso**: si calcola il
tasso del soggetto su ciascun livello del confondente, lo si applica ai volumi
effettivi, e si guarda il rapporto. Poi si legge quel rapporto **in
distribuzione**, non da solo: il numero dice qualcosa solo rispetto agli altri
casi confrontabili. Vedi `analyses/alcaraz-paul/smorzate/run.py`, dove il
confronto grezzo dava ragione alla tesi e quello controllato la smentiva.

**Escludi il soggetto dal riferimento**: se il tasso atteso è calcolato
includendo i match che stai valutando, il confronto è circolare.

**Controlla le metriche adiacenti prima di concludere, non dopo.** Quasi ogni
domanda ammette più modi di contare la stessa cosa, e sceglierne uno dopo aver
visto i risultati è il modo più facile per pubblicare un effetto che non esiste.
Nell'analisi delle smorzate, i soli *vincenti* davano 16,7% contro 29,7%
(p = 0,050, apparentemente forte); aggiungendo gli *errori forzati* — l'altro
modo in cui una smorzata porta il punto — il divario scendeva a 31,2% contro
40,0% (p = 0,22, rumore). Enumera gli esiti possibili **prima** di guardare i
numeri, e se le metriche vicine non concordano, il risultato è che non c'è un
risultato.

La tesi può risultare **falsa**: è un risultato, si scrive nel README e non si
cerca un taglio dei dati che la salvi.

### 5. Chiedi se serve un grafico

Quando i numeri sono pronti, **chiedi sempre**:

> Vuoi che rappresenti questo graficamente?

Non darlo per scontato in nessuna delle due direzioni. Se la risposta è sì,
**carica la skill `dataviz` prima di scrivere codice del grafico**, e proponi la
forma adatta al tipo di affermazione:

| La tesi riguarda… | Forma |
|---|---|
| un confronto fra due giocatori su più metriche | barre appaiate orizzontali |
| un andamento nel tempo o per match | linea con punti sui match reali |
| una distribuzione (lunghezza scambi, direzione servizio) | barre o heatmap del campo |
| due dimensioni insieme (volume vs efficacia) | scatter, o due dot plot affiancati con le stesse righe |
| un tasso che ha senso solo rispetto a un riferimento | **differenziale**: sottrai la linea di base e centra l'asse sullo zero |

I grafici vanno in `analyses/<matchup>/<analisi>/figures/`, e **sono versionati**: il
`.gitignore` ignora le immagini ovunque tranne lì, perché sono il risultato
visibile del lavoro e senza di loro i README mostrano link rotti su GitHub.

**Guarda il PNG dopo averlo generato.** Il validator della palette controlla i
colori, non la disposizione: le collisioni fra etichette, il testo che esce dal
bordo e i titoli disallineati si vedono solo aprendo il file.

### 6. README dell'analisi e del matchup

Dal template: **domanda → dati e filtri → metodo → risultato → limiti → come si
riproduce**. Nei limiti va sempre la distorsione misurata al passo 1.

Poi aggiungi l'analisi alla tabella del `README.md` del matchup, con il
risultato **in una riga**: è quella riga che la prossima volta, al passo 2,
servirà all'utente per scegliere se riusarla. Aggiorna anche l'indice in
`analyses/README.md`.

---

## Chiavi e collegamenti

Le fonti **non condividono una chiave**. Questo è il punto in cui si sbaglia.

| Fonte | Chiave | Data |
|---|---|---|
| TennisMyLife | `tml_match_id`, costruita da `load_tml()` | `tourney_date` = **inizio del torneo** |
| MCP | `match_id`, nel file | `date` = **giorno del match** |

- I due si collegano con `loaders.link_mcp_to_tml()`: coppia di giocatori
  normalizzata + turno + finestra di date (−3/+21 giorni). Aggancia il 92,5%
  dei match MCP; il resto sono Challenger, qualificazioni ed esibizioni.
- **I nomi vanno sempre passati da `loaders.normalize_name()`**: il MCP scrive
  `Felix Auger Aliassime` e `Christopher Oconnell`, TML `Felix Auger-Aliassime`
  e `Christopher O'Connell`.
- Dentro il MCP tutto si aggancia a `match_id`, che è leggibile:
  `20250603-M-Roland_Garros-QF-Carlos_Alcaraz-Tommy_Paul`.
- Nei merge usa `validate=`: è così che sono emersi i bug sulle chiavi.

---

## Riferimento: i file di input

### TennisMyLife — `data/raw/tml/<tour>/<anno>.csv`

Una riga per match. 200.058 match ATP (1968→2026). Fonte primaria per
l'universo dei match e le statistiche ufficiali.

| Colonna | Tipo | Contenuto |
|---|---|---|
| `tourney_id` `tourney_name` | str | identificativo e nome del torneo |
| `surface` | str | `Hard` `Clay` `Grass` |
| `indoor` | str | `I`/`O` — 8% vuoto |
| `draw_size` | float | dimensione del tabellone |
| `tourney_level` | str | `250` `500` `M` Masters `G` Slam `D` Davis `F` Finals `A` `O` Olimpiadi |
| `tourney_date` | int `YYYYMMDD` | **inizio del torneo**, non del match |
| `match_num` | float | progressivo nel torneo — **16,6% vuoto** |
| `winner_*` / `loser_*` | | `id` `seed` `entry` `name` `hand` `ht` `ioc` `age` `rank` `rank_points` |
| `score` | str | es. `7-6(6) 6-4 7-5` |
| `best_of` `round` `minutes` | | `minutes` 6,5% vuoto |
| `w_ace` `w_df` | float | ace e doppi falli |
| `w_svpt` | float | punti giocati al servizio |
| `w_1stIn` | float | prime palle **in campo** |
| `w_1stWon` `w_2ndWon` | float | punti vinti con la prima / con la seconda |
| `w_SvGms` | float | game al servizio |
| `w_bpSaved` `w_bpFaced` | float | palle break salvate / affrontate |
| `l_*` | | gli stessi per chi ha perso |

Le statistiche `w_*`/`l_*` sono vuote nel **6,6%** dei match del 2025 e quasi
sempre prima del 1991. `winner_seed` è vuoto nel 59%, `entry` nell'89%: sono
assenze normali (non tutti sono testa di serie), non dati mancanti.

**Punti di seconda** = `svpt − 1stIn`, e **includono i doppi falli**.

### MCP indice — `data/raw/mcp/charting-{m,w}-matches.csv`

Una riga per match annotato. 7.564 maschili, ~3.000 femminili.

Colonne grezze: `match_id`, `Player 1`, `Player 2`, `Pl 1 hand`, `Pl 2 hand`,
`Date`, `Tournament`, `Round`, `Time`, `Court`, `Surface`, `Umpire`, `Best of`,
`Final TB?`, `Charted by`.

`load_mcp_matches()` le rinomina in snake_case (`player_1`, `date`, `charted_by`…),
converte la data e **scarta le righe malformate** (2 su 7.564, colonne
disallineate, una duplica un `match_id`) con un warning.

`Charted by` è l'annotatore: utile per controllare se un effetto dipende da chi
ha annotato.

### MCP statistiche — `data/raw/mcp/charting-{m,w}-stats-<Nome>.csv`

Una riga per **match × giocatore × livello**. Il livello sta nella colonna `row`
(in `Overview` si chiama `set`) e **significa una cosa diversa in ogni file**.
C'è sempre una riga totale: **sommare tutte le righe conta gli stessi punti più
volte**.

| File | Livelli (`row`) | Colonne |
|---|---|---|
| **Overview** | `set`: `1`…`5`, `Total` | `serve_pts` `aces` `dfs` `first_in` `first_won` `second_in` `second_won` `bk_pts` `bp_saved` `return_pts` `return_pts_won` `winners` `winners_fh` `winners_bh` `unforced` `unforced_fh` `unforced_bh` |
| **ServeBasics** | `1` `2` `Total` (per prima/seconda) | `pts` `pts_won` `aces` `unret` `forced_err` `pts_won_lte_3_shots` `wide` `body` `t` |
| **ServeDirection** | `1` `2` `Total` | `deuce_wide` `deuce_middle` `deuce_t` `ad_wide` `ad_middle` `ad_t` + `err_net` `err_wide` `err_deep` `err_wide_deep` `err_foot` `err_unknown` |
| **ServeInfluence** | `1` `2` | `pts` `won_1+` … `won_10+`: quota di punti vinti quando lo scambio arriva ad almeno N colpi |
| **Rally** | `1-3` `4-6` `7-9` `10` `Total`, più i suffissi `-1`/`-2` (prima/seconda di servizio) | `server` `returner` `pts` `pl1_won` `pl1_winners` `pl1_forced` `pl1_unforced` + `pl2_*` |
| **KeyPointsServe** | `BP` `GP` `Deuce` `STotal` | `pts` `pts_won` `first_in` `aces` `svc_winners` `rally_winners` `rally_forced` `unforced` `dfs` |
| **KeyPointsReturn** | `BPO` `GPF` `DeuceR` `RTotal` | `pts` `pts_won` `rally_winners` `rally_forced` `unforced` |
| **ReturnDepth** | direzione `4` `5` `6`; lato `A` `D`; combinati `4A` `4D`…; colpo `fh` `bh` `gs` `sl`; servizio `v1st` `v2nd`; `Total` | `returnable` `shallow` `deep` `very_deep` `unforced` `err_*` |
| **ReturnOutcomes** | come sopra, più `7` `9` `89` | `pts` `pts_won` `returnable` `returnable_won` `in_play` `in_play_won` `winners` `total_shots` |
| **NetPoints** | `NetPoints` `Approach` `NetPointsRallies` `ApproachRallies` | `net_pts` `pts_won` `net_winner` `induced_forced` `net_unforced` `passed_at_net` `passing_shot_induced_forced` `total_shots` |
| **SnV** | `SnV` `SnV1st` `SnV2nd` `nonSnV` `nonSnV1st` `nonSnV2nd` | `snv_pts` `pts_won` `aces` `unret` `return_forced` `net_winner` … |
| **ShotTypes** | 32 codici, **gerarchici** — vedi sotto | `shots` `pt_ending` `winners` `induced_forced` `unforced` `serve_return` `shots_in_pts_won` `shots_in_pts_lost` |
| **ShotDirection** | `F` `B` `S` `Total` | `crosscourt` `down_middle` `down_the_line` `inside_out` `inside_in` |
| **ShotDirOutcomes** | `{F,B,S}-{XC,DTL,DTM,II,IO}` | `shots` `pt_ending` `winners` `induced_forced` `unforced` `shots_in_pts_won` `shots_in_pts_lost` |
| **SvBreakTotal** / **SvBreakSplit** | `4` `5` `6` `a` `d` e combinati | punti per direzione del servizio; `Split` separa prima e seconda |

**Codici di direzione del servizio** (verificati incrociando `ServeBasics`, che
ha le colonne esplicite): **`4` = wide, `5` = body, `6` = T**. `A` = ad court,
`D` = deuce court. Nei file di risposta le proporzioni differiscono un po'
perché lì si contano solo i servizi ribattibili, e gli ace si concentrano su
wide e T.

**Codici colpo:** `F` dritto, `B` rovescio, `S` slice — in `ShotDirection` questi
tre partizionano `Total`. In **`ShotTypes` no**: i 32 codici sono categorie
annidate, non una partizione. `Base` (94,6% dei colpi) e `Net` (5,4%) dividono
il totale; `Gs` (80,5%) sono i colpi a rimbalzo, di cui `F` (44,2%) e `B`
(36,3%); `Sl` slice, `Vo` volée (`V`/`Z` dritto/rovescio), `Dr` smorzata, `Lo`
pallonetto. **Non sommare mai righe diverse di questo file.**

### MCP punto per punto — `data/raw/mcp/charting-{m,w}-points-<era>.csv`

Una riga per punto, ~1,28 milioni per il maschile. Ere: `to-2009`, `2010s`, `2020s`.

| Colonna | Contenuto |
|---|---|
| `match_id` `Pt` | match e numero progressivo del punto |
| `Set1` `Set2` `Gm1` `Gm2` `Pts` | punteggio **prima** del punto (`Pts` es. `15-30`) |
| `Gm#` `TbSet` | game progressivo, se è un tie-break |
| `Svr` | chi serve: `1` o `2` (riferito a `player_1`/`player_2` dell'indice) |
| `1st` `2nd` | **sequenza dei colpi codificata** per prima e seconda di servizio |
| `Notes` `PtWinner` | note libere, chi ha vinto il punto (`1`/`2`) |

Le colonne `1st`/`2nd` non sono testo libero: sono un codice, un carattere per
colpo (es. `4b37y1r3n#`). **Va decodificato con la legenda del repo MCP prima di
usarlo come feature** — non improvvisare l'interpretazione.

### tennis-data.co.uk — `data/raw/tennis-data/<tour>/<anno>.xlsx`

Solo quote dei bookmaker (`B365W/L`, `PSW/L`, `MaxW/L`, `AvgW/L`), punteggio per
set e ranking. Nessuna statistica di gioco. Nomi abbreviati (`Alcaraz C.`), a
volte con spazi finali. `1/quota` non è una probabilità: usa
`loaders.implied_probabilities()`.

---

## Prima di consegnare un risultato

- gli ordini di grandezza somigliano al tennis? (prime in campo ~62%, punti
  vinti al servizio ~64%, ace ~8% con terra ~5% ed erba ~10%)
- ho controllato i confondenti, a partire dalla superficie? il riferimento
  esclude il soggetto?
- il numero è letto in distribuzione, o è appeso al nulla?
- le metriche adiacenti (altri modi di contare la stessa cosa) concordano?
- **ogni numero che ho comunicato è prodotto da `run.py`?** e lo script,
  rieseguito, dà gli stessi valori del README?
- i testi dei grafici sono costruiti dai dati, senza cifre scritte a mano?
- ho filtrato `row`/`set` al livello giusto?
- i denominatori sono quelli giusti? (`first_won` sta su `first_in`, non su `serve_pts`)
- i merge hanno `validate=` e il numero di righe è quello atteso?
- la distorsione del campione è dichiarata **prima** delle medie?
- se la tesi è risultata falsa, l'ho scritto invece di cercare un taglio che la salvi?
- se il matchup aveva già delle analisi, ho chiesto all'utente quali usare, e i
  numeri nuovi sono coerenti con quelli delle analisi scelte (o la differenza è
  spiegata)?
- l'analisi è nella tabella del `README.md` del matchup, con il risultato in
  una riga?
