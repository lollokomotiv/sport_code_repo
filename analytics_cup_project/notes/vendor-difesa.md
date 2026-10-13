# Responsabilità difensiva individuale: cosa offrono i vendor

Ricerca dell'11/10/2026. Domanda: quali vendor di dati calcistici attribuiscono a
un **singolo difensore** il valore concesso (threat, xT, EPV, qualità della
chance) per via della sua **posizione**, anche quando non ingaggia il portatore?
Per ciascuno: prodotto o metrica, dati, cosa misura, granularità, metodo pubblico
o chiuso.

Serve alla pista "Valore concesso per difensore" (ripartire il valore di ogni
passaggio avversario fra i difensori con una regola geometrica e interpretabile,
sul tracking broadcast open di SkillCorner A-League). Verifica ed estende la
tabella «Cosa esiste già (ricerca del 26/09/2026)» in
[plans/01-scegliere-la-domanda.md](../plans/01-scegliere-la-domanda.md).

**Fonti.** Solo fonti primarie: pagine prodotto, blog e documentazione dei vendor,
paper firmati da personale del vendor. Le pagine promozionali sono riportate come
**dichiarazioni** del vendor, non come fatti verificati.

**Cosa non è stato possibile consultare direttamente.**

- `statsperform.com` risponde 403 a ogni richiesta. Le pagine di Opta Vision e le
  definizioni degli eventi Opta sono lette da snapshot della Wayback Machine
  (20/07/2026, 09/02/2026, 05/09/2026), linkati in tabella.
- `secondspectrum.com` reindirizza a `performancestudio.geniussports.com`, che
  risponde 401 (serve un login). Il comunicato del 2020 sull'Insight feed della
  Premier League (BusinessWire) risponde 403. L'unica copia letta è una
  ripubblicazione su Manchester Digital: è una **fonte terza**, segnalata come tale.
- `catapult.com` (che ora ospita Impect) risponde 403: la pagina prodotto è letta
  da uno snapshot Wayback del 29/09/2026. La documentazione tecnica di Impect è
  invece pubblica su GitHub.
- `tracab.com` non ha risposto.
- La documentazione tecnica completa di DefR (Hudl StatsBomb), di Game
  Intelligence (SkillCorner) e dei feed Opta Vision è riservata ai clienti. Per
  DefR non esiste un paper o un whitepaper pubblico. Del modello sono pubblici
  solo gli input dichiarati e i nomi dei campi API.

---

## 1. Tabella

| Vendor | Prodotto / metrica | Dati | Cosa misura | Granularità | Metodo pubblico? | Fonte |
|---|---|---|---|---|---|---|
| **Hudl StatsBomb** | **DefR** — *Defensive Actions Above Expectation* e *Responsibility-Weighted OBV Conceded* ("DefR OBV") | eventi ("event-based framework"); l'uso dei freeze frame 360 non è menzionato | per ogni carry, passaggio, dribbling e tiro avversario stima quale difensore, in quale ruolo, avrebbe dovuto rispondere; l'OBV generato dall'avversario è ripartito fra i difensori **in proporzione alla responsabilità**, quindi anche chi non ingaggia | **per evento** (oggetto `defensive_responsibility` con le probabilità per difensore, Events API v10); per partita; per stagione e per 90' | **no**: dichiarati solo gli input (luogo dell'azione, "momentum" dell'attacco, "defensive shape"), nessun modello né paper | [blog 04/06/2026](https://www.hudl.com/blog/defensive-responsibility-defr-statsbomb), [release notes 04/06 e 23/06/2026](https://www.hudl.com/releases/statsbomb), [J-League](https://www.hudl.com/blog/defensive-responsibility-j-league-centre-back-analysis), [Serie A](https://www.hudl.com/blog/defensive-responsibility-serie-a-statsbomb) |
| **Impect** (Catapult) | **DEF_PXT_\*** (*Defensive Threat*), basata su un "responsibility model" | eventi raccolti da video TV, con il numero di avversari fra palla e porta e la pressione sul portatore; nessun tracking | variazione della minaccia di subire gol (modello pxT) per ogni passaggio, dribbling e calcio piazzato avversario, **distribuita ai giocatori responsabili**: premiati per le giocate regressive dell'avversario, penalizzati per quelle progressive. Comprende chi non tocca palla | **per evento** (`event_kpis`); per giocatore × posizione × partita | **parziale**: definizioni dei KPI e una frase sul modello ("attributes responsibility to defenders based on packing zones"); regola e pesi non documentati; output ispezionabile negli open data | [kpi_definitions.json](https://github.com/ImpectAPI/open-data/blob/main/data/kpi_definitions.json), [Documentation.pdf, App. 8](https://github.com/ImpectAPI/open-data/blob/main/Documentation.pdf), [open data](https://github.com/ImpectAPI/open-data), [Catapult (snapshot)](https://web.archive.org/web/20260929130150/https://www.catapult.com/solutions/impect) |
| **SkillCorner** | flag EPV sugli ingaggi: `beaten_by_movement`, `beaten_by_possession`, `reduce_possession_danger`, `stop_possession_danger`; profili dei centrali | tracking broadcast + eventi Wyscout sincronizzati [DE p.11] | flag **binari** sull'evento di ingaggio (OBE). `beaten_by_movement`: il difensore era "positioned to control the receiving player", ma non impedisce un aumento significativo delle probabilità di segnare fra passaggio e ricezione. Nessuna quantità di valore attribuita; nessun flag per chi non ha una riga OBE. I profili dei centrali sono frequenze in z-score (0–100) | per evento (OBE); profili per stagione | **parziale**: definizioni testuali, soglie in parte non numeriche ("significantly increase"); EPV (GNN-LSTM) chiuso | [Dynamic Events spec, p.10–11, 44–46](https://26560301.fs1.hubspotusercontent-eu1.net/hubfs/26560301/Guides/Dynamic%20Events/20250216%20-%20Dynamic%20Events%20CSV%20Specifications.pdf), [Out of Possession](https://landing.skillcorner.com/us/articles/game-intelligence-out-of-possession), [centrali](https://skillcorner.com/us/articles/game-intelligence-applied-analyzing-centre-backs-defensive-behaviour), [EPV](https://skillcorner.com/it/articles/introducing-skillcorners-expected-possession-value-epv) |
| **Stats Perform / Opta** | Opta Vision: *Pressure Intensity*, off-ball run difensive, *Shape Analysis*; evento Opta *Defensive Coverage* | Opta Vision: tracking da computer vision sincronizzato agli eventi Opta; Defensive Coverage: eventi | Pressure Intensity: livello di pressione dei **3 difensori più vicini** al portatore. Defensive Coverage: poligono e area della "zona di responsabilità" ricavata dalle azioni difensive del giocatore. **Nessuna** attribuzione di valore concesso | per frame o azione (pressione); per partita (coverage) | **no** (descrizioni di prodotto) | [Opta Vision (snapshot 20/07/2026)](https://web.archive.org/web/20260720131443/https://www.statsperform.com/products/opta-vision/), [Opta Vision 2024-25 (snapshot)](https://web.archive.org/web/20260209230244/https://www.statsperform.com/resource/introducing-opta-vision-for-2024-25/), [Opta event definitions (snapshot)](https://web.archive.org/web/20260905111156/http://www.statsperform.com/opta-event-definitions/), [The Analyst](https://theanalyst.com/articles/opta-vision-stats-tracking-data-premier-league) |
| **Second Spectrum** (Genius Sports) | Insight feed (Premier League) | tracking ottico in stadio, sincronizzato agli eventi | dichiarati: pressing, probabilità di passaggio, off-ball run. **Nessuna metrica pubblica** di valore concesso per difensore trovata | n.d. | **no**; documentazione dietro login | [Genius Sports, content hub](https://www.geniussports.com/content-hub/future-of-tracking-data/); [Manchester Digital](https://www.manchesterdigital.com/post/genius-sports/second-spectrum-premier-league) (*fonte terza*) |
| **Gradient Sports** (ex PFF FC) | *Player Grades*, fra cui *Positioning Grade* | tracking broadcast + eventi + video; voti assegnati da un "expert Grading team" | voto da −2 a +2 su ogni azione, anche senza palla: il Positioning Grade "captures every instance of a player being out of position". **Non** è legato a un modello di valore | per evento (oltre 2.000 voti a partita, dichiarati); aggregati 0–100 | **parziale**: scala e processo descritti, criteri di voto no | [Understanding grades](https://www.gradientsports.com/blog/understanding-gradient-player-grades-a-better-way-to-evaluate-performance), [Overall grade](https://www.gradientsports.com/blog/introducing-gradient-sports-overall-player-grades), [rebrand](https://www.gradientsports.com/blog/open-letter-from-the-ceo-pff-fc-rebrands-to-gradient-sports) |
| **Wyscout** (Hudl) | evento *Defensive positioning* | eventi codificati a mano da video | tag di una corsa difensiva per riprendere posizione; un errore di posizionamento è marcato con l'attributo `bad`. Nessun valore | per evento | definizione di una riga nel glossario | [glossario: defensive positioning](https://dataglossary.wyscout.com/defensive_positioning/), [defensive actions](https://dataglossary.wyscout.com/defensive_actions/) |
| **SciSports** | Performance module, gruppo "Defensive positioning" | non specificato nella pagina; il metodo dichiarato è VAEP | contributo alla probabilità di subire gol, per gruppi di azione | per azione, aggregato | VAEP pubblico; come si costruisce "Defensive positioning" non è descritto | [SciSports, 22/09/2020](https://www.scisports.com/assessing-player-performances-with-scisports/), [VAEP](https://arxiv.org/abs/1802.07127) |
| **DFL / Sportec Solutions** | Bundesliga Match Facts: *Most Pressed Player*, *Pressure Handling*, *Ball Recovery Time* | tracking in stadio | pressione subita dal portatore e recupero palla; nessuna attribuzione di valore concesso | per partita | descrizioni pubbliche (con AWS) | [Bundesliga Match Facts](https://www.bundesliga-group.com/topics/match-data/bundesliga-match-facts/) |
| **Kinexon, Sportlogiq** | — | Kinexon: GPS, LPS e sensore nel pallone; Sportlogiq: dati da video broadcast o tattico | nessuna metrica pubblica sul tema trovata | — | — | [Kinexon](https://kinexon-sports.com/sports/football/), [Sportlogiq](https://www.sportlogiq.com/soccer/) |

---

## 2. Vendor per vendor

### Hudl StatsBomb — DefR

È il caso più vicino all'idea. Il blog di lancio (04/06/2026, senza autore)
descrive DefR come "an event-based framework" che, per ogni carry, passaggio e
tiro avversario, "predicts — based on the location of the action, the momentum of
the attack, and the defensive shape of the team — which player in which role
would typically be expected to respond to it". Il risultato è una *Responsibility
Matrix*, da cui escono due metriche. *Defensive Actions Above Expectation* è
la differenza fra azioni difensive attese e fatte. *Responsibility-Weighted OBV
Conceded* distribuisce l'OBV dell'avversario "proportionally across the defending
players — weighted by their responsibility for each event"
([blog](https://www.hudl.com/blog/defensive-responsibility-defr-statsbomb)).
Rientra quindi anche il difensore che non ingaggia: il blog usa come esempi due
centrali "passivi" (Rüdiger, Delprato) che differiscono per OBV concesso.

Le [release notes](https://www.hudl.com/releases/statsbomb) aggiungono il dettaglio
che conta per la granularità:

- 04/06/2026, Events v9 → v10: "Added a `defensive_responsibility` object to
  each event, identifying which defenders were responsible and with what
  probability"; in Player Match Stats e Player Season Stats arrivano
  `expected_defensive_actions` e `defensive_actions_above_expectation`;
- 23/06/2026: `player_match_obv_conceded_responsibility_weighted` e
  `player_season_obv_conceded_responsibility_weighted_90`, "backfilled across
  the full data catalogue".

**Correzione alla tabella del 26/09:** DefR non è "livello stagione". La
responsabilità è consegnata **per evento**, come distribuzione di probabilità sui
difensori, e l'OBV pesato per partita e per stagione. Resta vero che il modello
non è descritto. Non si sa come venga stimata la "defensive shape" da soli dati
evento (il blog parla di "event data alone"), né quale modello produca le
probabilità. Nessun paper, nessuna validazione pubblicata oltre agli esempi
(Big 5 2025/26, J1 con soglia di 900 minuti nell'[articolo
J-League](https://www.hudl.com/blog/defensive-responsibility-j-league-centre-back-analysis)).
Nessuna fonte dichiara una derivazione da *Blame is easier than praise*.
L'impostazione però è la stessa: responsabilità attesa per ruolo, valore concesso
ripartito.

### Impect — DEF_PXT e "responsibility model"

Non era nella tabella del 26/09 ed è l'altro caso commerciale che fa esattamente
questo. Il modello pxT (*Packing Expected Threat*) è una catena di Markov su tre
variabili: distanza dalla porta, numero di avversari fra palla e porta, pressione
sul portatore. L'appendice 8 della documentazione degli open data dice: "The
defensive pxT model attributes responsibility to defenders based on packing
zones, evaluating both active and passive contributions to reducing goal threat"
([Documentation.pdf](https://github.com/ImpectAPI/open-data/blob/main/Documentation.pdf),
pp. 30–31). Le definizioni dei KPI
([kpi_definitions.json](https://github.com/ImpectAPI/open-data/blob/main/data/kpi_definitions.json))
precisano, per `DEF_PXT_PASS`: "The change in the threat of conceding is
distributed to the responsible players based on the responsability model. The
responsible players are rewarded for regressive passes by the opponent and
penalized for progressive passes". Lo stesso vale per dribbling e calci piazzati.
Per i tiri la quota è divisa in parti uguali fra tutti, con la raccomandazione di
non attribuirla ai singoli. La somma di tutti i `DEF_PXT` è pari al post-shot xG
dell'avversario.

**Verifica sugli open data** (Bundesliga 2023/24, partita 122838, file
`events_kpis`). `DEF_PXT_PASS` compare su 655 passaggi avversari, per 1–4
giocatori ciascuno (193 eventi con 1, 243 con 2, 61 con 3, 158 con 4). Il
destinatario non è mai l'autore del passaggio. In 412 eventi su 655 le quote sono
uguali; negli altri differiscono, con pesi non documentati. Esempio: su un
passaggio del terzino destro avversario la quota va all'ala sinistra e al terzino
sinistro della squadra in difesa. I dati evento di Impect contengono solo le
coordinate della palla, non quelle dei difensori fuori dall'azione. *Inferenza*:
la responsabilità sembra quindi assegnata per ruolo e zona di packing, non per la
posizione misurata di ciascun difensore.

### SkillCorner

Gli articoli indicati in `docs/Articles_SkillCorner.pdf`:

- *Game Intelligence Applied: Analyzing Centre-Backs' Defensive Behaviour*
  (L. Passarotti, 07/05/2026,
  [link](https://skillcorner.com/us/articles/game-intelligence-applied-analyzing-centre-backs-defensive-behaviour)).
  Costruisce indici (stepping forward, covering space in behind, recovery runs,
  penalty-area protection, shot prevention) filtrando gli OBE per altezza della
  linea, direzione, canale, posizione goal-side e probabilità di tiro, poi li
  normalizza con lo z-score su scala 0–100. Sono frequenze di comportamento fra
  pari; non c'è valore concesso. Confermato quanto scritto il 26/09;
- *Introducing SkillCorner's EPV* (L. Passarotti, 07/08/2026,
  [link](https://skillcorner.com/it/articles/introducing-skillcorners-expected-possession-value-epv)).
  Presenta l'EPV (probabilità di segnare entro 30 s o prima della prossima
  interruzione) come attribuzione di valore alle azioni, **offensiva**. Non
  annuncia metriche difensive per giocatore. Nota: la specifica dei Dynamic
  Events di febbraio 2025 dà un orizzonte di 90 s [DE p.11]. Le due fonti
  differiscono e potrebbero riferirsi a versioni diverse del modello.

La parte più vicina all'idea sono i flag EPV sugli ingaggi, definiti nella
[specifica dei Dynamic Events](https://26560301.fs1.hubspotusercontent-eu1.net/hubfs/26560301/Guides/Dynamic%20Events/20250216%20-%20Dynamic%20Events%20CSV%20Specifications.pdf)
(pp. 44–46) e presentati come prodotto in
[*Game Intelligence: Out of Possession*](https://landing.skillcorner.com/us/articles/game-intelligence-out-of-possession)
(C. Myers, 13/03/2025). `beaten_by_movement` è un'attribuzione individuale dovuta
alla **posizione** ("despite being positioned to control the receiving player at
some point"), ma ha tre limiti:

- è binaria;
- esiste solo su righe `on_ball_engagement`, quindi solo per chi ha ingaggiato;
- ha soglie non numeriche e "attenuating circumstances" (2v1, posizione goal-side
  più forte, ricezione di testa) applicate da regole non pubblicate.

Negli open data A-League (20 partite) i flag veri sono pochi: `beaten_by_movement`
187, `beaten_by_possession` 159, `reduce_possession_danger` 589,
`stop_possession_danger` 226, su 17.445 OBE [verificato]. Il dato è tracking
broadcast combinato con eventi Wyscout [DE p.11].

### Stats Perform / Opta

Opta Vision, secondo la pagina prodotto, offre tracking completo sincronizzato
agli eventi, *Phases of Play*, *Off-Ball Runs* ("attacking and defensive off-ball
runs"), *Shape Analysis*, *Pass Predictions* e *Pressure Intensity*.
Quest'ultima "dynamically assigns a pressure level to the three closest defending
players to the player in possession"
([snapshot 20/07/2026](https://web.archive.org/web/20260720131443/https://www.statsperform.com/products/opta-vision/)).
Il tracking è dichiarato "powered by Computer Vision"; la pagina non dice se da
broadcast o da telecamere dedicate
([snapshot](https://web.archive.org/web/20260209230244/https://www.statsperform.com/resource/introducing-opta-vision-for-2024-25/)).

Nelle definizioni degli eventi Opta c'è *Defensive Coverage*: "the area of
defensive responsibility implied by a player's defensive actions during a match",
consegnata come poligono e area in m²
([snapshot](https://web.archive.org/web/20260905111156/http://www.statsperform.com/opta-event-definitions/)).
È una responsabilità **spaziale ricavata dalle azioni**, per partita, senza
valore concesso. Non è stata trovata alcuna metrica Opta di xT o PV concesso per
difensore. Lato ricerca, il *Data-Driven Ghosting* (Le, Carr, Yue, Lucey, STATS
LLC, MIT Sloan 2017,
[PDF](https://la.disneyresearch.com/wp-content/uploads/Data-Driven-Ghosting-using-Deep-Imitation-Learning-Paper1.pdf))
confronta il movimento difensivo reale con quello di una squadra "media"
simulata e ne calcola l'expected goal value. È un controfattuale di **squadra**,
non una ripartizione per difensore; non risultano prodotti derivati documentati.

### Second Spectrum (Genius Sports)

Second Spectrum è il fornitore ufficiale di tracking della Premier League; il sito
ora rimanda a un'area clienti di Genius Sports (401). Il content hub di Genius
cita in termini generici "pressing and pressure and between the lines … things
that are happening off the ball"
([link](https://www.geniussports.com/content-hub/future-of-tracking-data/),
27/08/2025). Il comunicato del 2020 sull'Insight feed, letto solo in
ripubblicazione (fonte terza), elenca "pressing, shot velocity, passing
probabilities, and off-ball runs"
([Manchester Digital](https://www.manchesterdigital.com/post/genius-sports/second-spectrum-premier-league)).
Nessuna fonte pubblica descrive una metrica di valore concesso per difensore.
Questo non esclude che esista nel feed riservato ai club.

### Gradient Sports (ex PFF FC)

PFF FC si è separata da PFF nel giugno 2024 e si è rinominata Gradient Sports
([lettera del CEO](https://www.gradientsports.com/blog/open-letter-from-the-ceo-pff-fc-rebrands-to-gradient-sports)):
"PFF FC" e "Gradient Sports" della richiesta sono lo stesso soggetto. Il prodotto
pertinente sono i *Player Grades*: un voto da −2 a +2 su ogni azione, assegnato da
un "expert Grading team" con revisione senior, su oltre 2.000 eventi a partita e
25 categorie, anche senza palla
([Understanding grades](https://www.gradientsports.com/blog/understanding-gradient-player-grades-a-better-way-to-evaluate-performance),
26/03/2026). Il *Positioning Grade* "captures every instance of a player being
out of position, but also rewards positive decisions such as intercepting a pass
or timing a step up to set an offside trap"
([Overall grade](https://www.gradientsports.com/blog/introducing-gradient-sports-overall-player-grades),
31/03/2026). È una valutazione individuale della posizione, anche senza
ingaggio, ma **umana** e **non espressa in valore concesso**. Gradient produce
anche tracking broadcast: il tracking PFF del Mondiale 2022 è quello usato da
*Blame is easier than praise* (p.4 del paper).

### Wyscout (Hudl)

Nessuna metrica di valore concesso. Il glossario definisce l'evento *Defensive
positioning*, "a defensive run by a player to take back his position in the
defense", e precisa: "A error in positioning in defensive phase would be tagged
as a defensive positioning with special attribute set to bad"
([glossario](https://dataglossary.wyscout.com/defensive_positioning/)). È un tag
manuale di errore di posizione, per evento, senza valore. Non è stata trovata
alcuna indicazione che DefR sia disponibile anche in Wyscout. Gli eventi Wyscout
sono quelli che SkillCorner sincronizza con il proprio tracking [DE p.11].

### Altri, in breve

- **SciSports**: il Performance module ha un gruppo "Defensive positioning" fra i
  quattro difensivi, valutato con VAEP
  ([pagina del 2020](https://www.scisports.com/assessing-player-performances-with-scisports/)).
  VAEP valuta azioni con palla; la pagina non spiega come venga costruito il
  gruppo "positioning".
- **DFL / Sportec Solutions**: i Match Facts difensivi riguardano la pressione sul
  portatore e il recupero palla; nessuno attribuisce valore concesso per posizione
  ([Bundesliga Match Facts](https://www.bundesliga-group.com/topics/match-data/bundesliga-match-facts/)).
- **Kinexon**: GPS, LPS e pallone con sensore, orientati al carico fisico; nessuna
  metrica tattica difensiva pubblica
  ([pagina football](https://kinexon-sports.com/sports/football/)).
- **Sportlogiq** (acquisita da Teamworks): raccolta dati da video broadcast o
  tattico; nessuna metrica pubblica sul tema
  ([pagina soccer](https://www.sportlogiq.com/soccer/)).

### Lato accademico, per confronto

Nessuno dei tre lavori più vicini è firmato da personale di un vendor:

- *Blame is easier than praise* (Bischofberger, Ma et al.): Università di Vienna,
  VfB Stuttgart, DFB ([arXiv](https://arxiv.org/abs/2606.19931));
- DEFCON (Kim et al.): dati forniti dall'AFC Ajax
  ([arXiv](https://arxiv.org/abs/2512.10355));
- CDHMM (Groom et al.): Università di Birmingham e Nottingham Forest
  ([arXiv](https://arxiv.org/abs/2601.00748)).

Le schede sono in
[notes/letteratura/triage-paper.md](letteratura/triage-paper.md).

---

## 3. In cosa si distinguerebbe la pista

Il **concetto** non è nuovo. "Ripartire fra i difensori il valore concesso su ogni
azione avversaria, includendo chi non ingaggia" è già un prodotto commerciale in
almeno due casi, DefR (OBV) e Impect (`DEF_PXT`), entrambi per evento. In
letteratura c'è *Blame is easier than praise*, con regola geometrica, tracking
broadcast e codice pubblico. Le differenze vanno quindi cercate negli assi
seguenti.

| Asse | Pista (candidata) | DefR | Impect DEF_PXT | SkillCorner (flag OBE) | *Blame* (accademico) |
|---|---|---|---|---|---|
| Da cosa dipende la responsabilità | posizione **misurata** di ogni difensore nel frame del passaggio, con regola geometrica scritta | luogo dell'azione, momentum, "defensive shape" stimata da eventi; modello non descritto | ruolo e zona di packing (da documentazione); posizioni dei difensori non presenti nei dati | posizione e vicinanza durante l'ingaggio; regole non pubblicate | aree geometriche (cerchi di 5 m e corridoio) sul tracking, più una componente per ruolo |
| Dati | tracking broadcast SkillCorner, open (A-League, 20 partite) | eventi StatsBomb, proprietari | eventi Impect da video, proprietari (open solo un campione Bundesliga 2023/24) | tracking broadcast + eventi Wyscout | tracking e eventi, 516 partite, di cui 64 da broadcast (PFF) |
| Modello di valore | `xthreat` / EPV di SkillCorner: **chiusi** | OBV: chiuso | pxT: Markov descritto a grandi linee | EPV: chiuso | xT a griglia, addestrato su StatsBomb open |
| Metodo della responsabilità | aperto e riproducibile | chiuso | chiuso (una frase) | chiuso | aperto, codice pubblico |
| Granularità | per passaggio × difensore | per evento × difensore (probabilità) | per evento × giocatore | per ingaggio, binario | per passaggio × difensore |
| Individuale o squadra | individuale | individuale | individuale | individuale | individuale |
| Senza ingaggio | sì | sì | sì | no (solo chi ha un OBE) | sì |
| Interpretabilità della singola quota | la quota si legge sulla geometria del frame | probabilità senza spiegazione | quote spesso uguali, pesi non spiegati | flag con eccezioni non numeriche | geometrica |
| Validazione pubblicata | da fare; con 20 partite e squadre con 1–7 partite il confronto per giocatore è limitato | esempi, nessuna validazione | nessuna | nessuna | contro voti FIFA e valori di mercato |

**Distinzioni fattuali rispetto ai vendor.**

1. **Responsabilità dalla posizione osservata.** DefR e Impect lavorano su dati
   evento, dove la posizione di chi non tocca palla non è misurata: DefR la
   stima con un modello non descritto, Impect usa ruoli e zone (per inferenza
   dalla documentazione e dagli open data). La pista la prende dal tracking.
2. **Metodo aperto.** Nessuno dei due prodotti pubblica la regola. La pista
   renderebbe aperta la regola di responsabilità, **non** il valore: `xthreat` ed
   EPV di SkillCorner restano modelli chiusi, quindi il risultato dipende da un
   componente non ispezionabile.
3. **Quota interpretabile.** In DefR la quota è una probabilità di un modello; in
   Impect è spesso divisa in parti uguali. Nella pista ogni quota si
   ricondurrebbe a una distanza o a un'area misurabile nel frame.
4. **Dati broadcast open.** Nessun vendor espone la propria metrica su dati
   pubblici a questa granularità, tranne Impect con il suo campione. I flag di
   SkillCorner sono nei dati open, ma solo sugli ingaggi.

**Distinzioni rispetto a *Blame*, che non sono differenze di concetto.**
Responsabilità geometrica, tracking broadcast, metodo aperto e attribuzione senza
ingaggio sono già in *Blame*. Resterebbero diversi:

- il dataset (SkillCorner A-League al posto di PFF Mondiale 2022 e del tracking
  della Bundesliga);
- il modello di valore (`xthreat` / EPV SkillCorner al posto della griglia xT);
- la regola, se si usa il tempo di arrivo al posto del raggio fisso: la variante
  che *Blame* scarta "per semplicità" (p.9);
- l'uso del body pose, escluso esplicitamente dal paper (p.22).

**Limiti della pista dichiarabili fin d'ora.** Il tracking broadcast copre solo
in parte i giocatori fuori inquadratura, e le loro posizioni sono estrapolate. Su
una partita intera è osservato il 59% delle posizioni: l'87% entro 5 m dalla
palla, il 18% oltre i 40 m (`explorations/00-quanto-e-osservato.ipynb`). La
responsabilità calcolata su un giocatore estrapolato eredita l'errore
dell'estrapolazione. Il campione è di 16.072 passaggi su 20 partite, circa metà
dei 29.931 del Mondiale usato da *Blame* (triage, scheda *Blame*).
