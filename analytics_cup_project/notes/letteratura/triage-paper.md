# Triage dei paper in `docs/`

Stato al 04/10/2026.

È un triage, non un insieme di schede. Per ogni documento in `docs/` che non ha
ancora una scheda ho letto abstract, introduzione, sezione su dati e metodo,
risultati principali e conclusioni, e ho risposto a tre domande: cosa misura,
che dati richiede, se in linea di massima regge sui nostri. Le schede complete
(formato della skill `scheda-paper`) si scrivono dopo, solo per i paper legati
alle piste che sopravvivono.

Esclusi perché già schedati: Narizuka et al. ([pressione-tempo-arrivo](pressione-tempo-arrivo.md))
ed exPressV2 ([pressing-exPressV2](pressing-exPressV2.md)).

Le citazioni `p.N` sono pagine del PDF (numerazione del lettore, non quella
stampata). Il metro di confronto è quello di `CLAUDE.md` §4: 20 partite di
A-League 2024/25, tracking broadcast a 10 fps, 59% delle posizioni osservate
(87% entro 5 m dalla palla, 18% oltre i 40 m), body pose su 2 partite con il
45,7% utilizzabile. Due numeri contati per questo triage sulle 20 partite: 13
squadre, con un numero di partite per squadra che va da 1 a 7; 16.072 possessi
chiusi da un passaggio in `player_possession`.

**Verdetti.** *Replicabile*: il metodo gira sui nostri dati senza cambiarne il
nucleo. *Adattabile*: l'impostazione si trasferisce, un componente va
sostituito. *Solo ispirazione*: serve l'idea, non il metodo. *Non praticabile*:
né metodo né domanda reggono qui.

## Tabella

Ordinata per verdetto.

| Paper | Domanda | Dati richiesti | Metriche nostre già disponibili che userebbe | Verdetto | Tema 2.0 |
|---|---|---|---|---|---|
| **Pressing Intensity** (Bekkers 2025) | quanto ogni difensore mette sotto pressione ogni attaccante, frame per frame | tracking con velocità; nessun dataset di validazione nel paper | `unravelsports.PressingIntensity` (già provata su una partita), τ_opp, `time_to_impact`, `overall_pressure` | **Replicabile** | pressure |
| **Forcher et al. 2022**, pressione e recupero palla | la pressione è più alta nelle azioni difensive che finiscono con un recupero? | tracking 25 Hz + eventi, 153 partite Bundesliga | `get_pressure_on_player` di databallpy (stesso modello Herold), `player_possession`, `pressing_chain_end_type` | **Replicabile** | pressure |
| **EFPI** (Bekkers 2025) | che formazione ha una squadra e che posizione ha ogni giocatore, per frame o per segmento | tracking dei 10 di movimento | `unravelsports.EFPI`, `defensive_structure`, `team_*_width/length` delle fasi | **Replicabile**, ma misura anche l'estrapolazione | team shape |
| **Back-four** (Dash et al. 2025) | quali indicatori della linea a quattro distinguono le transizioni difese bene | tracking SkillCorner + eventi StatsBomb, 73 partite di due squadre | `last_defensive_line_*`, `inside_defensive_shape`, convex hull (mplsoccer/floodlight), fasi di gioco | **Replicabile** (le misure; il confronto per squadra no) | compactness, spacing, team shape |
| **Blame is easier than praise** (Bischofberger et al. 2025) | quanta colpa e quanto merito ha ogni difensore per i passaggi avversari, anche senza toccare palla | tracking + eventi sincronizzati, 516 partite (di cui 64 da broadcast) | passaggi di `player_possession` con esito e ricevitore, `xthreat`, EFPI o ruoli per i ruoli | **Adattabile** | marking, spacing, positioning |
| **Better Prevent than Tackle / DEFCON** (Kim et al. 2025) | quanto valore toglie (o concede) ogni difensore in ogni azione avversaria | tracking 25 Hz + ~1.400 eventi annotati per partita, 564 partite Eredivisie | `xpass_completion`, `passing_option_score`, `xthreat`, EPV di SkillCorner come componenti | **Adattabile** | defensive decision-making, marking |
| **VDEP** (Toda et al. 2022) | valutare la difesa di squadra con la probabilità di recuperare palla e di subire un attacco pericoloso | eventi + tracking 25 Hz di tutti i 22, 45 partite J1 | `xloss_*` e `xshot_*` (stimano le stesse due probabilità, solo sugli ingaggi), `lead_to_shot` | **Solo ispirazione** | valutazione difensiva di squadra |
| **Off-ball defensive role / CDHMM** (Groom et al. 2026) | chi marca chi e chi difende a zona sui corner, e quanto vale ogni difensore rispetto a un "fantasma" del suo ruolo | tracking 25 fps + eventi, 14.678 corner di 4 stagioni Premier League | nessuna diretta; `game_interruption_before` per isolare i corner | **Solo ispirazione** (la replica sui corner non è praticabile) | marking |
| **NFL Ghosts** (Yurko et al. 2025) | il difensore più vicino al ricevitore era messo meglio o peggio di un difensore "fantasma"? | tracking NFL 10 fps, 10.363 passaggi completati | inizio dei `player_possession`, avversario più vicino (`separation_start`), EPV o `xloss` come esito | **Solo ispirazione** — football americano | positioning individuale |
| **Spatio-Temporal Analysis of Team Sports** (Gudmundsson & Horton 2016) | survey dei metodi su dati spazio-temporali negli sport di squadra | — | — | **Solo ispirazione** (da consultare) | trasversale |
| **What Makes a Dribble Successful?** (Schepers et al. 2025) | quali feature di postura spiegano la riuscita di un dribbling | tracking 2D + pose 3D 25 Hz (Hawk-Eye), 125 partite di Champions | body pose (2 partite), `lib/pose.py` per l'orientamento del busto | **Solo ispirazione** — tema offensivo | (solo di sponda: postura del difensore) |
| **Revisiting EPV** (Overmeer et al. 2025) | come valutare un modello EPV, e un EPV migliore | tracking 10 Hz + eventi, 687 partite | EPV di SkillCorner (chiuso) | **Non praticabile** — non è un paper sulla difesa | nessuna |
| `Articles_SkillCorner.pdf` | non è un paper: due link ad articoli SkillCorner | — | — | non valutabile da qui | — |
| `Research_Directions_AnalyticsCup.pdf` | non è un paper: la mappa delle sei direzioni | — | — | — | tutti |

## Paper per paper

### Pressing Intensity — Bekkers 2025 (8 pp.)

Per ogni coppia difensore–attaccante calcola un tempo di intercetto alla
Spearman/Shaw, con tempo di reazione e una penalità per chi corre nella
direzione sbagliata (p.2–3, formule 1–3), lo trasforma in probabilità con una
logistica (T = 1,5 s, σ = 0,45, p.3) e combina i difensori assumendoli
indipendenti (p.3, formula 5). Aggiunge una soglia di velocità per il pressing
attivo (p.4). **Non c'è nessuna validazione empirica**: né dataset, né esiti, solo
figure (p.1–7). Ostacolo sui nostri dati: le velocità a 10 fps, e il fatto che la
matrice copre tutti gli attaccanti, compresi quelli lontani dalla palla; la
pressione sul portatore invece sta dove siamo all'87% osservato. Già provata su
una partita ([explorations/04](../../explorations/04-pressing-intensity-unravel.ipynb)).
Di nuovo, adattandola: la prima validazione contro esiti reali, e la penalità
sulla direzione calcolata con l'orientamento del busto invece che con la velocità.

### Forcher et al. 2022 — *The keys of pressing to gain the ball* (19 pp.)

153 partite di Bundesliga, TRACAB a 25 Hz con eventi sincronizzati (p.6).
Pressione con il modello ellittico di Andrienko nella versione di Herold et al.
(p.7, formule 1–5), misurata ogni secondo negli ultimi 10 s di un possesso
avversario, su tre scale: portatore, i 5 attaccanti più vicini alla palla, tutta
la squadra (p.8). Successo = recupero palla, su possessi di almeno 5 s e 3
passaggi (p.6). L'effetto è piccolo: 14,47 ± 16,82% contro 12,87 ± 15,31% (p.3,
p.10), una differenza di circa 0,1 deviazioni standard. E la pressione a t = 0,
alla fine di un'azione chiusa da un recupero, è alta quasi per costruzione. Sui
nostri dati il modello è già in databallpy; regge su portatore e gruppo, non
sulla scala "squadra", che dipende dai giocatori lontani. Di nuovo: il confronto
fra le tre scale con `is_detected`, cioè quanto cala il segnale quando si passa
da posizioni osservate a posizioni stimate.

### EFPI — Bekkers 2025 (11 pp.)

Riconosce la formazione con l'assegnazione ungherese fra giocatori e 65 modelli
di mplsoccer, dopo aver riscalato le posizioni alla larghezza e lunghezza dei
modelli (p.2–3); funziona per frame o su segmenti con la posizione media (p.4),
con un parametro di stabilità contro i cambi spuri (p.4–5). Nessuna validazione
quantitativa: un esempio su una partita del Mondiale 2022 (p.5–6). Il codice è in
`unravelsports` e legge SkillCorner via kloppy (p.6). L'ostacolo è strutturale:
usa tutti i 10 giocatori di movimento e il riscalamento dipende proprio dagli
estremi della squadra, cioè dai giocatori più lontani dalla palla, osservati al
18% oltre i 40 m. Di nuovo: un test di accordo gratuito con `defensive_structure`
di SkillCorner, e quanto cambia la formazione assegnata se si usano solo i frame
in cui la squadra è osservata.

### Prediction-based evaluation of back-four defense — Dash et al. 2025 (22 pp.)

Cinque indicatori a regole sulla linea dei quattro difendenti più arretrati
(stretch index, pressione entro 3 m, space score a zone pesate, altezza della
linea assoluta e relativa alla palla, p.5–6), su 2.413 transizioni negative di
Barcellona e Real Madrid, LaLiga 2023/24 (p.1, p.4–5). È l'unico paper del gruppo
che usa **tracking SkillCorner** (p.3), e non menziona mai le posizioni
estrapolate: la sola riserva è "minor spatial inaccuracies" (p.17). Da
verificare prima di citarlo: dichiara SkillCorner a 25 Hz (p.3), mentre i nostri
dati sono a 10 fps; i conteggi non tornano (2.413 sequenze, 1.434 + 979, poi 639
+ 624 nel dataset finale, p.5); il terzo difensivo è "ultimi 35 m" a p.4 e "palla
ad almeno 70 m dalla propria porta" a p.5; la tabella 4 dice che i modelli
d'insieme battono "linear baselines" che non ci sono (p.13); lo stretch index
somma un'area e una distanza (p.5). Sui nostri dati le misure si calcolano
subito, ma il confronto per squadra no (da 1 a 7 partite per squadra). Di nuovo:
la stessa linea calcolata con e senza posizioni estrapolate, su un caso in cui la
palla è nel terzo difensivo e la linea è relativamente vicina.

### Blame is easier than praise — Bischofberger et al. 2025 (27 pp.)

Distribuisce il valore di ogni passaggio avversario (ΔxT, griglia 16 × 12
addestrata su StatsBomb open data, p.6–7) fra i difensori dentro *defensive
pressure areas* geometriche: cerchi di 5 m attorno a passatore e ricevitore e il
corridoio fra i due (p.8–9). La responsabilità attesa è la media del
coinvolgimento per terna di ruoli passatore–ricevitore–difensore, con i ruoli da
template matching sulle formazioni (p.10–12). Validato su 516 partite, fra cui 64
del Mondiale 2022 con **tracking PFF ricavato da video broadcast** (p.4), contro
voti FIFA e valori di mercato; la "colpa" per i passaggi pericolosi concessi è la
misura più valida (p.1, p.23), con il confondente della forza della squadra
dichiarato (p.21–22). Codice pubblico (p.23). Sui nostri dati il coinvolgimento
attorno al passatore sta nella zona ben osservata; ricevitore e corridoio stanno
a distanze intermedie, dove la copertura va misurata. Il collo di bottiglia è la
parte per ruolo: soglie di 150–300 minuti per giocatore-ruolo (p.14–15) contro
squadre con 1–7 partite; abbiamo 16.072 passaggi, circa metà dei 29.931 del loro
Mondiale. Di nuovo: il paper esclude esplicitamente l'orientamento del corpo
(p.22), che il body pose fornisce; e la variante con un tempo di arrivo al posto
del raggio fisso, che gli autori scartano per semplicità (p.9).

### Better Prevent than Tackle / DEFCON — Kim et al. 2025 (28 pp.)

Scompone l'EPV in probabilità di scelta, di riuscita e valore condizionato di
ogni opzione (p.3–5), e divide la variazione di EPV fra i difensori in base alla
probabilità che ciascuno avrebbe intercettato il pallone (p.5–6), con regole per
ogni esito (p.6–8). Cinque GAT addestrati su 564 partite di Eredivisie a 25 Hz
con ~1.400 eventi annotati per partita (p.16–17). Il guadagno della rete sui
boosting è piccolo (AUC 0,9167 contro 0,9124 sul passaggio) e sui tiri vincono i
boosting (p.17–18). Come in *Blame*, il credito per ciò che si concede correla con
il valore di mercato più di quello per le azioni (0,754 per i centrali, p.19–20);
qui il confondente della forza della squadra è discusso solo per i crediti da
azione (p.19). Sui nostri dati non si addestra nulla di simile, ma i componenti
esistono già in SkillCorner (`xpass_completion`, `passing_option_score`,
`xthreat`, EPV), solo in pochi istanti e da modelli chiusi; manca la
responsabilità del difensore. Di nuovo: una responsabilità fisica (tempo di
arrivo sul corridoio) al posto della GAT, sopra componenti SkillCorner.

### VDEP — Toda et al. 2022 (15 pp.)

Sostituisce gol e gol subiti, troppo rari, con due eventi più frequenti:
recupero palla e "subire un attacco efficace" (tiro o ingresso in area) entro 5
eventi (p.3–4). Classificatori XGBoost con 139 feature, comprese le posizioni di
tutti i 22 giocatori (p.5–6), su 45 partite di J1 2019 (p.3). La validazione è a
livello squadra-stagione con 18 squadre: r = 0,397, p = 0,103 (p.11), che
l'abstract chiama "moderate" (p.1) mentre la scala adottata dal paper mette fra
0,20 e 0,40 la correlazione "low" (p.8). Sui nostri dati le due probabilità sono
già, in sostanza, `xloss` e `xshot` di SkillCorner, ma solo sugli ingaggi
difensivi; la validazione per squadra non è ripetibile con 13 squadre e 1–7
partite ciascuna. Di nuovo: poco, se non un indice `xloss − C·xshot` per ingaggio,
che eredita un modello chiuso.

### Off-ball defensive role / CDHMM — Groom et al. 2026 (40 pp.)

Un HMM con transizioni dipendenti da covariate, che stima frame per frame se ogni
difensore marca un avversario o tiene una zona sui corner, senza etichette (p.3–4);
ne derivano crediti individuali e "fantasmi" condizionati al ruolo (p.15–17,
p.19). Dati: 14.678 corner di quattro stagioni di Premier League, modelli
separati per squadra e tipo di battuta (p.4). È il lavoro citato dalle Research
Directions per i passaggi di consegne. Sui nostri dati i corner sono nell'ordine
di 200 in tutto (113 + 102 eventi con `game_interruption_before` = corner), cioè
una quindicina per squadra: la replica non è praticabile. Gli autori stessi dicono
che per il gioco aperto le distribuzioni di emissione vanno riscritte (p.19–20) e
che il modello non usa le velocità (p.19). Di nuovo: l'orientamento del busto
come covariata per distinguere chi segue l'uomo da chi guarda la palla.

### NFL Ghosts — Yurko, Nguyen, Pelechrinis 2025 (34 pp.)

**Football americano, non calcio.** Al momento della ricezione confronta la
posizione del difensore più vicino con una distribuzione 2D di posizioni
"fantasma", stimata con random forest per densità condizionate, e misura la
differenza in yard dopo la ricezione attese (p.6–7, p.12–14). Dati: Big Data Bowl
2021, stagione 2018, tracking RFID a 10 fps, 10.363 passaggi completati (p.3, p.8–9);
il modello delle yard si stabilizza dopo circa 8 settimane di dati, quello dei
fantasmi dopo circa 12 (p.12–13). Il "difensore più vicino al ricevitore" vive
nella zona dove i nostri dati sono buoni, e il volume (18.710 possessi) è dello
stesso ordine, ma il calcio non ha un esito equivalente alle yard dopo la
ricezione e il fantasma simula un comportamento medio (limiti in p.22–23). Di
nuovo: un fantasma condizionato per il solo difensore più vicino, che sposta
l'argomento "il ghosting richiede troppi dati" dal metodo alla domanda.

### Spatio-Temporal Analysis of Team Sports — Gudmundsson & Horton 2016 (42 pp.)

Survey del 2016, non un metodo. Utile per tre sezioni: modelli di moto e regioni
dominanti (p.9–13, dove compare il modello di Fujimura–Sugihara usato da
Narizuka), riconoscimento delle formazioni (p.23 e seguenti) e prestazione
difensiva (p.31–32), dove nota che le analisi difensive spaziali fatte nel basket
non esistevano nel calcio ("Open 7", p.32). Dieci anni dopo quel buco è in parte
chiuso dagli altri paper di questa lista. Nessun ostacolo di dati, perché non c'è
nulla da replicare. Da consultare per la bibliografia, non da leggere in blocco.

### What Makes a Dribble Successful? — Schepers et al. 2025 (15 pp.)

**Il tema è offensivo**: la riuscita del dribbling dell'attaccante. Fra le
feature di pose, quelle sul difensore sono due: l'angolo del corpo rispetto
all'attaccante e la gamba d'appoggio (p.5–6). Dati: 125 partite di Champions
2022/23, Hawk-Eye con 29 punti anatomici a 25 Hz, 1.736 dribbling (p.3–4); i
buchi di pose sono interpolati con spline (p.4). Il guadagno del pose è modesto,
e il testo non commenta un dato della sua tabella: Brier 0,2429 con 2D + 3D
contro 0,2444 con il solo 2D, ma 0,2356 per la baseline di sola posizione, che
resta migliore (p.8); con il 40% di successi un modello senza informazione ha
Brier 0,24. Sui nostri dati: pose su 2
partite e i giunti dei piedi sono i meno precisi (15–19 cm), quindi le feature
palla–piede non reggono; quelle di busto e anche sì. Di nuovo: l'orientamento del
difensore nell'1 contro 1, come dimostrazione di metodo.

### Revisiting Expected Possession Value — Overmeer et al. 2025 (16 pp.)

**Non riguarda la difesa.** Propone un benchmark di 50 coppie di situazioni
giudicate da esperti (p.5), non riesce a replicare l'EPV di Fernández et al. 2021
(p.1, p.5) e propone una U-net su 624 partite di Eredivisie e 63 del Mondiale,
tracking a 10 Hz (p.5). Il 78% sul benchmark (p.12) è 39 coppie su 50, con un
intervallo di confidenza di circa ±11 punti. Sui nostri dati non c'è niente da
addestrare: l'EPV lo fornisce già SkillCorner, chiuso. Di nuovo, al massimo: il
benchmark a coppie come modo di mettere alla prova l'EPV chiuso che erediteremmo
se una pista lo usasse come componente.

### `Articles_SkillCorner.pdf` (1 p.)

Non è un paper: una nota del 22/09/2026 con due link, *Game intelligence applied:
analyzing centre-backs' defensive behaviour* e *Introducing SkillCorner's EPV*
(p.1). Gli articoli non sono in `docs/` e non li ho letti: il primo è pertinente
al tema 2.0 e va recuperato prima di valutarlo.

### `Research_Directions_AnalyticsCup.pdf` (1 p.)

Non è un paper ma la mappa delle sei direzioni, già riassunta nel
[README](README.md). Due cose da tenere presenti leggendola: dice «circa 10
partite» (p.1), e sono 20; e rimanda a «il progetto che ho linkato prima» per il
limite del tracking estrapolato (p.1), link che nel PDF non c'è.

## Cosa suggerisce per le piste

Raggruppamenti di fatto, senza scegliere.

**1. La famiglia del tempo di arrivo e della pressione sul portatore.** Narizuka
(τ_opp), Pressing Intensity, Forcher/Herold, exPressV2, e a monte le regioni
dominanti del survey. Tutti misurano lo stato vicino alla palla, dove i nostri
dati sono migliori, e tre di queste misure le abbiamo già calcolate (τ_opp,
Pressing Intensity su una partita, `time_to_impact`). Pressing Intensity non ha
una validazione propria, Forcher trova un effetto di circa 0,1 deviazioni
standard: il confronto di più misure sugli stessi possessi, già avviato con il
test di equivalenza, è uno spazio che questi paper lasciano aperto. Tocca la
direzione 4 (trigger) solo se si passa dall'intensità a ciò che la precede.

**2. L'attribuzione individuale di ciò che si concede.** DEFCON, *Blame*,
exPressV2, Groom, NFL Ghosts. Stessa struttura: una variazione di valore della
squadra divisa fra i difensori secondo una responsabilità. Cambia come si stima
la responsabilità: rete neurale (DEFCON), regole geometriche e ruoli (*Blame*),
ruoli latenti da HMM (Groom), distribuzione fantasma (NFL). Due risultati
convergono in modo indipendente: la colpa per ciò che si concede è più legata al
valore dei giocatori del merito per tackle e intercetti (DEFCON p.19–20, *Blame*
p.1 e p.23). È la stessa famiglia della pista B, ma valuta il posizionamento,
non la scelta fra uscire e tenere. Su questi dati la versione praticabile è quella
geometrica, con componenti di valore presi da SkillCorner.

**3. L'organizzazione collettiva lontano dalla palla.** EFPI, back-four, VDEP,
la scala "squadra" di Forcher, la parte per ruoli di *Blame*. Dipendono dai
giocatori meno osservati. Un fatto trasversale: **nessuno dei 12 paper di questo
triage parla di posizioni estrapolate** (ricerca nel testo estratto), nemmeno i due che usano tracking ricavato da
broadcast (*Blame* con PFF, p.4; back-four con SkillCorner, p.3 e p.17). È la
direzione 5 (qualità dei dati broadcast), e qui ha oggetti concreti: EFPI contro
`defensive_structure`, l'altezza della linea con e senza posizioni stimate.

**4. La marcatura.** Groom sui corner, la proposta di un HMM per le coperture
nella discussione di NFL Ghosts (p.22), Franks et al. nel basket citati dal survey
(p.31). È la direzione 3. Sui nostri dati manca il volume per i corner, e per il
gioco aperto lo stesso Groom dice che il modello va riscritto (p.19–20).

**5. Il body pose.** Dribbling (guadagno modesto, tema offensivo), *Blame* che
esclude l'orientamento per scelta (p.22), Groom che non usa le velocità (p.19), la
penalità di direzione della Pressing Intensity. Il pose entra come aggiunta a un
metodo esistente, non come metodo a sé, e con 2 partite resta una dimostrazione.

**Fuori tema.** Revisiting EPV (valore del possesso, non difesa), il dribbling
(attaccante), NFL Ghosts (altro sport), il survey (generale). Restano utili come
componenti o come bibliografia, non come base di una pista sulla difesa.
