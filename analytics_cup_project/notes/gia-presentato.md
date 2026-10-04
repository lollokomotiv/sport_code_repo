# Già presentato: lavori sulla difesa all'Analytics Cup

Filtro "non già presentato" per le piste della 2.0 (tema calcio: *Defensive
Positioning*, vedi [`challenge.md`](challenge.md)). Ricerca del 04/10/2026.

## Cosa si è cercato e dove

**Un'edizione sola alle spalle.** "Edizione 2026" ed "edizione 1.0" sono la
stessa cosa: pretalx la presenta come *"the first SkillCorner X PySport Analytics
Cup"* ([pretalx](https://pretalx.pysport.org/skillcorner-x-pysport-analytics-cup-2026/)),
la pagina risultati di PySport come *"the first ever"*. Finale a SkillCorner HQ,
Parigi, 5 febbraio 2026; consegna entro il 28/12/2025; due track, Research e
Analyst.

**Fonti primarie usate:**

- **Elenco ufficiale delle consegne**: il CSV che alimenta la pagina
  [pysport.org/analytics-cup/results](https://pysport.org/analytics-cup/results),
  [analytics-cup-26.csv](https://pysport.org/static/media/analytics-cup-26.38772e0e6926c3495752.csv).
  50 righe (31 Research, 19 Analyst) con titolo, autore, track, finalista, menzione
  d'onore, dati usati e repo. È la base di questa nota: la colonna «Dati» sotto
  riporta il campo `Data Used` del CSV.
- **Finalisti e video**: [presentazioni SkillCorner](https://skillcorner.com/event-2026/analytics-cup-2026-presentations),
  [pagina evento](https://landing.skillcorner.com/us/event-2026/5-02-skillcorner-x-pysport-analytics-cup).
  Gli ID YouTube dei sei video stanno nel bundle JS di pysport.org.
- **Repo delle consegne**, letti uno per uno via `gh api` (README e, dove il
  README era quello del template, il notebook o l'abstract). Elenco di partenza:
  i fork di [PySport/analytics_cup_research](https://github.com/PySport/analytics_cup_research/forks)
  e di [PySport/analytics_cup_analyst](https://github.com/PySport/analytics_cup_analyst/forks).

**Esito.** 6 finalisti, vincitore Amar Shah (Research). 3 menzioni d'onore:
Tahmeed Tureen, Hitesh Gautam, Axel Gautrand. Nessuna menzione d'onore riguarda la
difesa.

**Cosa non si è potuto leggere:**

- **pretalx non pubblica le proposte.** Le pagine `/talk/` e `/schedule/` di
  entrambe le edizioni mostrano solo *"Right now we are busy reviewing
  proposals"*, e l'API delle submission risponde 401
  ([1.0](https://pretalx.pysport.org/skillcorner-x-pysport-analytics-cup-2026/talk/),
  [2.0](https://pretalx.pysport.org/analytics-cup-2/talk/)). Gli abstract
  arrivano quindi solo dai repo.
- **pysport.org è renderizzato in JavaScript.** Il testo non si legge da riga di
  comando: i dati vengono dal bundle `main.*.js` e dal CSV che quel bundle carica.
- **Tre repo del CSV danno 404**: *Schrödinger's Pitch Control* (Amal),
  *Finding Alvarez (and Others) in the A-League* (Karim Elgammal),
  *Phase-Dependent Physical Profiles in Football* (Ashot Akopov). Il primo, dal
  solo titolo, non si classifica.
- **I video delle presentazioni non sono stati visti.** Per i finalisti, domanda
  e metodo vengono dal README, non da quanto detto sul palco.

**Dataset.** Tutte le consegne usano lo stesso: SkillCorner open data, A-League
2024/25. Per la 1.0 pysport.org lo descrive come *"10 Games of XY Tracking Data +
Game Intelligence Dynamic Events"* più *"175 Games of Physical Aggregates"*
([pysport.org/analytics-cup](https://pysport.org/analytics-cup), testo letto dal
bundle JS). Il workbench oggi ne ha 20, ma all'epoca della 1.0 i dati
pubblicati erano quelle 10 partite.

## Lavori sulla difesa (edizione 1.0)

"Difensivo" qui vuol dire che l'oggetto della misura è la squadra o il giocatore
che non ha la palla. Le righe 9–13 sono **parziali**: la difesa è una componente
fra altre, o il lavoro è uno strumento descrittivo.

| # | Titolo | Autore | Edizione / esito | Domanda | Dati (CSV) | Metodo | Link |
|---|---|---|---|---|---|---|---|
| 1 | Simulated Annealing For Positional Optimization | Amar Shah | 1.0, Research, **vincitore** | Dove dovrebbero stare i difensori per massimizzare una superficie arbitraria calcolabile da un frame (pitch control, pressione, DAS), con un trade-off scelto dal coach? | Raw Tracking | Simulated annealing su frame singoli: piccole perturbazioni delle posizioni, nessun dato di training. Obiettivi mostrati: xT-weighted pitch control e pressione media. Ora è un modulo di [DataballPy](https://databallpy.readthedocs.io/en/latest/) (fonte: pysport.org) | [repo](https://github.com/amarshah1999/analytics_cup_research), [video](https://www.youtube.com/watch?v=UWhVXbS69BE) |
| 2 | From On-Ball Engagements to Off-Ball Pressure: Transferring Defensive Knowledge from Tracking Data | Gabriel Valadão Meira | 1.0, Research | Si può rilevare la pressione sugli attaccanti **senza palla**, che gli eventi non registrano? | Raw Tracking, Dynamic Events, Phases of Play | Griglie di pressione (prossimità, tempo di intercetto, velocità dei difensori). Soglie calibrate sugli `on_ball_engagement` come ground truth (max F1), per zona. Eventi OFFBALL con filtro di continuità; metriche di squadra e giocatore per game state e blocco. Dichiara ~600 eventi a partita, ~40% delle azioni difensive | [repo](https://github.com/gabrielvaladao13/skc_competition) |
| 3 | Football Pressure Metric | Riley Killip | 1.0, Analyst | Come misurare in modo continuo e confrontabile la pressione difensiva sul portatore? | Raw Tracking | Score continuo da prossimità, velocità di chiusura dei difensori e contesto di ingaggio; aggregato per finestre temporali. App Streamlit | [repo](https://github.com/rkillip/pressure-metric-project), [video](https://youtu.be/Ptz9yaRjDgQ) |
| 4 | A new defensive metric - Squeeeze | Stuart Macfarlane | 1.0, Analyst | Quanto spazio "stringe" ogni difensore, e uno spazio più stretto concede meno azioni d'attacco? | campo vuoto nel CSV; dal README: tracking 10 partite + eventi con fasi | Area del convex hull fra il giocatore e i 3 compagni più vicini, per frame, solo in fase difensiva. Correlazione a livello squadra con i *line-breaking passes* concessi; il README chiede ulteriore validazione | [repo](https://github.com/StuMacf89/analytics_cup_analyst) |
| 5 | Quantifying the Spatial Drivers of On-Field Reaction Time | Vaughn Hajra | 1.0, Research | Quanto ci mettono i difensori a reagire a uno stimolo della palla in partita, e da cosa dipende? | Raw Tracking | Stimoli = picchi di accelerazione della palla, risposta = prima variazione di accelerazione del difensore (entro 50 m e 1,25 s). Latenza e frequenza di risposta; OLS (R² 0,067), random forest (R² 0,126), GMM per i profili | [repo](https://github.com/vaughnhajra/analytics_cup_research) |
| 6 | Modelling Defensive Pressure Conversion into Possession Disruption Using Spatiotemporal Tracking Data | Aadit Pahuja | 1.0, Analyst | Con che probabilità la pressione si converte in palla persa, e con rendimenti crescenti o decrescenti? | Raw Tracking, Phases of Play | Pressione da funzioni di influenza sul tempo di arrivo (decadimento esponenziale), standardizzata per partita. Disruption = cambio di possesso entro 3 s; curve pressione–disruption per fase (alto, basso, caotico) | [repo](https://github.com/aadit1412/analytics_cup_analyst), [video](https://youtu.be/vm-nh2vJyKQ) |
| 7 | Pressure Point | Joel McLean | 1.0, Analyst | Chi applica e chi guida il pressing, e dove si rompe? | Dynamic Events | Solo `on_ball_engagement`: pressione individuale vs catene di pressing, *press setter share*, *press break* (mancati ingaggi durante un pressing), clustering in archetipi. App Streamlit | [repo](https://github.com/Zero9588/analytics_cup_analyst), [video](https://youtu.be/6sj8VumyYRg) |
| 8 | Proximity Score: A New Metric and its Effect on Expected Threat Outcomes | "fg" (falguni7) | 1.0, Research | La vicinanza dei difensori al portatore **e alle opzioni di passaggio** riduce l'xThreat generato? | Raw Tracking, Dynamic Events, Phases of Play | Distanza minima difensore–giocatore mediata sui frame dell'evento, per portatore e per opzioni. Confronto con xThreat increase e "potential xThreat reduction", aggregato per fase. Risultato descrittivo/visuale | [repo](https://github.com/falguni7/analytics_cup_research_fg) |
| 9 | The Off-Ball Rating: A Comprehensive Framework for Evaluating Player Movement Without Possession | Achraff Adjileye | 1.0, Research — *parziale* | Valutare il contributo senza palla; la componente difensiva (SCR) misura copertura dello spazio e blocco delle linee di passaggio | Raw Tracking | SCR = media di copertura delle zone ad alto xT e quota di linee di passaggio bloccabili, su finestre di 1 s. L'altra metà (OCR) è offensiva | [repo](https://github.com/akedjouadj/skillcorner_pysport_analytics_cup_research) |
| 10 | Evaluating Decision-Making and Pressing Performance from SkillCorner Tracking Data | Idriss Ben Mrad | 1.0, Research — *parziale* | Valutare i centrocampisti per qualità delle decisioni in possesso e per contributo al pressing | Raw Tracking, Dynamic Events, Physical Aggregate | Decisioni: V_real/V_best sulle opzioni (parte offensiva). Pressing: pressione continua sul portatore (distanza, velocità relativa, angolo) più OBE aggregati in un *pressing contribution score*. Il README è quello del template: contenuto da `submission.ipynb` | [repo](https://github.com/IdrissaBM/analytics_cup_research) |
| 11 | Team Shape Analyzer | Martin Steglich | 1.0, Analyst — *parziale* | Come varia la forma della squadra (larghezza, profondità, compattezza) fra possesso e non possesso, in una partita? | Raw Tracking, Phases of Play | Metriche geometriche descrittive da posizioni di giocatori e palla; app Streamlit | [repo](https://github.com/martin-steglich/analytics_cup_analyst) |
| 12 | Gamestate Tactical Analytics Toolkit | Hamza Adhnan Shakir | 1.0, Analyst — *parziale* | Come cambiano forma e linea (anche difensiva) col punteggio? | Raw Tracking, Dynamic Events, Phases of Play | Segmentazione in/out of possession × tempo × differenza reti; libreria Python e notebook | [repo](https://github.com/hamza-shakir/analytics_cup_analyst) |
| 13 | Football Match Intelligence | Tiago Monteiro | 1.0, Analyst — *parziale* | Piattaforma di analisi partita; una sezione è sulla struttura difensiva | Raw Tracking, Dynamic Events, Phases of Play | Altezza della linea (4 giocatori più arretrati), larghezza della linea, compattezza (area del poligono) in/out of possession; app Streamlit | [repo](https://github.com/DataKnight1/football-match-intelligence), [doc](https://github.com/DataKnight1/football-match-intelligence/blob/main/documentation/team_analysis.md) |

Dei sei finalisti solo Shah lavora sulla difesa. Le altre righe sono consegne
non premiate, ma pubbliche: per il filtro contano lo stesso, perché i giudici le
hanno lette.

### Lavori offensivi che usano la pressione come input

Non sono difensivi, ma toccano lo stesso materiale. Una riformulazione che finisce
qui va controllata con loro:

- **ASI (Ryan Inghilterra)**: quanto si muovono i compagni del portatore
  durante gli eventi di pressione.
  [repo](https://github.com/ringhilterra/analytics_cup_research)
- **xPRA (Alexandre L.)**: sollievo dalla pressione sul portatore prodotto dai
  movimenti senza palla, con un campo di pressione gaussiano.
  [repo](https://github.com/Joji-Stan/analytics_cup_research-Space-Architect)
- **Invisible Work (Gabriel Gausachs)**: introduce una *defensive density
  change* causata dai movimenti dei centrocampisti.
  [repo](https://github.com/GabrielGausachs/analytics_cup_research)
- **FootballTransformer (Hitesh Gautam, menzione d'onore)**: similarità di
  scena auto-supervisionata. Cita *pressing triggers* e *defensive block shapes*
  fra i concetti che il modello cattura, ma non li valuta.
  [repo](https://github.com/HiteshG/analytics_cup_research/tree/main)

## Lavori non difensivi (edizione 1.0)

Titolo e link, dal CSV ufficiale. Alcuni URL del CSV reindirizzano a repo
rinominati.

- Positional Player Scouting with Tracking Data (Hadi Sotudeh, finalista) — https://github.com/hadisotudeh/analytics_cup_research
- Off Ball Run Decision Making and Route Optimization (Zach Cochran, finalista) — https://github.com/zcochran4275/skill_corner_analytics_cup_tracking_data_research
- Calculating Worst-Case Scenario Running Demands in Soccer (Emaly Vatne, finalista) — https://github.com/emalyvatne/analytics_cup_research_VatneEmaly
- Dynamic Skills Finder (Oscar Bartolome Pato, finalista) — https://github.com/Data-Kicks/Dynamic-Skills-Finder
- SkPy Analytics Platform (Antoine Verdon, finalista) — https://github.com/2nzi/skillcorner-analytics
- Tempo Flexibility: A Player-Adjusted Bayesian Hierarchical Model… (Tahmeed Tureen, menzione d'onore) — https://github.com/tahmeed14/analytics_cup_research
- FootballTransformer: Self-supervised representation learning on football spatiotemporal data (Hitesh Gautam, menzione d'onore) — https://github.com/HiteshG/analytics_cup_research/tree/main
- A Modular SkillCorner Data Analysis Platform (Axel Gautrand, menzione d'onore) — https://github.com/AxelGautrand/analytics_cup_analyst
- Player Passing Decision Quality Analysis — https://github.com/gaelbuchy/analytics_cup_analyst
- Game Action A-League 24/25 — https://github.com/Twiist33/Expected_Threat
- Performance Profiling Report — https://github.com/superdeb-sys/football-tracking-dashboard
- Quantum Optimisation for Football Passing Decisions — https://github.com/niirvikk/SkillCorner-Submission
- Individual Pitch Control Evaluation — https://github.com/TustiWutsi/analytics_cup_research
- The Space Architect: Valuing Off-Ball Movement with xPRA — https://github.com/Joji-Stan/analytics_cup_research-Space-Architect
- Off-Ball Run Analysis: Quantifying Space Creation in Football — https://github.com/Ketchuphausen/analytics_cup_research
- Invisible Work: Quantifying Midfield Impact Runs — https://github.com/GabrielGausachs/analytics_cup_research
- Schrödinger's Pitch Control (repo 404, non classificato) — https://github.com/AmalAbdirahman/analytics_cup_research
- Finding Alvarez (and Others) in the A-League (repo 404) — https://github.com/KarimElgammal/analytics_cup_research
- Why xThreat Already Works: Validating Expected Threat with Tracking Data — https://github.com/yureed/analytics_cup_research
- The Active Support Index (ASI): Quantifying Off-Ball Movement During Pressure Events — https://github.com/ringhilterra/analytics_cup_research
- Forecasting Expected Threat (xThreat) in Soccer: An XGBoost Model Based on Geometric Player Configurations — https://github.com/pdiagne/analytics_cup_research_MD
- Run Value Added (RVA) Metric — https://github.com/rodmart21/analytics_cup_research
- Modeling Build-Up Play in Football with Graph Neural Networks — https://github.com/APScott4/analytics_cup_research
- WorkRate Metric — https://github.com/DominikZabron/analytics_cup_research
- AI Sports Analyst — https://github.com/adityamukherjee42/analytics_cup_analyst
- Identifying Similar Football Plays Through Ball Movement Patterns — https://github.com/shehab400/analytics_cup_analyst
- Explain the Game: Interactive Episodes for Coaching — https://github.com/MihirT906/analytics_cup_analyst
- Physics simulation of ball and its application using 3D ball tracking data — https://github.com/903124/analytics_cup_research
- GSI: Measuring Group Synchronization in Football Tracking Data — https://github.com/mateusz-gob1/analytics_cup_research
- SkillCorner Physical Scouting Tool — https://github.com/BigR-2000/analytics_cup_analyst
- GKLaunch: Measuring Significant Goalkeepers Contributions in the Attacking Phases — https://github.com/BogdanAlexandru09/analytics_cup_research
- PADI — Phase-Adjusted Decision Index — https://github.com/gcarbs1/analytics_cup_research
- Player Movement Dashboard — https://github.com/rodmart21/analytics_cup_analyst
- Phase-Dependent Physical Profiles in Football (repo 404) — https://github.com/AshotAkopov/analytics_cup_research
- Quick Game Recaps and Player Performance Explorer — https://github.com/KouroshGerayeli/Quick-Game-Recaps-and-Player-Performance-Explorer
- Decision-xT (D-xT): Context-Aware Penalization of xThreat Based on Decision Quality — https://github.com/anarabiyev/analytics_cup_research_submission
- Quantifying the Unseen: Valuing False Runs with Expected Possession Value (EPV) — https://github.com/Eshah7/analytics_cup_research

## Edizione 2.0: repo pubblici già visibili

Non sono lavori presentati: la 2.0 è in corso, chiude il 18/12/2026
([API pretalx](https://pretalx.pysport.org/api/events/), evento `analytics-cup-2`)
e non ha un elenco pubblico delle proposte. Però le consegne sono open source e
alcuni repo dichiarano già tema e domanda. Ricerca non esaustiva (`gh search
repos`), solo calcio, solo repo che si dichiarano esplicitamente Analytics Cup
2.0:

- **"Who should have picked him up?" (ahmadelbabaa)**: responsabilità di
  marcatura nel tempo rispetto a una filosofia scelta dal coach (da zonale a
  uomo), con buchi di responsabilità e passaggi di consegna tardivi. Work in
  progress. [repo](https://github.com/ahmadelbabaa/analytics-cup-2)
- **AnalyticsCup2_0 (ElliottTDon)**: EDA difensiva, con la domanda del tema
  copiata così com'è. [repo](https://github.com/ElliottTDon/AnalyticsCup2_0)
- **analytics_cup_2.0_defense (piperojas0618)**: README vuoto, contenuto non
  letto. [repo](https://github.com/piperojas0618/analytics_cup_2.0_defense)

## Cosa resta scoperto

Sui sei sotto-temi citati dal tema 2.0, solo rispetto alle 50 consegne 1.0 qui
sopra:

| Sotto-tema | Stato nella 1.0 | Chi |
|---|---|---|
| **Pressure** | **Saturo.** Pressione sul portatore come score continuo (3); conversione pressione→palla persa (6); pressing da eventi, con catene e rotture (7); pressione su chi non ha palla (2, 8); pressione come obiettivo di ottimizzazione (1); contributo al pressing per giocatore (10) | 1, 2, 3, 6, 7, 8, 10 |
| **Team shape** | **Coperto solo in forma descrittiva** (tool), più un approccio prescrittivo su frame singoli (1). Manca un lavoro che leghi la forma a un esito, o la confronti con un riferimento lungo un'azione | 1, 11, 12, 13 |
| **Compactness** | **Come sopra.** Area o poligono della squadra in/out of possession (11, 13), per game state (12); compattezza locale con 3 compagni, legata ai line-breaking passes concessi solo per correlazione a livello squadra (4) | 4, 11, 12, 13 |
| **Spacing** | **Parziale.** Spazio stretto attorno al difensore (4); copertura delle zone ad alto xT (9); pitch control come obiettivo (1). Nessun lavoro sulle distanze fra reparti o fra linee nel tempo | 1, 4, 9 |
| **Marking** | **Non coperto.** Nessun lavoro assegna chi marca chi o valuta le marcature. Ci vanno vicino il blocco delle linee di passaggio (9) e la pressione su chi non ha palla (2). Attenzione: nella 2.0 un repo pubblico lavora proprio sulla responsabilità di marcatura (ahmadelbabaa) | (2, 9) |
| **Defensive decision-making** | **Quasi non coperto.** Coperti solo la latenza di reazione allo stimolo (5) e i mancati ingaggi durante il pressing (7, *press break*). Nessun lavoro valuta la scelta del difensore (uscire o tenere, pressare o scappare, scalare) rispetto alle alternative. L'ottimizzazione di Shah (1) dà posizioni ottime su un frame, non decisioni nel tempo | (5, 7) |

Assenti anche dai lavori letti, fuori dai sei esempi del tema:

- gestione della linea difensiva e del fuorigioco come scelta collettiva (13 la
  misura, non la valuta);
- difesa nella transizione negativa, o rest defence;
- coordinamento difensivo come squadra (GSI misura la sincronia in generale, non
  in fase difensiva);
- portiere in fase difensiva (GKLaunch guarda solo l'attacco);
- palle inattive difensive.
