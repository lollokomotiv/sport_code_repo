# Risorse

Elenco ufficiale delle risorse indicate da PySport, annotato con quello che
serve sapere prima di sceglierle. *"A non-exhaustive set of open-source tools
and references to get you started. You are free to use anything else you like."*

---

## Lavorare con i dati

### [kloppy](https://github.com/PySport/kloppy) — `pip install kloppy`
Carica e standardizza tracking ed eventi da una decina di provider. **È di
PySport**, cioè dell'organizzazione che co-organizza la gara, e il tutorial
ufficiale SkillCorner lo usa.

Quello che conta qui:

```python
from kloppy import skillcorner
dataset = skillcorner.load_open_data(match_id=1886347)   # scarica da GitHub
df = (dataset
      .transform(to_orientation="STATIC_HOME_AWAY")      # attacchi sempre →
      .filter(lambda f: f.period.id == 1)
      .to_df(engine="polars"))
```

`load_open_data()` è **il modo previsto dal regolamento** per caricare i dati
nella submission: niente file grossi nel repo. `sample_rate` e `limit` servono
per lavorare su sottoinsiemi durante lo sviluppo.

Il `transform(to_orientation=...)` non è un dettaglio estetico: senza, il primo
e il secondo tempo hanno le direzioni di attacco invertite, e qualunque metrica
spaziale aggregata sui due tempi è sbagliata.

### [databallpy](https://github.com/Alek050/databallpy) — `pip install databallpy`
Sincronizzazione tracking ↔ eventi e feature di partita. **La sincronizzazione
qui non serve**: richiede un provider di eventi supportato (Opta, StatsBomb,
Sportec…), e i dynamic events di SkillCorner sono già allineati al frame.

Servono invece le feature. `get_game` non legge SkillCorner, ma
`get_game_from_kloppy` sì: pitch control (Fernández & Bornn), pressione sul
giocatore (Herold et al.), velocità e accelerazioni girano sui nostri dati. Il
Voronoi approssimato no. Vedi `metriche-disponibili.md` §3.2.

### [floodlight](https://github.com/floodlight-sports/floodlight) — `pip install floodlight`
Toolkit multi-sport con un modello dati proprio (`XY`, `Pitch`, `Events`) e
metriche spaziali già implementate. Ha una curva di apprendimento sua: conviene
se usi le sue metriche, non come semplice lettore. **Come lettore non funziona
qui**: il parser SkillCorner è per il dataset JSON del 2021 e fallisce sul JSONL
attuale. I modelli si usano costruendo gli `XY` a mano.

---

## Visualizzazione

### [skillcornerviz](https://github.com/SkillCorner/skillcorner-viz) — `pip install skillcornerviz`
Libreria di viz di SkillCorner, costruita sui **loro** aggregati: radar, swarm
plot, confronti fra giocatori. Parla nativamente il formato dei dati, quindi sui
file `aggregates/` è la strada più corta.

### [mplsoccer](https://github.com/andrewRowlinson/mplsoccer) — `pip install mplsoccer`
Campi e chart per matplotlib. **Già usato in `xgoals_project/`**, quindi lo stile
è trasferibile. Per il tracking servono `Pitch` con dimensioni reali lette da
`{id}_match.json`, non i default.

### [mplbasketball](https://github.com/mlsedigital/mplbasketball)
Campi da basket. Rilevante solo se l'edizione 2027 — annunciata come *"due
sport"* — include il basket. Da tenere d'occhio quando escono i dettagli.

---

## Modelli e metodi

### [ML-KULeuven](https://github.com/ML-KULeuven) — `socceraction`
Il gruppo di machine learning della KU Leuven: SPADL, **VAEP**, xT. Attenzione:
`socceraction` nasce su dati **evento**, non tracking. Qui serve come riferimento
per le superfici di valore (una griglia xT da usare come pesatura), non come
pipeline da eseguire.

### [Hyunsung Kim](https://github.com/hyunsungkim) — traiettorie e tracking
Ricerca su imputazione di traiettorie e modelli su dati di tracking. Rilevante
**proprio per il nostro limite principale**: i dati broadcast hanno buchi, e
l'imputazione di traiettorie è la letteratura che se ne occupa.

### [Soccer Analytics Handbook](https://github.com/devinpleuler/analytics-handbook)
Un notebook di esempi di Devin Pleuler (aggiornato a febbraio 2023), su dati
StatsBomb e Metrica: visualizzazione, clustering, difficoltà del passaggio con
XGBoost, e per il tracking traiettorie, corse ad alta intensità e **time to
intercept**. Il pitch control non c'è più nella versione attuale.

### [unravelsports](https://github.com/UnravelSports/unravelsports) — `pip install unravelsports`
Graph neural network su dati di tracking: ogni frame diventa un grafo
giocatori-nodi. Contiene anche **Pressing Intensity** (Bekkers 2025) e il
riconoscimento delle formazioni (EFPI). Richiede Python ≥3.11. Le dipendenze di
base sono leggere (kloppy, polars, scipy); tensorflow e torch servono solo per
addestrare le GNN. Installato nel venv; la Pressing Intensity gira sui nostri dati
via kloppy, vedi `explorations/04-pressing-intensity-unravel.ipynb`.

**Nota sul campione:** una GNN su questi volumi si addestra — exPressV2 lo fa
su 36 partite — ma nel loro caso guadagna 0,013 di AUC su una regressione
logistica con le stesse feature. Se vai di GNN devi poter mostrare che il
margine giustifica la perdita di interpretabilità. Vedi
`notes/letteratura/pressing-exPressV2.md`.

### [Visual Exploratory Behaviour](https://github.com/USSF-ARTS/) — U.S. Soccer Federation
Analisi dello *scanning*: quante volte un giocatore gira la testa prima di
ricevere. Collegato al **body pose** del dataset SkillCorner (29 giunti, 2 partite),
che è il filone più inesplorato dei dati disponibili — e anche il più limitato
come campione.

---

## Edizione precedente

- [Submission Analytics Cup 1.0](https://github.com/PySport) — tutte le entry con codice e presentazioni
- [Presentazioni 2026](https://skillcorner.com/event-2026/analytics-cup-2026-presentations)

Vale la pena leggere gli abstract dei finalisti 2026 **prima** di scegliere la
domanda, per due motivi: capire il livello, ed evitare di rifare qualcosa che è
già stato presentato.

---

## Materiale ufficiale nel repo dei dati

`SkillCorner/opendata` contiene molto più dei dati — vale la pena guardarlo prima
di scrivere codice da zero:

| Percorso | Cosa |
|---|---|
| `notebooks/tutorials/01_*` | viz, z-score, archetipi di attaccanti |
| `notebooks/tutorials/02_*` | dynamic events, fasi di gioco, merge con tracking, **"Build Your Own Metric: Cutback Opportunities"** |
| `notebooks/tutorials/03_*` | basi del tracking, getting started con kloppy |
| `notebooks/tutorials/05_*` | body pose |
| `src/features/` | `DynamicEventsAggregator.py`, `PhasesOfPlayAggregator.py` già pronti |
| `viz_tools/` | due HTML standalone: Tracking Viewer e Dynamic Events Explorer |

Il tutorial `02_Part6` è di fatto un percorso guidato verso una submission
Research: costruisce una metrica nuova dai dati grezzi e la valuta.

## Glossari SkillCorner

- [Physical Data Glossary](https://skillcorner.crunch.help/en/glossaries/physical-data-glossary)
- [Glossari generali](https://skillcorner.crunch.help/en/glossaries)
