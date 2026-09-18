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
Sincronizzazione tracking ↔ eventi e feature di partita. Il valore vero è la
sincronizzazione: con dati broadcast, allineare il timestamp dell'evento al
frame giusto è un problema serio, e averlo già risolto vale.

### [floodlight](https://github.com/floodlight-sports/floodlight) — `pip install floodlight`
Toolkit multi-sport con un modello dati proprio (`XY`, `Pitch`, `Events`) e
metriche spaziali già implementate. Ha una curva di apprendimento sua: conviene
se usi le sue metriche, non come semplice lettore.

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
Notebook introduttivi di Devin Pleuler. Buon punto di partenza per pitch control
e modelli spaziali se non li hai mai implementati.

### [unravelsports](https://github.com/UnravelSports/unravelsports) — `pip install unravelsports`
Graph neural network su dati di tracking: ogni frame diventa un grafo
giocatori-nodi. Richiede Python ≥3.11 e porta dietro tensorflow — pesante.

**Nota sul campione:** con 20 partite, addestrare una GNN è quasi certamente
sovradimensionato. Il vincitore 2026 ha scelto l'ottimizzazione matematica
*proprio perché* il ML non regge su questi volumi, e lo ha scritto nell'abstract.
Se vai di GNN devi poter difendere quella scelta.

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
