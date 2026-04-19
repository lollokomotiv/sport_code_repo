# xGoals Project

Modelli di **Expected Goals (xG)** e **Expected Assists (xA)** addestrati sui dati open-source di StatsBomb, con pipeline di inferenza per valutare singole partite.

---

## Obiettivo

Sviluppare modelli probabilistici per stimare la qualità di tiri e assist a partire da dati evento e dati di freeze frame (StatsBomb 360), e poi fare inferenza su una partita specifica per ottenere xG per ogni tiro.

---

## Struttura del progetto

```
xgoals_project/
├── xg_notebook_statsbomb.ipynb        # Notebook principale: training del modello xG (produzione)
├── expected-goals-xg-model.ipynb      # Versione legacy del modello xG (confronto tra algoritmi)
├── expected-assist-xg-model.ipynb     # Modello xA (Expected Assists)
├── xA-model.ipynb                     # Versione alternativa del modello xA
├── requirements.txt                   # Dipendenze Python
├── models/
│   ├── xg_model_360.joblib            # Modello xG addestrato con dati 360 (produzione)
│   ├── xg_model_360_no_penalty.joblib # Variante senza rigori
│   └── holdout_match_ids.json         # ID delle partite riservate per il test finale
└── inference/
    ├── score_match.py                 # Motore di inferenza principale
    └── run_holdout_scoring.py         # Script per valutare il modello sul holdout set
```

---

## Dati

**Fonte**: [StatsBomb Open Data](https://github.com/statsbomb/open-data)

La cartella dati si trova in:
```
/Users/lorenzoguercio/Documents/Projects/sport_data/open-data/data/
├── competitions.json
├── matches/<league_id>/<season_id>/
├── events/<match_id>.json       # Eventi di partita (tiri, passaggi, ecc.)
└── three-sixty/<match_id>.json  # Freeze frame al momento del tiro
```

I dati di freeze frame (360) contengono le posizioni di tutti i giocatori visibili nella ripresa video nel momento esatto del tiro — permettono di calcolare metriche avanzate come la distanza dei difensori e la posizione del portiere.

---

## Modello xG

### Approccio

Due modelli LightGBM addestrati sulle stesse partite, con split a livello di partita (train 70% / validation 15% / holdout test 15%) per evitare data leakage:

| Modello | Feature | Uso |
|---|---|---|
| **Baseline** | Geometria + contesto tiro | Riferimento senza dati 360 |
| **360-enhanced** | Baseline + freeze frame | Produzione (consigliato) |

### Feature utilizzate

**Geometriche:**
- `distance` — distanza dalla porta
- `angle` — angolo di tiro rispetto alla porta

**Contesto tiro:**
- `body_part`, `shot_type`, `shot_technique`, `play_pattern`
- `first_time`, `one_on_one`, `under_pressure`

**Freeze frame (360):**
- `nearest_defender_dist` — distanza dal difensore più vicino
- `n_defenders_within_1m/2m/3m` — pressione difensiva ravvicinata
- `keeper_dist_to_shot`, `keeper_dist_to_goal` — posizionamento del portiere
- `keeper_present` — portiere visibile nel frame
- `n_players_in_cone_to_goal` — difensori che bloccano il cono di tiro
- `visible_area_size` — copertura della ripresa video

### Training

Il notebook principale è `xg_notebook_statsbomb.ipynb`. Eseguendolo:
1. Carica tutti i match StatsBomb con dati 360 disponibili
2. Filtra i tiri (open play, esclusi corner e rigori)
3. Calcola le feature geometriche e di freeze frame
4. Addestra i due modelli e salva `models/xg_model_360.joblib`
5. Valuta su validation e holdout (log loss, Brier score, AUC)
6. Analizza calibrazione e importanza delle feature

---

## Inferenza su una partita

Il modulo `inference/score_match.py` carica il modello addestrato e calcola l'xG per ogni tiro di una partita.

### Da riga di comando

```bash
python3 inference/score_match.py \
  --data-root /path/to/statsbomb/data \
  --model-path models/xg_model_360.joblib \
  --match-id <MATCH_ID> \
  --require-360 \
  --shot-scope open_play
```

**Opzioni `--shot-scope`:**
- `open_play` — solo azione, esclude rigori e calci piazzati
- `all_non_penalty` — tutto tranne i rigori

### Output

Per ogni tiro: `match_id`, `team`, `player`, `xg`, `goal`, `distance`, `angle`, e le feature principali.

Riepilogo per squadra:
```
team     shots  xg    goals  goal_diff
Team A   12     8.34  7      -1.34
Team B   8      5.12  6      +0.88
```

### Valutazione sul holdout set

```bash
python3 inference/run_holdout_scoring.py
```

Usa i match ID salvati in `models/holdout_match_ids.json` (2 partite riservate durante il training).

---

## Installazione

```bash
pip install -r requirements.txt
```

Dipendenze principali: `lightgbm`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `joblib`.

---

## Modello xA

I notebook `expected-assist-xg-model.ipynb` e `xA-model.ipynb` contengono esperimenti per il modello di Expected Assists, con una pipeline parallela a quella xG applicata ai passaggi che generano tiri.
