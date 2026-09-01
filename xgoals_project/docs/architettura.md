# Architettura

Come sono collegati i pezzi del progetto, e dove ciascuno può rompersi.

---

## Il flusso

```
StatsBomb open data
  ├── events/<match_id>.json        3.464 match — tiri, passaggi, eventi
  └── three-sixty/<match_id>.json     326 match — posizione dei giocatori al momento dell'evento
                │
                ▼
        feature engineering          ← esiste in TRE copie (vedi "Il punto fragile")
                │
     ┌──────────┴──────────┐
     ▼                     ▼
  modello xG            modello P_shot
  P(gol | tiro)         P(tiro | passaggio)
     │                     │
     └──────────┬──────────┘
                ▼
         xA = P_shot × xG
                │
                ▼
     inference/score_match.py
                │
     ┌──────────┴──────────┐
     ▼                     ▼
  shot map              assist map
```

## I modelli

| File in `models/` | Cosa stima | Algoritmo | Addestrato da |
|---|---|---|---|
| `xg_model_360.joblib` | P(gol \| tiro), tutti i tiri | `LogisticRegression` | `xg_notebook_statsbomb.ipynb` |
| `xg_model_360_no_penalty.joblib` | P(gol \| tiro), rigori esclusi — **in produzione** | `LogisticRegression` | `xg_notebook_statsbomb.ipynb` |
| `xa_pshot_360.joblib` | P(il passaggio genera un tiro) | `LGBMClassifier` | `xa_notebook_statsbomb.ipynb` |
| `xa_zone_xg_lookup.joblib` | xG medio per zona del campo (fallback) | lookup, non un modello | `xa_notebook_statsbomb.ipynb` |

Tutti salvati come `Pipeline` sklearn complete: il preprocessing (imputazione, one-hot) viaggia col modello, quindi l'inferenza non deve replicarlo.

## Il calcolo dell'xA

```
passaggio completato
        │
        ├── ha generato un tiro?  ──SÌ──►  xA = P_shot × xG(tiro reale)
        │                                        ↑ modello xG applicato al tiro effettivo
        └──────────────────────────NO──►  xA = P_shot × xG_medio(zona d'arrivo)
                                                 ↑ lookup, l'anello debole (vedi plans/01)
```

## Il punto fragile

**La feature engineering esiste in tre copie**: nel notebook xG, nel notebook xA e in `inference/score_match.py`. Nessuna importa dalle altre.

Sono già divergute tre volte:

| Cosa | Sintomo | Doc |
|---|---|---|
| `visible_area` trattata come coppie invece che lista piatta | `TypeError` al training | — |
| Flag booleani `bool` invece di `int` | `SimpleImputer` rifiuta il dtype | — |
| Formula dell'angolo invertita | xG sovrastimato del doppio, silenziosamente | [indagine](indagini/01-disallineamento-feature.md) |

I primi due si manifestano come eccezioni: fastidiosi ma innocui, il codice si ferma. Il terzo **non ha prodotto errori** — ha solo prodotto numeri sbagliati, che sono sopravvissuti fino a diventare una diagnosi sbagliata sul modello xG.

La regressione è coperta da [`tests/compare_features.py`](../tests/compare_features.py); la soluzione strutturale — un modulo unico — è lo step 0 di [`plans/00`](../plans/00-xg-calibrazione.md).

## Esecuzione

I notebook si eseguono con [`run_notebook.py`](../run_notebook.py), non a mano cella per cella:

```bash
# smoke test (~7s): 8 match, modelli in dir temporanea
python3 run_notebook.py xa_notebook_statsbomb.ipynb \
    --limit-matches 8 --models-dir /tmp/smoke --figures-dir /tmp/smoke/figs

# run completo (~95s su 324 match)
python3 run_notebook.py xa_notebook_statsbomb.ipynb --figures-dir figures/xa
```

Esegue le celle in ordine in un solo processo e si ferma alla prima eccezione, stampando sorgente e traceback. Evita il problema delle celle accodate nel kernel di VS Code.

## Vincoli dei dati

- **326 match su 3.464** hanno i dati 360. Ogni feature basata sul freeze frame vive solo lì.
- I freeze frame sono **istantanee** al momento dell'evento, non tracking continuo: niente traiettorie, niente velocità.
- `visible_area` delimita l'inquadratura: i giocatori fuori campo visivo sono *ignoti*, non *assenti*.
- I tiri non-rigore disponibili sono ~8.200 nel subset 360 e ~82.000 su tutti i match.
