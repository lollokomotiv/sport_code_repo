# 00 — Modello xG: stato e miglioramenti

**Obiettivo:** portare il modello xG dal "funziona ed è centrato" al "regge un esame tecnico" — più dati, forma funzionale adeguata, calibrazione documentata.

**Stato:** 🟡 Il modello è **calibrato** (media centrata sul benchmark e sul tasso gol reale), ma il suo **potere discriminante è modesto** e il valore aggiunto dei dati 360 non è dimostrato. Piano di miglioramento, non di riparazione.

---

## Diagnosi chiusa (2026-08-20)

Il sospetto iniziale — "il modello xG sovrastima 2x e correla 0.61 con StatsBomb" — **era falso**. Quei numeri venivano da `extract_shot_features` del notebook xA, non dal modello.

Test: stessi 554 tiri (25 match con 360), feature calcolate nei due modi, stesso modello.

| Feature usate | media xG | corr. Pearson con `statsbomb_xg` |
|---|---|---|
| `score_match.py` (identiche al training xG) | **0.0981** | **0.8508** |
| `extract_shot_features` del notebook xA | 0.1796 | 0.7122 |
| `statsbomb_xg` (benchmark) | 0.1006 | — |
| tasso gol reale | 0.0903 | — |

Il modello è centrato sul benchmark e sul tasso di gol reale. Lo script del test: `/tmp/compare_features.py` — da spostare in `tests/` (vedi step 0).

**Nota:** questi 554 tiri provengono in larga parte da match di training. Non è una valutazione su holdout — quella resta da fare e da documentare (step 4).

### Cause del disallineamento (→ ora tracciate nel piano 01)

1. **Formula dell'angolo invertita.** Il notebook xA usa `atan2(W · |y − GOAL_Y|, den)` dove la formula standard vuole `atan2(W · |x − GOAL_X|, den)`. Risultato: angolo 0 (il peggiore) per i tiri centrali, valori alti per quelli defilati — esattamente al contrario. Diverge sul 99,1% dei tiri.
2. **Convenzione dei flag binari.** Training e `score_match.py` passano `True`/`None` (l'imputer riempie con la mediana); il notebook xA passa `0.0`/`1.0`. Diverge sul 100% delle righe.
3. Differenze minori sulle feature 360: il notebook xA esclude il portiere dagli "opponents" e include entrambe le squadre nel cono verso la porta.

---

## Performance reali del modello

Fonte autorevole: `xg_notebook_statsbomb.ipynb`, cella 63 (output già salvato nel notebook).
Split a livello di match su 321 partite: **5.065** tiri di training, **1.238** validation, **1.569** test.

| | log loss | Brier | AUC |
|---|---|---|---|
| VAL baseline | 0.2594 | 0.0704 | 0.7745 |
| VAL 360 | 0.2637 | 0.0704 | **0.7596** |
| TEST baseline | 0.2993 | 0.0858 | 0.7647 |
| TEST 360 | 0.2896 | 0.0829 | **0.7869** |

> Da non confondere con il controllo in "Diagnosi chiusa": quello misura **calibrazione e
> allineamento delle feature** (media predetta vs benchmark) su tiri in gran parte di training.
> Queste sono le metriche di **discriminazione** su dati non visti. Sono cose diverse: un modello
> può essere perfettamente centrato in media e distinguere male i tiri buoni dai cattivi.

### Due conclusioni

**1. I dati 360 non aiutano in modo affidabile.** Su validation il modello 360 è *peggiore*
del baseline (AUC 0.7596 vs 0.7745); su test è *migliore* (0.7869 vs 0.7647). Segni opposti su
due campioni da ~1.300-1.600 tiri: è rumore, non un effetto. Il modello dichiarato "consigliato
in produzione" **non ha una superiorità dimostrata** su quello che ignora i freeze frame.

**2. Il guadagno sul predittore costante è modesto.** Un modello che predice sempre il tasso
base ha Brier ≈ r(1−r) ≈ 0.087. Il 360 su test fa 0.0829: circa il **4% meglio**. Su validation
il margine è più ampio (0.070, ~19%), e la distanza fra i due valori misura quanto siano
instabili queste stime con così pochi tiri.

Entrambe le conclusioni puntano nella stessa direzione: **servono più dati** (step 1).

---

## Step

### 0. Impedire che succeda di nuovo 🔴
La stessa feature engineering esiste in tre copie (notebook xG, notebook xA, `score_match.py`) e sono già divergute tre volte: `visible_area`, i dtype booleani, l'angolo.

- [ ] Estrarre le funzioni di feature engineering in un modulo unico importabile dai notebook e dall'inferenza
- [ ] Spostare `compare_features.py` in `tests/` come regressione: fallisce se le implementazioni divergono di nuovo

### 1. Decuplicare i dati di training 🟠
Il modello baseline non usa i dati 360: gli bastano posizione, parte del corpo, tipo di tiro — presenti in **tutti** i 3.464 match.

| | Match | Tiri non-rigore |
|---|---|---|
| Oggi (subset 360) | 326 | 8.218 |
| Disponibili | 3.464 | ~82.270 (stima da campione) |

Con ~8.000 tiri un modello non lineare va in overfitting e il rumore di valutazione è alto — le performance qui sopra ne sono la prova diretta: l'AUC del modello 360 si muove di ±0.03 fra validation e test, cambiando segno rispetto al baseline. Con ~82.000 tiri quell'oscillazione si riduce e la domanda "il 360 serve?" diventa rispondibile.

- [ ] Addestrare il baseline su tutti i match disponibili
- [ ] Rispondere alla domanda che ne nasce: **i dati 360 aggiungono valore rispetto alla sola geometria?** Confronto a tre: baseline 82k vs baseline 8k vs 360-enhanced 8k

### 2. Dare al modello la forma giusta 🟠
La relazione distanza→probabilità non è lineare nei log-odds: crolla nei primi metri e si appiattisce. Una `LogisticRegression` su `distance` grezza non può rappresentarla.

- [ ] Logistica con feature non lineari: `log(distance)`, `1/distance`, interazione `distance × angle`, distinzione testa/piede
- [ ] Gradient boosting (LightGBM) sulle stesse feature
- [ ] Confronto onesto su holdout. **Se la logistica ben ingegnerizzata regge, è quello il risultato** — vale più di un GBM messo per abitudine

### 3. Calibrazione documentata 🟡
- [ ] `CalibratedClassifierCV` (isotonic) su fold separato
- [ ] Reliability diagram: xG predetto in bin vs frequenza reale di gol
- [ ] Verifica che la somma xG per partita sia dell'ordine dei gol reali

### 4. Protocollo di valutazione definitivo 🟡
- [ ] Su holdout mai toccato: log loss, Brier, AUC, curva di calibrazione
- [ ] Pearson **e Spearman** contro `statsbomb_xg` (distribuzione molto asimmetrica)
- [x] Baseline "tasso base costante": Brier ≈ 0.087, contro 0.0829 del modello su test — guadagno ~4%
- [ ] Baseline "solo distanza", per separare il contributo della geometria da quello del resto
- [ ] Ripetere il confronto baseline vs 360 dopo lo step 1, su un campione abbastanza grande da distinguere effetto da rumore

### 5. Dettagli di dominio 🟡
- [ ] Rigori: esclusi dal modello, assegnati a costante (~0.76 storico)
- [ ] Punizioni dirette: feature esplicita o modello separato
- [ ] Tiri deviati (`shot.deflected`): valutare se scartarli
- [ ] Teste vs piedi: interazione con la distanza, non solo dummy

### 6. Documentazione
- [x] README: corretta l'affermazione su LightGBM (i modelli xG sono `LogisticRegression`)
- [ ] README: sostituire la nota sui limiti con i numeri reali di validazione una volta fatto lo step 4

---

## Criterio di chiusura

- Modello addestrato su ~82k tiri per il baseline
- Su holdout: Spearman > 0.80 con `statsbomb_xg`, media entro il 15% del tasso gol reale
- Reliability diagram e confronto con le baseline pubblicati nel README
