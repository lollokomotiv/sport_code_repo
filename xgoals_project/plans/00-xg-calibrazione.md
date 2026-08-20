# 00 — Calibrazione del modello xG

**Obiettivo:** portare il modello xG a un livello difendibile, usando `statsbomb_xg` come benchmark di riferimento. È il prerequisito di tutto il resto: xA, metriche avanzate e webapp ereditano l'errore di questo modello.

**Stato:** 🔴 Da iniziare — priorità massima.

---

## Il problema

Sui 5.335 tiri generati da key pass (sanity check in cella 24 di `xa_notebook_statsbomb.ipynb`):

| | our_xg | statsbomb_xg |
|---|---|---|
| Media | **0.1611** | 0.0800 |
| Correlazione (Pearson) | **0.6147** | |

Sono due problemi distinti che vanno affrontati separatamente:

1. **Livello** — sovrastima di ~2x in media. Problema di calibrazione.
2. **Ordinamento** — correlazione 0.61 significa che i due modelli non concordano su *quali* tiri siano buoni. È il problema più grave: si può essere perfettamente calibrati in media e comunque sbagliare tiro per tiro.

---

## Diagnosi già fatta (2026-08-19)

**✅ Verificato — il modello salvato è una LogisticRegression, non LightGBM.**

```
xg_model_360_no_penalty.joblib → Pipeline(pre=ColumnTransformer, model=LogisticRegression)
class_weight = None
```

Il README dichiara "Due modelli LightGBM addestrati sulle stesse partite": **è sbagliato**. Il notebook `xg_notebook_statsbomb.ipynb` (celle 50 e 56) costruisce e salva `LogisticRegression(max_iter=1000, solver="liblinear")`. LightGBM non compare da nessuna parte nella pipeline xG.

Questo è il sospetto numero uno per la correlazione bassa: una regressione logistica sulle feature grezze è lineare nei log-odds, quindi non può catturare né la non-linearità della relazione distanza→probabilità né le interazioni (es. distanza × angolo × pressione difensiva). StatsBomb usa un modello non lineare.

**✅ Escluso — `class_weight`.** Era la mia prima ipotesi per la sovrastima 2x, ma il modello ha `class_weight=None`, quindi non è quello.

**✅ Escluso — `visible_area_size`.** Il notebook xG accoppia correttamente la lista piatta (cella 32 riga 50, cella 46 riga 32), coerente con `score_match.py:266`. Nessun disallineamento su questa feature.

---

## Step

### 1. Capire da dove viene la sovrastima 2x
- [ ] Verificare la calibrazione sul *training set stesso*: media di `our_xg` vs tasso di gol reale. Se già lì sovrastima, è un problema del modello; se no, è un problema di popolazione (i tiri da key pass sono diversi da quelli su cui è stato addestrato)
- [ ] Confrontare le distribuzioni delle feature tra training set e `shot_df` dell'xA
- [ ] Controllare lo `shot_scope` usato in training vs quello dei tiri valutati

### 2. Sostituire il modello
- [ ] Addestrare un LightGBM sulle stesse feature e confrontarlo con la LogisticRegression su holdout (log loss, Brier, AUC, correlazione con `statsbomb_xg`)
- [ ] Tenere la LogisticRegression come baseline dichiarata, non buttarla: il confronto lineare vs non lineare è materiale interessante da raccontare
- [ ] Se la LogisticRegression regge, dirlo — un risultato negativo onesto vale più di un modello complicato senza motivo

### 3. Calibrare esplicitamente
- [ ] Applicare `CalibratedClassifierCV` (isotonic o Platt) sul modello scelto
- [ ] Curva di calibrazione: xG predetto in bin vs frequenza reale di gol
- [ ] Verificare che la somma degli xG per partita sia dello stesso ordine dei gol reali

### 4. Ri-valutare contro il benchmark
- [ ] Su holdout: correlazione Pearson **e Spearman** con `statsbomb_xg` (Spearman è più informativa su una distribuzione così asimmetrica)
- [ ] Scatter plot our_xg vs statsbomb_xg con la bisettrice
- [ ] Identificare dove i due modelli divergono di più — è lì che si capisce cosa manca

### 5. Correggere la documentazione
- [ ] Aggiornare il README: rimuovere l'affermazione su LightGBM, descrivere il modello effettivamente in uso
- [ ] Documentare le metriche di validazione reali, non quelle attese

---

## Criterio di chiusura

Il filone si chiude quando, su holdout:
- la media di `our_xg` è entro ~15% del tasso di gol reale;
- la correlazione di Spearman con `statsbomb_xg` supera 0.80;
- esiste una curva di calibrazione che lo dimostra.
