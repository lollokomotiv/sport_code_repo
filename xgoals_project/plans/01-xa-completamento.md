# 01 — Completamento xA

**Obiettivo:** modello xA affidabile + assist map utilizzabile, con la stessa qualità di validazione già raggiunta sul modello xG.

**Stato:** 🟡 In corso — il notebook gira end-to-end (93s) e salva i modelli, ma tre problemi di correttezza restano aperti e vanno chiusi *prima* di considerare valide le visualizzazioni.

---

## Fatto (2026-08-19)

- [x] `run_notebook.py` — runner headless, esegue le celle in ordine in un solo processo (niente code VS Code). Knob `--limit-matches` per smoke test da ~7s, `--models-dir` per non sovrascrivere i modelli di produzione, `--figures-dir` per salvare le figure.
- [x] Fix dtype bool: `cross`/`switch`/`through_ball` erano `bool` Python, rifiutati da `SimpleImputer`. Corretto in `int()` sia nel notebook (cella 10) sia in `inference/score_match.py::_pass_meta`.
- [x] Fix `visible_area`: StatsBomb la serializza piatta `[x1,y1,x2,y2,...]`, il notebook la trattava come lista di coppie. Allineato alla convenzione di `score_match.py:266`.
- [x] Run completo su 324 match → salvati `xa_pshot_360.joblib`, `xa_zone_xg_lookup.joblib`, `xa_holdout_match_ids.json`.

**Risultati del run completo:**

| Metrica | Valore |
|---|---|
| Passaggi completati | 268.356 (5.334 shot assist, 1,99%) |
| Copertura 360 | 84,3% |
| P_shot baseline | AUC 0.9367 |
| P_shot 360-enhanced | (vedi log) |
| xA medio key pass | 0.1385 |
| xA medio non-key pass | 0.02503 |

---

## Aperto — da chiudere in ordine

### 1. `is_goal` è 0 su tutti i 5.335 tiri da key pass 🔴

`shot_df['is_goal'].sum()` restituisce 0. Verificato che i dati contengono goal da key pass (match 3788741: 1 goal su 19 tiri da key pass), quindi è un bug di estrazione, **causa non ancora identificata**.

Conseguenza: la calibrazione match-level (cella 29) confronta xA con goal sempre a zero — il grafico prodotto è privo di significato.

Prima ipotesi da verificare: in cella 24 `'is_goal'` viene impostato *prima* di `**feat`, quindi se `extract_shot_features` restituisce una chiave omonima la sovrascrive.

- [ ] Isolare la causa
- [ ] Correggere e ri-verificare che il tasso goal sia ~10-12% dei tiri da key pass

### 2. `extract_shot_features` calcola feature diverse dal training xG 🔴

**Causa identificata (2026-08-20).** La sovrastima di `our_xg` (0.1611 vs 0.0800 di StatsBomb) non veniva dal modello xG — che è sano — ma da questa funzione. Su 554 tiri, le stesse feature calcolate da `score_match.py` danno media 0.0981 e correlazione 0.8508; calcolate qui danno 0.1796 e 0.7122.

**a) Formula dell'angolo invertita** — diverge sul 99,1% dei tiri:
```python
# sbagliato (qui):     atan2(W * abs(y - GOAL_Y), den)
# corretto (standard): atan2(W * abs(x - GOAL_X), den)
```
L'angolo risulta 0 — il peggiore — per i tiri centrali, e massimo per quelli defilati.

**b) Convenzione dei flag binari** — diverge sul 100% delle righe. Training e inferenza passano `True`/`None` (l'imputer riempie con la mediana); qui si passa `0.0`/`1.0`.

**c) Feature 360 minori** — qui il portiere è escluso dagli "opponents" e il cono verso la porta include entrambe le squadre.

- [ ] **Eliminare la funzione e importare quella di `score_match.py`**, invece di correggerla: è la terza volta che le copie divergono (vedi piano 00, step 0)
- [ ] Rieseguire il notebook e verificare che `our_xg` torni ~0.10

> ⚠️ La stessa formula sbagliata dell'angolo è in `geometry_features` (cella 10) come `receiver_angle_goal`, feature del modello P_shot. Lì training e inferenza sono coerenti fra loro, quindi non è un bug bloccante — ma è una feature che misura il contrario di quel che dovrebbe, e correggerla dovrebbe migliorare il P_shot.

### 3. `zone_xg_lookup`: convenzione dei bin incoerente 🟠

In cella 26 i bin sono costruiti da `distance` e `angle` con una trasformazione proxy (`max(60, GOAL_X - d)`, `GOAL_Y + a*10`), mentre in inferenza `score_match.py::_zone_xg_lookup` riceve `end_x`/`end_y` reali. Sintomo coerente: solo **23 zone popolate su 96**.

Se le convenzioni non coincidono, l'xA dei non-key pass è sistematicamente sbagliato.

- [ ] Riscrivere i bin su `(x, y)` reali del tiro in entrambi i lati
- [ ] Verificare che le zone popolate salgano a copertura sensata

### 4. Visualizzazioni

- [ ] Generare la prima assist map reale (`--score-xa --plot-assist-map`) e verificarla a occhio
- [ ] Confrontarla con la shot map esistente per coerenza di stile
- [ ] Salvare un esempio in `figures/` da usare come materiale di portfolio

### 5. Validazione onesta

- [ ] Valutare xA sull'holdout (mai usato in training), non solo su validation
- [ ] Riportare log loss / Brier / AUC del P_shot e calibrazione dell'xA finale
- [ ] Confronto esplicito con `statsbomb_xg` come benchmark

---

## Note operative

```bash
# smoke test (~7s) — sempre prima di un run vero
python3 run_notebook.py xa_notebook_statsbomb.ipynb \
    --limit-matches 8 --models-dir /tmp/xa_smoke --figures-dir /tmp/xa_smoke/figs

# run completo (~95s)
python3 run_notebook.py xa_notebook_statsbomb.ipynb --figures-dir figures/xa
```
