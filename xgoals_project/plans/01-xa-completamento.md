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

### 2. `our_xg` sovrastima di ~2x rispetto a `statsbomb_xg` 🟠

`our_xg` medio 0.1611 vs `statsbomb_xg` medio 0.0800, correlazione 0.6147. Un modello xG sano dovrebbe stare vicino al benchmark in media e correlare ben oltre 0.8.

Da indagare: `class_weight='balanced'` nel training del modello xG gonfierebbe sistematicamente le probabilità; oppure disallineamento tra le feature calcolate qui e quelle del training xG.

- [ ] Confrontare feature per feature le due pipeline su uno stesso set di tiri
- [ ] Verificare `class_weight` del modello xG salvato
- [ ] Se serve, ricalibrare (Platt/isotonic) e ri-valutare

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
