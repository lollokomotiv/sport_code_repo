# 02 — Costruire la submission

**Bloccato da [01](01-scegliere-la-domanda.md).**

## Checklist dei requisiti formali

Dal template PySport. Violarli *"may result in a point deduction or
disqualification"*.

- [ ] Il repo è un **fork** di `PySport/analytics_cup_<track>`, edizione corretta
- [ ] Il lavoro è sul branch **`main`** del fork
- [ ] `submission.ipynb` nella **root**, **max 2000 parole**
- [ ] Tutto il resto del codice in `src/`, **importato** nel notebook
- [ ] Abstract nel `README.md`, **max 500 parole**, struttura
      Introduction / Methods / Results / Conclusion
- [ ] **Max 2 figure**, o 2 tabelle, o 1 figura + 1 tabella
- [ ] **Nessun file di dati grosso** nel repo — solo `load_open_data()`
- [ ] Il notebook gira su un **ambiente pulito** (testato da zero, non sul tuo venv)
- [ ] `requirements.txt` minimo: solo ciò che il notebook importa davvero
- [ ] Tutto in **inglese**
- [ ] Consegna su [pretalx.pysport.org](https://pretalx.pysport.org)

## Il passaggio dal workbench alla submission

Il codice nasce in `explorations/` e si trasferisce in `submission/src/` quando è
maturo. **L'unica cosa che va riscritta di proposito è il caricamento dati:**

```python
# workbench — clone locale, veloce
pd.read_json(DATA / "matches" / str(mid) / f"{mid}_tracking_extrapolated.jsonl", lines=True)

# submission — da GitHub, riproducibile ovunque
skillcorner.load_open_data(match_id=mid, coordinates="skillcorner")
```

Isolalo in una funzione sola invece di spargere percorsi nel codice: così la
riscrittura tocca un punto solo.

## Verifica finale dell'ambiente pulito

Non basta che giri sul tuo venv — quello ha dentro mezzo workbench.

```bash
python3 -m venv /tmp/ac-clean
/tmp/ac-clean/bin/pip install -r requirements.txt
/tmp/ac-clean/bin/jupyter nbconvert --execute --to notebook --inplace submission.ipynb
```

Se fallisce, manca una dipendenza nel `requirements.txt` o c'è un percorso locale
rimasto nel codice.

## Conteggio parole

I limiti (2000 nel notebook, 500 nell'abstract) sono sulle **parole**, e vanno
verificati, non stimati a occhio:

```bash
python3 -c "
import json,sys,re
nb=json.load(open('submission.ipynb'))
txt=' '.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='markdown')
print(len(re.findall(r'\S+', txt)), 'parole nelle celle markdown')
"
```
