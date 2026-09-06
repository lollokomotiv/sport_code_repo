# Notebook

Spazio per guardare i dati, non per produrre risultati. Quello che nasce qui e
merita di restare va spostato in un'analisi sotto [`../analyses/`](../analyses/),
con la sua domanda e i suoi limiti dichiarati.

| Notebook | A cosa serve |
|---|---|
| [`01-esplorare-i-file.ipynb`](01-esplorare-i-file.ipynb) | aprire un CSV qualsiasi, vederne lo schema e guardarci dentro |
| [`02-query-di-esempio.ipynb`](02-query-di-esempio.ipynb) | ricette pronte: scontri diretti, stagione di un giocatore, statistiche per superficie, dettaglio di un match |

## Avvio

```bash
cd tennis_project
pip install -r requirements.txt
jupyter lab notebooks/
```

La prima cella di ogni notebook risale le cartelle finché non trova `lib/`,
quindi funziona da qualunque posizione senza percorsi scritti a mano.

## Due funzioni che risparmiano tempo

```python
explore.read("Rally")      # apre il file da un pezzo del nome, senza il percorso
explore.schema(df)         # tipo, valori distinti, vuoti e valori di ogni colonna
```

## L'errore da non fare

I file di statistiche del Match Charting Project hanno **una riga per ogni
livello di aggregazione**, più una riga `Total`. Sommarle tutte conta gli stessi
punti più volte.

La colonna `row` contiene il livello, e significa una cosa diversa in ogni file:
la lunghezza dello scambio in `Rally`, il tipo di punto in `KeyPointsServe`, il
tipo di colpo in `ShotTypes`. Guarda sempre `df["row"].value_counts()` prima di
aggregare.

I file di TennisMyLife non hanno questo problema: una riga per match, punto.
