# <Titolo dell'analisi>

> Template. Copia questa cartella, rinominala, sostituisci i segnaposto ed
> elimina questa riga.

## Domanda

Una riga, verificabile. Non "analizzare il servizio", ma "quanto vale il
vantaggio del servizio su erba rispetto alla terra, a parità di giocatore?".

## Dati

- **Fonte**: (Match Charting Project / tennis-data.co.uk / altro — vedi
  [`../../docs/fonti-dati.md`](../../docs/fonti-dati.md))
- **Comando di download**: `python3 -m lib.download ...`
- **Filtri applicati**: periodo, superficie, livello di torneo, minimo di match
  per giocatore…
- **Righe rimaste** dopo i filtri: N

## Metodo

Cosa viene calcolato e perché quel calcolo risponde alla domanda. Se c'è un
modello: cosa predice, com'è diviso il campione, contro quale baseline è
confrontato.

## Risultato

Il numero o il grafico, con la sua incertezza. Un valore puntuale senza un
intervallo o un confronto è un aneddoto.

## Limiti

Cosa **non** dimostra questo risultato. Almeno:

- il campione (il MCP è annotato a mano e sbilanciato verso i big match);
- le confondenti non controllate;
- il salto da correlazione a causa, se è stato fatto.

## Come si riproduce

```bash
python3 -m lib.download <...>
python3 analyses/<slug>/run.py
```
