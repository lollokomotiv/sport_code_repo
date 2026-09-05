# Analisi

Ogni sottocartella è **un'analisi indipendente**: una domanda, i dati che
servono a risponderle, il risultato e i suoi limiti. Non è un progetto unico
spezzato in pezzi — due analisi possono usare fonti diverse, metodi diversi e
arrivare a conclusioni scollegate.

Ciò che le tiene insieme è solo il codice condiviso in [`../lib/`](../lib/) e i
dati in [`../data/`](../data/): nessuna analisi scarica dati per conto suo,
nessuna reimplementa un caricamento che esiste già.

## Indice

| Cartella | Domanda | Fonte | Stato |
|---|---|---|---|
| _(nessuna analisi ancora)_ | | | |

> Aggiorna questa tabella quando ne inizi una. Un'analisi che nessuno trova non
> esiste, e l'indice è l'unica cosa che qualcuno legge davvero.

## Come iniziarne una

```bash
cp -r analyses/_template analyses/<slug-in-inglese>
```

Poi, **prima di scrivere codice**, compila le prime due sezioni del README
(`Domanda` e `Dati`). Se la domanda non sta in una riga, non è ancora una
domanda; se i dati disponibili non possono risponderle, meglio scoprirlo su
carta che dopo tre giorni di feature engineering.

## Regole minime

1. **Una domanda per cartella.** Se ne emerge una seconda, è un'altra cartella.
2. **Niente dati dentro l'analisi.** I file grezzi stanno in `data/raw/`, i
   derivati riusabili in `data/processed/` (`lib.paths.processed_path()`).
3. **Il codice che serve a due analisi sale in `lib/`.** Alla seconda copia
   incolla, spostalo.
4. **I limiti stanno nel README dell'analisi**, non in una conversazione. Il
   Match Charting Project non è un campione casuale: quasi ogni conclusione
   costruita su di esso ha un limite da dichiarare.
5. **Un risultato negativo si tiene.** "Questa feature non aggiunge nulla"
   documentato vale più di un grafico che sembra buono.
