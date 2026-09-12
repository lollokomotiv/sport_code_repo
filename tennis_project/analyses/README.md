# Analisi

Le analisi sono raggruppate **per matchup**: una cartella per confronto diretto
(`alcaraz-zverev/`), e dentro una sottocartella per ogni analisi fatta su quel
confronto (`smorzate/`, `vinte-perse/`, `direzione-servizio/`…). Le analisi su
un giocatore solo, contro tutto il circuito, stanno nella cartella del giocatore
(`zverev/`).

```
analyses/
├── _template/                 # una singola analisi: si copia dentro un matchup
├── alcaraz-zverev/
│   ├── README.md              # copertura dell'H2H + indice delle analisi del matchup
│   ├── smorzate/              # README.md, run.py, figures/, posts/
│   └── vinte-perse/
└── zverev/                    # analisi su un giocatore solo
    └── doppi-falli/
```

Ogni sottocartella resta **un'analisi indipendente**: una domanda, i dati che
servono a risponderle, il risultato e i suoi limiti. Stare nello stesso matchup
non le lega: due analisi possono usare fonti diverse, metodi diversi e arrivare
a conclusioni scollegate.

Ciò che le tiene insieme è solo il codice condiviso in [`../lib/`](../lib/) e i
dati in [`../data/`](../data/): nessuna analisi scarica dati per conto suo,
nessuna reimplementa un caricamento che esiste già.

## Nomi

- **Matchup**: i due cognomi in ordine alfabetico, minuscoli, col trattino —
  `alcaraz-zverev`, mai `zverev-alcaraz`. Così lo stesso confronto non finisce
  in due cartelle a seconda di chi lo nomina per primo.
- **Giocatore singolo**: il cognome — `zverev`.
- **Analisi**: il tema, senza ripetere i giocatori — `smorzate`, non
  `alcaraz-zverev-smorzate`.

## Indice

| Matchup | Analisi | Stato |
|---|---|---|
| [alcaraz/](alcaraz/) | [smorzate/](alcaraz/smorzate/) — quante smorzate gioca Alcaraz rispetto al circuito? (circa il doppio: 3,01 ogni 100 colpi, 9° su 130 a parità di superficie) | Conclusa |
| [alcaraz-paul/](alcaraz-paul/) | [h2h/](alcaraz-paul/h2h/) — dove si è deciso il confronto diretto | In corso |
| | [smorzate/](alcaraz-paul/smorzate/) — Alcaraz gioca meno smorzate contro Paul? (no; l'unico effetto solido è che sale meno a rete) | Conclusa |

> Aggiorna questa tabella e il README del matchup quando inizi un'analisi.
> Un'analisi che nessuno trova non esiste, e l'indice è l'unica cosa che
> qualcuno legge davvero.

## Come iniziarne una

```bash
mkdir -p analyses/<matchup>
cp -r analyses/_template analyses/<matchup>/<analisi>
```

Se la cartella del matchup è nuova, crea anche il suo `README.md`: copertura
dell'H2H (incontri giocati, annotati, chi ha vinto i mancanti) e la tabella
delle analisi. Se esiste già, leggilo prima: dice cosa è già stato fatto e cosa
si può riusare.

Poi, **prima di scrivere codice**, compila le prime due sezioni del README
dell'analisi (`Domanda` e `Dati`). Se la domanda non sta in una riga, non è
ancora una domanda; se i dati disponibili non possono risponderle, meglio
scoprirlo su carta che dopo tre giorni di feature engineering.

## Regole minime

1. **Una domanda per sottocartella.** Se ne emerge una seconda, è un'altra
   sottocartella dello stesso matchup.
2. **Niente dati dentro l'analisi.** I file grezzi stanno in `data/raw/`, i
   derivati riusabili in `data/processed/` (`lib.paths.processed_path()`).
3. **Il codice che serve a due analisi sale in `lib/`.** Alla seconda copia
   incolla, spostalo — anche se le due analisi stanno nello stesso matchup.
4. **I limiti stanno nel README dell'analisi**, non in una conversazione. Il
   Match Charting Project non è un campione casuale: quasi ogni conclusione
   costruita su di esso ha un limite da dichiarare.
5. **Un risultato negativo si tiene.** "Questa feature non aggiunge nulla"
   documentato vale più di un grafico che sembra buono.
