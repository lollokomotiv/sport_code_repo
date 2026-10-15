# CLAUDE.md — Volleyball Score

## Cos'è

Un'app che, dal video di una partita di pallavolo indoor, ricostruisce la sequenza
degli scambi e l'esito di ciascuno: quale squadra fa il punto e come finisce lo
scambio. Gli utenti sono una squadra amatoriale reale; i video sono pochi, da una
camera fissa, a bassa risoluzione.

Progetto in fase di impostazione: la definizione di "score" per la prima versione
non è ancora fissata (vedi le domande aperte nel README).

## I video sono privati

Il repo è pubblico. I video ritraggono persone reali:

- non si versionano i video, né frame, ritagli, GIF o figure che mostrano i
  giocatori. Stanno in `data/raw/`, ignorata da git;
- le etichette fatte a mano stanno in `data/labels/`, anche quella ignorata finché
  non si decide cosa contengono (nomi, numeri di maglia);
- prima di produrre qualunque immagine destinata a uscire dal computer, chiedi.

## Validazione

- La verità di riferimento è l'annotazione a mano degli scambi. Ogni metrica si
  misura contro di essa e si riporta col denominatore: "41/47 scambi trovati",
  non "87%".
- Split per partita: il video su cui si regolano soglie e parametri non è quello
  su cui si misura il risultato.
- Con pochi video i numeri sono indicativi. Va scritto accanto al numero.

## Regole di gioco che servono come controllo

- **Rally point system**: ogni scambio assegna un punto, e chi lo vince batte lo
  scambio successivo. Il vincitore dello scambio *n* è il lato che serve lo
  scambio *n+1*; fa eccezione l'ultimo scambio del set.
- Le squadre cambiano campo a ogni set: la corrispondenza lato → squadra va
  fissata set per set.
- Set a 25, tie-break a 15, vantaggio di 2. Un punteggio finale ricostruito che
  non è un punteggio legale (26-20, 25-24) indica un errore nella pipeline.
- Numero di scambi del set = somma dei punti. Se non torna, mancano o avanzano
  scambi.

## Codice di terzi

Prima di copiare codice da una repo esterna controlla la licenza in
`docs/prior-work.md`. VolleyVision ha due licenze in conflitto (AGPL-3.0 nel file,
CC BY-NC-ND 4.0 nel README): non se ne copia codice.

## Convenzioni

- Commenti, docstring e documentazione in italiano; `README.md` in inglese.
- I commenti spiegano perché, non cosa.
