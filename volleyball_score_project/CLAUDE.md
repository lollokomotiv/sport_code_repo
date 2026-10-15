# CLAUDE.md — Volleyball Score

## Cos'è

Un'app che, dal video di una partita di pallavolo indoor, ricostruisce la sequenza
degli scambi e quale squadra vince ciascuno, e da lì il punteggio di ogni set. Gli
utenti sono una squadra amatoriale reale (FIPAV serie D).

Decisioni prese (dettagli nel README):

- **prima versione**: chi vince ogni scambio e il punteggio. Come finisce lo scambio
  (ace, attacco, muro, errore) è la seconda tappa, ma lo schema delle etichette ha
  già il campo;
- **flusso**: la pipeline gira in locale dopo la partita; una schermata di revisione
  permette di correggere. L'obiettivo è che correggere costi molto meno che fare lo
  scouting a mano, non zero errori;
- **una sola camera**, alta, dietro una linea di fondo, secondo
  `docs/protocollo-ripresa.md`. Niente seconda camera finché non serve davvero;
- il progetto è anche portfolio.

I prossimi passi stanno nella issue mappa
[#10](https://github.com/lollokomotiv/sport_code_repo/issues/10) e nelle sue figlie.
Nelle issue, che sono pubbliche, niente immagini dai video e niente nomi di squadre
o palestre.

## I video sono privati

Il repo è pubblico. I video ritraggono persone reali:

- non si versionano i video, né frame, ritagli, GIF o figure che mostrano i
  giocatori. Stanno in `data/raw/`, ignorata da git;
- le etichette fatte a mano stanno in `data/labels/`, anche quella ignorata finché
  non si decide cosa contengono (nomi, numeri di maglia);
- la squadra dà il consenso per sé, non per avversarie e arbitri. Le immagini
  pubbliche sono schemi del campo (traccia della palla, linea del tempo degli
  scambi) oppure clip con i volti sfocati. Mai clip in chiaro;
- prima di produrre qualunque immagine destinata a uscire dal computer, chiedi.

## Validazione

- La verità di riferimento è l'annotazione a mano degli scambi. Ogni metrica si
  misura contro di essa e si riporta col denominatore: "41/47 scambi trovati",
  non "87%".
- Divisione dei dati:
  - video vecchi: prima prova del tracker, caso peggiore;
  - partita 1 del campionato: sviluppo e regolazione delle soglie;
  - partita 2: solo misura finale. Non la si apre durante lo sviluppo;
  - partite pubbliche: immagini e prove, mai misure.
- Con poche partite i numeri sono indicativi. Va scritto accanto al numero.

## Regole di gioco che servono come controllo

Regolamento FIPAV: al meglio dei 5 set, set a 25, quinto set a 15, vantaggio di 2.

- **Rally point system**: ogni scambio assegna un punto, e chi lo vince batte lo
  scambio successivo. Il vincitore dello scambio *n* è il lato che serve lo
  scambio *n+1*.
- L'ultimo scambio del set lo vince per forza chi chiude il set: si ricava dal
  punteggio precedente, serve solo sapere che è l'ultimo.
- Le squadre cambiano campo a ogni set: la corrispondenza lato → squadra va
  fissata set per set (chi filma la annota).
- Un punteggio finale ricostruito che non è un punteggio legale (26-20, 25-24)
  indica un errore nella pipeline.
- Numero di scambi del set = somma dei punti. Se non torna, mancano o avanzano
  scambi. Eccezione rara: punti assegnati senza scambio (sanzioni, fallo di
  posizione).
- I punteggi dei set annotati a bordo campo sono un controllo indipendente.

## Codice di terzi

Prima di copiare codice da una repo esterna controlla la licenza in
`docs/prior-work.md`. VolleyVision ha due licenze in conflitto (AGPL-3.0 nel file,
CC BY-NC-ND 4.0 nel README): non se ne copia codice.

## Convenzioni

- Commenti, docstring e documentazione in italiano; `README.md` in inglese.
- I commenti spiegano perché, non cosa.
