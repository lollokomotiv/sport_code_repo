# CLAUDE.md — Volleyball Score

## Cos'è

Un'app che, dal video di una partita di pallavolo indoor, ricostruisce gli scambi e
i tocchi: chi vince ogni scambio, il punteggio di ogni set, il tipo di ogni tocco.
Gli utenti sono una squadra amatoriale reale (FIPAV serie D).

Decisioni prese (dettagli nel README):

- **cosa si produce**: scambi e chi li vince, punteggio; ogni tocco con momento,
  lato e tipo (battuta, ricezione, alzata, attacco, muro, difesa, free ball; codici
  DataVolley S R E A B D F), per entrambe le squadre; punti ed errori dall'ultimo
  tocco; battuta attribuita alla giocatrice dalla rotazione, numero facoltativo a
  mano sugli altri tocchi;
- **cosa no**: valutazioni DataVolley (`#` `+` `!` `-` `/` `=`) e riconoscimento
  automatico delle giocatrici. Sono ricerca, fuori portata;
- **il tipo di tocco viene dalla grammatica dello scambio**, non da un
  classificatore. La visione artificiale deve trovare momento e lato di ogni
  tocco: è lì il rischio, ed è lì che si misura;
- **flusso**: la pipeline gira in locale dopo la partita; una schermata di revisione
  permette di correggere. L'obiettivo è che correggere costi molto meno che fare lo
  scouting a mano, non zero errori;
- **una sola camera**: un iPhone a 1080p e 60 fps, nella stessa posizione della
  camera fissa della palestra del video di esempio (alta, dietro una linea di
  fondo), secondo `docs/protocollo-ripresa.md`. Niente seconda camera finché non
  serve davvero;
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
- le immagini pubbliche sono schemi del campo (traccia della palla, linea del
  tempo degli scambi) oppure clip con i volti sfocati. Mai clip in chiaro;
- prima di produrre qualunque immagine destinata a uscire dal computer, chiedi.

## Validazione

- La verità di riferimento è l'annotazione a mano dei tocchi, fatta secondo
  `docs/guida-etichette.md`. Ogni metrica si misura contro di essa e si riporta
  col denominatore: "41/47 scambi trovati", non "87%". Per i tocchi: trovati e
  mancati entro una tolleranza di tempo dichiarata, lato giusto, matrice di
  confusione dei tipi.
- Divisione dei dati:
  - video di esempio (camera della palestra, 15 fps): prima prova del tracker e
    collaudo della pagina di etichettatura, mai come riferimento;
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
- Al massimo tre tocchi per parte; il tocco a muro non conta tra i tre.
- Il libero non batte: l'ordine di battuta segue la rotazione delle sei in campo,
  con formazioni e sostituzioni dal referto.
- Un punteggio finale ricostruito che non è un punteggio legale (26-20, 25-24)
  indica un errore nella pipeline.
- Numero di scambi del set = somma dei punti. Se non torna, mancano o avanzano
  scambi. Eccezione rara: punti assegnati senza scambio (sanzioni, fallo di
  posizione).
- I punteggi dei set sul referto sono un controllo indipendente.

## Codice di terzi

Prima di copiare codice da una repo esterna controlla la licenza in
`docs/prior-work.md`. VolleyVision ha due licenze in conflitto (AGPL-3.0 nel file,
CC BY-NC-ND 4.0 nel README): non se ne copia codice.

## Convenzioni

- Venv: `.venv/` in questa cartella (Python 3.12.8, da `requirements.txt`, in
  `.gitignore`). Test: `.venv/bin/python -m pytest`.
- Le regole dello scambio stanno in `lib/scambio.py` (un solo scambio) e
  `lib/partita.py` (la sequenza: punteggio, chi batte). I test passano solo dalle
  funzioni pubbliche: `punteggio`, `esito`, `proponi_tipi`, `controlla_scambio`,
  `battitrici`.
- Commenti, docstring e documentazione in italiano; `README.md` in inglese.
- I commenti spiegano perché, non cosa.
