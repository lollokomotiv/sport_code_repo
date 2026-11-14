# CLAUDE.md — Table Tennis

## Cos'è

Un'app che, dal video di una partita amatoriale di singolare ripresa in diagonale da
un angolo, trova i colpi di entrambi i giocatori, ne dà lato, tipo ed esito, ricava
chi vince ogni punto e produce un report per il giocatore. I termini sono definiti in
[CONTEXT.md](CONTEXT.md); le decisioni non ovvie in [docs/adr/](docs/adr/).

Decisioni prese:

- **cosa si produce**: per la partita set, punteggio e chi serve; per lo scambio
  inizio, fine, vincitore del punto, let; per il colpo impatto, giocatore (vicino o
  lontano), lato, tipo, esito;
- **tipi di colpo**: servizio, top, flip, taglio, blocco, schiacciata, altro. *Altro* ha
  un sottotipo facoltativo (pallonetto, smorzata, chop block, difesa), così si possono
  creare classi nuove senza ritaggare;
- **l'esito si ricava** dall'ultimo colpo dello scambio e dal vincitore del punto, non
  lo stima un modello (ADR-0001). Niente rete o fuori, niente forzato o non forzato;
- **il vincitore del punto si legge dal segnapunti** (ADR-0003), oppure si inserisce a
  mano in revisione;
- **un colpo è un impatto** (ADR-0002): un istante, non inizio e fine;
- **impostazione a mano** a inizio video (ADR-0008): angoli del tavolo, segnapunti,
  giocatore vicino, mano, primo servizio, formato;
- **solo singolare, solo ripresa in diagonale** con treppiede fermo (ADR-0006);
- **posa con RTMPose** via `rtmlib` (ADR-0007). Impatti da posa e audio, con la soglia
  di ADR-0005; modelli sui pixel e tracciamento della pallina solo dopo;
- **flusso**: la pipeline gira in locale dopo la partita; il tagger è anche la
  schermata di revisione. L'MVP riesce se rivedere una partita richiede al massimo un
  terzo del tempo necessario per etichettarla da zero;
- **cosa no**: doppio, riconoscimento delle persone, mano automatica, mappe di
  atterraggio, direzione e velocità della pallina, lunghezza del servizio. Servono la
  pallina: sono in una Wave futura;
- l'MVP è non commerciale; il progetto è anche portfolio.

## Wave

I prossimi passi stanno nella issue mappa
[#20](https://github.com/lollokomotiv/sport_code_repo/issues/20) e nelle sue figlie
(#21-#27, una per Wave). Nelle issue, che sono pubbliche, niente immagini dai video e
niente nomi di giocatori, canali o circoli.

| Wave | Contenuto | Conclusa quando |
|---|---|---|
| 0 — Fondamenta | protocollo di ripresa, CSV dei video, script di ricerca su YouTube | circa 20 video conformi nel CSV, 2 partite pilota scaricate |
| 1 — Tagger e pilota | tagger con impostazione ed etichettatura da tastiera; 2 partite etichettate | ~700 colpi, coerenza con sé stessi su 100 colpi ritaggati, tempo per partita misurato |
| 2a — Impatti | posa, scelta dei due giocatori, baseline del polso, modello su posa e audio | metriche degli impatti sulla validazione, decisione secondo ADR-0005 |
| 2b — Segnapunti | cifre nel riquadro, controllo col regolamento, precompilazione nel tagger | vincitore giusto su quanti punti, quanti in revisione |
| 3 — Colpi | giocatore, lato e tipo dalla finestra dell'impatto | matrici di confusione che battono la baseline per posizione |
| 4 — Scambi ed esito | scambi da impatti, servizio e segnapunti; esito; set e cambi di campo | scambi trovati ed esiti giusti su quanti |
| 5 — Pipeline e report | pipeline completa, revisione, report HTML | misura finale su una partita mai aperta, tempo di revisione |

Il report dell'MVP, per giocatore: tabella lato × tipo con colpi, errori, vincenti e
percentuale di errore, anche set per set; punti vinti al servizio e in risposta;
scambi corti e lunghi; primo attacco e suo esito; attaccato o staccato dal tavolo;
tempo di gioco e durata totale; clic su una cella per vedere le clip.

## I video sono privati

Il repo è pubblico. I video amatoriali ritraggono persone reali:

- non si versionano video, frame, ritagli, GIF o figure che mostrano i giocatori.
  Stanno in `data/`, ignorata da git;
- il CSV dei video versionato contiene id, codice del canale (`c01`, `c02`...),
  formato, fps, risoluzione, audio, licenza e note, **non titoli né nomi**: molti
  canali hanno il nome di una persona. Titoli e corrispondenza codice → canale stanno
  in un file locale in `data/`;
- le immagini pubbliche sono schemi del tavolo, oppure clip con i volti sfocati;
- prima di produrre qualunque immagine destinata a uscire dal computer, chiedi;
- le partite filmate a protocollo richiedono il consenso dell'avversario.

## Dati

- **YouTube**: singolare, diagonale da un angolo, tavolo intero, almeno 1080p,
  segnapunti visibile nei video continui. 50-60 fps preferiti, 30 fps accettati e
  segnati, sotto i 30 fps si scarta. Colonne `formato` (continuo, completo-tagliato, highlights) e `audio`
  (reale, musica, assente).
- **Divisione** (ADR-0004): training, validazione e test per canale; validazione e test
  contengono solo video continui. Il test sono 1-2 canali YouTube tenuti da parte più
  almeno una partita filmata a protocollo, mai aperti durante lo sviluppo. I numeri
  del test si riportano separati per giocatori già visti nel training e mai visti.
- I video in cui gioca l'utente entrano nel training: le sue partite nel test contano
  come giocatore già visto.
- Al massimo 5 partite intere per canale nel training, perché un canale con decine di
  partite riempirebbe il training con la sua palestra.
- Gli highlights servono solo al classificatore dei colpi e alle classi rare. Il
  rilevatore degli impatti si allena e si misura solo su audio reale.
- **OpenTTGames** (CC BY-NC-SA 4.0, camera di lato centrata, 120 fps): banco di prova
  per impatti, segnapunti e pallina, mai misura.
- **Licenze**: i termini di YouTube non permettono di scaricare senza autorizzazione.
  Va bene per un MVP di ricerca, non per un prodotto: la licenza di ogni video sta nel
  CSV.

## Validazione

- La verità di riferimento sono le etichette fatte a mano nel tagger. Ogni metrica si
  riporta col denominatore ("412/451 impatti trovati", non "91%").
- Impatti: trovati, mancati e falsi entro ±100 ms. Poi giocatore giusto, matrice di
  confusione di lato e di tipo, scambi trovati, vincitore del punto, esito.
- Tutto separato per giocatore vicino e lontano, e per 30 e 60 fps.
- Baseline: per gli impatti il picco di velocità del polso; per il tipo il più frequente
  per posizione nello scambio. Un modello che non la batte non entra.
- Le incongruenze col regolamento vanno in revisione, non si correggono in silenzio.
- Con poche partite i numeri sono indicativi. Va scritto accanto al numero.

## Regole di gioco che servono come controllo

- Set a 11 con almeno 2 punti di vantaggio; partite al meglio di 3, 5 o 7 set.
- Ogni scambio assegna un punto, tranne il let.
- Il servizio cambia ogni 2 punti; dal 10-10 cambia a ogni punto.
- Si cambia campo a ogni set e, nel set decisivo, quando un giocatore arriva a 5.
- Nel set successivo serve per primo chi ha ricevuto per primo nel set precedente.
- Un punteggio finale che non è un punteggio legale (11-10, 12-9) indica un errore
  nella pipeline.

## Convenzioni

- Commenti, docstring e documentazione in italiano; `README.md` e `FEASIBILITY.md` in
  inglese.
- I commenti spiegano perché, non cosa.
- Il tagger nasce come copia adattata di `volleyball_score_project/strumenti/etichette.html`,
  non come libreria comune: i progetti sono indipendenti.
