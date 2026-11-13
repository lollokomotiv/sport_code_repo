# Dati divisi per canale, misura solo sulle partite filmate a protocollo

Se lo stesso giocatore compare nel training e nella validazione, il modello impara la
persona e i numeri si gonfiano. L'identità dei giocatori nei video di YouTube non è
nota, quindi la si approssima con il canale: un canale è di solito una persona, un
circolo o una squadra. Training e validazione sono canali diversi; la validazione
contiene solo video continui, perché è ciò che l'app riceverà.

La misura finale si fa solo sulle partite filmate secondo `docs/protocollo-ripresa.md`
(iPhone, 1080p, 60 fps, treppiede nell'angolo), con almeno una partita mai aperta
durante lo sviluppo.

## Consequences

- Gli highlights entrano solo nel training del classificatore dei colpi e nella ricerca
  delle classi rare: la loro distribuzione dei colpi è distorta e non hanno tempi
  morti.
- OpenTTGames (camera di lato centrata, CC BY-NC-SA 4.0) è un banco di prova per
  controlli veloci, mai una misura.
