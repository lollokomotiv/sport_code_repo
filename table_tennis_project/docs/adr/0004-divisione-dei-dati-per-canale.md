# Dati divisi per canale, test con giocatori visti e mai visti

Se lo stesso giocatore compare nel training e nella misura, il modello impara la
persona e i numeri si gonfiano. L'identità dei giocatori nei video di YouTube non è
nota, quindi la si approssima con il canale: un canale è di solito una persona, un
circolo o una squadra. Training, validazione e test sono canali diversi; validazione
e test contengono solo video continui, perché è ciò che l'app riceverà.

Il test è formato da 1-2 canali YouTube tenuti da parte e da almeno una partita
filmata secondo `docs/protocollo-ripresa.md` (iPhone, 1080p, 60 fps, treppiede
nell'angolo), che ha le condizioni esatte dell'app. Nessuno dei due si apre durante
lo sviluppo.

I video in cui gioca l'utente entrano nel training, perché sono dati buoni e il test
non deve dipendere dalle sole sue partite. Per non nascondere il vantaggio, i numeri
del test si riportano separati: giocatori già visti nel training (vale per chi ha già
partite nel dataset) e mai visti (vale per un utente nuovo).

## Consequences

- I canali del test non danno video né al training né alla validazione: servono più
  canali di quanti ne servirebbero con il solo test a protocollo.
- Al massimo 5 partite intere per canale nel training.
- Gli highlights entrano solo nel training del classificatore dei colpi e nella ricerca
  delle classi rare: la loro distribuzione dei colpi è distorta e non hanno tempi
  morti.
- OpenTTGames (camera di lato centrata, CC BY-NC-SA 4.0) è un banco di prova per
  controlli veloci, mai una misura.
