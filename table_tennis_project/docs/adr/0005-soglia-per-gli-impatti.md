# Soglia di decisione sugli impatti, fissata prima di misurare

Trovare gli impatti è il primo rischio del progetto: blocchi e tagli hanno un
movimento minimo, il giocatore lontano è piccolo, nelle palestre si sentono i colpi
degli altri tavoli. Il rilevatore parte da posa e audio. Se sulla validazione trova
almeno il 90% degli impatti del giocatore vicino entro ±100 ms, con al massimo il
10% di falsi, si prosegue; altrimenti si passa a un modello sui pixel prima di
costruire il resto. La soglia è scritta qui prima di vedere i numeri, per non
spostarla dopo.

## Consequences

- Per il giocatore lontano il numero si riporta senza soglia. Se è molto peggio del
  vicino, la scelta è tra migliorare la posa e restringere il report al giocatore
  vicino.
- Un modello che non batte la baseline (picco di velocità del polso) sulla validazione
  non entra.
