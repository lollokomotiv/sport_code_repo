# L'esito del colpo si ricava, non si apprende

Capire dal video se un colpo è finito in rete o fuori richiede di seguire la pallina,
che in un video amatoriale a 30-60 fps è piccola, sfocata e spesso invisibile. Per
questo l'esito non lo stima un modello: è in gioco per ogni colpo tranne l'ultimo
dello scambio, mentre l'ultimo è errore se chi l'ha giocato perde il punto e
vincente se lo vince. Bastano la sequenza dei colpi e il vincitore del punto (vedi
ADR-0003).

## Consequences

- Non si distingue tra rete e fuori, né tra errore forzato e non forzato. Sono rimandati
  alla fase in cui si segue la pallina.
- Se l'avversario manca del tutto la pallina, l'ultimo colpo resta un vincente.
- Un errore di servizio è l'ultimo colpo di uno scambio di un solo colpo: rientra nella
  regola senza eccezioni.
- Un let chiude uno scambio senza punto: i suoi colpi non hanno esito errore o
  vincente.
