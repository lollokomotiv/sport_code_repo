# 02 — Metriche avanzate

**Obiettivo:** estendere il progetto oltre xG/xA con metriche che rispondano a domande che i club si pongono davvero.

**Stato:** ⚪ Da iniziare — bloccato dal completamento di [01](01-xa-completamento.md).

---

## Principio guida

Meglio **una** metrica costruita bene, validata e comunicata, che quattro modelli a metà. Il criterio di scelta non è "quanto è difficile da implementare" ma "quale domanda risponde e chi la userebbe".

## Candidati (da scegliere, non da fare tutti)

| Metrica | Domanda a cui risponde | Dati disponibili? | Note |
|---|---|---|---|
| **Possession value / EPV** | Quanto ogni azione cambia la probabilità di segnare? | Sì (eventi + 360) | Il più richiesto nel settore; generalizza xG e xA in un unico framework |
| **Pass difficulty / pass completion model** | Questo passaggio era facile o difficile? Chi passa meglio del previsto? | Sì (già ho le feature dei passaggi da xA) | Riuso quasi diretto del dataset xA — costo marginale basso |
| **Shot quality vs. finishing** | Il giocatore segna più o meno di quanto dovrebbe? | Sì | Richiede prima che l'xG sia calibrato (vedi 01, punto 2) |
| **Pitch control / spazio** | Chi controlla quale zona del campo? | Parziale (360 = solo frame al momento dell'evento) | Attrattivo visivamente, ma i dati open non hanno tracking continuo |
| **Set piece analysis** | Quanto valgono i calci piazzati di questa squadra? | Sì | Nicchia concreta, i club ci investono molto |

## Step

- [ ] Chiudere 01 (senza un xG calibrato, tutto ciò che ci si costruisce sopra eredita l'errore)
- [ ] Scegliere **una** metrica dalla tabella
- [ ] Definire in anticipo come si valida: qual è il benchmark, cosa significa "funziona"
- [ ] Implementare seguendo lo schema già rodato: notebook di training → modello in `models/` → funzione di inferenza in `inference/` → visualizzazione
- [ ] Scrivere una pagina di analisi sui risultati, non solo il codice

## Vincolo da tenere presente

I dati aperti StatsBomb sono 3.464 match di eventi ma solo **326 con dati 360**. Ogni metrica che dipende dal freeze frame è limitata a quel sottoinsieme. Va detto esplicitamente nei risultati invece che nascosto.
