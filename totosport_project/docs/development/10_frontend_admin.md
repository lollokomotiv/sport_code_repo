# Fase 10 — Frontend Admin

Obiettivo: pannello admin completo per gestire stagioni, giornate, risultati, tabellone e giocatori.

---

## Pagine da implementare

### `/admin` — Dashboard

- Riepilogo stagione corrente: stato, giornate completate/aperte, giocatori attivi
- Card "Azioni rapide": Crea giornata, Inserisci risultati, Apri finestra mercato
- Ultima giornata con stato e N. previsioni inserite vs giocatori totali

### `/admin/rounds` — Gestione Giornate

**Lista:**
- Tabella con: nome, competizione, deadline, stato, N. partite, N. previsioni
- Filtro per stato (draft/open/closed/completed)
- Bottone "Nuova giornata"

**Dettaglio giornata (`/admin/rounds/:id`):**
- Header con nome, deadline, stato, bottone per cambio stato
- Lista partite con: squadre, orario, risultato (input se `closed`), N. previsioni
- Pulsante "Aggiungi partita" (manuale o da fixture staged)
- Se `closed`: form risultati inline per ogni partita + bottone "Calcola punteggi"
- Se `completed`: tabella con top scorer della giornata

**Form "Nuova giornata":**
```
Nome: [_________________________]
Competizione: [Serie A ▼]
Matchday: [34]
Deadline: [data] [ora]
[CREA]
```

**Aggiunta partita (modale):**
- Tab "Manuale": home, away, orario
- Tab "Da API-Football": mostra fixture staged per la giornata selezionata, click per aggiungere

### `/admin/fixtures` — Fixture API-Football

- Form fetch: `{competizione, giornata, stagione}` → bottone "Recupera da API-Football"
- Lista fixture staged (non ancora assegnate a un round): squadre, orario, matchday
- Per ogni fixture: bottone "Aggiungi a round" con select round disponibili

### `/admin/results` — Inserimento Risultati

Shortcut per inserire risultati senza entrare in ogni giornata singola.

- Dropdown "Seleziona giornata" (solo giornate in stato `closed` con partite senza risultato)
- Lista partite con input inline: `[__] - [__]`
- "Salva tutto" → PATCH per ogni partita → scoring automatico
- Contatore "X/N partite con risultato"

### `/admin/tabellone` — Gestione Tabellone Annuale

- Visualizza tutti i tabelloni dei giocatori (vista a griglia o per colonna)
- Sezione "Risultati stagione": form per inserire `SeasonOutcome` (tutti i valori reali)
- Bottone "Calcola punti tabellone" → `POST /admin/tabellone/score`
- Vista confronto: tabellone giocatore vs outcome reale, campo per campo

**Form SeasonOutcome:**
```
SERIE A
  Vincitore Scudetto: [_________] Punti: [__]
  Retrocesse: [_________] [_________] [_________]
  Capocannoniere: [_________] Gol: [__]

SERIE B
  Promosse dirette: [_________] [_________]
  Punti 1ª classificata: [__]
  Playoff disputati: [Sì ▼]
  ...
```

### `/admin/season` — Gestione Stagione

- Stato attuale con timeline visiva: `setup → active → mercato → active → closed`
- Bottoni transizione stato con conferma modale:
  - "Apri tabellone" (setup → active, sets `tabellone_deadline`)
  - "Apri finestra mercato" (active → mercato, sets `modification_deadline`)
  - "Chiudi finestra mercato" (mercato → active)
  - "Finalizza stagione" (→ closed, assegna bonus finali)
- Crea nuova stagione

### `/admin/players` — Gestione Giocatori

- Lista giocatori con: username, email, punti totali stagione, data registrazione
- "Invita giocatore": crea account (`POST /auth/register`)
- Click su giocatore → dettaglio con storico punti per giornata
- Disable/enable account

---

## Componenti admin-specific

- [ ] `StatusTransitionButton` — bottone con modale di conferma per transizioni di stato
- [ ] `ResultInput` — coppia input `home - away` per il risultato
- [ ] `RoundStatusTimeline` — visualizzazione stati round
- [ ] `TabelloneGrid` — tabella comparativa previsioni vs risultati reali
- [ ] `FixtureFetcher` — form + lista fixture staged

---

## Test di accettazione fase 10

1. Crea giornata → aggiungi 3 partite → porta a `open`
2. Aggiungi fixture da API-Football a un round
3. Inserisci risultati per tutte le partite → scoring calcolato automaticamente
4. Verifica punti in classifica corretti
5. Apri finestra mercato → un giocatore modifica tabellone → -5pt visibili
6. Inserisci SeasonOutcome → calcola punti tabellone → punti corretti in classifica
7. Finalizza stagione → bonus +10pt assegnati ai vincitori delle 3 classifiche
