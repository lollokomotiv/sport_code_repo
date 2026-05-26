# Fase 9 — Frontend Player

Obiettivo: tutte le pagine lato giocatore funzionanti e usabili su mobile.

---

## Pagine da implementare

### `/player` — Home (Giornate aperte)

- Lista delle giornate con status `open` o `completed`
- Card per ogni giornata: nome, competizione, deadline, badge stato, CTA
- Badge "⏰ X ore alla deadline" con countdown se < 24h
- Badge "✅ Previsioni inserite" se il giocatore ha già inserito tutto
- Badge "⚠️ Mancano N previsioni" se incompleto
- Empty state: "Nessuna giornata aperta al momento"

### `/player/rounds/:id` — Dettaglio Giornata + Form Previsioni

La pagina più importante dell'app. Deve essere ottimizzata per mobile.

**Layout:**
```
[Nome giornata — Deadline: Ven 10 Mag, 20:45]
[Totale gol giornata: ____]

[Juventus  vs  Inter]        [__] - [__]
[Milan     vs  Roma]         [__] - [__]
[Napoli    vs  Lazio]        [__] - [__]
...
[         SALVA PREVISIONI         ]
```

**Comportamento:**
- Input numerici (solo tastiera numerica su mobile: `inputMode="numeric"`)
- Segno derivato mostrato in tempo reale (1/X/2) accanto all'input
- Se deadline superata → form disabilitato, mostra risultati reali e punti guadagnati
- Se round `completed` → mostra risultati + punti per ogni partita con colori (verde/rosso/grigio)
- Bottone "Salva" fa upsert (POST `/predictions/match` per ogni partita + POST `/predictions/round-goals`)
- Ottimizzazione: salva tutto in un'unica chiamata batch o con `Promise.all`

**Componente `MatchRow`:**
```typescript
interface MatchRowProps {
  match: MatchOut
  prediction?: MatchPredictionOut
  isLocked: boolean  // deadline superata
}
```

### `/player/predictions` — Storico Previsioni

- Filtro per giornata (select o tab)
- Per ogni giornata: lista partite con previsione, risultato reale, punti guadagnati
- Totale punti per giornata, suddiviso per categoria (segni, esatti, gol totali, weekend bonus)
- Grafici opzionali (barchart punti per giornata con Recharts)

### `/player/leaderboard` — Classifica

- Tabella con: posizione, nome giocatore, punti totali, segni, esatti+gol, tabellone
- L'utente corrente evidenziato (sfondo diverso)
- Tab per switchare tra: Classifica Generale | Classifica Segni | Pieni+Gol | Tabellone
- Aggiornamento automatico ogni 30s (React Query `refetchInterval`)

### `/player/tabellone` — Previsioni Annuali

**Vista 1: compilazione (stagione in `setup`/`active` prima della deadline)**

Form con tutte le voci del tabellone:
- Sezione Serie A: Scudetto (+ punti), 3 retrocesse, Capocannoniere (+ gol)
- Sezione Serie B: 2 promosse dirette (+ punti prima), 6 playoff, promossa playoff, 3 retrocesse C, 2 playout, retrocessa playout, Capocannoniere B (+ gol)
- Sezione Coppe: Coppa Italia, Champions, Europa, Conference

Suggerimento UX: usa autocomplete/typeahead per i nomi delle squadre (lista statica per stagione).

**Vista 2: stato attuale (stagione `active` dopo deadline)**

Mostra le previsioni inserite con stato: "⏳ In attesa", "✅ N pt", "❌ 0 pt" (se outcome già disponibile).

**Vista 3: modifica post-mercato (stagione `mercato`)**

- Banner: "⚠️ Finestra di modifica aperta fino al [data]. Ogni modifica costa -5pt immediati e limita i punti al 50%."
- Ogni campo mostra un'icona di modifica (matita)
- Click su un campo → modale di conferma: "Modificando [campo] perdi 5pt immediatamente e il massimo guadagno passa da Xpt a Ypt. Confermi?"
- Dopo conferma → PATCH `/tabellone/me` con il campo modificato

---

## Componenti condivisi

- [ ] `CompetitionBadge` — chip colorato per Serie A / Serie B / Champions
- [ ] `PointsBadge` — chip con punti (verde se >0, grigio se 0)
- [ ] `DeadlineCountdown` — countdown live alla deadline
- [ ] `MatchScore` — display del risultato (es. "2 - 1") con animazione se appena inserito
- [ ] `SignIndicator` — mostra "1", "X" o "2" in base ai gol inseriti
- [ ] `TeamAutocomplete` — input con lista squadre per stagione (per tabellone)

---

## Note UX mobile

- Tutti i tap target ≥ 44px
- Input score: due campi affiancati piccoli, non troppo larghi
- Il bottone "Salva" deve essere sticky in fondo alla pagina su mobile
- Feedback visivo immediato sul salvataggio (toast "Salvato ✓" o spinner inline)
- Niente redirect dopo il salvataggio — rimani sulla pagina

---

## Test di accettazione fase 9

1. Login → `/player` → vedo le giornate aperte
2. Apro giornata → inserisco previsioni → salvo → vedo "✅ Salvato"
3. Dopo la deadline → form bloccato
4. Dopo scoring admin → vedo punti per partita con colori corretti
5. `/player/leaderboard` → mi vedo evidenziato
6. `/player/tabellone` → compilo e salvo
7. Finestra mercato → modifico campo → modale conferma → -5pt visibili in classifica
8. Test su viewport mobile (375px): tutto leggibile e tappable
