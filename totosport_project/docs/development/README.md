# Piano di Sviluppo — TotoSport

Piano operativo a fasi. Ogni file descrive obiettivi, checklist e note tecniche per una fase.
Leggi `CLAUDE.md` prima di iniziare qualsiasi fase.

---

## Fasi

| Fase | File | Contenuto | Dipende da |
|---|---|---|---|
| 1 | [01_foundation.md](01_foundation.md) | Docker, struttura cartelle, Alembic, CI base | — |
| 2 | [02_auth_and_users.md](02_auth_and_users.md) | Modelli User/Season, JWT, login/register | Fase 1 |
| 3 | [03_tabellone.md](03_tabellone.md) | Tabellone annuale, modifiche mercato, scoring fine stagione | Fase 2 |
| 4 | [04_rounds_and_matches.md](04_rounds_and_matches.md) | Round, Match, status machine, API-Football staging | Fase 2 |
| 5 | [05_predictions_and_scoring.md](05_predictions_and_scoring.md) | MatchPrediction, RoundPrediction, scoring engine | Fase 4 |
| 6 | [06_leaderboard_and_bonuses.md](06_leaderboard_and_bonuses.md) | Classifiche, weekend bonus, bonus fine stagione | Fase 5 |
| 7 | [07_api_football.md](07_api_football.md) | Integrazione API-Football, fetch fixture, auto-sync risultati | Fase 4 |
| 8 | [08_frontend_foundation.md](08_frontend_foundation.md) | Vite + React + TS, AuthContext, routing, layout, API client | Fase 2 |
| 9 | [09_frontend_player.md](09_frontend_player.md) | Pagine giocatore: login, giornate, previsioni, tabellone, classifica | Fase 8 |
| 10 | [10_frontend_admin.md](10_frontend_admin.md) | Pagine admin: gestione stagione, giornate, risultati, tabellone | Fasi 8, 3–6 |
| 11 | [11_deployment.md](11_deployment.md) | Docker prod, Nginx, env vars, health check | Tutte |

---

## Stato avanzamento

> Aggiorna questa tabella man mano che completi le fasi.

| Fase | Stato |
|---|---|
| 1 — Foundation | ✅ Completata |
| 2 — Auth & Users | ✅ Completata |
| 3 — Tabellone | ✅ Completata |
| 4 — Rounds & Matches | ✅ Completata |
| 5 — Predictions & Scoring | ✅ Completata |
| 6 — Leaderboard & Bonus | ✅ Completata |
| 7 — API-Football | ⬜ Non iniziata (inserimento partite/risultati manuale) |
| 8 — Frontend Foundation | ✅ Completata |
| 9 — Frontend Player | ✅ Completata (incl. pagina Regolamento, vista riepilogo tabellone) |
| 10 — Frontend Admin | ✅ Completata |
| 11 — Deployment | ⬜ Non iniziata — predisposti: PWA installabile, rifiniture mobile, opzioni hosting ([hosting_options.md](hosting_options.md)) |

### Aggiunte/affinamenti durante la review (oltre il piano base)

- Previsioni: **segno esplicito** su ogni partita + **risultato esatto** solo sulle partite flaggate dall'admin (`Match.requires_exact_score`).
- Tabellone mercato: penalità **−5 a voce** calcolata rispetto all'**originale** (rimodificare la stessa voce o tornare indietro non accumula); **compilazione tardiva** ammessa con **−30** una tantum.
- Classifica: colonne **Pieni / Gol / Weekend** separate.
- Fix cache cross-utente al login/logout.
- Mobile: navbar con **hamburger**, input **no-zoom iOS**, **safe-area** barre fisse.
- Pagina **Regolamento** (player + admin) che rende `docs/rules/REGOLAMENTO.md`.

> ⚠️ La pagina Regolamento usa una **copia** bundlata in `frontend/src/content/regolamento.md`
> (senza la §9 "Note per il Coding Agent"). Va rigenerata se si modifica il regolamento sorgente.

---

## Regole operative

- **Una fase alla volta.** Non iniziare la fase N+1 finché la N non è completata e i test passano.
- **Il REGOLAMENTO è la fonte di verità** per qualsiasi dubbio sulle regole di business.
- **Prima dei punti aperti, decidi.** I punti aperti in `CLAUDE.md` §11 bloccano alcune fasi — risolvili prima di arrivare a quella fase.
- **Migrazioni Alembic**: ogni modifica al DB è una migrazione, mai modificare tabelle a mano.
- **Test prima di merge**: ogni fase ha i suoi test minimi. Non procedere senza.
