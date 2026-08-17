# Roadmap operativa — modifiche a TotoSport in produzione

Come fare le modifiche più comuni **dopo** che l'app è online (percorso gratuito:
Vercel + Render + Neon). Tutto quello che segue si fa **gratis**.

Il deploy iniziale è descritto in [development/11_deployment.md](development/11_deployment.md).

---

## Regola d'oro

| Cosa cambi | Come | Sicurezza |
|---|---|---|
| **Codice** (frontend/backend) | `git push` → deploy automatico | rollback con 1 click |
| **Schema DB** | **sempre** migrazione Alembic (mai a mano) | **backup prima** |
| **Dati** nel DB | interfaccia admin, oppure SQL su Neon | backup se rischioso |

`git push` **non tocca mai** i dati: il DB (Neon) è separato dal codice.

---

## 1. Modifica al frontend o al backend

1. Modifichi il codice in locale e provi (`npm run dev` / test backend).
2. `git add -A && git commit -m "..." && git push`.
3. **Vercel** (frontend) e **Render** (backend) ridistribuiscono da soli in ~1–2 min.
   Stesso URL, dati intatti.

**Se cambi una variabile d'ambiente** (es. `ALLOWED_ORIGINS`, `SECRET_KEY`):
aggiornala nel pannello del servizio (Render/Vercel) → **redeploy**. Le env var
NON stanno nel repo.

**Rollback codice:** sia Render sia Vercel conservano i deploy precedenti → dal
pannello, "Rollback"/"Redeploy" della versione buona.

---

## 2. Modifica allo schema del DB (migrazioni)

> Le modifiche additive (aggiungere colonne/tabelle) **non cancellano** i dati.

1. **Backup prima** (vedi §4).
2. Modifichi i modelli SQLAlchemy in locale.
3. `alembic revision --autogenerate -m "descrizione"` → **rivedi** il file generato.
4. Provi in locale: `alembic upgrade head`.
5. **Committi modello + migrazione nello stesso commit**, poi `git push`.
6. Applichi in produzione (dal tuo Mac, o dalla Shell di Render):
   ```bash
   DATABASE_URL="postgresql+psycopg2://USER:PASS@HOST/DB?sslmode=require" \
     alembic upgrade head
   ```

**Rollback schema:** `alembic downgrade -1` (spesso *lossy* → i dati droppati non
tornano). Il vero piano B è il **backup**.

---

## 3. Modifica ai dati

Tre modi, tutti gratis:

**a) Interfaccia admin (consigliato)** — stagioni, giocatori, giornate, risultati,
tabellone: è il modo pensato per l'uso quotidiano.

**b) Editor SQL di Neon (browser)** — nel pannello Neon c'è una console SQL: lanci
query dirette senza installare nulla.

**c) Client SQL** (TablePlus / DBeaver / `psql`) con la connection string di Neon:
```bash
psql "postgresql://USER:PASS@HOST/DB?sslmode=require"
```

### Ricette SQL utili (le stesse usate in sviluppo)

> Sostituisci sempre i valori (`'mario'`, gli UUID, ecc.).

**Reset del tabellone di un giocatore** (azzera penalità/baseline, mantiene le scelte):
```sql
DELETE FROM table_prediction_modifications
 WHERE prediction_id IN (
   SELECT tp.id FROM table_predictions tp JOIN users u ON u.id=tp.player_id
   WHERE u.username='mario');
UPDATE table_predictions
   SET mercato_penalty=0, mercato_baseline=NULL, late_compile_penalty=0
 WHERE player_id=(SELECT id FROM users WHERE username='mario');
```

**Eliminare una giornata** (cancella in cascata partite, previsioni, punteggi):
```sql
DELETE FROM rounds WHERE id='<uuid-della-giornata>';
```

**Pulire la stagione** (eliminare TUTTE le giornate di test → la classifica si ripulisce
da sé, perché è ricalcolata dai RoundScore rimasti). Verifica sempre prima con la SELECT:
```sql
-- 1. controlla cosa stai per cancellare
SELECT r.id, r.name, r.status FROM rounds r
  JOIN seasons s ON s.id = r.season_id
 WHERE s.status <> 'closed' ORDER BY r.created_at;
-- 2. cancella tutte le giornate della stagione corrente
DELETE FROM rounds
 WHERE season_id = (SELECT id FROM seasons WHERE status <> 'closed' ORDER BY created_at DESC LIMIT 1);
```
Per azzerare anche i **tabelloni** di test: `DELETE FROM table_prediction_modifications ...` +
`DELETE FROM table_predictions WHERE season_id = (...)` (i tabelloni non c'entrano con le giornate).

**Vedere utenti / stagioni:**
```sql
SELECT username, email, role, is_active FROM users ORDER BY username;
SELECT name, status, tabellone_deadline, modification_deadline FROM seasons;
```

**Reset password di un utente** (serve l'hash bcrypt, generato dallo script/app):
```bash
# 1) genera l'hash (dal backend, con la venv attiva)
python -c "from app.services.auth import hash_password; print(hash_password('NUOVA_PWD'))"
```
```sql
-- 2) applica l'hash
UPDATE users SET password_hash='<hash-generato>', is_active=true WHERE username='mario';
```

---

## 4. Backup e restore del DB

**Backup** (fallo prima di ogni migrazione e periodicamente):
```bash
pg_dump "postgresql://USER:PASS@HOST/DB?sslmode=require" > backup_$(date +%F).sql
```

**Restore:**
```bash
psql "postgresql://USER:PASS@HOST/DB?sslmode=require" < backup_2026-01-01.sql
```

Neon offre anche **branching / point-in-time restore** dal pannello (comodo per
provare una migrazione su una copia prima di toccare la produzione).

---

## 5. Creare un nuovo giocatore / admin

- **Giocatori**: dall'interfaccia admin → "Giocatori" → "Crea giocatore"
  (comunichi tu username + password iniziale; nessuna mail viene inviata).
- **Un altro admin** (o il primo, su DB vuoto): `backend/scripts/create_admin.py`
  puntando al DB di produzione (`DATABASE_URL` = Neon).

---

## 6. Aggiornare il Regolamento

Il regolamento mostrato in-app è una **copia** in `frontend/src/content/regolamento.md`
(senza la sezione "Note per il Coding Agent"). Se modifichi la fonte
`docs/rules/REGOLAMENTO.md`:
1. rigenera la copia (togliendo la sezione §9),
2. `git push` → il frontend si aggiorna da solo.

*(Se lo si usa spesso, conviene uno script di sync — chiedilo pure.)*

---

## 7. Passare al piano a pagamento (se serve)

L'unico limite del percorso gratuito è il **cold start** del backend Render
(attesa ~30–60s al primo accesso dopo inattività). Per eliminarlo: dal pannello
Render passa il servizio a un piano **always-on** (~7 $/mese). Nient'altro
cambia: codice, dati e URL restano identici.
