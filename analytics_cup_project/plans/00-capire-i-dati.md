# 00 — Capire i dati e quantificarne i limiti

**Obiettivo:** sapere cosa questi dati permettono di chiedere, prima di decidere
cosa chiedere. Finché questo non è chiaro, qualunque scelta di domanda è un
azzardo.

## Perché viene prima di tutto

Il vincitore dell'edizione 2026 ha scelto l'ottimizzazione matematica invece del
machine learning, e nell'abstract lo ha motivato così: con dieci partite non c'è
abbastanza segnale per addestrare, e i modelli non sono interpretabili
tatticamente. **La scelta del metodo è nata dal vincolo dei dati.**

Lo stesso ragionamento va rifatto qui, con i numeri veri.

## Risposto

Misurato in [`explorations/00-quanto-e-osservato.ipynb`](../explorations/00-quanto-e-osservato.ipynb),
su una partita intera (956.076 osservazioni).

### Quanta parte del tracking è osservata e quanta estrapolata?

**59% osservato, 41% estrapolato.** Non distribuito a caso: la telecamera segue
la palla.

| | osservato |
|---|---|
| entro 5 m dalla palla | 87% |
| 10-20 m | 78% |
| 30-40 m | 40% |
| oltre 40 m | 18% |

Per ruolo lo stesso gradiente: esterni e mezzali 70-73%, terzini e punta ~62%,
centrali difensivi ~49%, **portiere 15%**.

Conseguenza per la scelta della domanda: i dati reggono le domande centrate
sull'azione e reggono male quelle sull'organizzazione collettiva lontano dalla
palla. Una metrica sul posizionamento del portiere descriverebbe soprattutto
l'algoritmo di estrapolazione.

### `is_detected` è accessibile via kloppy?

**No.** `frame.players_data[p].other_data` è vuoto e `to_df()` restituisce solo
x, y, distanza e velocità. Per misurare il limite principale di questi dati serve
leggere il JSONL grezzo — è il motivo per cui esiste `lib/data.py`, e un problema
da risolvere al momento della submission se la metrica finale dipende da
`is_detected`.

### Quante partite ci sono, e su che campo?

**20**, non le 10 del README: il dataset è cresciuto dopo l'edizione 2026.
Le dimensioni del campo variano — 105×68 (10 partite), 106×68 (6), 104×68 (4) —
quindi ogni metrica spaziale normalizzata va calcolata per partita.

## Domande aperte

### I numeri sopra valgono su tutte e 20 le partite?
Vengono da una sola. Da ripetere su tutte per sapere se il 59% è tipico o se
dipende dalla regia televisiva della singola partita.

### Quanto sono lunghi i buchi di detection?
41% di frame estrapolati sparsi è un problema diverso da 41% concentrato in
tratti lunghi: nel primo caso l'interpolazione è quasi innocua, nel secondo no.
Non ancora guardato.

### `image_corners_projection`
Dice quale porzione di campo era inquadrata. È la via per verificare
direttamente l'ipotesi della telecamera invece di dedurla dalla distanza dalla
palla.

### Che cosa contengono davvero gli altri file?
- `{id}_dynamic_events.csv` — 4,7 MB per partita, molto più grosso del tracking
  come densità informativa per riga. Quali categorie di evento?
- `{id}_phases_of_play.csv` — la tassonomia delle fasi
- `aggregates/` — physical, off-ball runs, passing a livello giocatore-stagione,
  filtrati sopra i 60 minuti
- `bodypose/` — 29 giunti, **solo 2 partite**: campione troppo piccolo per
  conclusioni, ma abbastanza per una dimostrazione di metodo

### Le dimensioni del campo variano fra partite?
Stanno in `{id}_match.json`. Se variano, ogni metrica spaziale normalizzata va
calcolata per partita, non con costanti globali.

## Problemi già emersi

**I file di tracking sono su Git LFS.** Appena clonati risultano da 133 byte:
sono puntatori. Servono `git lfs install` e `git lfs pull`; ogni file è ~90 MB,
il dataset completo ~1,9 GB. `lib/data.py` se ne accorge e ripiega sullo
streaming da GitHub (~25 s per partita), così i notebook girano comunque.

**`floodlight` e `socceraction` sono incompatibili** (lxml >=5.3 contro <5).
Tenuto `floodlight`; vedi il commento in `requirements.txt`.
