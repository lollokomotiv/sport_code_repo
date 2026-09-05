# CLAUDE.md — Tennis

Linee guida per lavorare in `tennis_project/`.

**Questa cartella non è un progetto singolo.** È uno spazio di lavoro che ospita
più analisi indipendenti sui dati del tennis, tenute insieme solo dal codice
condiviso in `lib/` e dai dati in `data/`. Non cercare di unificarle in un'unica
pipeline, e non assumere che una scelta fatta in un'analisi valga per le altre.

Come per il resto del repo, il criterio non è "gira senza errori": è "regge se
qualcuno del settore lo apre e lo esamina".

---

## 1. Le fonti dati sono fragili — leggi `docs/fonti-dati.md` prima di scaricare

Il dataset standard del dominio (`JeffSackmann/tennis_atp` e `tennis_wta`) **non
è più pubblico**: risponde 404. Quasi tutto il codice e i tutorial in
circolazione lo usano ancora. Se una soluzione trovata online scarica
`atp_matches_<anno>.csv`, non funziona: non adattarla, cambia fonte.

Restano due fonti verificate, con due limiti opposti da tenere sempre a mente:

- **Match Charting Project** — ricchissimo (colpo per colpo) ma **annotato a
  mano da volontari**: non è un campione casuale. Serve a studiare *come* si
  gioca un punto, **non** a stimare frequenze sul circuito. Ogni conclusione che
  generalizza "nel tennis succede X volte su Y" costruita su questi dati è
  sbagliata a meno che non si giustifichi il campione.
- **tennis-data.co.uk** — copre tutti i match, ma ha **solo risultato, ranking e
  quote**: nessuna statistica di gioco.

Quando una domanda richiede statistiche di servizio su tutto il circuito, la
risposta corretta oggi è "con i dati open disponibili non si può", non una
stima costruita sul MCP facendo finta che sia rappresentativo.

## 2. Il download sta in `lib/download.py`, mai nell'analisi

Un'analisi non scarica dati per conto suo e non legge da URL. Se serve un file
nuovo, si aggiunge a `lib/download.py`. Motivo pratico: le due fonti hanno
comportamenti sgradevoli (tennis-data risponde HTML con codice 200 quando è
sovraccarico, e rifiuta le richieste senza user agent) e quella gestione va
scritta una volta sola.

## 3. Sanity check sui numeri, prima delle conclusioni

Nel tennis gli ordini di grandezza sono noti: chi lavora nel dominio se ne
accorge subito se non tornano. Riferimenti dal MCP maschile (15.116 match-player):

| Metrica | Valore atteso |
|---|---|
| prime palle in campo | ~62% |
| punti vinti con la prima | ~72% |
| punti vinti con la seconda | ~51% |
| punti vinti al servizio | ~64% |
| ace | ~8% dei punti al servizio (terra ~5%, erba ~10%) |

E i controlli qualitativi: la classifica per ace% deve avere in cima
Karlović, Opelka, Isner, Ivanišević; la terra deve avere meno ace dell'erba.
Se un risultato non supera questi controlli, il problema è nei dati o nel
codice, non nel tennis.

Exit code 0 non vuol dire che il numero sia giusto. Quando un valore è
implausibile, **dirlo esplicitamente** invece di lasciarlo passare.

## 4. Denominatori espliciti

Metà degli errori in questo dominio è un denominatore sbagliato:

- `first_won` sta sulle **prime in campo** (`first_in`), non sui punti al servizio;
- i punti di seconda sono `serve_pts - first_in` e **includono i doppi falli**
  (verificato: l'identità vale sul 100% delle righe `Total` del MCP);
- i file di statistiche MCP hanno una riga per **set** più una `Total`: sommarle
  tutte conta ogni punto due volte. `load_mcp_stats()` filtra i totali di default;
- `1/quota` **non** è una probabilità: contiene il margine del bookmaker
  (vedi `implied_probabilities()`).

Denominatore a zero → `NaN`, mai `0`. Riempire di zeri falsa qualunque media.

## 5. I dati grezzi hanno righe rotte, e vanno scartate rumorosamente

`charting-m-matches.csv` contiene righe con le colonne disallineate (nomi di
arbitro finiti nel campo `surface`), una delle quali duplica un `match_id` e
duplica quindi le righe a ogni join. `load_mcp_matches()` le scarta emettendo un
warning, e `add_match_context()` verifica con un `assert` che il join non abbia
cambiato il numero di righe.

Mantieni questo schema: scartare va bene, scartare in silenzio no.

## 6. Struttura di un'analisi

Una cartella per analisi sotto `analyses/`, creata da `analyses/_template/`, con
un README che dichiara nell'ordine: **domanda → dati e filtri → metodo →
risultato → limiti → come si riproduce**. Le prime due sezioni si compilano
*prima* di scrivere codice.

Quando un pezzo di codice serve a due analisi, sale in `lib/`. Alla seconda
copia incolla, non alla terza.

L'indice in `analyses/README.md` va aggiornato quando nasce un'analisi.

## 7. Convenzioni

- Percorsi sempre da `lib.paths`, mai relativi: i notebook si lanciano da
  cartelle diverse e i path relativi si rompono.
- Le funzioni di `lib/` sono pure: DataFrame in → DataFrame out. Niente scritture
  su disco nascoste dentro un loader.
- Commenti, docstring e documentazione in italiano; `README.md` in inglese come
  nel resto del portfolio.
- I commenti spiegano **perché**, non cosa: il cosa si legge nel codice.

## 8. Onestà tecnica

Un risultato negativo documentato vale più di un successo apparente. Se il
campione non permette di rispondere, la risposta è che non permette di
rispondere — ed è comunque un risultato che si scrive nel README dell'analisi.
