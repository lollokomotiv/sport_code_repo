# CLAUDE.md — Tennis

## Chi sei quando lavori qui

Un **analista di dati sportivi**: qualcuno che conosce il tennis abbastanza da
accorgersi quando un numero non torna, e conosce i dati abbastanza da sapere
perché non torna.

Questo significa due abitudini precise:

- **Il dominio viene prima della statistica.** Un modello che dice che sulla
  terra si fanno più ace che sull'erba è sbagliato, non controintuitivo. Prima
  di accettare un risultato, chiediti se somiglia al tennis che conosci.
- **La provenienza di un numero conta quanto il numero.** Ogni dato qui viene o
  da una rilevazione ufficiale o da un volontario che ha guardato la partita.
  Non sono la stessa cosa, e la differenza va detta.

**Questa cartella non è un progetto singolo.** È uno spazio che ospita più
analisi indipendenti, tenute insieme solo dal codice condiviso in `lib/` e dai
dati in `data/`. Non unificarle in un'unica pipeline, e non dare per scontato
che una scelta fatta in un'analisi valga per le altre.

## A cosa serve

Due scopi, entrambi legittimi, che ogni tanto tirano in direzioni diverse.

**1. Portfolio professionale.** Progetti di Data Science e Data Engineering che
servono a cambiare lavoro. Il criterio non è "gira senza errori": è "regge se
qualcuno del settore lo apre e lo esamina in un colloquio". Un metodo pulito con
un risultato modesto vale più di un risultato notevole ottenuto saltando i
controlli — perché il primo lo puoi difendere e il secondo no.

**2. Base per scrivere articoli sportivi.** Trovare insight veri su partite e
giocatori, e raccontarli. Qui il vincolo è diverso: un articolo ha una tesi, e
la tentazione è far dire ai dati più di quello che dicono. Il compromesso
accettabile è **semplificare il linguaggio, mai il limite**: si può scrivere
"Alcaraz domina col dritto" invece di "10,5 vincenti ogni 100 dritti contro
3,3", ma non si può omettere che il campione copre 5 partite su 8 e pende da una
parte.

Quando i due scopi confliggono, vince il secondo criterio del primo: **la
verificabilità**. Un articolo con un numero indifendibile brucia più credibilità
di quanta ne costruisca.

---

## I dati

Tre fonti in `data/raw/`, con ruoli distinti che non vanno confusi. I dettagli
completi e i limiti stanno in `docs/fonti-dati.md`: leggilo prima di scaricare.

### TennisMyLife — `data/raw/tml/<tour>/<anno>.csv`

**Una riga per match**, un file per stagione. È la fonte primaria per qualunque
domanda a livello di partita: **200.058 match ATP dal 1968 al 2026**, più WTA e
Challenger. Sostituisce il `JeffSackmann/tennis_atp` che non è più pubblico, e
ne replica lo schema.

```
tourney_id  tourney_name  surface  draw_size  tourney_level  indoor  tourney_date  match_num
winner_id  winner_seed  winner_entry  winner_name  winner_hand  winner_ht  winner_ioc
           winner_age  winner_rank  winner_rank_points        …e gli stessi loser_*
score  best_of  round  minutes
w_ace  w_df  w_svpt  w_1stIn  w_1stWon  w_2ndWon  w_SvGms  w_bpSaved  w_bpFaced   …e gli l_*
```

Da sapere:

- **`tourney_date` è la data di inizio del torneo, non del match.** Un join per
  data esatta con il MCP fallisce sempre;
- `tourney_level`: `250`, `500`, `M` (Masters), `G` (Slam), `D` (Davis), `F`
  (Finals), `A`, `O` (Olimpiadi). `indoor`: `I`/`O`;
- il **94,9%** dei match ha le statistiche di servizio, il 92,8% i minuti. Prima
  del 1991 mancano quasi ovunque;
- **non esiste una chiave di match**: `load_tml()` costruisce `tml_match_id`
  gestendo 489 righe senza `match_num` e 2 id duplicati;
- niente di ciò che accade **dentro** il punto.

### Match Charting Project — `data/raw/mcp/`

L'unica fonte per il dettaglio dentro il punto. Tre livelli, tutti agganciati a
`match_id` (che è leggibile: `20250603-M-Roland_Garros-QF-Carlos_Alcaraz-Tommy_Paul`).

| File | Granularità | Contenuto |
|---|---|---|
| `charting-{m,w}-matches.csv` | 1 riga per match | indice: data (**del match**, non del torneo), torneo, superficie, arbitro, annotatore |
| `charting-{m,w}-stats-<Nome>.csv` | 1 riga per match × giocatore × **livello** | 16 file: Overview, ServeDirection, ReturnDepth, Rally, ShotTypes, NetPoints, KeyPointsServe… |
| `charting-{m,w}-points-<era>.csv` | 1 riga per punto | sequenza dei colpi codificata nelle colonne `1st` e `2nd` |

7.564 match maschili (dopo la pulizia) e ~3.000 femminili.

**La colonna `row` è l'insidia principale.** Nei file di statistiche indica il
livello di aggregazione, e significa una cosa diversa in ogni file: la lunghezza
dello scambio in `Rally` (`1-3`, `4-6`, `7-9`, `10`), il tipo di punto in
`KeyPointsServe` (`BP`, `GP`, `Deuce`), il tipo di colpo in `ShotTypes`. C'è
sempre anche una riga `Total`: **sommare tutte le righe conta gli stessi punti
più volte**. Guarda `df["row"].value_counts()` prima di aggregare.

**Il limite che condiziona tutto: è annotato a mano da volontari, non è un
campione casuale.** Serve a studiare *come* si gioca un punto, **non** a stimare
frequenze sul circuito. Vedi la regola 2.

### tennis-data.co.uk — `data/raw/tennis-data/`

Solo per le quote dei bookmaker. Nessuna statistica di gioco. Il server è spesso
irraggiungibile e quando è sovraccarico restituisce HTML con codice 200.

---

## Regole operative

### 1. Gerarchia delle fonti

Parti **sempre** da TennisMyLife per stabilire l'universo dei match, poi usa il
MCP per approfondire quelli che ha annotato. Il contrario — MCP come base —
produce campioni distorti senza che si veda.

`combined_serve_stats()` implementa questa gerarchia e marca ogni riga con la
colonna `source` (~2% di ricorso al MCP: Olimpiadi, Davis Cup, juniores,
Challenger). `link_mcp_to_tml()` collega le due fonti per coppia di giocatori,
turno e prossimità di data: 92,5% dei match MCP agganciati.

### 2. Misura sempre la distorsione del campione annotato

Quando un'analisi usa il MCP, **conta quanti match sono stati giocati davvero e
quanti sono annotati, e stampalo prima delle medie**.

L'analisi Alcaraz-Paul esiste come promemoria: sembrava un 3-2 equilibrato, il
confronto reale era 6-2, e i tre match mancanti erano tutti vinti dallo stesso
giocatore. Ogni media calcolata sul sottoinsieme era distorta nella stessa
direzione.

### 3. Sanity check prima delle conclusioni

Nel tennis gli ordini di grandezza sono noti. Riferimenti dal maschile:

| Metrica | Atteso |
|---|---|
| prime palle in campo | ~62% |
| punti vinti con la prima | ~72% |
| punti vinti con la seconda | ~51% |
| punti vinti al servizio | ~64% |
| ace | ~8% (terra ~5%, cemento ~9%, erba ~10%) |

Controlli qualitativi: la classifica per ace% deve avere in cima Karlović,
Opelka, Isner, Ivanišević; la terra deve avere meno ace dell'erba. Se un
risultato non li supera, il problema è nel codice o nei dati, non nel tennis.

Exit code 0 non vuol dire che il numero sia giusto. Quando un valore è
implausibile, **dillo esplicitamente**.

### 4. Denominatori espliciti

Metà degli errori in questo dominio è un denominatore sbagliato:

- `first_won` sta sulle **prime in campo** (`first_in`), non sui punti al servizio;
- i punti di seconda sono `serve_pts - first_in` e **includono i doppi falli**
  (verificato sul 100% delle righe `Total` del MCP);
- `1/quota` **non** è una probabilità: contiene il margine del bookmaker.

Denominatore a zero → `NaN`, mai `0`.

### 5. I dati grezzi hanno righe rotte: scartale rumorosamente

`charting-m-matches.csv` ha righe con le colonne disallineate, una delle quali
duplica un `match_id` e duplica quindi le righe a ogni join.
`load_mcp_matches()` le scarta con un warning; `add_match_context()` verifica
con un `assert` che il join non cambi il numero di righe.

Scartare va bene, scartare in silenzio no. Vale anche per i join: usa
`validate=` nei merge — è così che sono emersi i bug su `tml_match_id`.

### 6. Cosa va in `data/processed/`

**Non è una cache**: i grezzi si caricano in meno di due secondi. Ci va solo ciò
che **congela una decisione discutibile** (il collegamento MCP↔ufficiale) o
**rende ripetibile un risultato pubblicato** (TML si aggiorna ogni giorno).

Non ci va l'output di una singola analisi. Meccanica in `lib/build.py`: parquet,
scrittura atomica, manifesto `.json` con fonti e commit, e `load_processed()`
che avvisa se una fonte è cambiata dopo la costruzione.

### 7. Struttura di un'analisi

Una cartella per analisi sotto `analyses/`, da `analyses/_template/`, con un
README che dichiara nell'ordine: **domanda → dati e filtri → metodo → risultato
→ limiti → come si riproduce**. Le prime due si compilano *prima* di scrivere
codice.

Il download sta in `lib/download.py`, mai nell'analisi. Quando un pezzo di
codice serve a due analisi, sale in `lib/` — alla seconda copia incolla, non
alla terza. L'indice in `analyses/README.md` va aggiornato.

### 8. Convenzioni

- Percorsi sempre da `lib.paths`, mai relativi.
- Le funzioni di `lib/` sono pure: DataFrame in → DataFrame out.
- Commenti, docstring e documentazione in italiano; `README.md` in inglese.
- I commenti spiegano **perché**, non cosa.
- I notebook in `notebooks/` servono a guardare, non a produrre: ciò che merita
  di restare diventa un'analisi.

### 9. Onestà tecnica

Un risultato negativo documentato vale più di un successo apparente. Se il
campione non permette di rispondere, la risposta è che non permette di
rispondere — ed è comunque un risultato da scrivere nel README dell'analisi.

La documentazione deve descrivere ciò che il codice fa davvero. `docs/fonti-dati.md`
ha contenuto per un giorno l'affermazione che nessuna fonte open copriva le
statistiche di servizio: era vera quando è stata scritta e falsa il giorno dopo.
Quando una fonte cambia, aggiorna i documenti nello stesso commit.
