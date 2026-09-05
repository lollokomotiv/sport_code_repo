# Fonti dati — tennis

Stato verificato il **5 settembre 2026**. Le fonti tennis cambiano: prima di
fidarsi di questa pagina, ricontrolla che i download funzionino ancora.

---

## Premessa: il dataset standard non è più disponibile

Per anni il riferimento per i dati tennis sono stati i repo di Jeff Sackmann
`JeffSackmann/tennis_atp` e `JeffSackmann/tennis_wta`: risultati di tutti i
match dal 1968, statistiche di servizio dal 1991, ranking settimanali,
anagrafica giocatori. Quasi tutti i tutorial e i paper open citano quelli.

**Oggi rispondono 404.** Dell'account resta pubblico il solo
`tennis_MatchChartingProject`. Non è chiaro se sia una rimozione temporanea o
definitiva.

Conseguenze pratiche:

- codice ed esempi trovati online che scaricano `atp_matches_<anno>.csv` da
  quel repo **non funzionano più**;
- esistono fork di terzi, ma sono cristallizzati all'anno del fork e senza
  garanzia di integrità: usabili per un esperimento, non da citare come fonte;
- le **statistiche di servizio su tutto il circuito** (ace, punti al servizio,
  palle break per ogni match ATP) al momento non hanno un sostituto open
  equivalente. Chi ne ha bisogno deve accettare la copertura parziale del
  Match Charting Project o passare a una fonte a pagamento.

Questo limite va dichiarato nei risultati, non aggirato in silenzio.

---

## 1. Match Charting Project — *fonte primaria*

`github.com/JeffSackmann/tennis_MatchChartingProject` · licenza CC BY-NC-SA 4.0

Match annotati **a mano, colpo per colpo**, da volontari. È il dataset open più
ricco che esista sul tennis: per ogni punto si sa chi serviva, dove ha servito,
la sequenza dei colpi e come è finito lo scambio.

```bash
python3 -m lib.download mcp --gender m            # elenco match + 16 file di statistiche
python3 -m lib.download mcp --gender m --points   # + punto per punto (~130 MB)
```

**Cosa contiene** (verificato sul download di oggi, uomini):

| File | Contenuto | Dimensione |
|---|---|---|
| `charting-m-matches.csv` | 7.566 match, dal 1960 al maggio 2026: giocatori, torneo, superficie, arbitro | 1,1 MB |
| `charting-m-stats-Overview.csv` | statistiche per match/giocatore/set: `serve_pts`, `aces`, `dfs`, `first_in`, `first_won`, `second_won`, `bk_pts`, `bp_saved`, `return_pts_won`, `winners`, `unforced` (con split dritto/rovescio) | 6,7 MB |
| `charting-m-stats-*.csv` | 15 altri tagli: direzione del servizio, profondità della risposta, scambi, rete, serve&volley, tipi di colpo | 4–33 MB l'uno |
| `charting-m-points-*.csv` | punto per punto, con la notazione dello scambio nelle colonne `1st` e `2nd` | 36–56 MB |

Il femminile esiste con lo stesso schema (`charting-w-*`), circa 3.000 match.

**Limiti — i più importanti di questa pagina:**

- **non è un campione casuale.** Sono i match che qualcuno ha scelto di
  annotare: finali di Slam, big match, giocatori popolari. Ricavarne frequenze
  "del circuito" è sbagliato. Va bene per studiare *come* si gioca un punto,
  non *quanto spesso* accade qualcosa nel tennis in generale;
- **annotazione umana**, con qualità variabile per annotatore (la colonna
  `Charted by` permette almeno di controllarne l'effetto);
- copertura molto disomogenea nel tempo: densa dal 2010, rada prima;
- **il file `charting-m-matches.csv` contiene righe malformate** (colonne
  disallineate su alcuni match di Davis Cup: un nome di arbitro finito nel campo
  `surface`). Sono 2 su 7.566 e una duplica un `match_id`, quindi duplica le
  righe a ogni join. `lib.loaders.load_mcp_matches()` le scarta segnalandolo;
- la notazione dei colpi nelle colonne `1st`/`2nd` **non è testo libero**: è un
  codice (un carattere per colpo: tipo di colpo, direzione, esito). Va decodificata
  con la legenda del repo prima di usarla come feature.

**Statistiche aggregate: attenzione al livello.** Ogni file di statistiche ha
una riga per match, giocatore **e set**, più una riga `set == "Total"`. Sommare
tutte le righe conta ogni punto due volte. `load_mcp_stats()` tiene di default
solo i totali.

---

## 2. tennis-data.co.uk — *copertura completa + quote*

`www.tennis-data.co.uk` · un file `.xlsx` per stagione e per tour

Tutti i match del circuito ATP (dal 2000) e WTA (dal 2007), con punteggio per
set, ranking e punti dei due giocatori, e **quote dei bookmaker**.

```bash
python3 -m lib.download td --tour atp --from 2015 --to 2025
```

Colonne principali: `Location`, `Tournament`, `Date`, `Series`, `Court`,
`Surface`, `Round`, `Best of`, `Winner`, `Loser`, `WRank`/`LRank`,
`WPts`/`LPts`, `W1..W5`/`L1..L5` (giochi per set), `Wsets`/`Lsets`, `Comment`,
più le quote `B365W/L`, `PSW/L` (Pinnacle), `MaxW/L`, `AvgW/L`.

**Limiti:**

- **nessuna statistica di gioco**: niente ace, niente punti al servizio. Solo
  risultato, punteggio e mercato;
- i nomi dei giocatori sono abbreviati (`Popyrin A.`) e **non hanno un id**:
  incrociarli con il MCP (`Alexei Popyrin`) richiede un matching sui nomi, che
  è lavoro sporco e va validato a campione;
- `Comment` distingue `Completed` da ritiri e walkover: filtrarlo è quasi
  sempre necessario, perché un ritiro al primo game non è un match;
- il sito risponde a intermittenza e rifiuta le richieste senza user agent;
  quando è sovraccarico restituisce una pagina HTML **con codice 200** al posto
  del file. `lib.download` se ne accorge controllando la firma del file;
- le quote sono di chiusura e già comprensive del margine: `1/quota` non è una
  probabilità. Vedi `lib.loaders.implied_probabilities()`.

---

## 3. Altre piste, non ancora usate

| Fonte | Cosa dà | Perché non è (ancora) qui |
|---|---|---|
| Fork di `tennis_atp` su GitHub | i vecchi CSV Sackmann | fermi all'anno del fork, provenienza non verificabile |
| Tennis Abstract (`tennisabstract.com`) | classifiche e leaderboard elaborate | pagine web, niente export ufficiale: servirebbe scraping |
| ATP/WTA siti ufficiali | risultati e statistiche ufficiali | nessuna API pubblica, ToS restrittivi |
| Ultimate Tennis Statistics | database ricco, molte metriche derivate | dump non liberamente scaricabile |
| Provider commerciali (Sportradar, Infosys) | dati di tracking, colpo per colpo | a pagamento |

Se una di queste diventa necessaria, va aggiunta qui **con i suoi limiti**,
non solo con l'URL.
