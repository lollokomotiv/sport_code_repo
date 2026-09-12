# Quante smorzate gioca Alcaraz rispetto al resto del circuito?

## Domanda

Che quota dei suoi colpi Alcaraz gioca in smorzata, e quanto è rispetto agli
altri giocatori del circuito, a parità di superficie?

## Dati

- **Fonte**: Match Charting Project, `charting-m-stats-ShotTypes.csv` (riga `Dr`
  = smorzate, riga `Total` = tutti i colpi), più l'indice dei match per data e
  superficie. TennisMyLife solo per la copertura.
- **Download**: `python3 -m lib.download mcp --gender m` e
  `python3 -m lib.download tml --tour atp`
- **Soggetto**: i 221 match annotati di Alcaraz (62.688 colpi, 1.887 smorzate).
- **Riferimento**: tutti gli altri giocatori, nei match annotati **dalla data del
  primo match annotato di Alcaraz** (aprile 2019): 3.717 match, 525 giocatori,
  2,03 milioni di colpi. Per il confronto fra giocatori, quelli con **almeno
  3.000 colpi** annotati (130, circa 15 match): sotto quella soglia il tasso di un
  giocatore oscilla troppo.

### Copertura: quanto di Alcaraz è annotato

Dei **371 match ufficiali** di Alcaraz in TennisMyLife, **212 sono annotati**
(57%). Altri 10 match annotati non hanno un corrispettivo ufficiale
(Challenger, qualificazioni, esibizioni). Il campione annotato pende verso il
cemento: è annotato il 62% dei suoi match sul cemento e il 50% di quelli sulla
terra. Siccome sulla terra si giocano più smorzate, **il tasso grezzo sottostima
leggermente Alcaraz**: è uno dei motivi del confronto a parità di superficie.

### Cosa conta il denominatore

`Total` sono i **colpi dopo il servizio, risposta compresa**, e non include i
servizi. `run.py` lo verifica decodificando i punti dei 438 giocatori-match di
Alcaraz e dei suoi avversari negli anni 2020. `Total` vale in mediana 0,92 volte
i colpi dopo il servizio e 0,72 volte i colpi più i servizi. Lo scarto dell'8%
dalla prima misura è quello già noto della decodifica in `lib/points.py` (vedi
`alcaraz-zverev/vinte-perse`). "Smorzate ogni 100 colpi" vuol dire quindi ogni
100 colpi giocati nello scambio.

Una riga `Dr` mancante vuol dire **zero smorzate**: il file non scrive le righe a
zero. Fino a questa analisi `lib/shots.load_shot_type` scartava quei
giocatori-match, gonfiando il tasso di chi gioca poche smorzate. È stato
corretto in `lib/`.

## Metodo

Tre letture dello stesso numero, perché nessuna da sola basta:

- **contro il circuito aggregato**: tutte le smorzate degli altri diviso tutti i
  loro colpi. Pesa di più chi è annotato di più (Sinner, Djokovic, gli
  avversari frequenti di Alcaraz);
- **contro il giocatore mediano** fra i 130 con almeno 3.000 colpi, e la
  posizione di Alcaraz in quella classifica;
- **a parità di superficie**: le smorzate attese di ogni giocatore sono i suoi
  colpi su ciascuna superficie per il tasso del circuito su quella superficie,
  **calcolato senza di lui**. Il rapporto osservate/attese vale 1 per un
  giocatore nella media. Serve perché sulla terra si giocano quasi il doppio
  delle smorzate che sul cemento, e il calendario annotato cambia da un
  giocatore all'altro.

L'intervallo di confidenza ricampiona **i match** di Alcaraz (10.000 volte,
seme fisso), non i colpi: le smorzate di uno stesso match non sono indipendenti.

Il riferimento parte dalla data del primo match annotato di Alcaraz. Il tasso
del circuito cresce nel tempo: da 0,85 ogni 100 colpi nel 2014 a 1,79 nel 2025
in [`../../zverev/smorzate/`](../../zverev/smorzate/), dove il circuito
comprende Alcaraz. Qui, senza di lui, dal 2019 oscilla fra 1,41 e 1,72. Per questo c'è anche un controllo a parità di anno e
superficie.

Il calcolo sta in `lib/shots.py` (`rates_by_player`, `bootstrap_player`,
`rank_of`, `total_vs_decoded`) ed è condiviso con l'analisi su Zverev.

## Risultato

**Alcaraz gioca circa il doppio delle smorzate del circuito: 3 ogni 100 colpi,
10° su 130 giocatori.**

| | smorzate ogni 100 colpi | Alcaraz, in multipli |
|---|---|---|
| **Alcaraz** (221 match, 62.688 colpi) | **3,01** [2,85 – 3,17] | — |
| resto del circuito, aggregato | 1,56 | 1,9× |
| giocatore mediano (130 con ≥ 3.000 colpi) | 1,41 | 2,1× |
| a parità di superficie | rapporto **1,85** [1,76 – 1,95] | 9° su 130 (mediana 0,86) |

Per superficie il distacco resta simile, ed è più ampio sul cemento, dove le
smorzate sono più rare per tutti:

| | Alcaraz | circuito | multiplo |
|---|---|---|---|
| terra | 3,72 | 2,22 | 1,7× |
| erba | 3,27 | 1,85 | 1,8× |
| cemento | 2,52 | 1,21 | 2,1× |

**Plausibilità.** In cima alla classifica ci sono i giocatori noti per la
smorzata: Bublik (6,18, da solo il doppio di Alcaraz), Moutet, Paire, Kyrgios.
In fondo ci sono Millman, Struff e Rublev (0,25 su 171 match). Sulla terra se ne
giocano più che sull'erba, e sull'erba più che sul cemento.

**I controlli non cambiano il risultato.** A parità di anno e superficie il
rapporto è 1,83 (10° su 130). Nei match annotati vinti gioca 2,96 smorzate ogni
100 colpi, in quelli persi 3,06: se il campione pende da una parte, il tasso non
ne risente.

**Alcaraz è alto, non estremo.** I nove giocatori davanti a lui hanno fra 15 e
97 match annotati; lui ne ha 221, quindi il suo tasso è di gran lunga il più
stabile della parte alta della classifica. Con la soglia a 5.000 colpi sale 8° (7°
a parità di superficie), con quella a 2.000 scende 13°: la posizione dipende
dalla soglia, il "circa il doppio" no.

## Limiti

- **Il Match Charting Project non è un campione casuale.** Sono annotati
  soprattutto i match importanti e i giocatori popolari: il "circuito" qui è il
  circuito annotato, dove i big giocano più partite dei giocatori di fascia
  bassa. Il giocatore mediano è un confronto più robusto dell'aggregato per
  questo motivo.
- **Di Alcaraz è annotato il 57% dei match ufficiali**, e meno sulla terra (50%)
  che sul cemento (62%). Il confronto a parità di superficie corregge il
  calendario, non la selezione dei match dentro ciascuna superficie.
- **Chi annota decide cos'è una smorzata.** Il confine fra smorzata e colpo
  corto tagliato è una scelta dell'annotatore, e qui non è controllato.
- **Il volume non dice nulla sulla resa.** Quante smorzate vincono il punto è
  un'altra domanda (parziale risposta in
  [`../../alcaraz-zverev/smorzate/`](../../alcaraz-zverev/smorzate/), contro gli
  avversari di Alcaraz).
- **Contro chi**: il tasso dipende anche dall'avversario (contro Zverev ne
  gioca 1,46 volte il suo solito). Qui ogni giocatore è misurato contro i
  propri avversari, che non sono gli stessi.

## Come si riproduce

```bash
python3 -m lib.download tml --tour atp
python3 -m lib.download mcp --gender m --points
python3 analyses/alcaraz/smorzate/run.py
```
