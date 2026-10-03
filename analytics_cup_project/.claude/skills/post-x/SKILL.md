---
name: post-x
description: >-
  Scrive post e thread in inglese per X (Twitter) a partire da un risultato del
  workbench dell'Analytics Cup sui dati SkillCorner — un'esplorazione, una
  replica di un paper, un'analisi, un controllo di qualità dei dati — con
  immagini, GIF o video del tracking. Usare quando l'utente dice "scriviamo un
  post", "un tweet su questo", "facciamo un thread", "come lo comunichiamo",
  "pubblichiamolo su X", "una GIF di questa azione per X", oppure chiede di
  trasformare un notebook, un report o un'animazione in contenuto social. Copre
  la scelta della fonte, la promozione dei numeri da notebook a script, il
  controllo di ciò che i dati reggono, i media (PNG, GIF, MP4) con lib/social.py,
  il testo e l'archivio.
---

# Post su X dai dati SkillCorner

La skill vale per **qualunque** risultato del workbench: una pista del piano,
un'esplorazione, una nota sulla qualità dei dati. Non presuppone una domanda
di ricerca in particolare. Presuppone però che il risultato esista già: se
l'utente vuole un post su qualcosa che non è ancora stato calcolato, prima si fa
l'analisi, poi il post.

Due regole valgono per tutti i passi.

**1. Il post non produce numeri.** Ogni cifra del post, del grafico e dell'alt
text deve stare in un report in `reports/`, generato da uno script in
`scripts/`. Un post è l'unico artefatto del repo che qualcuno può citare senza
vedere il codice: un numero che non risale a un comando rieseguibile è
indifendibile proprio quando serve difenderlo. Se il numero sta solo in un
notebook di `explorations/`, va **promosso** prima: leggi
[`references/promozione.md`](references/promozione.md). Se il taglio richiede un
numero che il report non ha, lo si aggiunge allo script e si rigenera il report;
non lo si calcola nella shell.

**2. Ciò che la telecamera non ha visto si dichiara.** È tracking broadcast: una
parte delle posizioni è stimata fuori inquadratura (`is_detected`). Ogni
immagine del campo mostra le posizioni stimate col contorno bianco e lo dice in
legenda; ogni affermazione che dipende da posizioni dice quanta parte era
osservata, se il report lo misura. Un frame in cui il giocatore chiave è
stimato, presentato come osservato, è l'errore che qualcuno ti farà notare in
pubblico.

---

## Il flusso, in ordine

### 1. Scegli la fonte con l'utente, poi rileggila

Elenca i report disponibili (`reports/*.md`) e, se l'utente parla di un
notebook, il notebook in questione. **Chiedi quali usare** — `AskUserQuestion`
con `multiSelect: true` se sono al massimo quattro, altrimenti in testo. Non
scegliere al suo posto.

Se la fonte è solo un notebook, fermati qui e proponi la promozione
(`references/promozione.md`). Dillo in una riga: "questo numero sta solo nel
notebook 0X, su una partita; prima di pubblicarlo va portato in uno script su
tutte le partite". La promozione può cambiare il numero: è il motivo per farla.

Per ogni report scelto, rileggilo per intero (non dal ricordo della
conversazione) ed estrai:

- **il fatto**: il risultato, con il suo numero e il suo confronto;
- **il campione**: quante partite, quanti eventi o possessi, quali filtri;
- **la copertura**: quanta parte dei dati usati era osservata;
- **le metriche di terzi**: se il risultato usa output di modelli SkillCorner
  (`xthreat`, `time_to_impact`, `overall_pressure`, gli eventi dinamici stessi),
  vanno nominati come tali;
- **ciò che il risultato non dice**: i limiti, in una riga.

Un risultato negativo o deludente è materiale buono: "la cosa che si dà per
scontata non si vede nei dati" è un gancio più forte di una conferma.

### 2. Due controlli prima di scrivere

**La gara.** Il workbench prepara una submission all'Analytics Cup. Se il
risultato è il contributo centrale di qualcosa che l'utente potrebbe
presentare, pubblicarlo prima può toglierle novità. Chiedi in una riga se il
risultato è destinato alla submission; se sì, proponi di pubblicare la parte di
metodo o di qualità dei dati e tenere il risultato principale. Decide l'utente.

**Le persone.** I giocatori sono persone vere, in un campionato piccolo. Un
giudizio negativo su un giocatore nominato, costruito su una sola azione o su
posizioni stimate, non si pubblica. Gli aggregati di squadra e di campionato
vanno bene; una clip singola illustra un meccanismo, non giudica chi c'è dentro.

### 3. Tesi, contenuto e taglio li dà l'utente

Servono tre cose, e se ne manca una si chiede: **la tesi** (cosa afferma il
post), **il contenuto** (quali numeri), **il taglio** (a chi parla: di norma
asciutto, per chi fa football analytics).

Poi confronta la tesi con ciò che il report regge. Con questi dati le derive
tipiche sono sei:

| Deriva | Come si riconosce | Cosa fare |
|---|---|---|
| **Lo stimato come osservato** | il post descrive posizioni, distanze o corse senza dire che una parte è stimata | aggiungere la copertura, o restringere agli osservati se il report lo fa |
| **La partita come campionato** | un numero da una partita (o da un notebook) detto "in A-League" | promuovere su tutte le partite, o dire "in one match" |
| **L'A-League come il calcio** | "teams do X", "defenders do Y" | "in 20 A-League 2024/25 matches" |
| **Il modello come misura** | una metrica SkillCorner presentata come un fatto osservato | "SkillCorner's model rates…", non "it was…" |
| **La clip come regola** | un'animazione presentata come tipica senza un numero dietro | la clip illustra; il numero (dal report) generalizza |
| **Il meccanismo al posto della correlazione** | un "because" che il report non ha misurato | togliere il perché, o dichiararlo ipotesi |

Se la tesi cade in una di queste, **dillo in una riga e proponi la versione
onesta della stessa storia** — non rifiutare il post.

### 4. Si semplifica il linguaggio, mai il limite

Quando il post non ci sta, **si taglia un numero, mai una condizione**. Un post
con un numero e il suo limite è pubblicabile; tre numeri senza limiti no.

| Non scrivere | Scrivi |
|---|---|
| "Defenders close down the carrier in 0.9 s" | "the fastest opponent could reach the carrier in 0.9 s (median, motion model)" |
| "Broadcast tracking can't be trusted for pressing" | "for the opponent nearest in time to the ball carrier, it's estimated in only 3.3% of possessions" |
| "Newcastle press higher than anyone" | "Newcastle press highest of the teams in these 20 matches" |
| "Player X never tracks back" | non si pubblica (giudizio su una persona, vedi §2) |

### 5. La forma, e il media giusto

| La storia è… | Forma | Media |
|---|---|---|
| un numero solo, sorprendente | post singolo | grafico PNG |
| un meccanismo nello spazio (chi arriva prima, dove si apre il varco) | post singolo | frame PNG del campo con sovrapposizioni |
| un movimento (una corsa, una sequenza di pressing) | post singolo o 2 post | GIF breve in loop (≤ 35 s, meglio ≤ 10 s) |
| un'azione lunga o da fermare e rivedere | post singolo | MP4 (≤ 140 s) |
| un risultato più il suo controllo | thread da 2 | grafico nel primo, controllo nel secondo |
| un risultato aggregato più un esempio | thread da 2 | grafico nel primo, clip nel secondo |

GIF o MP4: la GIF parte da sola e gira in loop, quindi funziona per azioni di
pochi secondi; oltre i 35 s (350 frame a 10 fps) X non la accetta e serve l'MP4.

Regole di X che cambiano il testo:

- **280 caratteri** è il confine: la timeline taglia lì, e la prima parte deve
  reggere da sola;
- **il primo post di un thread** contiene il fatto e il campione, perché verrà
  ripubblicato senza il resto;
- **il link** (al repo o al notebook) va in una risposta, mai nel post principale;
- **niente hashtag**, emoji al massimo una;
- **non inventare handle.** SkillCorner e PySport si citano per nome; un handle
  si usa solo se l'utente lo conferma.

### 6. I media si generano con `lib/social.py`, dallo script

Il codice del media sta **nello script che produce i numeri del post** (una
funzione `figura_post()` o un flag `--post`), non nella shell e non nel
notebook: così il media si rigenera con i numeri. Il testo nel media viene da
f-string sui dati, mai scritto a mano. I file vanno in `figures/post/<slug>.<ext>`.

Funzioni disponibili (leggi le docstring in `lib/social.py` per i dettagli):

| Funzione | Cosa fa |
|---|---|
| `figura(titolo, sottotitolo, campo_meta=None)` | figura 16:9 con titolo, sottotitolo e fonte; con `campo_meta` disegna il campo |
| `legenda_campo(ax, meta, extra)` | squadre + contorno dell'estrapolazione, nell'intestazione |
| `salva_png(fig, path)` | salva a 1800×1012 e controlla i limiti |
| `anima(match_id, frame_start, frame_end, path)` | GIF o MP4 secondo l'estensione, in inglese, con fonte e colori delle squadre |
| `anima_evento(match_id, event_id, path)` | lo stesso, partendo da un evento di `dynamic_events.csv` |
| `controlla(path)` | misura il file contro i limiti di X; solleva se li supera, avvisa sopra i 5 MB per le GIF |
| `fotogrammi(path, out_dir, n)` | estrae fotogrammi da GIF/MP4 per guardarli |

Per una clip, prima di animarla controlla quanto il protagonista era osservato
con `lib.viz.osservabilita_evento`: se è sotto metà dei frame, scegli un'altra
azione o dillo nel post.

**Carica la skill `dataviz` prima di scrivere un grafico.** Il titolo dice il
risultato, non l'asse.

**Guarda sempre il media prima di consegnarlo**: il PNG direttamente, le
animazioni con `fotogrammi()` (inizio, metà, fine). Il controllo dei limiti non
vede etichette sovrapposte, un'azione tagliata o un protagonista fuori campo.

Scrivi l'**alt text** (fino a 1.000 caratteri): cosa mostra il media e il
numero principale. Per un'animazione, descrivi l'azione in ordine.

### 7. Inglese da football analytics

Il post è in inglese, asciutto. Struttura: il fatto in una riga, senza
preamboli ("I looked at…", "Thread 🧵"); il numero con il suo confronto; la
sorpresa o il contro-fatto, se c'è; campione e fonte in chiusura.

Termini attesi: `ball carrier`, `press` / `pressing`, `time to reach` o
`arrival time`, `off-ball run`, `line-breaking pass`, `final third`,
`half-space`, `defensive line`, `broadcast tracking`, `positions estimated off
camera`, `motion model`. Per le metriche SkillCorner usa il loro nome
(`Off-Ball Runs`, `Dynamic Events`) e attribuiscile.

Da evitare: superlativi non misurati (`insane`, `elite`), calchi dall'italiano,
"proves". Numeri arrotondati a una decimale, o interi quando la precisione non
aggiunge nulla; mai più cifre di quante il report ne giustifichi.

### 8. Consegna e archivio

Mostra il post come testo pronto da copiare, con il conteggio dei caratteri:

```
[1/2 — 268 chars]
<testo>

[media — figures/post/<slug>.gif · 1200×675 · 0.8 MB · 62 frames]
[alt text — 190 chars]
<testo>
```

Salva il testo in `posts/<AAAA-MM-GG>-<slug>.md`, con in testa: il report o i
report da cui vengono i numeri, lo script e il comando che li rigenera, i media
e il commit corrente (`git rev-parse --short HEAD`). Serve a ritrovare cosa è
uscito e con quali numeri, e ad accorgersi se un aggiornamento dei dati rende
falso un post già pubblicato.

**Non pubblicare nulla.** La skill scrive il testo e i media; pubblica l'utente.

---

## Esempio

**Il taglio dell'utente**: "Il tracking broadcast è inaffidabile per studiare
il pressing: il 41% delle posizioni è inventato".

**Cosa regge**:

- il 41% (59% osservato) viene da `explorations/00`, **una partita sola, solo
  notebook**: non è citabile finché non è promosso;
- `reports/tau_opp.md` dice invece che l'avversario che arriverebbe per primo
  sul portatore è stimato nel **3,3%** dei 17.566 possessi. Il tasso medio e
  quello del giocatore che conta sono cose diverse;
- se il taglio volesse aggiungere "e succede soprattutto quando la pressione è
  bassa", quel numero non è nel report: va aggiunto a `scripts/report_tau.py`
  prima di citarlo.

**Versione sbagliata**:

> Broadcast tracking is 41% made up. Forget using it to study pressing.

Numero non promosso, estrapolato a tutto il dataset, e conclusione contraria a
quella del report.

**Versione pubblicabile** (post singolo, con il 41% solo dopo la promozione):

> In SkillCorner's broadcast tracking, many positions are estimated off camera.
>
> For pressing on the ball carrier it barely matters: the opponent who'd get
> there first is estimated in only 3.3% of possessions.
>
> 17,566 possessions, 20 A-League 2024/25 matches. SkillCorner open data.

La sorpresa è proprio il contrasto col luogo comune.

---

## Prima di consegnare

- ho chiesto all'utente quali report usare?
- ogni numero (testo, media, alt text) sta in un report generato da uno script?
- il post dice il campione, e quanta parte era osservata se conta?
- le metriche SkillCorner sono attribuite come modelli, non come fatti?
- ho chiesto se il risultato è destinato alla submission?
- nessun giudizio negativo su un giocatore nominato da una clip o da posizioni stimate?
- c'è un "because" che il report non ha misurato?
- il primo post regge da solo?
- il media è generato dallo script, l'ho guardato, `controlla()` è passato?
- c'è l'alt text?
- nessun handle inventato, nessun hashtag, link in risposta?
- ho salvato il post in `posts/` con fonte, comando, media e commit?
- quando ho tagliato, ho tagliato un numero e non un limite?
