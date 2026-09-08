---
name: articolo-social
description: >-
  Scrive post e thread in inglese, stile Twitter/X, a partire da un'analisi già
  conclusa in tennis_project. Usare quando l'utente dice "scriviamo un post",
  "facciamo un thread", "un tweet su questa analisi", "raccontiamo questo
  risultato", "articolo", "pubblichiamo", oppure quando dopo un'analisi chiede
  come comunicarne il risultato. Copre tesi e taglio, controllo di ciò che i
  dati reggono davvero, scrittura del post, grafico per il timeline e
  archiviazione del testo.
---

# Articolo social (stile X)

Questa skill viene **dopo** `nuova-analisi`. Presuppone che esista già
un'analisi conclusa sotto `analyses/<slug>/`, con il suo `run.py` e il suo
README. Se non esiste, non è il momento di questa skill: si torna a
`nuova-analisi` e si fa prima l'analisi.

## Regola che vale per tutti i passi: il post non produce numeri

**Ogni cifra che compare nel post deve già stare nell'output di
`analyses/<slug>/run.py`.** Il post è un atto di comunicazione, non di calcolo.

Se il taglio richiede un numero che l'analisi non ha — "e contro i mancini?",
"e negli ultimi due anni?" — la risposta non è calcolarlo nella shell. La
risposta è **aggiungere quel calcolo a `run.py`, rieseguirlo, e aggiornare il
README**. Poi il post lo cita.

Il motivo è che un post è la cosa più esposta che questo repo produce: è l'unico
artefatto che qualcuno può citare senza avere accesso al codice. Un numero che
non si può risalire a una riga di script è indifendibile esattamente quando
serve difenderlo.

---

## Il flusso, in ordine

### 1. Rileggi l'analisi prima di parlarne

Apri il README dell'analisi **e** l'output di `run.py`. Non scrivere dal
ricordo della conversazione: il README contiene i limiti, e i limiti sono metà
del lavoro.

Estrai tre cose, in questa forma:

- **il fatto**: il risultato principale, con il suo numero e il suo confronto;
- **il campione**: quanti match, quanti punti, quanto è annotato;
- **ciò che il risultato NON dice**: la sezione Limiti, riassunta in una riga.

Se il README dice che la tesi è risultata falsa, **quello è il post**. Un
risultato negativo è materiale eccellente: "la cosa che tutti danno per scontata
non si vede nei dati" è un gancio più forte di una conferma.

### 2. Tesi, contenuto e taglio li dà l'utente

L'utente porta tre cose, e se ne manca una si chiede:

- **la tesi**: cosa vuole affermare il post;
- **il contenuto**: quali numeri vuole dentro;
- **il taglio**: a chi parla e con che tono — analitico e asciutto, oppure
  divulgativo, oppure polemico verso un luogo comune.

Poi, **prima di scrivere**, confronta la tesi con ciò che l'analisi regge
davvero. Qui si gioca tutto, e il modo in cui una tesi devia dai dati è quasi
sempre uno di questi quattro:

| Deriva | Come si riconosce | Cosa fare |
|---|---|---|
| **Il meccanismo al posto della correlazione** | il taglio contiene un "perché" che l'analisi non ha misurato | tenere il fatto, togliere il perché — o darlo come ipotesi dichiarata |
| **Il numero giusto letto male** | la metrica ha un denominatore che cambia fra i gruppi confrontati | usare la metrica grezza, o spiegare il denominatore |
| **Il campione annotato scambiato per il circuito** | il post dice "sul tour", "in carriera", "sempre" | dire "nei match annotati", e quanti sono |
| **L'assenza di prova come prova di assenza** | il post afferma che una cosa *non* accade, da un test non significativo | dire che i dati non la vedono, non che non esiste |

Se la tesi dell'utente cade in una di queste, **dillo in una riga e proponi la
versione onesta della stessa storia** — non rifiutare il post. Quasi sempre la
versione onesta è più interessante, perché contiene una sorpresa in più.

### 3. Il vincolo che non si negozia

Dal CLAUDE.md del progetto, ed è la regola che governa questa skill:

> **Si può semplificare il linguaggio, mai il limite.**

Si può scrivere "Alcaraz dominates with the forehand" invece di "10,5 vincenti
ogni 100 dritti contro 3,3". **Non** si può omettere che il campione copre 5
partite su 8 e pende da una parte.

In pratica, quando il post non ci sta nei caratteri: **si taglia un numero, mai
una condizione**. Un post con un numero e il suo limite è pubblicabile; un post
con tre numeri senza limiti non lo è.

Tre riscritture che valgono sempre:

| Non scrivere | Scrivi |
|---|---|
| "Alcaraz wins 62.6% of drop-shot points" | "62.6% in 131 charted drop shots" |
| "Zverev's deep return position forces the drop shot" | "Alcaraz drop-shots Zverev more than anyone — the charting data can't say why" |
| "Sinner never does X" | "X doesn't show up in Sinner's 41 charted matches" |

### 4. Scegli la forma

| La storia è… | Forma |
|---|---|
| un fatto solo, sorprendente | **post singolo** con grafico |
| un fatto più il suo controllo ("regge anche togliendo la superficie") | **post singolo**, controllo nella prima risposta |
| due risultati opposti (volume sì, resa no) | **thread da 2-3**, uno per affermazione |
| un confronto fra molti giocatori | **post singolo** con grafico, il testo nomina solo gli estremi |

Regole di forma su X che cambiano il testo:

- **280 caratteri è il confine che conta.** Anche con un account che ne permette
  di più, la timeline taglia lì con "Show more": la prima parte deve stare in
  piedi da sola.
- **Il primo post di un thread è quello che gira.** Deve contenere il fatto e il
  campione, perché verrà ripubblicato senza il resto.
- **I link deprimono la portata**: il link all'analisi va in una risposta, mai
  nel post principale.
- **Niente hashtag.** Su X non aggiungono nulla nel tennis analitico e fanno
  sembrare il post promozionale. Emoji al massimo una, spesso zero.
- **Non inventare handle.** Il Match Charting Project si cita per nome; se non
  sei certo dell'handle esatto di un account, scrivi il nome per esteso invece
  di indovinare una menzione che finirebbe su un profilo sbagliato.

### 5. Scrivi in inglese, e in inglese da tennis

Il post è in inglese anche se tutto il resto del progetto è in italiano.

Struttura che funziona, in ordine:

1. **il fatto**, in una riga, senza preamboli — mai "I looked at…", "Here's an
   interesting stat…";
2. **il numero con il suo confronto** — un numero da solo non significa niente:
   `45% more than expected` vive solo accanto a `median: 0%`;
3. **la sorpresa o il contro-fatto**, se c'è;
4. **il campione e la fonte**, in chiusura, in forma compatta.

Termini da usare, perché sono quelli che il lettore anglofono di tennis si
aspetta: `drop shot`, `unforced error`, `forced error`, `winner`,
`first-serve points won`, `break point saved`, `hold`, `break`, `deuce/ad
court`, `best-of-five`, `charted matches`, `surface-adjusted`.

Da evitare: calchi dall'italiano (`short ball` per smorzata, `direct point`),
superlativi non misurati (`insane`, `unbelievable`), e il presente storico.

Numeri: arrotonda a una cifra decimale, o a intero quando la precisione non
aggiunge (`45% more`, non `44.8% more`). Le percentuali di punti vinti a una
decimale.

### 6. Il grafico del post non è quello del README

Il grafico dell'analisi è fatto per essere letto fermi, su GitHub. Quello del
post viene visto piccolo, in movimento, spesso su telefono. Cambia:

- **proporzioni**: X mostra le immagini singole ritagliate verso il 16:9 nella
  timeline. Un grafico 13,5×8,6 con 27 righe diventa illeggibile;
- **meno righe**: nel post si tengono il soggetto e pochi riferimenti, non tutta
  la distribuzione;
- **caratteri più grandi**, e il titolo che dice il risultato, non l'asse;
- **la fonte resta nel grafico**, perché l'immagine viene ripubblicata da sola.

Il codice del grafico del post **sta anch'esso in `analyses/<slug>/run.py`** (o
in un `post_figure()` nella stessa cartella) e la figura va in
`analyses/<slug>/figures/`, che è versionata. Vale la regola di
`nuova-analisi`: nessuna cifra scritta a mano nei testi del grafico, tutto da
f-string.

**Carica la skill `dataviz` prima di scrivere il codice del grafico**, e
**guarda il PNG dopo averlo generato**.

Scrivi sempre anche l'**alt text** (X ne accetta fino a 1.000 caratteri):
descrive cosa mostra il grafico e ripete il numero principale, perché è ciò che
legge chi non vede l'immagine.

### 7. Consegna e archivio

Mostra all'utente il post **come testo pronto da copiare**, con il conteggio dei
caratteri di ogni parte, così:

```
[1/2 — 268 caratteri]
<testo>

[alt text — 190 caratteri]
<testo>
```

Poi salva il testo in `analyses/<slug>/posts/<AAAA-MM-GG>-<slug-breve>.md`, con
in testa una riga che dice da quale analisi viene e a quale figura si appoggia.
Serve a due cose: ritrovare cosa è stato pubblicato e con quali numeri, e
accorgersi se un aggiornamento dei dati rende falso un post già uscito.

**Non pubblicare nulla.** Questa skill scrive il testo; a pubblicare è l'utente.

---

## Esempio completo

Dall'analisi `analyses/alcaraz-zverev-smorzate/`.

**Il taglio proposto dall'utente**: "Alcaraz massacra Zverev di smorzate perché
Zverev sta troppo indietro, e gli rende tantissimo".

**Cosa regge e cosa no**, dal README:

- *regge*: rapporto 1,45 osservate/attese, 1° su 27 avversari, mediana 0,98,
  p = 0,000012, robusto a leave-one-out, anno, formato e annotatore;
- *non regge*: la resa. Il differenziale +11,2 contro +9,5 sembra un vantaggio,
  ma i tassi grezzi sono 62,6% contro 62,9%: identici. Il differenziale è più
  alto solo perché contro Zverev Alcaraz vince meno punti in generale;
- *non misurato*: la posizione in campo di Zverev. Il MCP non la registra.

**Versione sbagliata** — tre errori in due righe:

> Alcaraz's drop shot destroys Zverev — it wins him 11.2% more points than
> usual, because Zverev returns from three metres behind the baseline.

Il "because" non è nei dati; "11.2% more points than usual" legge male il
differenziale; manca il campione.

**Versione pubblicabile** (post singolo, 277 caratteri):

> Alcaraz drop-shots Zverev more than any other opponent.
>
> Surface-adjusted, that's 45% more than expected — top of his 27 most-charted
> opponents.
>
> They don't work any better, though: 62.6% of points won vs 62.9% against the
> rest.
>
> 8 charted matches. Match Charting Project data.

Il contro-fatto ("they don't work any better") non indebolisce il post: è la
parte che lo rende non ovvio.

---

## Prima di consegnare un post

- ogni numero del post sta nell'output di `run.py`?
- il post dice **quanti** match o punti, e che sono **annotati**?
- c'è un "because" o un "so" che l'analisi non ha misurato?
- una percentuale è confrontata con qualcosa, o è appesa al nulla?
- se il risultato è negativo, il post dice "i dati non lo vedono" e non "non
  esiste"?
- il primo post regge da solo, senza il resto del thread?
- il grafico è leggibile piccolo, e l'ho **guardato**?
- c'è l'alt text?
- il Match Charting Project è citato, senza handle inventati?
- ho salvato il testo in `analyses/<slug>/posts/`?
- quando ho dovuto tagliare, ho tagliato un numero e non un limite?
