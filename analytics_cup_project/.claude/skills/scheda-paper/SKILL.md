---
name: scheda-paper
description: >-
  Legge un paper della letteratura in analytics_cup_project/docs/ e ne produce
  una scheda in notes/letteratura/, con verdetto di fattibilità sui dati
  SkillCorner open. Usare quando l'utente dice "leggi questo paper", "cosa dice
  X", "aggiungi alla letteratura", "stato dell'arte", "questo metodo si può
  replicare?", nomina un file in docs/, oppure chiede di confrontare più lavori
  fra loro. Copre estrazione del testo (PDF e Apple Pages), lettura guidata,
  scheda, indice comparativo e ricaduta sulla scelta della domanda di ricerca.
---

# Scheda di un paper

## Regola che vale per tutti i passi: la scheda non è un riassunto

Un riassunto di un paper si trova già nel suo abstract. **Quello che l'abstract
non dice, e che a questo progetto serve, è se il metodo sopravvive ai nostri
dati.**

I paper in `docs/` lavorano su dati che noi non abbiamo. Il paper sulla pressione
difensiva usa 306 partite di J-League con evento e tracking sincronizzati; noi
abbiamo 20 partite di broadcast tracking con il 41% delle posizioni estrapolate.
Un metodo può essere ottimo e inapplicabile, e la scheda deve dirlo **prima** di
spiegare quanto è elegante.

Quindi ogni scheda si chiude con due cose che un riassunto non ha: un **verdetto
di fattibilità** motivato sui numeri, e il **buco che il paper lascia aperto** —
che è ciò che alimenta `plans/01-scegliere-la-domanda.md`.

Se al termine della lettura non sai dire quale delle due cose vale per questo
paper, non hai finito di leggerlo.

---

## Il flusso, in ordine

### 1. Estrai il testo

```bash
V=~/Documents/Projects/sport_venvs/analytics_cup_project/bin
S=.claude/skills/scheda-paper/scripts/estrai_testo.py

$V/python $S docs --elenco                    # cosa c'è, con le pagine
$V/python $S docs/<file>.pdf                  # tutto
$V/python $S docs/<file>.pdf --pagine 1-4     # solo una parte
$V/python $S docs/<file>.pages                # Apple Pages
```

**Sui `.pages`:** l'estrazione è approssimativa — accenti spezzati, ordine dei
frammenti non garantito. Va bene per documenti brevi. Se un `.pages` conta
davvero, chiedi all'utente di esportarlo in PDF o Markdown invece di lavorare
sull'estrazione.

**Controlla i duplicati prima di leggere.** Se due file hanno lo stesso numero
di pagine e titoli simili, verificali prima di produrre due schede: confronta il
testo estratto e i metadati (`PdfReader(f).metadata`).

Attenzione a non liquidarli come copie identiche. In `docs/` c'era il paper di
Bekkers in **due versioni** — v1 di gennaio 2025 e v2 arXiv di giugno — non due
copie dello stesso file. In quel caso si tiene quella con DOI e arXiv ID nei
metadati, che è quella citabile.

### 2. Leggi cercando sei risposte, non il senso generale

Non leggere linearmente sperando che emerga qualcosa. Cerca queste, in
quest'ordine — le prime due decidono tutto il resto:

1. **Che dati richiede?** Evento, tracking, o entrambi sincronizzati? A che
   frequenza? Posizioni di tutti i 22 giocatori o solo di alcuni?
2. **Su quante partite è stato validato?** È il numero da confrontare con le
   nostre 20.
3. **Dipende dai giocatori lontani dalla palla?** È il filtro che uccide più
   metodi: da noi oltre i 40 metri dalla palla solo il 18% delle posizioni è
   osservato.
4. **Serve la palla tracciata?** Nel nostro tracking manca nel 26% dei frame.
5. **Richiede annotazioni che non abbiamo?** Etichette di ruolo, eventi taggati
   a mano, esiti che SkillCorner non fornisce.
6. **Il suo risultato principale sta in una figura?** Vincolo di consegna: max
   2 figure. Un metodo che ha bisogno di sei pannelli non è presentabile, anche
   se funziona.

### 3. Scrivi la scheda

Una per paper, in `notes/letteratura/<nome-breve>.md`. Nome breve e parlante:
`pressione-tempo-arrivo.md`, non il nome del PDF.

Il formato è fisso — serve a rendere le schede **confrontabili fra loro**, che è
l'unico motivo per cui si scrivono invece di leggere e basta:

```markdown
# <Titolo>

**Riferimento:** autori, anno, sede di pubblicazione. [link se disponibile]
**File:** `docs/<nome>.pdf` (N pagine)

## Domanda e metodo

Tre o quattro righe, senza gergo. Cosa si chiede e come risponde.

## Dati richiesti

| | |
|---|---|
| Tipo | evento / tracking / entrambi sincronizzati |
| Frequenza | |
| Partite usate | |
| Completezza posizionale | tutti i 22? solo chi è vicino alla palla? |
| Annotazioni extra | |

## Verdetto di fattibilità

**<Replicabile / Replicabile con riserve / Adattabile / Non replicabile>**

Perché, con i numeri. Cosa si romperebbe sui nostri dati e dove.

## Cosa servirebbe per replicarlo

Stima onesta: quali pezzi esistono già (kloppy, dynamic_events, lib/), quali
andrebbero scritti, quanto è realistico nel tempo disponibile.

## Il buco che lascia

Cosa il paper non fa, e che con i nostri dati sarebbe affrontabile. È la riga
che alimenta `plans/01-scegliere-la-domanda.md`.

## Note
```

Testo in italiano, come il resto del workbench.

### 4. Aggiorna l'indice

`notes/letteratura/README.md` tiene una tabella comparativa: è lì che si vede a
colpo d'occhio quali metodi reggono. Una riga per paper:

| Scheda | Cosa misura | Dati richiesti | Verdetto |

Ordina per verdetto, non alfabeticamente: i replicabili in alto.

### 5. Ricaduta sui piani

Se il buco individuato è una direzione di ricerca praticabile, **annotalo in
`plans/01-scegliere-la-domanda.md`** invece di lasciarlo nella scheda. È lì che
la scelta si decide, e la regola del progetto è che un problema che non si chiude
subito va nel piano pertinente, non nella conversazione.

---

## Il vaglio di fattibilità: i nostri vincoli reali

Questi sono i numeri con cui confrontare ogni metodo. Misurati, non stimati —
vengono da `explorations/00` e `01`, e sono documentati in `notes/dati/`.

| Vincolo | Valore |
|---|---|
| Partite | **20**, A-League australiana 2024/25 |
| Frequenza | 10 fps |
| Posizioni osservate | **59%** complessivo |
| entro 5 m dalla palla | 87% |
| oltre 40 m dalla palla | **18%** |
| portiere | **15%** |
| Frame con tutti e 22 i giocatori | 74% (il resto è vuoto, non parziale) |
| Frame senza palla tracciata | 26% |
| Gioco osservabile | ~72 min su 98 nominali |

**Il vincolo che decide più casi è il terzo.** Le metriche difensive collettive —
altezza della linea, compattezza, forma del blocco — dipendono per costruzione
dai giocatori lontani dalla palla, che sono proprio quelli che la telecamera non
vede. Un metodo che le calcola sui nostri dati sta in buona parte misurando
l'algoritmo di estrapolazione di SkillCorner.

Non significa "scartare": significa che **quantificare quella fragilità è a sua
volta un contributo**, ed è una delle direzioni di ricerca già individuate.

### Cosa abbiamo già pronto

Prima di dire "servirebbe implementare X", controlla se X c'è già:

- **`dynamic_events.csv`** contiene modelli proprietari SkillCorner: `xthreat`,
  `xpass_completion`, `xshot`, `xloss`, EPV, `overall_pressure`,
  `space_constraint`, `time_to_impact`, `reception_difficulty`. Molti paper
  costruiscono proprio queste quantità da zero.
- **`phases_of_play.csv`** dà fase di gioco e già `width`/`length` di entrambe
  le squadre: compattezza senza derivarla.
- **`lib/data.py`** legge il tracking grezzo con `is_detected`, che kloppy non
  espone.
- **`lib/viz.py`** anima un evento o un intervallo di frame.

**Ma attenzione:** appoggiarsi alle metriche modellate di SkillCorner significa
ereditare assunzioni di un modello chiuso. È legittimo e va dichiarato — un
risultato costruito sopra non è difendibile fino in fondo davanti a una giuria
che chiede come è calcolato.

### Il vincolo sul campione

Con 20 partite, ghosting e graph neural network in stile DEFCON non sono
praticabili, e dirlo è più forte che provarci. Il vincitore dell'edizione 2026 ha
scelto l'ottimizzazione matematica **proprio perché** il ML non reggeva su quel
volume, e lo ha scritto nell'abstract.

Gli approcci che reggono: **fisici** (modelli di tempo di arrivo, controllo dello
spazio), **interpretabili** (template matching, regole), **bayesiani** (HMM,
modelli gerarchici). Quando un paper usa una di queste famiglie, il verdetto
parte avvantaggiato.

---

## Prima di consegnare una scheda

- il verdetto di fattibilità è motivato **con i numeri**, non con un'impressione?
- ho confrontato le partite usate dal paper con le nostre 20?
- ho verificato se il metodo dipende dai giocatori lontani dalla palla?
- ho controllato se quello che il paper costruisce esiste già in
  `dynamic_events.csv`, prima di stimare il lavoro di implementazione?
- il "buco che lascia" è una direzione concreta, o una frase generica?
- se il buco è praticabile, l'ho annotato in `plans/01-scegliere-la-domanda.md`?
- la scheda è nell'indice `notes/letteratura/README.md`, ordinata per verdetto?
- il paper era un duplicato di uno già schedato?
- il nome del file è breve e parlante, non il nome del PDF?
