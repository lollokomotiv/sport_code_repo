# CLAUDE.md — xGoals Project

Linee guida per lavorare in questo progetto. Contesto: modelli di **expected goals / expected assists** su dati StatsBomb open data, con pipeline di inferenza e visualizzazioni.

Questi progetti servono anche come **portfolio per lavorare nell'analytics sportivo**. Il criterio non è "gira senza errori", è "regge se qualcuno del settore lo apre e lo esamina".

---

## 1. Il benchmark viene prima del modello

In questo dominio esiste un riferimento esterno gratuito: `shot.statsbomb_xg` è già nei dati evento. Ogni modello xG va confrontato con quello — non per copiarlo, ma perché una divergenza forte è un segnale che qualcosa non va, e va spiegata.

**Prima di dichiarare un modello funzionante:**
- media predetta vs tasso reale dell'evento (calibrazione di livello)
- correlazione con il benchmark, Pearson **e** Spearman (le distribuzioni xG sono molto asimmetriche)
- curva di calibrazione a bin, non solo una metrica aggregata
- valutazione su **holdout**, non su validation riusata per scegliere gli iperparametri

AUC alta non basta. Un modello può ordinare bene i tiri e sbagliare le probabilità del doppio — ed è esattamente il caso in cui ci si trova oggi (vedi `plans/00-xg-calibrazione.md`).

## 2. Split a livello di match, sempre

I tiri della stessa partita sono correlati. Uno split casuale sulle righe è data leakage. Nel progetto lo split è già a livello di `match_id` — mantenerlo in ogni nuovo modello.

## 3. Training e inferenza devono calcolare le feature allo stesso modo

Il rischio ricorrente qui è la **doppia implementazione**: le feature esistono nei notebook (training) e in `inference/score_match.py` (produzione). Quando divergono, il modello riceve in inferenza input diversi da quelli su cui è stato addestrato e nessuno se ne accorge.

Casi già emersi:
- `visible_area` è una lista **piatta** `[x1,y1,x2,y2,...]` in StatsBomb — va convertita in coppie prima di calcolare l'area
- flag booleani (`cross`, `switch`, `through_ball`) vanno passati come `int`, non `bool`: `SimpleImputer` rifiuta il dtype bool

**Regola:** quando modifichi una funzione di feature engineering in un notebook, controlla se esiste la gemella in `inference/` e allineala nello stesso commit.

## 4. Smoke test prima di ogni run lungo

I notebook rileggono ~2.2 GB di JSON. Non eseguire celle a mano accodandole: usa il runner headless.

```bash
# smoke test (~7s) — cattura ogni errore di codice perché tocca tutte le celle
python3 run_notebook.py <notebook>.ipynb \
    --limit-matches 8 --models-dir /tmp/smoke --figures-dir /tmp/smoke/figs

# run completo
python3 run_notebook.py <notebook>.ipynb --figures-dir figures/<nome>
```

Il runner esegue le celle in ordine in un solo processo e si ferma alla prima eccezione, stampando sorgente e traceback. Uno smoke test da 8 match non valida i *risultati* (i numeri da campione piccolo non significano nulla) ma cattura tutti i bug di codice.

**Mai far scrivere a uno smoke test nella cartella `models/` di produzione**: usa `--models-dir` su una dir temporanea.

## 5. Sanity check sui numeri, non solo sull'assenza di errori

Exit code 0 non vuol dire che il risultato sia giusto. Un run può completare e produrre `is_goal = 0` su 5.335 tiri — il che è impossibile e va notato.

Prima di accettare un output, chiedersi: gli ordini di grandezza sono plausibili? Il tasso di gol è ~10%? La somma degli xG per partita assomiglia ai gol reali? Le zone popolate sono quelle attese?

Quando un numero è implausibile, **dirlo esplicitamente** invece di lasciarlo passare.

## 6. I dati e i loro limiti

- Fonte: `/Users/lorenzoguercio/Documents/Projects/sport_data/open-data/data/`
- 3.464 match con eventi, ma solo **326 con dati 360** (freeze frame)
- I freeze frame sono istantanee al momento dell'evento, **non** tracking continuo: metriche che richiedono traiettorie non sono costruibili con questi dati
- `visible_area` indica cosa era inquadrato: i giocatori fuori campo visivo semplicemente non ci sono, non sono "assenti"

I limiti dei dati vanno dichiarati nei risultati, non nascosti. È esattamente ciò che distingue un'analisi credibile.

## 7. Coerenza con il codice esistente

- Le funzioni di inferenza in `score_match.py` sono pure: DataFrame in → DataFrame/Figure out. Mantenere questa forma rende riusabile il codice in una webapp senza riscritture.
- I modelli si salvano come `Pipeline` sklearn complete (preprocessing incluso), mai il solo classificatore.
- Le visualizzazioni usano `mplsoccer` con lo stile già stabilito (`VerticalPitch`, sfondo `#1a1a2e`, oro per gli eventi che diventano gol).
- Commenti e output in italiano, coerentemente col resto del progetto.

## 8. Pianificazione

I filoni di lavoro stanno in `plans/`, numerati per priorità, con `plans/README.md` come indice. Quando emerge un problema che non si chiude subito, **va annotato nel piano pertinente** invece di restare in una conversazione.

Ordine attuale: `00` calibrazione xG → `01` completamento xA → `02` metriche avanzate → `03` webapp. Non saltare avanti: ogni filone eredita gli errori del precedente.

## 9. Onestà tecnica

Un risultato negativo documentato vale più di un successo apparente. Se una feature non aggiunge nulla, se un modello complesso non batte la baseline, se una metrica non è validabile con questi dati — dirlo e scriverlo.

La documentazione deve descrivere ciò che il codice fa davvero. Il README di questo progetto ha dichiarato a lungo "due modelli LightGBM" mentre il modello in produzione era una `LogisticRegression`: è il tipo di discrepanza che, in un colloquio, costa più di un modello mediocre.
