# Indagine: perché l'xG sembrava sbagliato del doppio

**Data:** 2026-08-20
**Esito:** il modello xG era sano. Il difetto era in una terza copia della feature engineering.

---

## Il sintomo

Al primo run completo del notebook xA, un sanity check già presente nel codice ha prodotto questo:

```
Shot dataset: 5,335 tiri da key pass
our_xg medio:       0.1611
statsbomb_xg medio: 0.0800
Correlazione our_xg vs statsbomb_xg: 0.6147
```

Due segnali distinti:
- **livello** — il modello prediceva in media il doppio del benchmark;
- **ordinamento** — correlazione 0.61, cioè i due modelli non concordavano nemmeno su *quali* tiri fossero buoni.

## Le ipotesi, e come sono cadute

**1. `class_weight='balanced'` nel training.** Riequilibrare le classi gonfia sistematicamente le probabilità predette, ed è un errore comune su target sbilanciati. Caricato il modello: `class_weight = None`. **Esclusa.**

Il controllo ha però rivelato altro: il modello salvato è una `LogisticRegression`, mentre il README dichiarava "due modelli LightGBM". La documentazione non descriveva il codice.

**2. Il modello è troppo semplice.** Una regressione logistica è lineare nei log-odds e non può rappresentare la curvatura della relazione distanza→probabilità. Ipotesi plausibile per la correlazione bassa — ma non spiega la sovrastima.

**3. Disallineamento fra le feature di training e quelle usate qui.** Diventata l'ipotesi principale grazie a un argomento aritmetico.

## L'argomento che ha ristretto il campo

Il tasso di gol sui tiri non-rigore del training set è **9,59%** (8.218 tiri su 326 match).

Una regressione logistica con intercetta, stimata per massima verosimiglianza, ha una proprietà: **la media delle probabilità predette sui dati di training coincide con il tasso base**. Il modello, sui suoi dati, deve produrre in media ~0,096.

Ne produceva 0,161. E i tiri in questione erano, secondo StatsBomb, chance *peggiori* della media (0,080 contro ~0,10 globale): il modello si muoveva nella direzione opposta alla realtà.

Nessuna spiegazione basata sul modello regge questi numeri. Restava l'input.

## Il test

Stessi tiri, feature calcolate nei due modi, stesso modello. Riferimento: `inference/score_match.py`, la cui funzione è **identica** a quella usata nel notebook xG per addestrare il modello.

Script: [`tests/compare_features.py`](../../tests/compare_features.py) — 554 tiri da 25 match.

```
feature                         % righe diverse    media ref     media xA
--------------------------------------------------------------------------
distance                                   0.0%       18.089       18.089
angle                                     99.1%        0.431        0.250  <<<
nearest_defender_dist                      2.5%        2.872        2.908
n_players_in_cone_to_goal                 11.9%        1.392        1.540
visible_area_size                          0.0%     1837.613     1837.613
first_time                               100.0%            -            -  <<<
one_on_one                               100.0%            -            -  <<<
under_pressure                           100.0%            -            -  <<<
body_part / shot_type / …                  0.0%            -            -
```

## La causa

### Formula dell'angolo invertita

```python
# notebook xA (sbagliato)
angle = atan2(GOAL_WIDTH * abs(y - GOAL_Y), den)

# formula standard, usata nel training
angle = atan2(GOAL_WIDTH * abs(x - GOAL_X), den)
```

Al numeratore va la distanza dalla **linea di porta**, non lo scostamento **laterale**. Scambiandoli, la feature misura il contrario di quello che dovrebbe:

| Posizione | training | notebook xA |
|---|---|---|
| dischetto, centrale | 1.095 | **0.000** |
| 10m centrale | 0.702 | **0.000** |
| 10m molto defilato | 0.231 | 0.339 |
| a lato della porta | 0.108 | 0.578 |

Un tiro dal dischetto in posizione frontale riceveva l'angolo peggiore possibile; uno da posizione defilata, il migliore.

### Convenzione dei flag binari

`first_time`, `one_on_one`, `under_pressure`: il training passa `True`/`None` e lascia che l'imputer riempia i mancanti con la mediana; il notebook xA passa `0.0`/`1.0`. Stesse feature, semantica diversa.

## L'esito

| Feature usate | media xG | corr. Pearson con `statsbomb_xg` |
|---|---|---|
| `score_match.py` (identiche al training) | **0.0981** | **0.8508** |
| notebook xA | 0.1796 | 0.7122 |
| `statsbomb_xg` (benchmark) | 0.1006 | — |
| tasso gol reale | 0.0903 | — |

Il modello xG è centrato sul benchmark e sul tasso di gol reale, e correla 0.85 con StatsBomb. La "sovrastima del doppio" non è mai esistita nel modello.

*(Nota: i 554 tiri includono match di training — non è una valutazione su holdout, che resta da fare.)*

## Cosa se ne impara

**Il bug silenzioso è peggio di quello rumoroso.** Nello stesso notebook c'erano altri due disallineamenti — `visible_area` e i dtype booleani — entrambi risolti in minuti perché sollevavano eccezioni. Quello dell'angolo non ha prodotto alcun errore: solo numeri plausibili e sbagliati, sopravvissuti fino a diventare una diagnosi errata sul modello.

**Il sanity check ha fatto il suo lavoro.** La correlazione con `statsbomb_xg` era già nel notebook, scritta prima che servisse. Senza quel confronto, l'xA sarebbe stato costruito sopra un xG raddoppiato senza che nulla lo segnalasse.

**La duplicazione del codice è il difetto strutturale.** Tre copie della stessa feature engineering, tre divergenze. La correzione puntuale non basta: serve un modulo unico — [`plans/00`](../../plans/00-xg-calibrazione.md), step 0.

**Un argomento aritmetico batte una ricerca a tentoni.** La proprietà della regressione logistica — media predetta = tasso base sul training — ha escluso in un colpo tutte le ipotesi sul modello e ha indirizzato sull'input. Trenta minuti invece di una giornata.
