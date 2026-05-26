# TotoSport — Regolamento Ufficiale

> **Fonti:** Foglio Excel `regolamento+ tabellone 2023-24.xls` + messaggi WhatsApp del 16/05/2026.
> In caso di discrepanza tra le due fonti, prevale il foglio Excel.

---

## Indice

1. [Struttura generale](#1-struttura-generale)
2. [Tabellone — Previsioni di Inizio Stagione](#2-tabellone--previsioni-di-inizio-stagione)
3. [Previsioni Settimanali (Giornata)](#3-previsioni-settimanali-giornata)
4. [Punteggio per Partita](#4-punteggio-per-partita)
5. [Bonus Settimanale (Weekend)](#5-bonus-settimanale-weekend)
6. [Bonus Finali (Fine Stagione)](#6-bonus-finali-fine-stagione)
7. [Modifiche al Tabellone](#7-modifiche-al-tabellone)
8. [Riepilogo Punti — Tabella Rapida](#8-riepilogo-punti--tabella-rapida)
9. [Note per il Coding Agent](#9-note-per-il-coding-agent)

---

## 1. Struttura Generale

Il TotoSport è un gioco di pronostici calcistici con **due livelli di partecipazione**:

| Livello | Quando | Descrizione |
|---|---|---|
| **Tabellone** | Inizio stagione | Previsioni annuali su Serie A, Serie B, Coppe |
| **Giornata** | Ogni weekend | Pronostici partita per partita + previsione gol totali |

I punti delle due componenti si sommano in un'unica **classifica generale**.

---

## 2. Tabellone — Previsioni di Inizio Stagione

Ogni giocatore compila il tabellone **una volta a inizio stagione**. I punti vengono assegnati al termine della stagione (o delle singole competizioni).

### 2.1 Serie A

| Domanda | Punti base | Note |
|---|---|---|
| Chi vincerà lo Scudetto? | **25 pt** | — |
| ↳ Con quanti punti finirà? | **+8 pt bonus** | Solo se hai indovinato il vincitore |
| Squadra retrocessa in Serie B — 1ª | **25 pt** | — |
| Squadra retrocessa in Serie B — 2ª | **25 pt** | — |
| Squadra retrocessa in Serie B — 3ª | **25 pt** | — |
| Chi vincerà la Classifica Marcatori Serie A? | **15 pt** | In caso di parità tra più vincitori reali: punti divisi per il numero di vincitori |
| ↳ Con quante reti? | **+8 pt bonus** | Solo se hai indovinato il capocannoniere |

### 2.2 Serie B

| Domanda | Punti base | Note |
|---|---|---|
| Squadra promossa direttamente in A — 1ª | **30 pt** | Se promozione avviene tramite playoff: **50% (15 pt)** |
| Squadra promossa direttamente in A — 2ª | **30 pt** | Se promozione avviene tramite playoff: **50% (15 pt)** |
| ↳ Punti della 1ª classificata? | **+8 pt bonus** | Solo se hai indovinato la 1ª classificata |
| Squadre qualificate ai playoff (6 squadre) | **20 pt** ciascuna | — |
| Squadra promossa in A tramite playoff | **30 pt** | Se promossa direttamente: **50% (15 pt)** |
| ↳ *Alternativa: se i playoff NON si disputano* | **40 pt** | Questa voce sostituisce le 6 squadre ai playoff + la promossa con playoff |
| Squadra retrocessa in C direttamente — 1ª | **30 pt** | — |
| Squadra retrocessa in C direttamente — 2ª | **30 pt** | — |
| Squadra retrocessa in C direttamente — 3ª | **30 pt** | — |
| Squadre qualificate ai playout (2 squadre) | **20 pt** ciascuna | — |
| Squadra retrocessa in C tramite playout | **30 pt** | — |
| ↳ *Alternativa: se i playout NON si disputano* | **35 pt** | Sostituisce le 2 squadre ai playout + la retrocessa con playout |
| Chi vincerà la Classifica Marcatori Serie B? | **15 pt** | In caso di parità: punti divisi |
| ↳ Con quante reti? | **+8 pt bonus** | Solo se hai indovinato il capocannoniere |

### 2.3 Coppe

| Domanda | Punti base | Note |
|---|---|---|
| Chi vincerà la Coppa Italia? | **25 pt** | — |
| Chi vincerà la Champions League? | **25 pt** | — |
| Chi vincerà l'Europa League? | **25 pt** | — |
| Chi vincerà la Conference League? | **25 pt** | — |

---

## 3. Previsioni Settimanali (Giornata)

Per ogni giornata il giocatore deve inserire:

1. **Il segno (1/X/2)** per ogni partita in programma.
2. **Il risultato esatto** per ogni partita in programma (es. 2-1).
3. **Il totale gol della giornata** (somma di tutti i gol di tutte le partite).

Tutte le previsioni devono essere inserite **prima del calcio d'inizio della prima partita della giornata** (deadline).

---

## 4. Punteggio per Partita

### 4.1 Segno corretto

| Esito | Punti |
|---|---|
| Segno giusto (1, X, o 2) | **1 pt** |
| Segno sbagliato | 0 pt |

### 4.2 Risultato esatto corretto

Se il giocatore indovina il risultato esatto, riceve i punti del segno **più** un bonus che dipende dal tipo di risultato:

| Tipo di risultato esatto | Calcolo | **Totale punti** |
|---|---|---|
| Vittoria casalinga (es. 2-1) | 1 (segno) + 5 (bonus esatto) | **6 pt** |
| Pareggio (es. 1-1) | 1 (segno) + 7 (bonus esatto) | **8 pt** |
| Vittoria esterna (es. 1-2) | 1 (segno) + 9 (bonus esatto) | **10 pt** |

### 4.3 Bonus partite con 5+ gol

Se una partita termina con **5 o più gol complessivi**, chi ha indovinato il **risultato esatto** riceve un ulteriore bonus di **+2 pt** (in aggiunta ai punti dell'esatto).

Il bonus non viene assegnato a chi ha indovinato solo il segno.

### 4.4 Totale gol della giornata

| Previsione | Punti |
|---|---|
| Totale gol della giornata esatto | **3 pt** |
| Totale gol sbagliato | 0 pt |

---

## 5. Bonus Settimanale (Weekend)

Al termine di ogni giornata, i **3 giocatori con il punteggio più alto della giornata** ricevono un bonus:

| Posizione nella giornata | Bonus |
|---|---|
| 1° classificato | **+6 pt** |
| 2° classificato | **+4 pt** |
| 3° classificato | **+2 pt** |

> **Parità:** in caso di pari punteggio, il bonus viene assegnato integralmente a **ciascuno** dei giocatori a pari merito (non diviso).

---

## 6. Bonus Finali (Fine Stagione)

Al termine della stagione vengono assegnati tre bonus separati, uno per ciascuna classifica:

| Classifica | Bonus |
|---|---|
| Miglior classifica finale — Solo segni (1/X/2) | **+10 pt** |
| Miglior classifica finale — Pieni + gol (risultati esatti + totale gol giornate) | **+10 pt** |
| Miglior classifica finale — Tabellone (previsioni annuali) | **+10 pt** |

> **Parità:** in caso di pari punteggio, il bonus viene assegnato integralmente a ciascun giocatore a pari merito.

---

## 7. Modifiche al Tabellone

### 7.1 Quando è possibile modificare

La finestra di modifica si apre alla **chiusura del mercato invernale** e resta aperta fino a una data comunicata dall'admin (di norma il giorno prima della ripresa del campionato). Dopo quella data non sono più ammesse modifiche.

### 7.2 Costo di ogni modifica

Ogni voce del tabellone che viene cambiata comporta **due penalità separate e cumulative**:

| Penalità | Importo | Quando |
|---|---|---|
| **Penalità immediata in classifica** | **−5 pt** per ogni voce modificata | Sottratti nel momento in cui si salva la modifica |
| **Cap sui punti ottenibili** | **Massimo 50% dei punti normali** (arrotondati per difetto) | Applicato al momento dello scoring finale |

> **Esempio:** modifichi il pronostico sullo Scudetto (vale 25 pt).
> Paghi subito **−5 pt** in classifica. A fine stagione, anche se il nuovo pronostico è corretto, guadagni al massimo **12 pt** invece di 25.

### 7.3 Costo per voce: tabella completa

Ogni riga rappresenta **una singola modifica** (= una voce del tabellone). Le domande "bonus" (punti/gol del vincitore) sono **incluse nella stessa modifica** del pronostico principale e non costano separatamente, a meno che non si voglia cambiare solo il bonus senza toccare la previsione principale.

#### Serie A

| Voce modificata | Punti normali | Max dopo modifica | Penalità immediata |
|---|---|---|---|
| Scudetto (squadra) | 25 pt | **12 pt** | −5 pt |
| ↳ Punti scudetto (bonus) | 8 pt | **4 pt** | −5 pt |
| Retrocessa in B — 1ª | 25 pt | **12 pt** | −5 pt |
| Retrocessa in B — 2ª | 25 pt | **12 pt** | −5 pt |
| Retrocessa in B — 3ª | 25 pt | **12 pt** | −5 pt |
| Capocannoniere A (giocatore) | 15 pt | **7 pt** | −5 pt |
| ↳ Gol capocannoniere A (bonus) | 8 pt | **4 pt** | −5 pt |

#### Serie B

| Voce modificata | Punti normali | Max dopo modifica | Penalità immediata |
|---|---|---|---|
| Promossa diretta 1ª | 30 pt (15 se da playoff) | **15 pt** (7 se da playoff) | −5 pt |
| Promossa diretta 2ª | 30 pt (15 se da playoff) | **15 pt** (7 se da playoff) | −5 pt |
| ↳ Punti 1ª classificata (bonus) | 8 pt | **4 pt** | −5 pt |
| Ai playoff — squadra 1 | 20 pt | **10 pt** | −5 pt |
| Ai playoff — squadra 2 | 20 pt | **10 pt** | −5 pt |
| Ai playoff — squadra 3 | 20 pt | **10 pt** | −5 pt |
| Ai playoff — squadra 4 | 20 pt | **10 pt** | −5 pt |
| Ai playoff — squadra 5 | 20 pt | **10 pt** | −5 pt |
| Ai playoff — squadra 6 | 20 pt | **10 pt** | −5 pt |
| Promossa con playoff | 30 pt (15 se diretta) | **15 pt** (7 se diretta) | −5 pt |
| *Alt: no playoff* | 40 pt | **20 pt** | −5 pt |
| Retrocessa C diretta — 1ª | 30 pt | **15 pt** | −5 pt |
| Retrocessa C diretta — 2ª | 30 pt | **15 pt** | −5 pt |
| Retrocessa C diretta — 3ª | 30 pt | **15 pt** | −5 pt |
| Ai playout — squadra 1 | 20 pt | **10 pt** | −5 pt |
| Ai playout — squadra 2 | 20 pt | **10 pt** | −5 pt |
| Retrocessa C con playout | 30 pt | **15 pt** | −5 pt |
| *Alt: no playout* | 35 pt | **17 pt** | −5 pt |
| Capocannoniere B (giocatore) | 15 pt | **7 pt** | −5 pt |
| ↳ Gol capocannoniere B (bonus) | 8 pt | **4 pt** | −5 pt |

#### Coppe

| Voce modificata | Punti normali | Max dopo modifica | Penalità immediata |
|---|---|---|---|
| Coppa Italia | 25 pt | **12 pt** | −5 pt |
| Champions League | 25 pt | **12 pt** | −5 pt |
| Europa League | 25 pt | **12 pt** | −5 pt |
| Conference League | 25 pt | **12 pt** | −5 pt |

### 7.4 Logica di arrotondamento

I punti a 50% vengono **arrotondati per difetto** all'intero più vicino (es. 25 × 50% = 12,5 → **12 pt**). Questo evita frazioni di punto nel sistema di scoring.

### 7.5 Esempio completo

Un giocatore modifica 3 voci dopo il mercato di gennaio:

1. Squadra vincitrice Scudetto → −5 pt immediati, max 12 pt a fine anno
2. Capocannoniere Serie A → −5 pt immediati, max 7 pt a fine anno
3. Vincitrice Champions → −5 pt immediati, max 12 pt a fine anno

**Penalità totale immediata: −15 pt** dalla classifica nel momento delle modifiche.

---

## 8. Riepilogo Punti — Tabella Rapida

```
TABELLONE (inizio stagione)
  Scudetto                      25 pt  (+8 bonus punti)
  Retrocesse Serie A (x3)       25 pt ciascuna
  Capocannoniere Serie A        15 pt  (+8 bonus gol)
  Promosse B→A dirette (x2)     30 pt ciascuna (15 pt se da playoff)
  Bonus punti 1ª classificata   +8 pt
  Ai playoff Serie B (x6)       20 pt ciascuna
  Promossa con playoff          30 pt  (15 pt se diretta)
  [Alt: no playoff]             40 pt
  Retrocesse B→C (x3)           30 pt ciascuna
  Ai playout (x2)               20 pt ciascuna
  Retrocessa con playout        30 pt
  [Alt: no playout]             35 pt
  Capocannoniere Serie B        15 pt  (+8 bonus gol)
  Coppa Italia                  25 pt
  Champions League              25 pt
  Europa League                 25 pt
  Conference League             25 pt

GIORNATA (ogni weekend)
  Segno giusto (1/X/2)          1 pt
  Risultato esatto — casa       6 pt totali  (1+5)
  Risultato esatto — pareggio   8 pt totali  (1+7)
  Risultato esatto — trasferta  10 pt totali (1+9)
  Bonus partita 5+ gol          +2 pt (solo se esatto indovinato)
  Totale gol giornata           3 pt

BONUS WEEKEND
  1° della giornata             +6 pt
  2° della giornata             +4 pt
  3° della giornata             +2 pt

BONUS FINE STAGIONE
  Best classifica segni         +10 pt
  Best classifica pieni+gol     +10 pt
  Best classifica tabellone     +10 pt
```

---

## 9. Note per il Coding Agent

Questa sezione riassume le **entità dati** e le **regole di business** necessarie per implementare l'applicazione.

### Entità principali

```
Season
  - id, name (es. "2024-2025"), status (active | closed)

Player
  - id, name, email

TablePrediction (Tabellone)
  - player_id, season_id
  - scudetto_team, scudetto_points_guess
  - relegated_a_1/2/3
  - top_scorer_a_name, top_scorer_a_goals_guess
  - promoted_b_direct_1/2, promoted_b_direct_1_points_guess
  - playoff_b_teams[6]
  - promoted_b_playoff
  - relegated_b_c_direct_1/2/3
  - playout_b_teams[2]
  - relegated_b_c_playout
  - top_scorer_b_name, top_scorer_b_goals_guess
  - coppa_italia_winner
  - champions_winner
  - europa_winner
  - conference_winner
  - is_modified_after_mercato (bool) — per applicare le penalità §7

Round (Giornata)
  - id, season_id, round_number, deadline (datetime), status (open | closed | scored)

Match
  - id, round_id, home_team, away_team
  - home_goals (null until played), away_goals (null until played)

RoundPrediction (Previsioni per Giornata)
  - player_id, round_id
  - total_goals_guess (intero)

MatchPrediction (Previsione per Singola Partita)
  - player_id, match_id
  - home_goals_guess, away_goals_guess
  - Derivato: sign_guess = "1"|"X"|"2"

Score (calcolato post-giornata)
  - player_id, round_id
  - match_points (somma punti partite)
  - total_goals_points (0 o 3)
  - weekend_bonus (0, 2, 4, o 6)
  - total_round_points
```

### Regole di scoring (pseudocodice)

```python
# Per ogni partita, dopo l'inserimento del risultato reale:
def score_match_prediction(pred, result):
    home_g, away_g = result.home_goals, result.away_goals
    pred_h, pred_a = pred.home_goals_guess, pred.away_goals_guess

    sign_correct = sign(pred_h, pred_a) == sign(home_g, away_g)
    exact_correct = (pred_h == home_g) and (pred_a == away_g)
    total_goals = home_g + away_g

    points = 0
    if sign_correct:
        points += 1  # punto base per il segno
    if exact_correct:
        outcome = sign(home_g, away_g)  # "1", "X", or "2"
        bonus = {"1": 5, "X": 7, "2": 9}[outcome]
        points += bonus
        if total_goals >= 5:
            points += 2  # bonus partite con 5+ gol (solo se esatto indovinato)
    return points

# Totale gol della giornata:
def score_total_goals(pred, round_matches):
    actual_total = sum(m.home_goals + m.away_goals for m in round_matches)
    return 3 if pred.total_goals_guess == actual_total else 0

# Bonus weekend (calcolato dopo aver totalizzato i punti della giornata per tutti):
def assign_weekend_bonus(round_scores):
    sorted_scores = sorted(round_scores, key=lambda s: s.total, reverse=True)
    bonuses = {1: 6, 2: 4, 3: 2}
    # Gestione parità: assegna il bonus della posizione a ogni giocatore a pari merito
    # (non ridistribuisce i punti, li assegna integralmente a tutti i pari merito)
    ...
```

### Punti aperti

Nessun punto aperto — tutte le discrepanze tra fonti sono state risolte a favore del foglio Excel.
