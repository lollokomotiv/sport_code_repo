# `{id}_tracking_extrapolated.jsonl` — le posizioni

**Cos'è:** una riga JSON per frame, **10 fps**. È il file grosso (~90 MB per
partita) e l'unico su **Git LFS**: appena clonato è un puntatore da 133 byte,
serve `git lfs pull`.

Il nome dice già la cosa importante: *extrapolated*. Le posizioni fuori
inquadratura sono stimate, non osservate.

## Struttura di un frame

```json
{
  "frame": 12000,
  "timestamp": "00:19:59.00",
  "period": 1,
  "ball_data": {"x": 39.28, "y": 20.69, "z": 0.2, "is_detected": true},
  "possession": {"player_id": null, "group": "home team"},
  "image_corners_projection": {"x_top_left": 34.46, "y_top_left": 39.0, ...},
  "player_data": [{"x": -41.18, "y": 3.23, "player_id": 51009, "is_detected": false}, ...]
}
```

| Campo | Contenuto |
|---|---|
| `frame` | numero di frame, chiave di join con tutti gli altri file |
| `timestamp` | tempo di gioco, precisione 1/10 s |
| `period` | 1 o 2 |
| `ball_data` | `x`, `y`, `z` (altezza), `is_detected` — **anche la palla può essere estrapolata** |
| `possession` | `group` (`home team` / `away team`) e `player_id` quando identificato, spesso `null` |
| `image_corners_projection` | i 4 angoli della porzione di campo inquadrata, come 8 chiavi `x_*`/`y_*` |
| `player_data` | lista di giocatori: `x`, `y`, `player_id`, `is_detected` |

## Coordinate

Metri, **origine al centro del campo**. L'asse `x` è il lato lungo, l'asse `y` il
lato corto. Per un campo 104×68: `x ∈ [-52, 52]`, `y ∈ [-34, 34]`.

`mplsoccer` le gestisce nativamente con `pitch_type="skillcorner"`, senza
conversioni.

## Copertura: il tracking è tutto-o-niente

Misurato su `1886347` (una partita intera):

| | frame | quota |
|---|---|---|
| totali nel file | 59.061 | 100% |
| con **22 giocatori** | 43.458 | 74% |
| con **0 giocatori** | 15.603 | 26% |
| senza palla tracciata | 15.617 | 26% |

Non esistono frame parziali: o ci sono tutti e 22, o nessuno. Il 26% vuoto è gioco
fermo, replay, inquadrature che non mostrano il campo.

Quindi il gioco osservabile è **~72 minuti su 98 di durata nominale**. Le durate in
`match.json` sono nominali e non vanno usate come denominatore.

I numeri di frame sono contigui (nessun salto), quindi i frame vuoti sono presenti
come righe, non omessi.

## `is_detected`: il limite principale di questi dati

Distingue le posizioni **osservate** da quelle **estrapolate**. Sulla stessa
partita, considerando solo i frame popolati:

| | osservato |
|---|---|
| complessivo | **59%** |
| entro 5 m dalla palla | 87% |
| 10-20 m | 78% |
| 30-40 m | 40% |
| oltre 40 m | 18% |

Per ruolo: esterni e mezzali 70-73%, terzini e punta ~62%, centrali difensivi
~49%, **portiere 15%**.

La telecamera segue la palla: il dato è solido intorno all'azione e
progressivamente ricostruito allontanandosi. Analisi e metodo in
[`explorations/00-quanto-e-osservato.ipynb`](../../explorations/00-quanto-e-osservato.ipynb).

> **Kloppy non espone `is_detected`.** `frame.players_data[p].other_data` è vuoto e
> `to_df()` restituisce solo `x`, `y`, distanza e velocità. Per misurare
> l'estrapolazione serve leggere il JSONL grezzo — è il motivo per cui esiste
> `lib/data.py`.

## Come leggerlo

**Nel workbench**, dal clone locale (veloce):

```python
from lib import data

for frame in data.iter_tracking(1886347, limit=1000):
    ...

df = data.tracking_long(1886347, limit=1000)   # una riga per (frame, giocatore)
```

**Nella submission**, obbligatoriamente da remoto:

```python
from kloppy import skillcorner

ds = skillcorner.load_open_data(match_id=1886347, coordinates="skillcorner",
                                sample_rate=1/2, limit=1000)
df = ds.transform(to_orientation="STATIC_HOME_AWAY").to_df(engine="polars")
```

A 10 fps una partita intera è ~956.000 righe in formato lungo: durante lo sviluppo
conviene sempre un `limit` o un `sample_rate`.

## Le traiettorie arrivano già lisciate

Misurato sulla partita 1886347. Derivando le velocità per differenze semplici,
senza nessuna lisciatura:

| | valore |
|---|---|
| velocità sopra i 10 m/s | **0,0%** |
| accelerazione al 99° percentile | **4,5 m/s²** |
| accelerazioni sopra gli 8 m/s² | 0,05% |
| variazione di velocità fra frame consecutivi, 99° percentile | 0,46 m/s |

Sono valori fisiologicamente plausibili, e una lisciatura Savitzky–Golay da 5 a
15 frame li sposta di pochi centesimi. SkillCorner consegna quindi traiettorie
già filtrate: **le velocità si possono ricavare con una derivata centrale
semplice**, senza lisciatura aggiuntiva. Aggiungerla introdurrebbe solo ritardo.

Riferimento esterno per le velocità di punta: `psv99` negli aggregati fisici ha
mediana 28,3 km/h (7,85 m/s) e massimo stagionale 32,3 km/h (8,97 m/s).

**Le posizioni estrapolate sono lisce quanto quelle osservate.** L'estrapolazione
produce traiettorie plausibili, quindi non la si riconosce dalla cinematica: per
distinguerla serve `is_detected`, sempre.

## `image_corners_projection`

Non ancora sfruttato. Dà la porzione di campo inquadrata frame per frame, quindi
permette di verificare **direttamente** perché un giocatore è estrapolato, invece
di dedurlo dalla distanza dalla palla.
