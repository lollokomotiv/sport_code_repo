# `bodypose/` — 29 giunti 3D per giocatore

**Cos'è:** il tracking 2D arricchito con **29 giunti per giocatore**, coordinate
3D sul campo, **25 fps**, ognuno con un errore dichiarato in centimetri
(`p90_mae_cm`: il raggio entro cui cade il 90% delle stime).

È il dato più vicino all'**intenzione** che SkillCorner fornisca: il tracking
dice dove sta un giocatore, il pose da che parte è girato.

Numeri misurati in
[`explorations/02-body-pose.ipynb`](../../explorations/02-body-pose.ipynb) sulla
partita `1925299`.

## Dove sta

**Due partite su venti**, e i file interi non sono nel repo dei dati: 3,3 GB di
JSON ciascuno, ~600 MB compressi, su
[Hugging Face](https://huggingface.co/datasets/SkillCorner/opendata-bodypose).

| Match | Partita | Data |
|---|---|---|
| 1925299 | Brisbane Roar v Perth Glory | 2024-12-21 |
| 1996435 | Sydney FC v Adelaide United | 2025-02-01 |

Nel repo c'è solo `sample_1925299_phase406.jsonl.gz`: **una fase di 12,3
secondi**, 309 frame. Serve per sviluppare senza scaricare, ma non è una partita
in miniatura — un tempo solo, nessuna sostituzione, nessun tratto senza pose.
Codice provato solo lì si rompe sulla partita vera.

```python
from lib import pose

pose.scarica(1925299)                      # una volta, ~600 MB, verifica lo sha256
for fr in pose.iter_match(1925299): ...    # streaming, memoria costante
```

## Struttura

```json
{
  "frame": 124487, "timestamp": "00:49:47.48", "period": 2,
  "ball_data": {"x": 0.35, "y": -22.22, "z": 7.074, "is_detected": true},
  "possession": {"player_id": null, "group": "away team"},
  "image_corners_projection": {"...": "..."},
  "player_data": [
    {"player_id": 809166, "x": -17.36, "y": -22.06, "is_detected": true,
     "joints": {"lAnkle": {"xyz": [-25.042, 0.86, 0.204], "p90_mae_cm": 12.39}, "...": "..."}}
  ]
}
```

`joints` è **`null`** quando la posa non è stata risolta, anche se `x`/`y` ci
sono.

I 29 giunti: `nose`, `neck`, `lEye`, `rEye`, `lEar`, `rEar`, `lShoulder`,
`rShoulder`, `lElbow`, `rElbow`, `lWrist`, `rWrist`, `lThumb`, `rThumb`,
`lPinky`, `rPinky`, `midHip`, `lHip`, `rHip`, `lKnee`, `rKnee`, `lAnkle`,
`rAnkle`, `lHeel`, `rHeel`, `lBigToe`, `rBigToe`, `lSmallToe`, `rSmallToe`.

## Le tre cose che cambiano rispetto al tracking

**1. 25 fps, non 10.** `pose_frame = 2.5 × tracking_frame`, verificato esatto.
Solo un frame di pose su 5 cade su un frame di tracking senza interpolazione.
`pose.a_frame_tracking()` fa la conversione arrotondando per eccesso a metà.

**2. Denominatori diversi.** Il file pose elenca **32 giocatori per frame** — la
rosa intera, panchina compresa — e **non ha frame vuoti**; il tracking ne ha 22
o zero, con il 22% di frame vuoti.

| | tracking | pose |
|---|---|---|
| frame | 61.301 | 153.252 |
| giocatori per frame | 22 o 0 | 32 |
| frame vuoti | 22% | nessuno |
| player-frame | 1.054.724 | 4.902.560 |

Conseguenza pratica: la copertura va calcolata sui **giocatori effettivamente in
campo**, ricavati dai minuti giocati in `match.json`. Sul denominatore ingenuo
esce **31,4%**, che è il "circa un terzo" del README ufficiale; su quello giusto
è **45,7%**.

**3. `z` è relativa al centroide del giocatore**, non alle coordinate del campo.
Non è un'altezza dal suolo e può essere negativa. Per direzioni sul piano si
usano solo `x` e `y`.

Nota anche: pose e tracking sono generati separatamente, quindi piccoli
disallineamenti sono attesi. In pratica coincidono quasi perfettamente —
`is_detected` e presenza dei giunti concordano nel 99,8% dei casi.

## Copertura: eredita il limite del rilevamento

**Il pose esiste dove e solo dove il giocatore è stato rilevato.** Non aggiunge
un limite nuovo, amplifica quello che già conosciamo.

Sui giocatori in campo, partita intera: **45,7%**.

| distanza dalla palla | con pose |
|---|---|
| entro 5 m | 77% |
| 5-10 m | 70% |
| 10-20 m | 62% |
| 20-30 m | 45% |
| 30-40 m | 28% |
| oltre 40 m | **10%** |

Per ruolo: portiere **15%** (lo stesso valore del tracking), centrali difensivi
35-38%, terzini ~44%, attaccanti ed esterni 50-54%, centrocampisti 55-61%.

## Non tutti i giunti sono affidabili

Errore medio dichiarato, dal più preciso al peggiore:

| giunti | errore |
|---|---|
| collo, spalle | **6 cm** |
| naso, occhi, orecchie, anche | 8-9 cm |
| gomiti, bacino | 10 cm |
| ginocchia | 12 cm |
| caviglie, polsi | 15-16 cm |
| talloni, dita dei piedi, pollici | 18-19 cm |
| mignoli | **21 cm** |

**I giunti che servirebbero per capire cosa fa un giocatore con i piedi sono i
meno affidabili**, quelli che sostengono una misura di orientamento i migliori.
Non è un caso che l'esempio ufficiale di SkillCorner
(`src/features/pose_orientation.py`) usi le spalle.

## Orientamento del busto

L'angolo è l'asse spalla sinistra → destra **ruotato di +90°**. `lib/pose.py` lo
calcola con i cinque filtri documentati da SkillCorner: posa presente, entrambe
le spalle, spalle non coincidenti, larghezza fra 0,15 e 0,6 m, errore sotto i
15 cm. Ognuno dei cinque, se ignorato, produce un numero plausibile e sbagliato.

> **Il segno è facile da sbagliare.** Invertirlo dà un angolo ruotato di 180°,
> che sembra normale finché non lo si confronta con qualcosa di noto. La prima
> versione del nostro calcolo dava uno scarto mediano di 160° rispetto alla
> direzione di corsa, e il 96% dei giocatori "in corsa all'indietro" a 5 m/s:
> implausibile, quindi bug. Verificare analiticamente su casi noti.

**La validazione da fare sempre:** a velocità alta il busto deve allinearsi alla
corsa. Con il segno corretto lo scarto mediano scende da **21° a 13°** salendo di
velocità, e la quota di movimento "all'indietro" passa dal 16% allo 0%.

Quel residuo a bassa velocità non è rumore: è chi si muove senza guardare dove
va — difensori che accompagnano indietro tenendo d'occhio la palla.

## Cosa regge e cosa no

**Regge:** orientamento del busto per giocatori **vicini alla palla**, in gioco
attivo — duelli, ingaggi, pressione sul portatore.

**Non regge:** qualunque cosa richieda i piedi, l'altezza dal suolo, i giocatori
lontani dall'azione, o una popolazione statistica. Con due partite non si
descrive un campionato: si dimostra un metodo, ed è una cosa diversa che va
dichiarata.

## Riferimenti

- `data/bodypose/README.md` nel repo dei dati
- `src/data/pose_loading.py` e `src/features/pose_orientation.py` (MIT)
- Tutorial: `notebooks/tutorials/05_Body_Pose/`
