# `{id}_match.json` — metadati della partita

**Cos'è:** un JSON unico per partita, ~33 KB. Contiene tutto ciò che serve per
interpretare gli altri tre file: chi è chi, quanto è grande il campo, dove
iniziano e finiscono i tempi.

**È il file da leggere per primo.** Senza, i `player_id` del tracking sono numeri
senza squadra e senza ruolo.

## Struttura

| Chiave | Tipo | Contenuto |
|---|---|---|
| `id` | int | id della partita |
| `date_time` | str | data e ora di inizio |
| `home_team`, `away_team` | dict | `id`, `name`, `short_name`, `acronym` |
| `home_team_score`, `away_team_score` | int | risultato finale |
| `home_team_side` | list[str] | **verso di attacco della squadra di casa, per tempo** |
| `pitch_length`, `pitch_width` | int | dimensioni in metri |
| `match_periods` | list[dict] | un elemento per tempo: `start_frame`, `end_frame`, `duration_frames`, `duration_minutes` |
| `players` | list[dict] | le due rose, titolari e panchina |
| `home_team_kit`, `away_team_kit` | dict | colori maglia (`jersey_color`, `number_color`) |
| `home_team_playing_time` | dict | `minutes_tip` / `minutes_otip` (in / out of possession) |
| `stadium` | dict | `name`, `city`, `capacity` |
| `competition_edition`, `competition_round` | dict | competizione, stagione, giornata |
| `ball` | dict | `trackable_object` (l'id della palla nel tracking) |
| `referees` | list | **vuota in questi dati** |
| `status` | str | stato della partita |

## Il campo di un giocatore

```json
{
  "id": 51009,
  "first_name": "...", "last_name": "...", "short_name": "...",
  "number": 10,
  "team_id": 4177,
  "trackable_object": 55,
  "player_role": {"id": 15, "acronym": "CF", "name": "Center Forward",
                  "position_group": "Center Forward"},
  "start_time": "00:00:00", "end_time": "01:25:21",
  "playing_time": {"total": {"minutes_played": 86.65, "minutes_tip": 29.55,
                             "minutes_otip": 18.76, "start_frame": 10,
                             "end_frame": 52009},
                   "by_period": [...]},
  "goal": 0, "own_goal": 0, "yellow_card": 0, "red_card": 0, "injured": false,
  "birthday": "...", "gender": "..."
}
```

36 giocatori per partita (titolari + panchina). Chi non è sceso in campo non ha
`player_role`.

I due mapping che servono ovunque:

```python
squadra_di = {p["id"]: p["team_id"] for p in meta["players"]}
ruolo_di   = {p["id"]: p["player_role"]["acronym"]
              for p in meta["players"] if p.get("player_role")}
```

## Due cose da non sbagliare

**`home_team_side` cambia fra i tempi.** Nell'esempio `1886347` vale
`["right_to_left", "left_to_right"]`: la squadra di casa attacca verso sinistra nel
primo tempo e verso destra nel secondo. Qualunque metrica spaziale aggregata sui
due tempi senza normalizzare il verso è sbagliata.

Kloppy risolve il problema da solo con
`dataset.transform(to_orientation="STATIC_HOME_AWAY")`. Se leggi il JSONL grezzo
devi farlo a mano.

**Le dimensioni del campo variano fra partite.** Fra le 20 disponibili: 105×68
(10 partite), 106×68 (6), 104×68 (4). Vanno lette da qui, non assunte — e ogni
normalizzazione spaziale va fatta per partita.

## Durate nominali

`match_periods` dà l'intervallo di frame di ciascun tempo. Attenzione: è la durata
*nominale*, non il gioco osservabile. Nell'esempio, 59.040 frame dichiarati contro
43.458 frame che contengono effettivamente giocatori — vedi
[`tracking-extrapolated.md`](tracking-extrapolated.md).
