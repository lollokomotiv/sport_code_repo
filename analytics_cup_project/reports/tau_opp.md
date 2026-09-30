# τ_opp sulle 20 partite

Generato da `scripts/report_tau.py` su `reports/tau_opp.csv` (prodotto da `scripts/calcola_tau.py`). Solo numeri: l'interpretazione è da fare.

Metodo: Narizuka et al., α = 1,0 s⁻¹, V_max = 10,0 m/s; velocità per derivata centrale sulle posizioni, senza lisciatura; x_b = posizione di tracking del portatore. Scheda: `notes/letteratura/pressione-tempo-arrivo.md`.

## Copertura

- partite: 20
- player_possession esclusi i portieri: 17674 (coppie match_id, event_id duplicate: 0)
- τ_opp calcolato all'inizio: 17566 (99.4%); al rilascio: 17663 (99.9%)
- direct play (frame_start == frame_end): 4432 (25.1%)
- avversari con velocità definita, start: mediana 11, possessi con meno di 11: 0.0%
- avversari con velocità definita, end: mediana 11, possessi con meno di 11: 0.0%

|  | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| tau_opp_start | 17566.000 | 1.147 | 0.540 | 0.054 | 0.743 | 1.032 | 1.460 | 3.936 |
| tau_opp_end | 17663.000 | 0.957 | 0.462 | 0.058 | 0.637 | 0.857 | 1.154 | 3.871 |

τ_opp(t_get) − τ_opp(t_rel), possessi non direct play (N = 13145): media 0.255 s, deviazione standard 0.378 s (paper: 0,276 s e 0,400 s).

## 1. Correlazione con time_to_impact di SkillCorner

Spearman fra τ_opp (s) e `time_to_impact_{start,end}_id` (1 = very_easy … 5 = very_hard), sugli stessi istanti.

| istante | N | ρ di Spearman | p |
|---|---|---|---|
| inizio possesso | 16045 | -0.864 | 0 |
| rilascio | 16122 | -0.770 | 0 |

τ_opp per categoria di time_to_impact, inizio:

| time_to_impact_start_id | count | median | mean |
|---|---|---|---|
| 1.0 | 3898.000 | 1.889 | 1.900 |
| 2.0 | 4421.000 | 1.183 | 1.205 |
| 3.0 | 3798.000 | 0.845 | 0.872 |
| 4.0 | 2547.000 | 0.645 | 0.669 |
| 5.0 | 1381.000 | 0.564 | 0.605 |

τ_opp per categoria di time_to_impact, rilascio:

| time_to_impact_end_id | count | median | mean |
|---|---|---|---|
| 1.0 | 2453.000 | 1.740 | 1.775 |
| 2.0 | 2385.000 | 1.128 | 1.159 |
| 3.0 | 3927.000 | 0.862 | 0.880 |
| 4.0 | 4410.000 | 0.666 | 0.689 |
| 5.0 | 2947.000 | 0.571 | 0.606 |

## 2. Quota di possessi con l'avversario più vicino estrapolato

"Più vicino" = l'avversario che realizza τ_opp (argmin del tempo di arrivo). Per confronto anche l'avversario più vicino in distanza, e il portatore stesso.

| istante | N | avversario di τ_opp estrapolato | avversario più vicino in distanza estrapolato | portatore estrapolato | i due avversari non coincidono |
|---|---|---|---|---|---|
| inizio possesso | 17566 | 3.3% | 3.1% | 2.5% | 10.0% |
| rilascio | 17663 | 2.9% | 2.9% | 3.0% | 10.1% |

Possessi in cui l'avversario di τ_opp è estrapolato in almeno uno dei due istanti: 4.9% (N = 17566).

Per partita, all'inizio: min 1.1%, mediana 3.2%, max 5.9%.

## 3. Pressione all'inizio → avanzamento della palla (paper, Fig. 6)

Possessi non direct play, portieri esclusi: N = 13145. Progressione in metri (positiva = verso la porta avversaria).

Spearman τ_opp(t_get) vs progressione: ρ = 0.278, p = 1.9e-231.

| bin | N | media | sd | P(>0 m) | P(>3 m) | P(>6 m) |
|---|---|---|---|---|---|---|
| 0.00–0.25 | 56.000 | -0.760 | 8.368 | 0.429 | 0.250 | 0.179 |
| 0.25–0.50 | 645.000 | 0.262 | 6.990 | 0.394 | 0.202 | 0.110 |
| 0.50–0.75 | 1851.000 | 0.935 | 6.621 | 0.485 | 0.223 | 0.128 |
| 0.75–1.00 | 2651.000 | 1.133 | 6.860 | 0.496 | 0.206 | 0.128 |
| 1.00–1.25 | 2360.000 | 1.412 | 6.139 | 0.595 | 0.226 | 0.125 |
| 1.25–1.50 | 1798.000 | 2.029 | 5.798 | 0.679 | 0.244 | 0.143 |
| 1.50–1.75 | 1242.000 | 2.564 | 5.633 | 0.752 | 0.292 | 0.143 |
| 1.75–2.00 | 1020.000 | 2.861 | 5.256 | 0.800 | 0.304 | 0.160 |
| 2.00–2.25 | 819.000 | 3.242 | 4.620 | 0.869 | 0.368 | 0.173 |
| 2.25–2.50 | 443.000 | 4.833 | 6.201 | 0.932 | 0.481 | 0.273 |
| ≥ 2.50 | 260.000 | 5.543 | 6.493 | 0.915 | 0.554 | 0.381 |

### Zona < 35 m dalla porta (N = 1737; Spearman ρ = 0.253, p = 7.6e-27)

| bin | N | media | sd | P(>0 m) | P(>3 m) | P(>6 m) |
|---|---|---|---|---|---|---|
| 0.00–0.25 | 23.000 | -3.238 | 8.421 | 0.391 | 0.217 | 0.130 |
| 0.25–0.50 | 154.000 | -1.030 | 4.539 | 0.344 | 0.143 | 0.026 |
| 0.50–0.75 | 403.000 | 0.254 | 4.743 | 0.494 | 0.196 | 0.087 |
| 0.75–1.00 | 505.000 | 0.448 | 4.870 | 0.552 | 0.208 | 0.097 |
| 1.00–1.25 | 368.000 | 1.308 | 4.708 | 0.698 | 0.296 | 0.120 |
| 1.25–1.50 | 196.000 | 1.935 | 4.618 | 0.760 | 0.291 | 0.158 |
| 1.50–1.75 | 53.000 | 2.427 | 5.034 | 0.849 | 0.377 | 0.132 |
| 1.75–2.00 | 26.000 | 4.803 | 5.183 | 0.808 | 0.500 | 0.308 |
| 2.00–2.25 | 7.000 | 2.861 | 2.174 | 0.857 | 0.429 | 0.000 |
| 2.25–2.50 | 0.000 | nan | nan | nan | nan | nan |
| ≥ 2.50 | 2.000 | 0.682 | 0.621 | 1.000 | 0.000 | 0.000 |

### Zona 35–70 m dalla porta (N = 7228; Spearman ρ = 0.238, p = 1.6e-93)

| bin | N | media | sd | P(>0 m) | P(>3 m) | P(>6 m) |
|---|---|---|---|---|---|---|
| 0.00–0.25 | 19.000 | 1.016 | 6.560 | 0.421 | 0.316 | 0.211 |
| 0.25–0.50 | 330.000 | 1.102 | 6.665 | 0.455 | 0.255 | 0.155 |
| 0.50–0.75 | 976.000 | 1.387 | 6.907 | 0.491 | 0.235 | 0.148 |
| 0.75–1.00 | 1414.000 | 1.765 | 7.059 | 0.505 | 0.225 | 0.156 |
| 1.00–1.25 | 1283.000 | 1.702 | 6.348 | 0.602 | 0.236 | 0.143 |
| 1.25–1.50 | 1064.000 | 2.446 | 5.797 | 0.699 | 0.261 | 0.157 |
| 1.50–1.75 | 788.000 | 2.898 | 5.639 | 0.772 | 0.312 | 0.156 |
| 1.75–2.00 | 666.000 | 2.887 | 5.100 | 0.811 | 0.308 | 0.161 |
| 2.00–2.25 | 425.000 | 3.213 | 5.002 | 0.849 | 0.374 | 0.169 |
| 2.25–2.50 | 196.000 | 4.651 | 6.498 | 0.903 | 0.449 | 0.260 |
| ≥ 2.50 | 67.000 | 4.880 | 7.011 | 0.791 | 0.478 | 0.373 |

### Zona > 70 m dalla porta (N = 4180; Spearman ρ = 0.371, p = 2e-136)

| bin | N | media | sd | P(>0 m) | P(>3 m) | P(>6 m) |
|---|---|---|---|---|---|---|
| 0.00–0.25 | 14.000 | 0.900 | 9.910 | 0.500 | 0.214 | 0.214 |
| 0.25–0.50 | 161.000 | -0.224 | 9.072 | 0.317 | 0.149 | 0.099 |
| 0.50–0.75 | 472.000 | 0.581 | 7.299 | 0.464 | 0.222 | 0.123 |
| 0.75–1.00 | 732.000 | 0.385 | 7.502 | 0.439 | 0.169 | 0.097 |
| 1.00–1.25 | 709.000 | 0.940 | 6.382 | 0.528 | 0.171 | 0.094 |
| 1.25–1.50 | 538.000 | 1.240 | 6.103 | 0.610 | 0.191 | 0.110 |
| 1.50–1.75 | 401.000 | 1.926 | 5.654 | 0.701 | 0.242 | 0.120 |
| 1.75–2.00 | 328.000 | 2.653 | 5.549 | 0.777 | 0.280 | 0.146 |
| 2.00–2.25 | 387.000 | 3.281 | 4.205 | 0.891 | 0.359 | 0.181 |
| 2.25–2.50 | 247.000 | 4.978 | 5.965 | 0.955 | 0.506 | 0.283 |
| ≥ 2.50 | 191.000 | 5.827 | 6.317 | 0.958 | 0.586 | 0.387 |

## 4. Pressione al rilascio → perdita di palla (paper, Fig. 7a)

Passaggi in gioco aperto: N = 13847 (possession play 10414, direct play 3433). Esclusi 1334 passaggi di possessi iniziati da palla inattiva.

### direct play (N = 3433; perdita media 24.0%; Spearman τ_opp(t_rel) vs perdita ρ = -0.029, p = 0.084)

| bin | N | P_perdita |
|---|---|---|
| 0.00–0.25 | 14.000 | 0.357 |
| 0.25–0.50 | 362.000 | 0.235 |
| 0.50–0.75 | 1044.000 | 0.255 |
| 0.75–1.00 | 989.000 | 0.236 |
| 1.00–1.25 | 546.000 | 0.267 |
| 1.25–1.50 | 232.000 | 0.237 |
| 1.50–1.75 | 124.000 | 0.194 |
| 1.75–2.00 | 52.000 | 0.135 |
| 2.00–2.25 | 38.000 | 0.053 |
| 2.25–2.50 | 25.000 | 0.080 |
| ≥ 2.50 | 7.000 | 0.000 |

### possession play (N = 10414; perdita media 12.9%; Spearman τ_opp(t_rel) vs perdita ρ = -0.054, p = 4.2e-08)

| bin | N | P_perdita |
|---|---|---|
| 0.00–0.25 | 61.000 | 0.131 |
| 0.25–0.50 | 854.000 | 0.124 |
| 0.50–0.75 | 2389.000 | 0.143 |
| 0.75–1.00 | 2710.000 | 0.149 |
| 1.00–1.25 | 1725.000 | 0.156 |
| 1.25–1.50 | 932.000 | 0.121 |
| 1.50–1.75 | 715.000 | 0.090 |
| 1.75–2.00 | 487.000 | 0.051 |
| 2.00–2.25 | 334.000 | 0.033 |
| 2.25–2.50 | 141.000 | 0.028 |
| ≥ 2.50 | 66.000 | 0.015 |

## Differenze dal paper da tenere presenti

- 10 fps invece di 25; nessuna lisciatura aggiuntiva (il paper usa Savitzky–Golay e spline); traiettorie broadcast con posizioni estrapolate.
- Intervalli di possesso e esiti presi da `player_possession` di SkillCorner, non da una sincronizzazione evento-tracking propria.
- Perdita di palla: nessuna etichetta per passaggio ordinario/filtrante; incluso ogni `end_type == pass` in gioco aperto. "Gioco fermo dopo l'azione" è coperto solo tramite pass_outcome (unsuccessful/offside).
- 20 partite contro 306.
