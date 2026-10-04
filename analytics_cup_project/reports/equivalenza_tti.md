# time_to_impact al posto di τ_opp?

Generato da `scripts/equivalenza_tti.py` su `reports/tau_opp.csv`. Protocollo in `plans/01-scegliere-la-domanda.md`, «Il test di equivalenza». Solo numeri: l'interpretazione è da fare.

Segni attesi opposti: τ_opp in secondi (più alto = meno pressione), `time_to_impact_id` da 1 (very_easy) a 5 (very_hard). Ogni confronto usa gli stessi possessi, quelli in cui esistono entrambe le misure.

## 1. Pressione all'inizio → avanzamento (paper, Fig. 6)

Possessi non direct play, portieri esclusi, con entrambe le misure all'inizio: N = 12804.

| campione | N | ρ τ_opp | p τ_opp | ρ time_to_impact | p time_to_impact |
|---|---|---|---|---|---|
| tutti | 12804.000 | 0.277 | 0.000 | -0.154 | 0.000 |
| zona < 35 m | 1706.000 | 0.262 | 0.000 | -0.084 | 0.001 |
| zona 35–70 m | 7067.000 | 0.238 | 0.000 | -0.093 | 0.000 |
| zona > 70 m | 4031.000 | 0.367 | 0.000 | -0.281 | 0.000 |

Progressione per classe di `time_to_impact` all'inizio:

| time_to_impact_start_id | N | media | P(>0 m) | P(>6 m) |
|---|---|---|---|---|
| 1.0 | 3748.000 | 2.556 | 0.781 | 0.148 |
| 2.0 | 4101.000 | 1.569 | 0.593 | 0.128 |
| 3.0 | 2694.000 | 1.746 | 0.560 | 0.154 |
| 4.0 | 1263.000 | 1.288 | 0.518 | 0.158 |
| 5.0 | 998.000 | 1.338 | 0.509 | 0.166 |

## 2. Pressione al rilascio → perdita di palla (paper, Fig. 7a)

Passaggi in gioco aperto con entrambe le misure al rilascio: N = 12895.

| tipo | N | ρ τ_opp | p τ_opp | ρ time_to_impact | p time_to_impact |
|---|---|---|---|---|---|
| direct play | 2666.000 | -0.066 | 0.001 | 0.146 | 0.000 |
| possession play | 10229.000 | -0.056 | 0.000 | 0.152 | 0.000 |

Perdita per classe di `time_to_impact` al rilascio:

| tipo · classe | N | P_perdita |
|---|---|---|
| direct play · classe 1 | 140.000 | 0.029 |
| direct play · classe 2 | 289.000 | 0.138 |
| direct play · classe 3 | 992.000 | 0.155 |
| direct play · classe 4 | 1028.000 | 0.239 |
| direct play · classe 5 | 217.000 | 0.272 |
| possession play · classe 1 | 2081.000 | 0.045 |
| possession play · classe 2 | 1903.000 | 0.090 |
| possession play · classe 3 | 2520.000 | 0.148 |
| possession play · classe 4 | 2686.000 | 0.179 |
| possession play · classe 5 | 1039.000 | 0.184 |

## 3. Informazione aggiuntiva, nei due sensi

Dentro ogni strato di una misura, ρ dell'altra con l'esito. Media pesata per N sugli strati con almeno 30 possessi. I quintili di τ_opp sono calcolati sul campione del confronto.

### avanzamento

τ_opp dentro le classi di `time_to_impact` (media pesata ρ = 0.294, su N = 12804):

| classe time_to_impact_start_id | N | ρ | p |
|---|---|---|---|
| 1.0 | 3748.000 | 0.368 | 0.000 |
| 2.0 | 4101.000 | 0.339 | 0.000 |
| 3.0 | 2694.000 | 0.251 | 0.000 |
| 4.0 | 1263.000 | 0.152 | 0.000 |
| 5.0 | 998.000 | 0.125 | 0.000 |

`time_to_impact` dentro i quintili di τ_opp (media pesata ρ = 0.142, su N = 12804):

| quintile di τ_opp | N | ρ | p |
|---|---|---|---|
| Q1 | 2561.000 | 0.030 | 0.123 |
| Q2 | 2561.000 | 0.188 | 0.000 |
| Q3 | 2560.000 | 0.152 | 0.000 |
| Q4 | 2561.000 | 0.210 | 0.000 |
| Q5 | 2561.000 | 0.130 | 0.000 |

### perdita, possession play

τ_opp dentro le classi di `time_to_impact` (media pesata ρ = 0.103, su N = 10229):

| classe time_to_impact_end_id | N | ρ | p |
|---|---|---|---|
| 1.0 | 2081.000 | -0.044 | 0.043 |
| 2.0 | 1903.000 | 0.106 | 0.000 |
| 3.0 | 2520.000 | 0.164 | 0.000 |
| 4.0 | 2686.000 | 0.160 | 0.000 |
| 5.0 | 1039.000 | 0.100 | 0.001 |

`time_to_impact` dentro i quintili di τ_opp (media pesata ρ = 0.165, su N = 10229):

| quintile di τ_opp | N | ρ | p |
|---|---|---|---|
| Q1 | 2046.000 | 0.072 | 0.001 |
| Q2 | 2046.000 | 0.156 | 0.000 |
| Q3 | 2045.000 | 0.163 | 0.000 |
| Q4 | 2046.000 | 0.259 | 0.000 |
| Q5 | 2046.000 | 0.174 | 0.000 |

### perdita, direct play

τ_opp dentro le classi di `time_to_impact` (media pesata ρ = 0.072, su N = 2666):

| classe time_to_impact_end_id | N | ρ | p |
|---|---|---|---|
| 1.0 | 140.000 | -0.054 | 0.525 |
| 2.0 | 289.000 | 0.031 | 0.598 |
| 3.0 | 992.000 | 0.138 | 0.000 |
| 4.0 | 1028.000 | 0.050 | 0.109 |
| 5.0 | 217.000 | 0.007 | 0.913 |

`time_to_impact` dentro i quintili di τ_opp (media pesata ρ = 0.155, su N = 2666):

| quintile di τ_opp | N | ρ | p |
|---|---|---|---|
| Q1 | 534.000 | 0.073 | 0.091 |
| Q2 | 533.000 | 0.184 | 0.000 |
| Q3 | 533.000 | 0.146 | 0.001 |
| Q4 | 533.000 | 0.125 | 0.004 |
| Q5 | 533.000 | 0.246 | 0.000 |

### Sintesi

| esito | N | ρ τ_opp (tutti) | ρ τ_opp dentro le classi tti | ρ tti (tutti) | ρ tti dentro i quintili τ_opp |
|---|---|---|---|---|---|
| avanzamento | 12804.000 | 0.277 | 0.294 | -0.154 | 0.142 |
| perdita, possession play | 10229.000 | -0.056 | 0.103 | 0.152 | 0.165 |
| perdita, direct play | 2666.000 | -0.066 | 0.072 | 0.146 | 0.155 |

## 4. Copertura

| istante | possessi | entrambe | solo τ_opp | solo time_to_impact | nessuna | solo τ_opp, di cui direct play |
|---|---|---|---|---|---|---|
| inizio | 17674 | 16045 | 1521 | 1 | 107 | 1180 |
| rilascio | 17674 | 16122 | 1541 | 1 | 10 | 1180 |

Direct play senza `time_to_impact` all'inizio: 1190 su 4432 (26.9%).

Campioni rispetto a `reports/tau_opp.md` (solo τ_opp): avanzamento 12804 su 13145; perdita 12895 su 13847.

