# Lavori esistenti

Repo e risorse valutate per il progetto. Metadati GitHub (licenza, ultimo push, stelle)
letti dall'API il 2026-10-15: cambiano, ricontrollali prima di dipendere da una repo.

## Riconoscimento da video

| Repo | Cosa fa | Licenza | Ultimo push | Uso qui |
|---|---|---|---|---|
| [asigatchov/fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference) | Tracking della palla su CPU (ONNX), `ball.csv` → tracce → segmentazione in scambi. Variante con campo e giocatori: tocchi, tecnica del tocco, distingue lo scambio dalla palla passata per la battuta successiva (`PLAYER_CONTACTS.md`) | MIT | 2026-10-15 | **Punto di partenza.** Fa già la segmentazione in scambi, che è il primo pezzo del punteggio. Gli esempi della variante con giocatori sono su beach (`--beach`): l'indoor va verificato |
| [asigatchov/vball-net](https://github.com/asigatchov/vball-net) | Training dei modelli VballNetV1 / VballNetFastV1, derivati da TrackNetV4 | MIT | 2026-06-08 | Solo se serve un fine-tuning della palla sui nostri video |
| [shukkkur/VolleyVision](https://github.com/shukkkur/VolleyVision) | YOLO per palla, giocatori, azioni (block, defense, serve, set, spike), segmentazione del campo; dataset su Roboflow | **In conflitto**: AGPL-3.0 per GitHub, CC BY-NC-ND 4.0 nel badge del README | 2025-06-20 | Dataset Roboflow e idee. Non copiarne codice finché la licenza non è chiara. La pagina DeepWiki è documentazione generata della stessa repo |
| [openvolley/ovml](https://github.com/openvolley/ovml) | YOLO v3/4/7 in R (torch), più una variante per la palla; esiste `ovmlpy` in Python | MIT | 2026-01-16 | Riferimento. Lo dice la repo stessa: è più lenta delle implementazioni Python |
| [FinityFly/tobio](https://github.com/FinityFly/tobio) | Progetto universitario full-stack: YOLOv11 per campo, palla e giocatori, riconoscimento della battuta, FastAPI + React, agente Gemini | **Nessuna** (tutti i diritti riservati) | 2026-04-20 | Solo come esempio di architettura dell'app. Richiede CUDA |
| Beach Volley Vision | Non identificato con certezza; su Roboflow c'è `Beach_Volley_BB` (102 immagini) | – | – | Fuori dominio se si gioca indoor |
| [inovex — tesi con Baden Volleys](https://www.inovex.de/de/?p=50292) | Statistiche per giocatore: YOLOv5x, tracking multi-oggetto, rilevamento e classificazione delle azioni | Articolo | – | Riferimento per la scomposizione in moduli |

## Scouting manuale

| Repo | Cosa fa | Licenza | Ultimo push | Uso qui |
|---|---|---|---|---|
| [openvolley/ovscout2](https://github.com/openvolley/ovscout2) | App R Shiny per lo scouting stile DataVolley, sincronizzato col video | Non dichiarata in modo standard | 2026-09-12 | Riferimento per il formato degli esiti (codici DataVolley) e per il flusso "la macchina propone, la persona conferma" |
| [ellemenno/vbstats](https://github.com/ellemenno/vbstats) | Registratore manuale di statistiche (Svelte) | MIT | 2025-05-04 | Riferimento per l'interfaccia di inserimento rapido |

Non verificato: la pagina oosmetrics su VolleyVision.
