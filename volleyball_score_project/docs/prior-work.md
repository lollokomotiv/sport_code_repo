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

## Dataset pubblici

Letti il 2026-10-18. Tutti di alto livello e quasi tutti da riprese televisive (zoom,
cambi di inquadratura, replay): servono a pre-addestrare il modello sul tipo di tocco,
non a misurarlo. La misura resta sulla partita 2 del campionato.

### Con video ed etichette delle azioni

| Dataset | Cosa contiene | Licenza | Uso qui |
|---|---|---|---|
| [MultiSports](https://github.com/MCG-NJU/MultiSports), parte pallavolo (ICCV 2021) | 800 clip da YouTube (Olimpiadi, Coppa del Mondo), 720p/1080p; 7.645 azioni in 12 classi (Serve, Block, First Hit Pass, Defend, Protect, Second Hit Pass, Adjust, Save, Second Attack, Spike, Dink, No Offensive Attack); inizio e fine di ogni azione e riquadro della giocatrice per fotogramma a 25 fps, etichettati da pallavolisti con un manuale scritto. Dati su Hugging Face | CC BY-NC 4.0 | **Candidato principale** per pre-addestrare il classificatore: azioni con un tempo, come i nostri tocchi. Le 12 classi vanno ricondotte alle 7 nostre. Il manuale è un riferimento per `guida-etichette.md` |
| [Volleyball Dataset](https://github.com/mostafa-saad/deep-activity-rec), Ibrahim et al. (CVPR 2016) | 55 video YouTube, 4.830 clip; etichettato solo il fotogramma centrale di ogni clip: riquadri, azione individuale (setting, digging, spiking, blocking, jumping, falling, moving, standing, waiting), azione di squadra | Codice BSD-2; termini del dataset da verificare | Classificatore sulla singola giocatrice (ritaglio o posa). Niente tempi del tocco |
| [Volleyball Activity Dataset 2014](https://tugraz.at/index.php?id=17751), TU Graz | 6 partite del campionato austriaco 2011/12, 1080p a 25 fps; 7 classi (Serve, Reception, Setting, Attack, Block, Stand, Defense/Move); 36.178 annotazioni su 18.960 fotogrammi | Nessuna dichiarata; chiedono di citare gli articoli | Livello più vicino al nostro, ma piccolo. Formato delle annotazioni da vedere |
| Roboflow (dataset di VolleyVision) | Immagini con riquadri per Attack, Block, Defense, Serve, Set | Da verificare per dataset | Rilevamento delle azioni fotogramma per fotogramma, senza tempi |

### Solo scouting, senza video

| Dataset | Cosa contiene | Licenza | Uso qui |
|---|---|---|---|
| [VREN](https://arxiv.org/html/2209.13846v2) (UCSB, 2022) | 1.632 scambi, 12.112 tocchi, NCAA Division I e nazionali maschili. Per tocco: squadra, zona, qualità della ricezione, tipo di alzata e di attacco, numero di muratori, motivo del punto. Un solo CSV ([repo](https://github.com/haotianxia/VREN)), senza tempi | Nessuna nella repo | Frequenze delle sequenze di tocchi (attacco di seconda, free ball, muro) per tarare la grammatica |
| [ovdata](https://openvolley.r-universe.dev/ovdata/ovdata.pdf) (openvolley) | Pochi file DataVolley di esempio: una partita della PlusLiga polacca 2018/19, campionato sloveno, lega australiana (VBStats) | MIT | Come VREN, su pochi file |

Su Kaggle non risulta nulla al livello del tocco, solo statistiche aggregate.

### Partite femminili

Verificato il 2026-10-18:

- **MultiSports**, train e validation (file delle annotazioni, scaricato con l'accesso
  a Hugging Face il 2026-10-19): 43 video YouTube, 532 clip, 4.843 azioni. **27 video
  femminili** (Londra 2012, Rio 2016, preolimpici 2016 e 2020, VNL 2018, Challenger Cup
  2019, finale NCAA 2019, Pechino 2008): **347 clip e 3.265 azioni, il 67%**. 13
  maschili (155 clip, 1.270 azioni), 3 non più su YouTube (30 clip, 308 azioni). Il
  genere viene dal titolo del video; l'unico titolo che non lo dice (Russia-Giappone,
  Rio 2016) è femminile, perché il Giappone maschile a Rio non c'era. Nel test set, dai
  nomi delle clip nel file d'esempio della repo, 5 video femminili su 11 (68 clip su
  128).

  Le azioni di MultiSports sono movimenti di una giocatrice, non tocchi di palla: nel
  femminile ci sono 977 muri e 339 battute, quindi ogni muratrice ha la sua azione,
  probabilmente anche quando non tocca la palla (da verificare sul manuale). Per
  usarle come tocchi vanno ricondotte.
- **VREN**: solo maschile, lo dice l'articolo.
- **TU Graz**: non dichiarato. L'unica figura dell'articolo mostra una partita
  maschile, ripresa da una camera fissa dietro la linea di fondo: più simile alla
  nostra delle riprese televisive. Le altre partite non sono verificate.
- **Volleyball Dataset (Ibrahim)**: non dichiarato. I video sono numerati senza fonte;
  per saperlo bisogna scaricare il dataset (circa 58 GB).
- **ovdata**: la partita della PlusLiga è maschile; i file sloveni e australiani non
  sono verificati.
