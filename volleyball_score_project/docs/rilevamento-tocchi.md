# Rilevamento dei tocchi dalla traiettoria

Ricerca del 2026-11-10 per la issue #19. Metadati GitHub (licenza, ultimo push) letti
dall'API lo stesso giorno: cambiano, ricontrollali prima di dipendere da una repo.

## Domanda

Come si trovano i tocchi (istante e lato del campo) dalla traiettoria della palla in un
video da una sola camera, nella pallavolo e negli sport di rete simili? In particolare:

1. come si filtrano le rotture false della traiettoria (rimbalzi, occlusioni, rumore
   del tracker, palla vicino alla rete);
2. come si separano due tocchi ravvicinati come attacco e muro (0,1-0,3 s);
3. come si stima il lato con una camera alta dietro la linea di fondo;
4. quali dataset e metodi con codice esistono, con che licenza.

Punto di partenza: il prototipo `prototipi/tocchi_19.py` (rottura della velocità oltre
la gravità, finestra di 5 campioni, soppressione dei non massimi in 0,3 s, più i buchi
della traccia fino a 0,5 s). Sulla partita di sviluppo: 132/151 tocchi trovati entro
±0,3 s, 109/241 rotture inventate, muri 3/11, lato dalla proiezione sul pavimento
69/132, lato dal raggio apparente 96/116 (esplorativo, regola decisa dopo aver visto i
dati).

## Sintesi

Cosa dicono le fonti:

- **La rottura locale della traiettoria è la baseline che tutti battono.** Su badminton,
  MonoTrack riporta per una baseline basata sulle derivate F1 0,699 contro 0,946 del suo
  modello con ottimizzazione globale (dataset TrackNetV2, partite di prova tenute da
  parte) ([MonoTrack](https://ar5iv.arxiv.org/html/2204.01899)). Su pallavolo, Volley3D
  riporta per un'euristica sui cambi di segno del gradiente F1 0,759 a ±1 fotogramma su
  tracce TrackNetV3, contro 0,898 di una piccola rete temporale 1D (TCN) sulla sola
  traccia 2D ([Volley3D](https://arxiv.org/html/2610.38358)). Il vincitore della CoachAI
  Badminton Challenge 2023 scrive che contare le inversioni verticali "non funziona
  bene" per il rumore del rilevatore
  ([Chen et al. 2023](https://ar5iv.labs.arxiv.org/html/2308.12645)).
- **Il filtro più efficace riportato è una decodifica globale con le regole dello
  sport**, non una soglia migliore: MonoTrack sceglie i tocchi con la programmazione
  dinamica imponendo numero di tocchi circa pari alla durata dello scambio in secondi,
  almeno mezzo secondo tra due tocchi, alternanza tra i giocatori; il passo porta
  l'accuratezza da 76,4% a 89,7% ([MonoTrack](https://ar5iv.arxiv.org/html/2204.01899)).
  La repo del nostro tracker fa una cosa simile per l'attribuzione: possessi di al
  massimo tre tocchi per lato
  ([PLAYER_CONTACTS.md](https://github.com/asigatchov/fast-volleyball-tracking-inference/blob/main/PLAYER_CONTACTS.md)).
- **Nessuna fonte trovata risolve attacco e muro dalla sola traiettoria.** VNL-STES
  (pallavolo, VNL) indica la confusione schiacciata/muro "per vicinanza temporale e
  somiglianza visiva" come limite, senza rimedio
  ([progetto STES](https://hoangqnguyen.github.io/stes)); Volley3D assume che non ci
  siano contatti con la rete e non tratta il muro come caso di segmentazione
  ([Volley3D](https://arxiv.org/html/2610.38358)).
- **La soppressione dei non massimi, nei modelli di event spotting, è per classe e
  stretta**: in E2E-Spot è fatta separatamente per etichetta con finestra predefinita di
  1 fotogramma ([codice, `util/eval.py`](https://github.com/jhong93/spot/blob/main/util/eval.py));
  T-DEED ottiene i risultati migliori con NMS a 1 fotogramma o Soft-NMS a 3
  ([T-DEED](https://arxiv.org/html/2404.05392v2)). Nel dataset Tennis di E2E-Spot il
  lato è dentro la classe (tocco "near-court" e "far-court" sono classi diverse)
  ([E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213)), quindi due tocchi di lati
  opposti non si sopprimono a vicenda.
- **La dimensione apparente della palla dà la profondità solo se la palla è grande
  nell'immagine.** Su basket, con palle di 14-45 px di diametro, l'errore 3D è del 10%
  della distanza dalla camera ([Van Zandycke e De Vleeschouwer 2022](https://ar5iv.labs.arxiv.org/html/2204.00003));
  per il tennistavolo televisivo (palla sotto i 10 px) TT3D lo giudica inattuabile
  ([TT3D](https://arxiv.org/html/2504.10035)); per tennis e calcio "Where Is The Ball"
  dice che la camera è troppo lontana ([Ponglertnapakorn e Suwajanakorn 2025](https://arxiv.org/html/2506.05763v1)).
- **La ricostruzione 3D da una camera funziona per tratti tra due tocchi, con vincoli
  fisici e di campo.** MonoTrack: gravità più resistenza quadratica, punto iniziale nella
  metà campo di chi colpisce, velocità verso l'avversario, altezze 0-3 m. Volley3D:
  stessi modelli fisici, estremi del tratto ancorati alle mani delle giocatrici
  ricostruite in 3D, 0,27 m di RMSE su 10 scambi ripresi da 4 camere. Dong et al.
  (pallavolo, una camera in tribuna a 60 fps): 82,5% dei fotogrammi entro 200 mm su 20
  sequenze ([MVA 2021](https://www.mva-org.jp/Proceedings/2021/papers/P1-14.pdf)).

Cosa propongo di provare (proposte mie, non delle fonti):

- **Rotture inventate**: sostituire la soglia locale con una segmentazione globale in
  tratti parabolici con penalità per tratto (come TT3D, codice MIT) e poi una decodifica
  dello scambio con le regole che abbiamo già in `lib/scambio.py` (battuta, al massimo
  tre tocchi per lato, muro che non conta). Soglia sul residuo proporzionale al raggio
  apparente della palla invece che 6 px fissi (idea di Dong et al.).
- **Attacco e muro**: togliere la soppressione indifferenziata di 0,3 s e sopprimere
  solo tra rotture dello stesso lato, con finestra corta; accettare una seconda rottura
  0,1-0,3 s dopo un attacco se la palla torna verso il campo dell'attaccante. Il
  protocollo a 60 fps raddoppia i campioni tra i due tocchi.
- **Lato**: combinare tre indizi deboli invece di cercarne uno forte: il raggio
  apparente (già 96/116), la scala della gravità in pixel nel tratto (l'accelerazione
  verticale apparente diminuisce con la distanza) e la grammatica dello scambio (chi
  batte è noto dal punteggio, il lato cambia solo attraversando la rete). Le teste
  trovate da RAVEL-VB possono fare da quarto indizio.

## 1. Rotture false

### Cosa fanno le fonti

**Decodifica globale con vincoli.** MonoTrack (badminton, CVPRW 2022) classifica ogni
fotogramma come "nessun colpo", "colpo del giocatore vicino", "colpo del giocatore
lontano" con una piccola GRU (circa 16 mila parametri) che legge posizione del volano,
pose dei due giocatori e angoli del campo su 12 fotogrammi. Poi sceglie i fotogrammi dei
colpi risolvendo un problema vincolato con la programmazione dinamica: il numero di
colpi è circa la durata dello scambio in secondi ("il tempo medio tra due colpi è circa
1 secondo"), due colpi distano almeno mezzo secondo, i colpi si alternano tra i due
giocatori; l'obiettivo somma (punteggio − τ) e così penalizza i colpi in più. Sul
dataset TrackNetV2 (26 partite di singolo, 3 di prova tenute da parte), Tabella 1:
baseline sulle derivate acc 53,8%, F1 0,699; HitNet da solo acc 76,4%, F1 0,866;
HitNet più ottimizzazione acc 89,7%, F1 0,946. La fonte non dichiara una tolleranza in
fotogrammi per l'abbinamento ([MonoTrack](https://ar5iv.arxiv.org/html/2204.01899)).

**Segmentazione in tratti parabolici.** TT3D (tennistavolo, CVSports 2025) divide la
traccia 2D in tratti di polinomi di secondo grado con la programmazione dinamica; il
costo somma errore di adattamento, una penalità λ per ogni nuovo tratto e un termine sul
mosso della palla. Il rimbalzo è l'intersezione di due tratti consecutivi; la
racchettata si distingue dal rimbalzo perché cambia la direzione lungo l'asse lungo del
tavolo. Servono almeno tre fotogrammi visibili dopo il rimbalzo; λ non è indicato
([TT3D](https://arxiv.org/html/2504.10035)). Il codice di segmentazione è in
[`tt3d/traj_seg/segmenter.py`](https://github.com/cogsys-tuebingen/tt3d/blob/main/tt3d/traj_seg/segmenter.py),
licenza MIT: adatta parabole in x(t) e y(t) prima e dopo e prende come istante
l'intersezione, quindi con precisione sotto il fotogramma.

**Soglia scalata sulla palla.** Dong et al. (pallavolo, MVA 2021) aggiornano il modello
fisico del tratto quando la distanza tra punto osservato e punto riproiettato supera "il
raggio della volleyball sul piano immagine"; sotto quella soglia il tratto continua
([MVA 2021](https://www.mva-org.jp/Proceedings/2021/papers/P1-14.pdf)). È di fatto un
criterio di rottura con soglia proporzionale alla dimensione apparente.

**Reti temporali sulla traccia.** Volley3D (pallavolo, arXiv ottobre 2026) segmenta la
traccia con una TCN 1D che legge solo (X, Y, dX, dY) per fotogramma, con etichette
sfumate da una gaussiana attorno al contatto (σ = 1, raggio 3 fotogrammi), finestre di 17
fotogrammi, picchi sopra 0,35. Sul Volleyball dataset di Ibrahim et al. con annotazioni
proprie (377 clip, 1.621 sottotraiettorie nel test), F1 a ±1 fotogramma: euristica Y(t)
0,759, HitNet ristretto alla traccia 0,815, loro 0,898, su tracce TrackNetV3. Non c'è un
filtro dedicato alle rotture false oltre a soglia e picchi; l'unico limite dichiarato è
che "le euristiche standard sono molto sensibili al rumore del tracking 2D"
([Volley3D](https://arxiv.org/html/2610.38358)). Codice non rilasciato: la repo
[Noambsf/Volley3D](https://github.com/Noambsf/Volley3D) contiene solo la pagina del
progetto ("Coming Soon"). "Where Is The Ball" usa un'uscita di "fine traiettoria" con lo
stesso ruolo, ma come segnale ausiliario, non per tagliare
([Ponglertnapakorn e Suwajanakorn 2025](https://arxiv.org/html/2506.05763v1)).

**Esempi negativi.** TTNet (tennistavolo) aggiunge al training sequenze senza eventi
("empty events"), in numero simile agli eventi, "per la robustezza della CNN e contro
l'overfitting"; l'obiettivo attorno all'evento è sfumato con sin(nπ/8) per |n| < 4
fotogrammi, perché nel fotogramma esatto la palla può essere mossa
([TTNet](https://ar5iv.labs.arxiv.org/html/2004.09927)).

**Occlusioni e buchi.** TrackNetV3 (badminton) individua dalla traccia predetta i tratti
da riparare e li ricostruisce con una rete di inpainting
([README](https://github.com/qaz812345/TrackNetV3)). Volley3D usa proprio TrackNetV3 con
InpaintNet a monte della segmentazione e attribuisce a InpaintNet gran parte del
vantaggio sul tracking ([Volley3D](https://arxiv.org/html/2610.38358)). La repo del
nostro tracker unisce due tracce separate da al massimo 0,5 s se finiscono e iniziano
vicine con una giocatrice a portata ("il tocco spiega il buco") e taglia la traccia se
nessun rilevatore vede la palla per 0,8 s
([PLAYER_CONTACTS.md](https://github.com/asigatchov/fast-volleyball-tracking-inference/blob/main/PLAYER_CONTACTS.md)).

**Rimbalzi a terra.** Volley3D classifica un evento come rimbalzo quando la palla è più
vicina alle caviglie che ai polsi della giocatrice più vicina
([Volley3D](https://arxiv.org/html/2610.38358)); serve la posa. TT3D separa rimbalzo e
colpo con la direzione lungo il tavolo (sopra).

**Rete.** Nessuna fonte trovata tratta il tocco della rete come rottura da scartare:
Volley3D lo esclude per ipotesi, TT3D non ne parla
([Volley3D](https://arxiv.org/html/2610.38358), [TT3D](https://arxiv.org/html/2504.10035)).

### Cosa ne ricavo per noi (proposte)

- Le nostre 109 rotture inventate su 241 sono coerenti con quello che le fonti dicono
  delle euristiche locali. Il passo con il guadagno riportato più grande è la decodifica
  globale (MonoTrack, +13 punti di accuratezza). Le nostre regole sono diverse da quelle
  del badminton (fino a tre tocchi per lato, muro fuori conteggio), ma sono già scritte
  in `lib/scambio.py`: si può dare a ogni rottura un punteggio e scegliere con la
  programmazione dinamica la sequenza di tocchi legale con punteggio massimo.
- Il vincolo "almeno mezzo secondo tra due tocchi" di MonoTrack **non va copiato**:
  ucciderebbe i muri. Il minimo va legato al lato (sotto).
- La soglia `RESIDUO_MIN = 6` px è uguale vicino e lontano dalla camera. Con la camera
  dietro la linea di fondo la palla lontana è piccola e la stessa spinta produce pochi
  pixel: una soglia in raggi apparenti (Dong et al.) è più coerente.
- Una TCN come quella di Volley3D è stata addestrata su più di 1.600 eventi; noi ne
  abbiamo 151 su una partita. Un modello piccolo sulla traccia (o un gradient boosting su
  feature della rottura) è provabile solo con la validazione su partite diverse.
- Se un giorno si addestra un modello, servono esempi negativi presi apposta (TTNet):
  palla che passa sopra una giocatrice, apice della parabola, rotture di rumore.

## 2. Attacco e muro

### Cosa fanno le fonti

- **Soppressione per classe.** In E2E-Spot (ECCV 2022) la NMS raggruppa le previsioni
  per etichetta e sopprime solo dentro la stessa etichetta, con finestra predefinita di 1
  fotogramma ([`util/eval.py`](https://github.com/jhong93/spot/blob/main/util/eval.py));
  l'articolo dice che la NMS può fare danni con tolleranza ≤ 1 fotogramma
  ([E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213)). T-DEED: migliori risultati
  con NMS di 1 fotogramma e Soft-NMS di 3; la Soft-NMS migliora sempre la mAP su
  FigureSkating e FineDiving ([T-DEED](https://arxiv.org/html/2404.05392v2)).
- **Rappresentazioni che distinguono fotogrammi vicini.** T-DEED nasce proprio dal
  problema che fotogrammi adiacenti hanno rappresentazioni simili e le previsioni si
  impastano ([T-DEED](https://arxiv.org/html/2404.05392v2)). Lavora sul video, non sulla
  traccia.
- **Pallavolo.** VNL-STES (CVPRW 2025): schiacciata e muro si confondono "per vicinanza
  temporale e somiglianza visiva"; come lavoro futuro citano pesi sulle classi, audio,
  rilevamento della palla. Il loro modello STES ha mTAP 68,44 (tolleranza 0-4 fotogrammi,
  25 fps) contro 58,58 di T-DEED sul loro test set di 115 scambi
  ([progetto STES](https://hoangqnguyen.github.io/stes)). Volley3D non tratta il muro come
  caso di segmentazione ([Volley3D](https://arxiv.org/html/2610.38358)).
- **Tocchi ravvicinati dello stesso giocatore.** La testa "touch" di RAVEL-VB-013 produce
  doppi picchi: 123 dei 258 tocchi trovati solo dal modello cadono 7-20 fotogrammi dopo un
  tocco vero, e la repo consiglia di unire i picchi dello stesso giocatore più vicini di
  circa 20 fotogrammi ([PLAYER_CONTACTS.md](https://github.com/asigatchov/fast-volleyball-tracking-inference/blob/main/PLAYER_CONTACTS.md)).
  Qui la soppressione è per giocatore, non globale.
- **Risoluzione temporale.** TT3D nota che il colpo può cadere troppo vicino al rimbalzo
  per formare un tratto separato e chiede tre fotogrammi visibili dopo il rimbalzo
  ([TT3D](https://arxiv.org/html/2504.10035)).

### Cosa ne ricavo per noi (proposte)

- I muri soppressi (3/11, attacco 0,16-0,3 s prima) sono l'effetto atteso di una NMS
  indifferenziata di 0,3 s. Le fonti sopprimono per classe o per giocatore. L'analogo per
  noi è sopprimere solo tra rotture dello stesso lato: un attacco e un muro sono di lati
  opposti per definizione.
- Serve allora un'ipotesi di lato già per la rottura, prima della NMS. Indizio specifico
  del muro: la palla va verso la rete e torna verso il campo dell'attaccante (o cade
  subito dietro il muro). Una regola da provare: dopo una rottura etichettabile come
  attacco vicino alla rete, accettare una seconda rottura entro 0,1-0,3 s se l'inversione
  della componente verso la rete è netta.
- A 30 fps, 0,16 s sono circa 5 fotogrammi: due parabole da almeno 3 punti ciascuna ci
  stanno appena, e le stime di velocità su finestre di 5 campioni (il nostro `FINESTRA`)
  mescolano i due tocchi. Il protocollo prevede 60 fps: con 10 fotogrammi tra attacco e
  muro la segmentazione parabolica di TT3D diventa praticabile. Questo va misurato, è una
  stima.
- L'intersezione delle due parabole (codice TT3D) dà l'istante anche quando il tocco cade
  in un buco della traccia, che per noi è frequente (#13).

## 3. Lato del campo

### Cosa fanno le fonti

**Dimensione apparente.** Van Zandycke e De Vleeschouwer (basket, CVSports 2022) stimano
il diametro in pixel con una piccola CNN e ne ricavano la posizione 3D da calibrazione e
diametro reale. Con palle di 14-45 px (DeepSport) e 19-40 px (loro set): errore sul
diametro 1,6 px, errore 3D 2,3 m e 1,8 m, cioè circa il 10% della distanza dalla camera;
la trasformata di Hough fa 4,9 px e 6,3 m. Le stime sono per immagine, senza filtro
temporale ([arXiv 2204.00003](https://ar5iv.labs.arxiv.org/html/2204.00003)). Contro:
TT3D giudica la profondità dal diametro "inattuabile" per palle sotto i 10 px
([TT3D](https://arxiv.org/html/2504.10035)); "Where Is The Ball" dice che in tennis e
calcio la camera è troppo lontana e il mosso degrada la stima
([arXiv 2506.05763](https://arxiv.org/html/2506.05763v1)). La palla da pallavolo ha
circonferenza 65-67 cm (regola 3.1), cioè circa 21 cm di diametro
([Regole FIVB 2025-2028](https://www.fivb.com/wp-content/uploads/2025/06/FIVB-Volleyball_Rules2025_2028-FR-v04.pdf)):
poco meno di un pallone da basket. Conta il diametro in pixel, non quello reale: se
la nostra palla sta tra 10 e 40 px siamo nel regime del basket, sotto i 10 px in quello
che TT3D scarta (deduzione mia; il diametro in pixel sui nostri video non l'ho misurato).

**Ricostruzione 3D fisica per tratti.**

- MonoTrack: calibrazione DLT da sei punti (quattro angoli e le cime dei pali); per ogni
  tratto incognite posizione e velocità iniziali e coefficiente di resistenza; perdita =
  errore di riproiezione più distanza dell'inizio dal giocatore che colpisce e della fine
  da chi riceve (piedi, altezza fissa 2 m) più distanza di atterraggio fuori campo.
  Vincoli: altezze 0-3 m, velocità ≤ 120 m/s, inizio nella metà di chi colpisce,
  velocità verso l'avversario. Su dati sintetici l'errore scende da 14,9 a 8,0 cm con i
  vincoli; su dati reali l'errore di riproiezione è 8,1-9,8 px con colpi e volano veri,
  23,3-37,1 px end-to-end ([MonoTrack](https://ar5iv.arxiv.org/html/2204.01899)).
- Volley3D: gravità più resistenza (C_d = 0,035 m⁻¹), il tratto tra due deviazioni va da
  una mano 3D all'altra (o al pavimento), unica incognita la velocità iniziale, nessuna
  perdita di riproiezione. Su VolliQ (4 camere sincronizzate, 10 scambi, 101
  sottotraiettorie): RMSE 3D 0,27 m con resistenza, 0,38 m balistico puro, 1,56 m per
  MonoTrack ([Volley3D](https://arxiv.org/html/2610.38358)).
- Dong et al.: gravità più resistenza, modello aggiornato quando il residuo supera il
  raggio apparente, correzione con le caratteristiche tipiche di ogni azione (battuta,
  ricezione, attacco). Una camera in tribuna, 1080p a 60 fps, 20 sequenze, verità da un
  sistema multi-camera: 73,3% dei fotogrammi entro 150 mm, 82,5% entro 200 mm; la
  deriva del modello piccola resta un limite dichiarato
  ([MVA 2021](https://www.mva-org.jp/Proceedings/2021/papers/P1-14.pdf)).
- Chen et al. (pallavolo, 2012) fanno tracking e approssimazione 3D con fisica e
  calibrazione da una camera fissa "dietro il campo", il nostro stesso punto di vista
  ([NYCU, scheda](https://ir.lib.nycu.edu.tw/handle/11536/16404); DOI
  [10.1007/s11042-011-0833-y](https://doi.org/10.1007/s11042-011-0833-y)). Non ho potuto
  leggere il testo completo: metodo e numeri non verificati.

**Grammatica e giocatori.**

- MonoTrack assegna i giocatori a vicino/lontano per distanza dalla camera, tiene solo le
  pose coi piedi dentro il campo, e l'alternanza imposta dalla decodifica corregge gli
  errori singoli ([MonoTrack](https://ar5iv.arxiv.org/html/2204.01899)).
- Il nostro tracker: lato della palla per voto su 0,25 s di proiezioni sul campo, con
  zona morta di 0,35 m attorno alla rete e pesi che riducono le rilevazioni col raggio
  lontano dalla mediana locale; quando due giocatori di lati opposti sono quasi
  ugualmente plausibili decide il possesso (al massimo tre tocchi, dopo la battuta il
  lato cambia) ([`src/player_contacts.py`](https://github.com/asigatchov/fast-volleyball-tracking-inference/blob/main/src/player_contacts.py),
  [PLAYER_CONTACTS.md](https://github.com/asigatchov/fast-volleyball-tracking-inference/blob/main/PLAYER_CONTACTS.md)).
  La proiezione sul pavimento è la stessa che sul nostro video ha dato 69/132.
- Volley3D attribuisce il tocco alla giocatrice col polso più vicino alla palla, pesato
  sulla velocità del polso: 85,3% di giocatrici giuste con traccia vera, 83,3% con traccia
  predetta ([Volley3D](https://arxiv.org/html/2610.38358)).
- Il dataset Tennis di E2E-Spot ha il lato nella classe (servizio, colpo, rimbalzo, ognuno
  vicino/lontano) e E2E-Spot arriva a 96,1 mAP a ±1 fotogramma dal solo video
  ([E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213)): da riprese televisive
  il lato si impara dai fotogrammi, con 19 partite di training.

**Attraversamento della rete.** Nessuna fonte trovata stima esplicitamente l'istante in cui
la palla attraversa il piano della rete da una camera dietro il fondo. Ci si avvicina il
vincolo di MonoTrack (velocità iniziale verso l'avversario) e il campo `ball_above_net`
del nostro tracker.

### Cosa ne ricavo per noi (proposte)

- La proiezione sul pavimento fallisce per un motivo geometrico: la palla in aria sta
  sopra il campo e la sua proiezione cade più lontano lungo il raggio visivo. Con la
  camera dietro il fondo il raggio visivo è quasi parallelo alla lunghezza del campo, cioè
  proprio la direzione che decide il lato. Il raggio apparente misura la distanza lungo
  lo stesso raggio, quindi è l'indizio giusto per questa camera.
- Ordine di grandezza (calcolo mio): Z = f·D/d, l'errore relativo su Z è circa quello su
  d. Con d = 10 px e 1 px di errore si sbaglia il 10% della distanza: a 15 m sono 1,5 m,
  abbastanza per il lato tranne vicino alla rete. Il 10% è anche l'errore riportato su
  basket.
- Secondo indizio, indipendente dal raggio: in un tratto di volo libero l'accelerazione
  verticale apparente in pixel è circa f·g/Z (approssimazione mia, peggiora con la
  camera inclinata e coi tratti lunghi in profondità). La gravità in pixel calcolata alla
  rete in `gravita_px` può diventare un confronto: un tratto che accelera meno della
  gravità alla rete è oltre la rete. La versione completa è l'adattamento fisico di
  MonoTrack o Dong et al., con la calibrazione che abbiamo da #19.
- Terzo indizio, la grammatica: chi batte è noto (rally point: il vincitore dello scambio
  precedente), il lato cambia solo con un passaggio della rete, al massimo tre tocchi per
  lato più il muro. Una decodifica che sceglie la sequenza di lati più coerente con gli
  indizi di ogni tocco corregge gli errori isolati, come in MonoTrack.
- Quarto indizio, le teste di RAVEL-VB: una testa vicina alla palla al tocco, con
  dimensione compatibile con quella della palla, indica il lato. Da provare dopo i primi
  tre.
- Il 96/116 col raggio è stato deciso dopo aver visto i dati: la regola va fissata e
  misurata sulla partita 2.

## 4. Metodi e dataset

### Metodi

| Progetto / articolo | Cosa fa | Licenza | Ultimo aggiornamento | Uso per noi |
|---|---|---|---|---|
| [MonoTrack](https://ar5iv.arxiv.org/html/2204.01899) (badminton, CVPRW 2022) | Colpi da traccia + pose + campo con GRU, decodifica vincolata con programmazione dinamica, ricostruzione 3D fisica per tratti | Articolo; codice non trovato | 2022-04 | **Riferimento principale per la decodifica globale** e per i vincoli della ricostruzione 3D. Il vincolo di 0,5 s tra colpi non vale per il muro |
| [Volley3D](https://arxiv.org/html/2610.38358) (pallavolo, arXiv) | TCN 1D sulla traccia per le deviazioni, attribuzione al polso più vicino, 3D ancorato alle mani | Articolo CC BY-NC-SA 4.0; [repo](https://github.com/Noambsf/Volley3D) senza licenza e senza codice | articolo 2026-10-28, repo 2026-11-04 | Il confronto più vicino al nostro caso: numeri dell'euristica contro la TCN su pallavolo. Non c'è codice da copiare |
| [TT3D](https://arxiv.org/html/2504.10035) (tennistavolo, CVSports 2025) | Segmentazione della traccia in parabole con programmazione dinamica, rimbalzo all'intersezione, 3D con fisica e spin | [Codice](https://github.com/cogsys-tuebingen/tt3d) MIT | 2026-10-15 | **Codice copiabile** (con nota di copyright) per la segmentazione parabolica e l'istante all'intersezione |
| [Dong et al., MVA 2021](https://www.mva-org.jp/Proceedings/2021/papers/P1-14.pdf) (pallavolo) | 3D da una camera con gravità e resistenza, aggiornamento quando il residuo supera il raggio apparente | Articolo; codice non menzionato | 2021-07 | Soglia di rottura in raggi apparenti |
| [Chen et al. 2012](https://doi.org/10.1007/s11042-011-0833-y) (pallavolo) | Tracking e 3D da camera fissa dietro il campo | Articolo, testo non letto | 2012 | Da leggere se si trova il testo: stessa inquadratura |
| [Van Zandycke e De Vleeschouwer](https://ar5iv.labs.arxiv.org/html/2204.00003) (basket, CVSports 2022) | Posizione 3D dal diametro apparente stimato da una CNN | [Codice deepsport](https://github.com/gabriel-vanzandycke/deepsport) **CC BY-NC-ND 4.0** | 2026-05-11 | Solo l'idea e i numeri. NoDerivatives: una versione modificata non si può distribuire, quindi niente codice nel repo pubblico |
| [E2E-Spot](https://ar5iv.labs.arxiv.org/html/2207.10213) (ECCV 2022) | Event spotting preciso dal video (RegNet + GRU); dataset Tennis con lato nella classe | [Codice](https://github.com/jhong93/spot) BSD-3-Clause | 2023-02-15 | NMS per classe; candidato se si passa a un modello sul video attorno alla rottura |
| [T-DEED](https://arxiv.org/html/2404.05392v2) (CVSports 2024) | Event spotting preciso, feature che distinguono fotogrammi vicini, Soft-NMS | [Codice](https://github.com/arturxe2/T-DEED) **GPL-3.0** | 2026-01-07 | Idee. Copiare codice obbligherebbe a distribuire il derivato sotto GPL; il repo oggi non ha licenza |
| [STES / VNL-STES](https://hoangqnguyen.github.io/stes) (pallavolo, CVPRW 2025) | Istante e posizione (x, y) di battuta, ricezione, alzata, schiacciata, muro, punto | [Codice](https://github.com/hoangqnguyen/spot) BSD-3-Clause (fork di E2E-Spot) | 2026-03-18 | Unico spotting preciso su pallavolo con codice. Riprese televisive VNL |
| [F3Set / F3ED](https://arxiv.org/html/2504.08222v2) (ICLR 2025) | Eventi di tennis (e badminton, tennistavolo) al fotogramma, ±1 fotogramma | [Repo](https://github.com/F3Set/F3Set) **senza licenza** | 2024-10-02 | Solo lettura. Dataset: solo URL di YouTube, uso accademico |
| [TTNet](https://ar5iv.labs.arxiv.org/html/2004.09927) (tennistavolo, CVPRW 2020) | Rimbalzi e rete dal video con esempi negativi espliciti | Articolo; [reimplementazione](https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch) senza licenza | 2024-08-02 | Idea degli esempi negativi e dell'obiettivo sfumato |
| [Chen et al. 2023, CoachAI](https://ar5iv.labs.arxiv.org/html/2308.12645) (badminton) | Colpi da clip video (X3D) con classe del giocatore; scarta l'euristica delle inversioni | [Codice](https://github.com/jean50621/Badminton_Challenge) CC0-1.0 | 2023-05-24 | Esempio di lato dal video invece che dalla traccia. Nessun numero sul colpo nell'articolo |
| [Where Is The Ball](https://arxiv.org/html/2506.05763v1) (CVSports 2025) | 3D da traccia 2D con LSTM addestrate su simulazioni | Articolo CC BY 4.0; codice "coming soon" | 2025-06 | Riferimento se si vuole un sollevamento 3D appreso |
| [Uplifting Table Tennis](https://arxiv.org/html/2511.20250) (WACV 2026) | 3D e spin da traccia 2D con transformer addestrato su simulazioni | [Codice](https://github.com/KieDani/UpliftingTableTennis) **GPL-3.0** | 2026-02-06 | Solo idee |
| [TrackNetV3](https://github.com/qaz812345/TrackNetV3) (badminton) | Tracker con inpainting dei tratti occlusi | MIT con estensione ai pesi (GitHub la legge come NOASSERTION) | 2026-09-03 | Inpainting dei buchi prima di cercare le rotture |
| [WASB](https://github.com/nttcom/WASB-SBDT) (BMVC 2023) | Tracker della palla per cinque sport, pallavolo inclusa | MIT | 2023-11-23 | Alternativa al tracker, se VballNet non basta |
| [fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference) | Il nostro tracker: contatti da rottura + giocatrici, possessi, testa "touch" RAVEL-VB-013 | MIT, con un paragrafo che estende la licenza ai pesi: GitHub ora la legge come NOASSERTION | 2026-11-04 | Già in uso. La licenza resta MIT nel testo |
| [PathFinder](https://arxiv.org/abs/2309.14753) (pallavolo, ICDM workshop 2023) | Traiettoria dell'alzata da una camera dietro una squadra, filtro a maggioranza sulla direzione | [Repo](https://github.com/volleyIEEE/VolleyStats) **senza licenza** | 2023-07-18 | Poco: scarta esplicitamente la segmentazione in parabole |

### Dataset

| Dataset | Cosa contiene | Licenza | Uso per noi |
|---|---|---|---|
| Volleyball (Ibrahim) + annotazioni di Volley3D | Più di 1.600 cambi di traiettoria, 654 collisioni nel test | Annotazioni "saranno pubbliche"; non rilasciate al 2026-11-10 | Il pre-training più vicino per un segmentatore della traccia, se esce |
| [VNL](https://hoangqnguyen.github.io/stes) (STES) | 8 partite VNL 2022-2023, 1.028 scambi, 6.137 eventi con fotogramma e (x, y), 25 fps | **Non dichiarata**; video "con permesso di Volleyball World" | Solo con chiarimento degli autori. Riprese televisive |
| Tennis (E2E-Spot) | 3.345 punti da 28 partite, 33.791 eventi in 6 classi col lato | Codice BSD-3; licenza dei dati non trovata | Solo come riferimento per il formato "lato nella classe" |
| [OpenTTGames](https://lab.osai.ai/) | Tennistavolo a 120 fps, 4.271 eventi (rimbalzo, rete, vuoto) | CC BY-NC-SA 4.0 | Fuori dominio; utile solo per provare idee sugli eventi negativi |
| [CoachAI 2023 track 1](https://sites.google.com/view/coachai-challenge-2023/tasks/track1) | Badminton BWF, 30 fps, fotogramma del colpo, chi colpisce, posizioni | Non dichiarata; accesso con modulo | No |
| [Shuttlecock Trajectory Dataset](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw) | 26 video, solo posizione e visibilità del volano | Non dichiarata | No |
| [ShuttleSet](https://arxiv.org/abs/2306.04948) | 36.492 colpi di badminton annotati | Codice della repo [CoachAI-Projects](https://github.com/wywyWang/CoachAI-Projects) MIT; dati da verificare | No |
| [DeepSport / Ball 3D localization](https://github.com/DeepSportradar/ball-3d-localization-challenge) | Basket, diametro della palla in pixel | Repo senza licenza (la sezione "License" è nell'indice ma manca); versione estesa sotto NDA | No: solo per l'idea del diametro |

## 5. Domande aperte

- **Muro**: nessuna fonte con un metodo dalla traiettoria. Le proposte della sezione 2
  sono da misurare; con 11 muri etichettati la misura sarà molto incerta.
- **Tocco della rete**: nessuna fonte lo tratta. Non sappiamo quante delle nostre rotture
  inventate siano palle che toccano la rete o il nastro.
- **Attraversamento del piano della rete** da una camera dietro il fondo: non trovato.
- **Chen et al. 2012** usa la nostra stessa inquadratura, ma non ho letto il testo
  completo (accesso chiuso, PDF del repository NYCU non scaricabile).
- **Volley3D** è un preprint di ottobre 2026, non revisionato e senza codice; i suoi numeri
  sono su riprese televisive, con tolleranze di 1-3 fotogrammi a una frequenza che la
  parte letta non indica.
- Le tolleranze non sono confrontabili: MonoTrack non la dichiara, CoachAI usa ±2
  fotogrammi a 30 fps, E2E-Spot e F3Set ±1, VNL-STES 0-4 a 25 fps; noi ±0,3 s, circa 9
  fotogrammi. I numeri delle fonti non si confrontano coi nostri.
- **Licenza del nostro repo**: non c'è un file LICENSE. Copiare codice MIT o BSD richiede
  di tenerne la nota di copyright; per GPL o CC BY-NC-ND la risposta è no finché la
  questione non è decisa.

## Fonti

Articoli:

- Liu, Wang. MonoTrack: Shuttle trajectory reconstruction from monocular badminton video. CVPRW 2022. https://ar5iv.arxiv.org/html/2204.01899
- Boussouf, Maglo, Chaouch, Saito. Volley3D: Unified 3D Full-Sequence Reconstruction from Monocular Broadcasts. arXiv 2610.38358, 2026. https://arxiv.org/html/2610.38358 — pagina: https://noambsf.github.io/Volley3D
- Gossard, Ziegler, Zell. TT3D: Table Tennis 3D Reconstruction. CVSports 2025. https://arxiv.org/html/2504.10035
- Dong, Cheng, Ikenaga. Multi-physical and Temporal Feature Based Self-correcting Approximation Model for Monocular 3D Volleyball Trajectory Analysis. MVA 2021. https://www.mva-org.jp/Proceedings/2021/papers/P1-14.pdf
- Chen, Tsai, Lee, Yu. Ball tracking and 3D trajectory approximation with applications to tactics analysis from single-camera volleyball sequences. Multimedia Tools and Applications 60(3), 2012. https://doi.org/10.1007/s11042-011-0833-y — scheda: https://ir.lib.nycu.edu.tw/handle/11536/16404
- Van Zandycke, De Vleeschouwer. 3D Ball Localization From A Single Calibrated Image. CVSports 2022. https://ar5iv.labs.arxiv.org/html/2204.00003
- Ponglertnapakorn, Suwajanakorn. Where Is The Ball: 3D Ball Trajectory Estimation From 2D Monocular Tracking. CVSports 2025. https://arxiv.org/html/2506.05763v1 — pagina: https://where-is-the-ball.github.io/
- Kienzle et al. Uplifting Table Tennis. arXiv 2511.20250. https://arxiv.org/html/2511.20250
- Hong et al. Spotting Temporally Precise, Fine-Grained Events in Video (E2E-Spot). ECCV 2022. https://ar5iv.labs.arxiv.org/html/2207.10213
- Xarles et al. T-DEED. CVSports 2024. https://arxiv.org/html/2404.05392v2
- Nguyen et al. VNL-STES. CVPRW 2025. https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Nguyen_VNL-STES_A_Benchmark_Dataset_and_Model_for_Spatiotemporal_Event_Spotting_CVPRW_2025_paper.html — pagina: https://hoangqnguyen.github.io/stes
- Liu et al. F3Set. ICLR 2025. https://arxiv.org/html/2504.08222v2
- Voeikov, Falaleev, Baikulov. TTNet. CVPRW 2020. https://ar5iv.labs.arxiv.org/html/2004.09927
- Chen et al. An All Deep System for Badminton Game Analysis. 2023. https://ar5iv.labs.arxiv.org/html/2308.12645
- Xia et al. Advanced Volleyball Stats for All Levels (PathFinder). ICDM workshop 2023. https://arxiv.org/abs/2309.14753
- Wang et al. ShuttleSet. KDD 2023. https://arxiv.org/abs/2306.04948
- Ertner et al. SynthNet (tennis, MMSports 2024): https://researcher.itu.dk/en/publications/synthnet-leveraging-synthetic-data-for-3d-trajectory-estimation-f/ — colpi e rimbalzi con una GRU; numeri non letti.
- Chen, Chu. How it flies and why it flies? Volleyball trajectory segmentation and classification. IEEE TCAS-II 68(5), 2021. https://lida.sport-iat.de/dvv/Record/4068785 — due camere, fuori dal nostro caso.

Repository (licenza e ultimo push dall'API GitHub, 2026-11-10):

- https://github.com/asigatchov/fast-volleyball-tracking-inference (`PLAYER_CONTACTS.md`, `src/player_contacts.py`, `LICENSE`)
- https://github.com/Noambsf/Volley3D
- https://github.com/cogsys-tuebingen/tt3d (`tt3d/traj_seg/segmenter.py`)
- https://github.com/cogsys-tuebingen/blurball
- https://github.com/KieDani/UpliftingTableTennis
- https://github.com/jhong93/spot (`util/eval.py`)
- https://github.com/hoangqnguyen/spot
- https://github.com/arturxe2/T-DEED
- https://github.com/F3Set/F3Set
- https://github.com/qaz812345/TrackNetV3
- https://github.com/nttcom/WASB-SBDT
- https://github.com/gabriel-vanzandycke/deepsport
- https://github.com/DeepSportradar/ball-3d-localization-challenge
- https://github.com/jean50621/Badminton_Challenge
- https://github.com/wywyWang/CoachAI-Projects
- https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch
- https://github.com/volleyIEEE/VolleyStats

Dataset e regole:

- OpenTTGames: https://lab.osai.ai/
- CoachAI Badminton Challenge 2023, track 1: https://sites.google.com/view/coachai-challenge-2023/tasks/track1
- Shuttlecock Trajectory Dataset: https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw
- Regole FIVB 2025-2028 (francese), regola 3.1: https://www.fivb.com/wp-content/uploads/2025/06/FIVB-Volleyball_Rules2025_2028-FR-v04.pdf
