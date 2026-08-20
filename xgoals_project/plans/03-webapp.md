# 03 — Webapp

**Obiettivo:** un'interfaccia dove generare le visualizzazioni (shot map, assist map, metriche) a partire dai dati di una partita.

**Stato:** ⚪ Da iniziare.

---

## Decisione da prendere per prima

Due prodotti molto diversi si nascondono dietro "webapp":

**A. Strumento personale** — carichi i JSON StatsBomb di una partita, ottieni le viz.
Pubblico: io. Valore: comodità.

**B. Demo pubblica deployata** — match open data StatsBomb già caricati, chiunque apre un URL, sceglie una partita, esplora le visualizzazioni.
Pubblico: chiunque riceva il link, incluso chi legge un CV. Valore: dimostrativo.

B costa poco più di A (la differenza è precaricare i dati e deployare) ma vale molto di più come portfolio. **Raccomandazione: puntare a B**, con A come sottoinsieme naturale.

## Step

- [ ] Decidere A o B
- [ ] Verificare la licenza StatsBomb open data per la ridistribuzione pubblica (uso non commerciale con attribuzione — da confermare prima di deployare)
- [ ] Scegliere lo stack. L'opzione a minor attrito: le funzioni di `inference/score_match.py` sono già pure (DataFrame in → figura out), quindi basta un layer sottile sopra
- [ ] Precaricare un set di match significativi (finali, partite note) invece di tutti i 326
- [ ] Deploy con URL stabile
- [ ] Aggiungere il link al README e al CV

## Nota

Il vero collo di bottiglia non è il codice della webapp: è avere sotto metriche che reggono a un esame. Una demo bella su un xG scalibrato è peggio di nessuna demo, perché chi la valuta se ne accorge. Ordine corretto: 01 → 02 → 03.
