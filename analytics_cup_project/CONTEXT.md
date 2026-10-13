# Analytics Cup

Preparazione alla SkillCorner X PySport Analytics Cup su tracking broadcast open
data: si sceglie una domanda di ricerca e la si porta a una submission.

## Language

**Pista**:
Una domanda di ricerca candidata, descritta da dato che serve, cosa la minaccia,
perché regge e figura finale. Le piste vivono in `plans/01`.
_Avoid_: zona di lavoro, filone, idea

## Valutazione difensiva

**Coinvolgimento**:
Quanto un difensore era vicino a un passaggio avversario (passatore, ricevitore,
linea di passaggio), misurato sulle posizioni del tracking. Chi era lontano ha
coinvolgimento zero, anche se avrebbe dovuto esserci.
_Avoid_: responsabilità (è un'altra cosa), involvement, pressione

**Responsabilità**:
Quanto un passaggio avversario spettava a un difensore: il coinvolgimento che ci
si aspetterebbe da lui in quella situazione, anche se non c'era.
_Avoid_: colpa, blame, coinvolgimento, marcatura (che è un'assegnazione, non un peso)

**Valore concesso**:
Quanta probabilità di segnare ha guadagnato la squadra in possesso in una
situazione d'attacco. Ripartito per coinvolgimento o per responsabilità, diventa
il valore concesso da un difensore.
_Avoid_: colpa, danno, conceded threat
