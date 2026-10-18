"""Regole che legano gli scambi tra loro: punteggio dei set e della partita, e
chi batte a ogni scambio.

Le squadre qui sono "nostra" e "avversaria". Gli scambi parlano di lati
dell'inquadratura; `lato_nostro` dice, set per set, in che metà campo gioca la
nostra squadra.
"""

from dataclasses import dataclass, field

from lib.scambio import Problema, Scambio

PUNTI_SET = 25
PUNTI_TIE_BREAK = 15
SET_DECISIVO = 5
SET_PER_VINCERE = 3

NOSTRA = "nostra"
AVVERSARIA = "avversaria"


@dataclass
class Punteggio:
    # (nostri, avversari) per ogni set, in ordine
    set: list[tuple[int, int]]
    # set vinti (nostri, avversari)
    partita: tuple[int, int]
    problemi: list[Problema] = field(default_factory=list)


def set_chiuso(numero: int, a: int, b: int) -> bool:
    obiettivo = PUNTI_TIE_BREAK if numero == SET_DECISIVO else PUNTI_SET
    return max(a, b) >= obiettivo and abs(a - b) >= 2


def punteggio(scambi: list[Scambio], lato_nostro: dict[int, str]) -> Punteggio:
    punti: dict[int, list[int]] = {}
    problemi = []
    precedente: Scambio | None = None
    vinti = [0, 0]
    for s in scambi:
        if max(vinti) >= SET_PER_VINCERE:
            problemi.append(Problema(s.set, s.indice, "scambio dopo la fine della partita"))
            continue
        set_corrente = punti.setdefault(s.set, [0, 0])
        if set_chiuso(s.set, *set_corrente):
            problemi.append(Problema(s.set, s.indice, "scambio dopo la fine del set"))
            continue
        # Rally point system: chi vince lo scambio batte il successivo. Tra un set
        # e l'altro no, perché le squadre cambiano campo e la prima battuta alterna.
        if precedente is not None and precedente.set == s.set and s.tocchi:
            if s.tocchi[0].lato != precedente.lato_vincitore:
                problemi.append(Problema(s.set, s.indice, "non batte chi ha vinto lo scambio precedente"))
        precedente = s
        if s.lato_vincitore == lato_nostro[s.set]:
            set_corrente[0] += 1
        else:
            set_corrente[1] += 1
        if set_chiuso(s.set, *set_corrente):
            vinti[0 if set_corrente[0] > set_corrente[1] else 1] += 1

    per_set = [tuple(punti[n]) for n in sorted(punti)]
    return Punteggio(set=per_set, partita=tuple(vinti), problemi=problemi)


@dataclass(frozen=True)
class Sostituzione:
    """Un cambio come lo riporta il referto."""

    set: int
    squadra: str
    esce: int
    entra: int
    # (punti della squadra, punti avversari) nel momento del cambio
    punteggio: tuple[int, int]


def battitrici(
    scambi: list[Scambio],
    lato_nostro: dict[int, str],
    formazioni: dict[tuple[int, str], tuple[int, ...]],
    sostituzioni: list[Sostituzione] = (),
) -> list[int]:
    """Numero di chi batte a ogni scambio, dalle formazioni di partenza (posizioni I-VI)."""
    numeri = []
    in_campo = {chiave: list(formazione) for chiave, formazione in formazioni.items()}
    rotazioni: dict[tuple[int, str], int] = {}
    punti: dict[tuple[int, str], int] = {}
    precedente: tuple[int, str] | None = None
    for s in scambi:
        squadra = NOSTRA if s.tocchi[0].lato == lato_nostro[s.set] else AVVERSARIA
        chiave = (s.set, squadra)
        # Il referto registra il cambio col punteggio del momento: si applica
        # prima dello scambio giocato su quel punteggio. Chi entra prende la
        # posizione di chi esce.
        for c in sostituzioni:
            if c.set != s.set:
                continue
            avversaria = AVVERSARIA if c.squadra == NOSTRA else NOSTRA
            if c.punteggio == (punti.get((s.set, c.squadra), 0), punti.get((s.set, avversaria), 0)):
                formazione = in_campo[(s.set, c.squadra)]
                formazione[formazione.index(c.esce)] = c.entra
        # Cambio palla: la squadra che conquista la battuta ruota. All'inizio del
        # set non ruota nessuno, batte chi è in I.
        if precedente is not None and precedente[0] == s.set and precedente != chiave:
            rotazioni[chiave] = rotazioni.get(chiave, 0) + 1
        formazione = in_campo[chiave]
        numeri.append(formazione[rotazioni.get(chiave, 0) % len(formazione)])
        precedente = chiave
        vincitrice = NOSTRA if s.lato_vincitore == lato_nostro[s.set] else AVVERSARIA
        punti[(s.set, vincitrice)] = punti.get((s.set, vincitrice), 0) + 1
    return numeri
