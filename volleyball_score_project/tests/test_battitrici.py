"""Test di chi batte a ogni scambio, ricavato dalla rotazione.

Le formazioni elencano le sei titolari nelle posizioni I-VI: in I c'è chi batte.
Quando una squadra conquista la battuta in ricezione ruota di una posizione, e
batte chi era in II. Il libero non batte e non compare nelle formazioni.
"""

from lib.partita import AVVERSARIA, NOSTRA, Sostituzione, battitrici
from lib.scambio import LONTANO, VICINO, Scambio, Tocco

LATO = {"V": VICINO, "L": LONTANO}

FORMAZIONI = {
    (1, NOSTRA): (1, 2, 3, 4, 5, 6),
    (1, AVVERSARIA): (11, 12, 13, 14, 15, 16),
}


def set_giocato(numero, vincitori, prima_battuta="V"):
    """Scambi di un set da una stringa di vincitori; batte chi ha vinto il precedente."""
    scambi = []
    battuta = prima_battuta
    for i, vincitore in enumerate(vincitori, start=1):
        battitrice = Tocco(t=float(i), lato=LATO[battuta], tipo="S")
        scambi.append(Scambio(set=numero, indice=i, tocchi=(battitrice,), lato_vincitore=LATO[vincitore]))
        battuta = vincitore
    return scambi


def test_finche_la_squadra_fa_punto_in_battuta_batte_sempre_la_stessa():
    scambi = set_giocato(1, "VVV")

    assert battitrici(scambi, lato_nostro={1: VICINO}, formazioni=FORMAZIONI) == [1, 1, 1]


def test_dopo_un_cambio_palla_si_ruota():
    # Batte la nostra 1; le avversarie conquistano la battuta e ruotano (12),
    # poi la riconquistiamo noi e ruotiamo (2).
    scambi = set_giocato(1, "VLLVV")

    assert battitrici(scambi, lato_nostro={1: VICINO}, formazioni=FORMAZIONI) == [1, 1, 12, 12, 2]


def test_una_sostituzione_cambia_chi_batte_in_quella_posizione():
    # Sul 2-0 per noi entra la 7 al posto della 1, che era al servizio.
    scambi = set_giocato(1, "VVV")
    cambio = Sostituzione(set=1, squadra=NOSTRA, esce=1, entra=7, punteggio=(2, 0))

    numeri = battitrici(scambi, lato_nostro={1: VICINO}, formazioni=FORMAZIONI, sostituzioni=[cambio])

    assert numeri == [1, 1, 7]
