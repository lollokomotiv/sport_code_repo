"""Test del punteggio di una partita, ricostruito dagli scambi.

Regolamento FIPAV: al meglio dei 5 set, set a 25, quinto set a 15, vantaggio di 2.
I punteggi attesi sono scritti a mano, non ricalcolati come fa il codice.
"""

from lib.partita import punteggio
from lib.scambio import LONTANO, VICINO, Scambio, Tocco

LATO = {"V": VICINO, "L": LONTANO}


def set_giocato(numero, vincitori, prima_battuta="V"):
    """Scambi di un set da una stringa di vincitori ("V" vicino, "L" lontano).

    Ogni scambio ha un solo tocco, la battuta, e batte chi ha vinto lo scambio
    precedente: è il caso regolare.
    """
    scambi = []
    battuta = prima_battuta
    for i, vincitore in enumerate(vincitori, start=1):
        battitrice = Tocco(t=float(i), lato=LATO[battuta], tipo="S")
        scambi.append(Scambio(set=numero, indice=i, tocchi=(battitrice,), lato_vincitore=LATO[vincitore]))
        battuta = vincitore
    return scambi


def test_set_vinto_25_23():
    scambi = set_giocato(1, "VL" * 23 + "VV")

    risultato = punteggio(scambi, lato_nostro={1: VICINO})

    assert risultato.set == [(25, 23)]
    assert risultato.partita == (1, 0)
    assert risultato.problemi == []


def test_scambio_dopo_il_25_20_e_un_problema():
    # 25-20 al 45° scambio: il set è finito, il 46° non doveva esserci.
    scambi = set_giocato(1, "VL" * 20 + "V" * 5 + "V")

    risultato = punteggio(scambi, lato_nostro={1: VICINO})

    assert [(p.set, p.scambio) for p in risultato.problemi] == [(1, 46)]


def test_ai_vantaggi_26_24_e_legale():
    scambi = set_giocato(1, "VL" * 24 + "VV")

    risultato = punteggio(scambi, lato_nostro={1: VICINO})

    assert risultato.set == [(26, 24)]
    assert risultato.problemi == []


def test_il_quinto_set_finisce_a_15():
    # Stessa sequenza in ogni set, ma le squadre cambiano campo: 2-2 dopo quattro set.
    quattro_set = [s for n in range(1, 5) for s in set_giocato(n, "VL" * 23 + "VV")]
    # 15-13 al 28° scambio del quinto; il 29° non doveva esserci.
    quinto = set_giocato(5, "VL" * 13 + "VV" + "V")
    lati = {1: VICINO, 2: LONTANO, 3: VICINO, 4: LONTANO, 5: VICINO}

    risultato = punteggio(quattro_set + quinto, lato_nostro=lati)

    assert risultato.set == [(25, 23), (23, 25), (25, 23), (23, 25), (15, 13)]
    assert risultato.partita == (3, 2)
    assert [(p.set, p.scambio) for p in risultato.problemi] == [(5, 29)]


def test_se_non_batte_chi_ha_vinto_lo_scambio_prima_e_un_problema():
    scambi = set_giocato(1, "VVL")
    # Il 2° scambio l'ha vinto il lato vicino, ma il 3° risulta battuto dal lontano.
    terzo = scambi[2]
    scambi[2] = Scambio(set=1, indice=3, tocchi=(Tocco(t=3.0, lato=LONTANO, tipo="S"),),
                        lato_vincitore=terzo.lato_vincitore)

    risultato = punteggio(scambi, lato_nostro={1: VICINO})

    assert [(p.set, p.scambio) for p in risultato.problemi] == [(1, 3)]


def test_dopo_il_terzo_set_vinto_non_ci_sono_altri_scambi():
    tre_set = [s for n in range(1, 4) for s in set_giocato(n, "VL" * 23 + "VV")]
    quarto = set_giocato(4, "V")

    risultato = punteggio(tre_set + quarto, lato_nostro={1: VICINO, 2: VICINO, 3: VICINO, 4: VICINO})

    assert risultato.partita == (3, 0)
    assert [(p.set, p.scambio) for p in risultato.problemi] == [(4, 1)]


def test_un_set_non_concluso_non_conta_come_vinto():
    # Il video può finire a metà set, come quello di esempio.
    scambi = set_giocato(1, "VL" * 10 + "VV")

    risultato = punteggio(scambi, lato_nostro={1: VICINO})

    assert risultato.set == [(12, 10)]
    assert risultato.partita == (0, 0)
