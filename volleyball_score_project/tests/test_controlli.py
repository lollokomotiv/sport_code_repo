"""Test dei controlli su uno scambio etichettato a mano.

Segnalano le sequenze che il regolamento non permette, così un errore di
etichettatura si vede mentre si etichetta e non quando si misura.
"""

from lib.scambio import LONTANO, VICINO, Scambio, Tocco, controlla_scambio

LATO = {"V": VICINO, "L": LONTANO}


def scambio(sequenza, vince="V"):
    """Scambio da una sequenza tipo "SV RL EL AL": tipo del tocco e lato."""
    tocchi = tuple(
        Tocco(t=float(i), lato=LATO[codice[1]], tipo=codice[0])
        for i, codice in enumerate(sequenza.split())
    )
    return Scambio(set=1, indice=7, tocchi=tocchi, lato_vincitore=LATO[vince])


def dove(problemi):
    return [(p.scambio, p.tocco) for p in problemi]


def test_quattro_tocchi_dalla_stessa_parte_sono_un_problema():
    assert dove(controlla_scambio(scambio("SV RL EL EL AL"))) == [(7, 5)]


def test_muro_piu_tre_tocchi_dalla_stessa_parte_e_regolare():
    assert controlla_scambio(scambio("SV RL EL AL BV DV EV AV")) == []


def test_ricezione_non_preceduta_da_una_battuta_e_un_problema():
    # Il quinto tocco segue un attacco: è una difesa, non una ricezione.
    assert dove(controlla_scambio(scambio("SV RL EL AL RV"))) == [(7, 5)]


def test_muro_non_preceduto_da_un_attacco_e_un_problema():
    # Murare la battuta non è permesso.
    assert dove(controlla_scambio(scambio("SV BL"))) == [(7, 2)]


def test_tocco_dopo_la_fine_dello_scambio_e_un_problema():
    # Come lo scambio 1 della prima giornata: battuta a 27,2 s, fine segnata a 24,6 s.
    s = Scambio(set=1, indice=1, tocchi=(Tocco(t=27.2, lato=VICINO, tipo="S"),),
                lato_vincitore=LONTANO, t_fine=24.6)

    assert dove(controlla_scambio(s)) == [(1, 1)]


def test_scambio_che_non_comincia_con_una_battuta_e_un_problema():
    assert dove(controlla_scambio(scambio("DL EL AL"))) == [(7, 1)]
