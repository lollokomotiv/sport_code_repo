"""Test dell'esito di uno scambio: chi fa il punto, e con quale tocco.

L'esito si ricava dall'ultimo tocco e da chi vince lo scambio. I casi sono
quelli dello scouting DataVolley: una ricezione sbagliata è un ace di chi batte,
una difesa che non tiene è un punto in attacco dell'altra parte.
"""

from lib.scambio import LONTANO, VICINO, Scambio, Tocco, esito

LATO = {"V": VICINO, "L": LONTANO}


def scambio(sequenza, vince):
    """Scambio da una sequenza tipo "SV RL EL AL": tipo del tocco e lato."""
    tocchi = tuple(
        Tocco(t=float(i), lato=LATO[codice[1]], tipo=codice[0])
        for i, codice in enumerate(sequenza.split())
    )
    return Scambio(set=1, indice=1, tocchi=tocchi, lato_vincitore=LATO[vince])


def descrivi(e):
    return (e.tocco.tipo, e.tocco.lato, e.modo)


def test_ricezione_sbagliata_e_un_ace_di_chi_batte():
    assert descrivi(esito(scambio("SV RL", vince="V"))) == ("S", VICINO, "punto")


def test_difesa_che_non_tiene_e_un_punto_in_attacco_dell_altra_parte():
    assert descrivi(esito(scambio("SV RL EL AL DV", vince="L"))) == ("A", LONTANO, "punto")


def test_mani_out_e_un_punto_in_attacco():
    # La palla tocca il muro del lato vicino ed esce: il punto è dell'attacco.
    assert descrivi(esito(scambio("SV RL EL AL BV", vince="L"))) == ("A", LONTANO, "punto")


def test_attacco_fuori_e_un_errore_in_attacco():
    assert descrivi(esito(scambio("SV RL EL AL", vince="V"))) == ("A", LONTANO, "errore")


def test_muro_che_chiude_lo_scambio_e_un_punto_a_muro():
    assert descrivi(esito(scambio("SV RL EL AL BV", vince="V"))) == ("B", VICINO, "punto")
