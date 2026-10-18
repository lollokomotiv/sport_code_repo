"""Test della grammatica dello scambio: il tipo di ogni tocco da lati e tempi.

È la proposta che la pagina di etichettatura mostra e che la pipeline userà sui
tocchi trovati dal tracker. Muro e difesa si distinguono dal tempo trascorso
dall'attacco: la soglia è provvisoria, quindi i casi qui sono netti (0,1 s
contro 0,8 s) e non la fissano.
"""

from lib.scambio import LONTANO, VICINO, Tocco, proponi_tipi

LATO = {"V": VICINO, "L": LONTANO}


def tocchi(sequenza):
    """Tocchi da una sequenza tipo "V:0 L:1.2": lato e istante in secondi."""
    return [
        Tocco(t=float(istante), lato=LATO[lato])
        for lato, istante in (codice.split(":") for codice in sequenza.split())
    ]


def test_battuta_ricezione_alzata_attacco():
    assert proponi_tipi(tocchi("V:0 L:1 L:2 L:3")) == ["S", "R", "E", "A"]


def test_tocco_subito_dopo_l_attacco_dall_altra_parte_e_un_muro():
    assert proponi_tipi(tocchi("V:0 L:1 L:2 L:3 V:3.1")) == ["S", "R", "E", "A", "B"]


def test_tocco_dall_altra_parte_lontano_nel_tempo_dall_attacco_e_una_difesa():
    assert proponi_tipi(tocchi("V:0 L:1 L:2 L:3 V:3.8")) == ["S", "R", "E", "A", "D"]


def test_il_muro_non_conta_tra_i_tre_tocchi():
    sequenza = "V:0 L:1 L:2 L:3 V:3.1 V:4 V:5 V:6"
    assert proponi_tipi(tocchi(sequenza)) == ["S", "R", "E", "A", "B", "D", "E", "A"]


def test_secondo_tocco_che_passa_la_rete_e_un_attacco_di_seconda():
    assert proponi_tipi(tocchi("V:0 L:1 L:2 V:3")) == ["S", "R", "A", "D"]
