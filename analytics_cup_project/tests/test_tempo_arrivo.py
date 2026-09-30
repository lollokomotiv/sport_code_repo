"""Test del tempo di arrivo (Narizuka et al., modello di Fujimura-Sugihara).

Questi test sono il criterio di stop del `/goal` che implementa `lib/pressione.py`.
Sono stati scritti e approvati **prima** del codice, e il loop non li modifica:
se un test sembra sbagliato, ci si ferma e se ne discute.

Nessun test dipende dai dati SkillCorner. Verificano la fisica del modello su
casi con risposta nota, così un errore nell'implementazione non può essere
nascosto da un risultato che "sembra plausibile".

Il modello (scheda: notes/letteratura/pressione-tempo-arrivo.md):

    c_p(t) = x_p(0) + (1 - exp(-a t)) / a * v_p(0)
    r_p(t) = V_max * (t - (1 - exp(-a t)) / a)

Il tempo di arrivo in x è il primo t per cui ||x - c_p(t)|| <= r_p(t).
"""

import math

import numpy as np
import pytest

from lib.pressione import ALPHA, V_MAX, tau_opp, tempo_arrivo, velocita

# --------------------------------------------------------------------------
# Decisione presa prima del loop: come si ricavano le velocità.
#
# None  = derivata centrale sulle posizioni, senza lisciatura aggiuntiva.
# int   = finestra Savitzky-Golay in frame (dispari, >= 5), grado 2.
#
# Misurato sulla partita 1886347: le traiettorie SkillCorner arrivano già
# lisciate (accelerazione al 99° percentile 4,5 m/s² senza lisciatura, nessuna
# velocità sopra i 10 m/s), e una finestra in più cambia i numeri di pochi
# centesimi. Vedi notes/dati/tracking-extrapolated.md.
# --------------------------------------------------------------------------
FINESTRA_LISCIATURA = None

TOL = 1e-4


def _fermo(d, alpha=1.0, vmax=10.0):
    """Tempo di arrivo di un giocatore fermo a distanza d, risolvendo il modello."""
    from scipy.optimize import brentq

    if d == 0:
        return 0.0
    return brentq(lambda t: vmax * (t - (1 - math.exp(-alpha * t)) / alpha) - d, 1e-12, 120)


# --------------------------------------------------------------------------
# Parametri fissati
# --------------------------------------------------------------------------

def test_parametri_del_paper():
    """alpha e V_max sono quelli del paper: è una replica, non una calibrazione."""
    assert ALPHA == 1.0
    assert V_MAX == 10.0


# --------------------------------------------------------------------------
# Giocatore fermo: la risposta è nota
# --------------------------------------------------------------------------

@pytest.mark.parametrize("d, atteso", [
    (1, 0.483183),
    (5, 1.198290),
    (10, 1.841406),
    (20, 2.947531),
])
def test_fermo_valori_di_riferimento(d, atteso):
    t = tempo_arrivo((0.0, 0.0), (0.0, 0.0), (d, 0.0))
    assert t == pytest.approx(atteso, abs=TOL)


@pytest.mark.parametrize("d", [0.3, 2, 7, 15, 35])
def test_fermo_soddisfa_l_equazione(d):
    """Il tempo restituito deve annullare l'equazione del modello."""
    t = tempo_arrivo((0.0, 0.0), (0.0, 0.0), (0.0, d))
    residuo = V_MAX * (t - (1 - math.exp(-ALPHA * t)) / ALPHA) - d
    assert abs(residuo) < 1e-3


def test_distanza_zero():
    assert tempo_arrivo((3.0, -2.0), (0.0, 0.0), (3.0, -2.0)) == pytest.approx(0.0, abs=TOL)


def test_monotono_nella_distanza():
    distanze = [0.5, 1, 2, 5, 10, 20, 40]
    tempi = [tempo_arrivo((0.0, 0.0), (0.0, 0.0), (d, 0.0)) for d in distanze]
    assert all(a < b for a, b in zip(tempi, tempi[1:]))


# --------------------------------------------------------------------------
# La velocità conta, e nel verso giusto
# --------------------------------------------------------------------------

def test_correre_verso_il_bersaglio_accorcia():
    fermo = tempo_arrivo((0.0, 0.0), (0.0, 0.0), (10.0, 0.0))
    verso = tempo_arrivo((0.0, 0.0), (5.0, 0.0), (10.0, 0.0))
    assert verso < fermo


def test_correre_in_direzione_opposta_allunga():
    fermo = tempo_arrivo((0.0, 0.0), (0.0, 0.0), (10.0, 0.0))
    via = tempo_arrivo((0.0, 0.0), (-5.0, 0.0), (10.0, 0.0))
    assert via > fermo


def test_velocita_laterale_fra_i_due():
    """Correre di lato è peggio che verso il bersaglio e meglio che in direzione opposta."""
    verso = tempo_arrivo((0.0, 0.0), (5.0, 0.0), (10.0, 0.0))
    lato = tempo_arrivo((0.0, 0.0), (0.0, 5.0), (10.0, 0.0))
    via = tempo_arrivo((0.0, 0.0), (-5.0, 0.0), (10.0, 0.0))
    assert verso < lato < via


# --------------------------------------------------------------------------
# Invarianze: il risultato non dipende dal sistema di riferimento
# --------------------------------------------------------------------------

def test_invariante_per_traslazione():
    a = tempo_arrivo((0.0, 0.0), (2.0, 1.0), (8.0, 3.0))
    b = tempo_arrivo((30.0, -20.0), (2.0, 1.0), (38.0, -17.0))
    assert a == pytest.approx(b, abs=TOL)


def test_invariante_per_rotazione():
    ang = math.radians(37)
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    pos, vel, bers = np.array([1.0, 2.0]), np.array([3.0, -1.0]), np.array([9.0, 6.0])
    a = tempo_arrivo(tuple(pos), tuple(vel), tuple(bers))
    b = tempo_arrivo(tuple(R @ pos), tuple(R @ vel), tuple(R @ bers))
    assert a == pytest.approx(b, abs=TOL)


# --------------------------------------------------------------------------
# tau_opp: il minimo fra gli avversari
# --------------------------------------------------------------------------

def test_tau_opp_e_il_minimo():
    bersaglio = (0.0, 0.0)
    posizioni = [(10.0, 0.0), (3.0, 0.0), (0.0, 20.0)]
    velocita_ = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    atteso = min(tempo_arrivo(p, v, bersaglio) for p, v in zip(posizioni, velocita_))
    assert tau_opp(bersaglio, posizioni, velocita_) == pytest.approx(atteso, abs=TOL)
    assert tau_opp(bersaglio, posizioni, velocita_) == pytest.approx(_fermo(3.0), abs=TOL)


def test_tau_opp_conta_la_velocita_non_solo_la_distanza():
    """Un avversario più lontano ma già lanciato può arrivare prima di uno vicino e fermo."""
    bersaglio = (0.0, 0.0)
    vicino_fermo = ((4.0, 0.0), (0.0, 0.0))
    lontano_lanciato = ((-6.0, 0.0), (8.0, 0.0))
    t_vicino = tempo_arrivo(*vicino_fermo, bersaglio)
    t_lontano = tempo_arrivo(*lontano_lanciato, bersaglio)
    assert t_lontano < t_vicino
    assert tau_opp(bersaglio, [vicino_fermo[0], lontano_lanciato[0]],
                   [vicino_fermo[1], lontano_lanciato[1]]) == pytest.approx(t_lontano, abs=TOL)


# --------------------------------------------------------------------------
# Velocità dalle posizioni
# --------------------------------------------------------------------------

def test_lisciatura_e_quella_decisa():
    """La scelta sulla lisciatura è fissata qui sopra, non dal loop."""
    import inspect
    default = inspect.signature(velocita).parameters["finestra"].default
    assert default == FINESTRA_LISCIATURA


def test_velocita_moto_rettilineo_uniforme():
    t = np.arange(30) / 10          # 3 secondi a 10 fps
    x, y = 2.0 * t + 5, -1.5 * t + 1
    vx, vy = velocita(x, y, fps=10)
    assert np.allclose(vx, 2.0, atol=1e-6)
    assert np.allclose(vy, -1.5, atol=1e-6)


def test_velocita_giocatore_fermo():
    x, y = np.full(20, 12.0), np.full(20, -3.0)
    vx, vy = velocita(x, y, fps=10)
    assert np.allclose(vx, 0.0) and np.allclose(vy, 0.0)


def test_velocita_stessa_lunghezza_dell_input():
    x = np.linspace(0, 10, 40)
    vx, vy = velocita(x, x, fps=10)
    assert len(vx) == len(x) and len(vy) == len(x)
