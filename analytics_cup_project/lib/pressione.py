"""Pressione sul portatore come tempo minimo di arrivo dell'avversario (τ_opp).

Replica di Narizuka et al., scheda in notes/letteratura/pressione-tempo-arrivo.md.
Modello del moto di Fujimura-Sugihara: dopo un tempo t un giocatore può trovarsi
in un cerchio di centro c_p(t) e raggio r_p(t),

    c_p(t) = x_p(0) + (1 - exp(-a t)) / a * v_p(0)
    r_p(t) = V_max * (t - (1 - exp(-a t)) / a)

e il tempo di arrivo in x è il primo t per cui ||x - c_p(t)|| <= r_p(t).

Perché il primo t e non una radice qualunque: con velocità iniziale non nulla i
cerchi non sono annidati nei primi istanti (il centro si sposta più in fretta di
quanto cresce il raggio), quindi un solver su [0, T] potrebbe trovare un
attraversamento successivo. Si scandisce una griglia fine e si raffina con brentq
nel primo intervallo in cui cambia segno.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.signal import savgol_filter

# Parametri del paper, calibrati su sprint di J-League. Non si toccano.
ALPHA = 1.0    # s^-1
V_MAX = 10.0   # m/s

# Griglia di scansione per il primo attraversamento.
_DT = 0.01     # s
_T_MAX = 30.0  # s: a 10 m/s copre ben oltre la diagonale del campo


def _eccesso(t, pos, vel, bersaglio, alpha, vmax):
    """r_p(t) - ||x - c_p(t)||: negativo finché il bersaglio è fuori portata."""
    g = (1 - np.exp(-alpha * t)) / alpha
    cx = pos[0] + g * vel[0]
    cy = pos[1] + g * vel[1]
    r = vmax * (t - g)
    return r - np.hypot(bersaglio[0] - cx, bersaglio[1] - cy)


def tempo_arrivo(pos, vel, bersaglio, alpha=ALPHA, vmax=V_MAX) -> float:
    """Tempo minimo perché un giocatore in `pos` con velocità `vel` raggiunga `bersaglio`."""
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    bersaglio = np.asarray(bersaglio, dtype=float)
    if not (np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(bersaglio).all()):
        return np.nan
    if _eccesso(0.0, pos, vel, bersaglio, alpha, vmax) >= 0:
        return 0.0

    t0 = 0.0
    while True:
        t = np.arange(t0, t0 + _T_MAX + _DT, _DT)
        f = _eccesso(t, pos, vel, bersaglio, alpha, vmax)
        idx = np.flatnonzero(f >= 0)
        if idx.size:
            k = idx[0]
            if f[k] == 0:
                return float(t[k])
            return float(brentq(_eccesso, t[k - 1], t[k],
                                args=(pos, vel, bersaglio, alpha, vmax), xtol=1e-9))
        t0 = t[-1]


def tempi_arrivo(bersaglio, posizioni, velocita_, alpha=ALPHA, vmax=V_MAX) -> np.ndarray:
    """Tempo di arrivo di ciascun giocatore al bersaglio."""
    return np.array([tempo_arrivo(p, v, bersaglio, alpha, vmax)
                     for p, v in zip(posizioni, velocita_)], dtype=float)


def tau_opp(bersaglio, posizioni, velocita_, alpha=ALPHA, vmax=V_MAX) -> float:
    """Tempo minimo di arrivo fra i giocatori dati (gli avversari del portatore)."""
    tempi = tempi_arrivo(bersaglio, posizioni, velocita_, alpha, vmax)
    return float(np.nanmin(tempi)) if np.isfinite(tempi).any() else np.nan


def velocita(x, y, fps=10, finestra=None):
    """Velocità (vx, vy) in m/s dalle posizioni campionate a `fps`.

    finestra=None: derivata centrale (np.gradient), unilaterale agli estremi.
    finestra=int: Savitzky-Golay di grado 2 su quella finestra, derivata prima.

    I NaN (frame senza tracking) si propagano ai vicini: dove manca un vicino la
    derivata centrale non esiste e il risultato resta NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dt = 1.0 / fps
    if finestra is None:
        return np.gradient(x, dt), np.gradient(y, dt)
    return (savgol_filter(x, finestra, 2, deriv=1, delta=dt),
            savgol_filter(y, finestra, 2, deriv=1, delta=dt))
