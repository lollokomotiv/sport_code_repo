"""Test e intervalli per proporzioni, condivisi fra le analisi.

Qui si finisce quasi sempre per la stessa ragione: nel tennis annotato a mano i
denominatori sono piccoli — decine di punti per avversario, non migliaia — e una
percentuale senza incertezza a quelle dimensioni suggerisce differenze che i
dati non sostengono.
"""

from __future__ import annotations

import math


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Intervallo di confidenza al 95% per una proporzione (metodo di Wilson).

    Preferito a quello normale perché resta dentro [0, 1] e non degenera quando
    la proporzione è vicina agli estremi o `n` è piccolo — entrambe situazioni
    ordinarie con 20-50 smorzate per avversario.

    Ritorna gli estremi **in punti percentuali**. Denominatore nullo -> NaN.
    """
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    semi = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return 100 * (centro - semi), 100 * (centro + semi)


def z_two_proportions(k1: float, n1: float, k2: float, n2: float) -> tuple[float, float]:
    """Test z sulla differenza fra due proporzioni. Ritorna (z, p bilaterale).

    Usa la proporzione comune per l'errore standard (ipotesi nulla: stessa
    proporzione nei due gruppi). Denominatore nullo -> NaN, mai 0.
    """
    if not n1 or not n2:
        return float("nan"), float("nan")
    p_comune = (k1 + k2) / (n1 + n2)
    if p_comune in (0.0, 1.0):
        return float("nan"), float("nan")
    se = math.sqrt(p_comune * (1 - p_comune) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def z_observed_vs_expected(observed: float, expected: float, trials: float) -> tuple[float, float]:
    """Test z fra un conteggio osservato e uno atteso, su `trials` occasioni.

    È il test che accompagna il rapporto prodotto da `shots.observed_vs_expected`:
    dire che un rapporto vale 1,45 non basta, serve sapere se 1,45 si distingue
    da 1 con quel numero di colpi.

    Il tasso atteso è trattato come **noto**, non stimato: vale quando il
    riferimento poggia su un campione molto più grande del soggetto (qui decine
    di migliaia di colpi contro qualche migliaio). Se i due campioni fossero
    confrontabili, questo test sarebbe troppo generoso e servirebbe quello a due
    proporzioni.
    """
    if not trials or not expected:
        return float("nan"), float("nan")
    p0 = expected / trials
    if p0 in (0.0, 1.0):
        return float("nan"), float("nan")
    se = math.sqrt(trials * p0 * (1 - p0))
    z = (observed - expected) / se
    return z, math.erfc(abs(z) / math.sqrt(2))
