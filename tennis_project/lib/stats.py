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


# ------------------------------------------------------------------ conteggi

def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) per X ~ Poisson(lam), sommando i termini in scala logaritmica."""
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0
    return min(1.0, sum(math.exp(i * math.log(lam) - lam - math.lgamma(i + 1))
                        for i in range(int(k) + 1)))


def poisson_interval(k: int, conf: float = 0.95) -> tuple[float, float]:
    """Intervallo esatto (Garwood) per la media di un conteggio di Poisson.

    Serve per eventi rari contati su pochi match — i doppi falli contro un
    avversario affrontato 4 volte sono una ventina — dove l'approssimazione
    normale produce intervalli simmetrici e sbagliati. Si trova per bisezione,
    senza dipendere da scipy.
    """
    alfa = (1 - conf) / 2

    def bisezione(f, lo, hi):
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2

    alto = max(10.0, 5 * (k + 1))
    # limite inferiore: il lambda per cui P(X >= k) = alfa
    basso = 0.0 if k == 0 else bisezione(lambda l: 1 - _poisson_cdf(k - 1, l) < alfa, 0.0, alto)
    # limite superiore: il lambda per cui P(X <= k) = alfa
    sopra = bisezione(lambda l: _poisson_cdf(k, l) > alfa, 0.0, alto)
    return basso, sopra


def poisson_test(observed: int, expected: float) -> float:
    """p bilaterale esatto: il conteggio osservato è compatibile con quello atteso?

    Doppio della coda più piccola, limitato a 1. L'atteso è trattato come noto:
    vale quando il riferimento poggia su molti più eventi del soggetto.
    """
    if expected <= 0:
        return float("nan")
    coda_bassa = _poisson_cdf(observed, expected)
    coda_alta = 1 - _poisson_cdf(observed - 1, expected)
    return min(1.0, 2 * min(coda_bassa, coda_alta))


def holm(p_values: list[float]) -> list[float]:
    """p corretti per confronti multipli (Holm-Bonferroni), nello stesso ordine.

    Con 74 avversari, qualche p sotto 0,05 esce per puro caso: questa correzione
    dice quali estremi restano tali dopo averne tenuto conto.
    """
    m = len(p_values)
    ordine = sorted(range(m), key=lambda i: p_values[i])
    corretti, massimo = [0.0] * m, 0.0
    for rango, i in enumerate(ordine):
        massimo = max(massimo, min(1.0, (m - rango) * p_values[i]))
        corretti[i] = massimo
    return corretti
