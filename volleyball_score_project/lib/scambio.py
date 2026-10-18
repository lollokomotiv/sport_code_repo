"""Lo scambio e i suoi tocchi, come escono dalle etichette o dalla pipeline.

I lati sono quelli dell'inquadratura, non le squadre: "vicino" è la metà campo
dal lato della camera. Le squadre cambiano campo a ogni set, quindi il passaggio
da lato a squadra si fa set per set, fuori da qui.

Tipi di tocco con i codici DataVolley: S battuta, R ricezione, E alzata,
A attacco, B muro, D difesa, F free ball.
"""

from dataclasses import dataclass

VICINO = "vicino"
LONTANO = "lontano"

TENTATIVI_DI_SALVARE = ("R", "D", "B")


@dataclass(frozen=True)
class Tocco:
    t: float
    lato: str
    tipo: str | None = None
    numero: int | None = None


@dataclass(frozen=True)
class Scambio:
    set: int
    indice: int
    tocchi: tuple[Tocco, ...]
    lato_vincitore: str


@dataclass(frozen=True)
class Problema:
    set: int
    scambio: int
    motivo: str
    # Posizione del tocco nello scambio, da 1. None se riguarda lo scambio intero.
    tocco: int | None = None


@dataclass(frozen=True)
class Esito:
    # Il tocco che decide lo scambio: di chi vince se modo è "punto",
    # di chi perde se modo è "errore".
    tocco: Tocco
    modo: str


def esito(scambio: Scambio) -> Esito:
    vincitore = scambio.lato_vincitore
    ultimo = scambio.tocchi[-1]
    # Ricezione, difesa e muro di chi perde sono tentativi di salvare la palla:
    # il punto è del tocco precedente di chi vince (ace, attacco, mani-out).
    if ultimo.lato != vincitore and ultimo.tipo in TENTATIVI_DI_SALVARE:
        decisivo = next(t for t in reversed(scambio.tocchi) if t.lato == vincitore)
        return Esito(decisivo, "punto")
    if ultimo.lato != vincitore:
        return Esito(ultimo, "errore")
    return Esito(ultimo, "punto")


# Tipo proposto per posizione nel possesso: primo, secondo, terzo tocco della
# stessa squadra. Il primo, quando arriva dall'altra parte, dipende da cosa
# arriva; qui c'è solo il caso in cui segue un muro della stessa squadra.
TIPI_PER_POSIZIONE = {1: "D", 2: "E", 3: "A"}

# Secondi tra l'attacco e il tocco dall'altra parte sotto cui il tocco è un muro.
# Provvisoria: da un attacco a pochi metri dalla rete la palla arriva al muro in
# 0,05-0,2 s, in seconda linea in circa mezzo secondo. Da regolare sulla partita 1.
SOGLIA_MURO = 0.3


def proponi_tipi(tocchi: list[Tocco]) -> list[str]:
    tipi: list[str] = []
    posizione = 0
    for i, tocco in enumerate(tocchi):
        if i == 0:
            tipi.append("S")
            continue
        if tocco.lato != tocchi[i - 1].lato:
            # Un'alzata che passa la rete è un attacco di seconda.
            if tipi[-1] == "E":
                tipi[-1] = "A"
            if tipi[-1] == "A" and tocco.t - tocchi[i - 1].t < SOGLIA_MURO:
                # Il muro non conta tra i tre tocchi della squadra.
                posizione = 0
                tipi.append("B")
            else:
                posizione = 1
                tipi.append("R" if tipi[-1] == "S" else "D")
        else:
            posizione += 1
            tipi.append(TIPI_PER_POSIZIONE[posizione])
    return tipi


MAX_TOCCHI = 3


def controlla_scambio(scambio: Scambio) -> list[Problema]:
    problemi = []
    possesso = 0
    for i, tocco in enumerate(scambio.tocchi):
        precedente = scambio.tocchi[i - 1] if i > 0 else None
        # Il muro non conta tra i tre tocchi: dopo un muro il possesso riparte da zero.
        if tocco.tipo == "B":
            possesso = 0
        elif precedente is not None and precedente.lato == tocco.lato:
            possesso += 1
        else:
            possesso = 1
        if possesso > MAX_TOCCHI:
            problemi.append(Problema(scambio.set, scambio.indice, "più di tre tocchi dalla stessa parte", i + 1))
        if tocco.tipo == "R" and not dall_altra_parte(precedente, tocco, "S"):
            problemi.append(Problema(scambio.set, scambio.indice, "ricezione non preceduta da una battuta", i + 1))
        if tocco.tipo == "B" and not dall_altra_parte(precedente, tocco, "A"):
            problemi.append(Problema(scambio.set, scambio.indice, "muro non preceduto da un attacco", i + 1))
    return problemi


def dall_altra_parte(precedente: Tocco | None, tocco: Tocco, tipo: str) -> bool:
    """Vero se il tocco precedente è del tipo dato e viene dall'altra metà campo."""
    return precedente is not None and precedente.tipo == tipo and precedente.lato != tocco.lato
