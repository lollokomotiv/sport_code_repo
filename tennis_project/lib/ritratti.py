"""Ritratti dei giocatori nei grafici: il volto al posto del punto del soggetto.

Condiviso fra le analisi che evidenziano un giocatore in un dot plot. La foto è
un'aggiunta, non un requisito: se il file manca (per esempio perché la licenza
non è ancora chiarita e la foto è stata tolta), `aggiungi_volto` non disegna
nulla e ritorna False, così l'analisi ricade sul punto colorato invece di rompersi.
"""

from __future__ import annotations

from pathlib import Path


def volto_circolare(path: Path, colore_bordo: str, fondo: str):
    """Il ritratto ritagliato a cerchio, su un disco pieno, con un anello colorato.

    La foto va prima **composta sul disco** e solo dopo ritagliata. I ritratti
    scontornati hanno lo sfondo trasparente (quello di Zverev per il 43% dei
    pixel), e sotto la trasparenza il colore è nero: applicare la maschera
    circolare sostituendo il canale alfa renderebbe quei pixel neri e opachi.

    L'anello fa parte dell'immagine invece di essere disegnato a parte: così
    resta concentrico e proporzionato qualunque sia la dimensione a cui la foto
    viene resa. Ritorna un array RGBA pronto per `OffsetImage`.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.open(path).convert("RGBA")
    lato = min(img.size)
    sx, sy = (img.width - lato) // 2, (img.height - lato) // 2
    img = img.crop((sx, sy, sx + lato, sy + lato))

    # Si lavora a risoluzione quadrupla e poi si riduce: è il modo più semplice
    # per avere bordi del cerchio antialiasati invece che seghettati.
    s = 4
    grande = img.resize((lato * s, lato * s), Image.LANCZOS)
    disco = Image.new("RGBA", grande.size, fondo)
    grande = Image.alpha_composite(disco, grande)          # rispetta la trasparenza
    maschera = Image.new("L", grande.size, 0)
    ImageDraw.Draw(maschera).ellipse((0, 0, grande.width - 1, grande.height - 1), fill=255)
    grande.putalpha(maschera)                               # ora l'alfa è solo il cerchio
    spessore = int(grande.width * 0.045)
    ImageDraw.Draw(grande).ellipse((0, 0, grande.width - 1, grande.height - 1),
                                   outline=colore_bordo, width=spessore)
    return np.asarray(grande.resize((lato, lato), Image.LANCZOS))


def aggiungi_volto(ax, x: float, y: float, path: Path, diametro_pollici: float,
                   colore_bordo: str, fondo: str, zorder: int = 5) -> bool:
    """Disegna il ritratto centrato su (x, y) in coordinate dei dati.

    Ritorna True se l'ha disegnato, False se la foto non c'è: il chiamante
    decide allora di disegnare il punto normale.
    """
    if not Path(path).exists():
        return False
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    volto = volto_circolare(path, colore_bordo, fondo)
    # OffsetImage misura in punti tipografici: pixel × zoom = punti, 72 per pollice.
    zoom = diametro_pollici * 72 / volto.shape[0]
    ax.add_artist(AnnotationBbox(OffsetImage(volto, zoom=zoom), (x, y),
                                 frameon=False, zorder=zorder))
    return True
