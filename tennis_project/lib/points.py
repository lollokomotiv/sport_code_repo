"""Decodifica delle sequenze di colpi del Match Charting Project.

Le colonne `1st` e `2nd` dei file punto per punto non sono testo libero: sono un
codice, un carattere per colpo (es. `4b37y1r3n#`). Il CLAUDE.md del progetto
avverte di non improvvisarne l'interpretazione, ed è il motivo per cui la
decodifica sta qui invece che dentro un'analisi: una sola implementazione, con
la legenda scritta accanto, invece di una rilettura a ogni uso.

## La legenda, dalla fonte ufficiale

Verificata sulla *quick start guide* del Match Charting Project
(tennisabstract.com, 23-09-2015):

- **servizio**: `4` wide, `5` body, `6` T — è sempre il primo carattere;
- **colpi**: `f` dritto, `b` rovescio, `s` slice di rovescio, `r` slice di
  dritto; la guida cita anche `v` volée, `l` pallonetto, `o` smash;
- **direzione** dopo il colpo: `1` verso il dritto di un destro, `2` al centro,
  `3` verso il rovescio di un destro;
- **profondità della risposta**: `7` corta, `8` mediamente profonda, `9` molto
  profonda;
- **esito**: `*` vincente, `#` errore forzato, `@` errore gratuito;
- **tipo di errore**: `n` rete, `w` larga, `d` lunga, `x` larga e lunga;
- `+` colpo di avvicinamento, `-` a rete, `=` a fondo campo.

## Ciò che la legenda letta NON documenta

`z y m p h i j k u t q`, i simboli `^ ; !`, e la `c` iniziale. Su questi il
modulo **non** decide: `BACKHAND` e `FOREHAND` contengono solo i codici
documentati, e chi vuole allargare la definizione lo fa esplicitamente passando
un insieme diverso, dichiarandolo nell'analisi.

La `c` compare solo in prima posizione e sempre prima del codice di direzione
del servizio (42 sequenze su 2.262 nei match Alcaraz-Zverev, l'1,9%): viene
trattata come annotazione del servizio e saltata. Non tocca l'identificazione
dei colpi, che comincia dopo il servizio.
"""

from __future__ import annotations

import pandas as pd

# Tutte le lettere che aprono un colpo, comprese quelle non documentate: servono
# a spezzare la sequenza nel punto giusto anche quando non si sa cosa siano.
SHOT_LETTERS = frozenset("fbrsvzopuylmhijktq")

# Solo i codici documentati. Volée, pallonetti e smash restano fuori di
# proposito: "rovescio contro rovescio" nel tennis è lo scambio da fondo campo.
BACKHAND = frozenset("bs")
FOREHAND = frozenset("fr")


def played_sequence(points: pd.DataFrame) -> pd.Series:
    """La sequenza effettivamente giocata: la seconda se c'è, altrimenti la prima.

    Quando la prima è fallita, `1st` contiene solo il servizio sbagliato
    (es. `4w`) e il punto vero sta in `2nd`. Prendere sempre `1st` conterebbe
    gli scambi dei soli punti di prima.
    """
    return points["2nd"].fillna(points["1st"])


def shot_letters(seq: str) -> list[str]:
    """Lettere dei colpi dopo il servizio, in ordine di esecuzione.

    I colpi si alternano fra i due giocatori: l'elemento 0 è la risposta (di chi
    riceve), l'1 è il colpo successivo di chi serve, e così via. Digits, simboli
    di esito e codici di errore vengono ignorati.
    """
    if not isinstance(seq, str) or not seq:
        return []
    s = seq[1:] if seq.startswith("c") else seq
    return [ch for ch in s[1:] if ch in SHOT_LETTERS]


def has_exchange(seq: str, side: frozenset[str] = BACKHAND) -> bool:
    """Vero se il punto contiene due colpi consecutivi dello stesso lato.

    Due colpi consecutivi sono per costruzione di giocatori opposti, perché nello
    scambio i colpi si alternano: `b` seguito da `b` è un rovescio a cui l'altro
    risponde di rovescio, cioè uno scambio rovescio-contro-rovescio.
    """
    c = shot_letters(seq)
    return any(c[i] in side and c[i + 1] in side for i in range(len(c) - 1))


def load_points(gender: str = "m", eras: str | list[str] = "2020s") -> pd.DataFrame:
    """Punti con la sequenza giocata già estratta, senza righe inutilizzabili.

    Scarta rumorosamente i punti senza sequenza o senza vincitore: sono pochi e
    non si possono usare, ma sparire in silenzio falserebbe i denominatori.
    """
    from . import loaders

    p = loaders.load_mcp_points(gender, eras)
    p = p.copy()
    p["seq"] = played_sequence(p).astype("string")

    prima = len(p)
    p = p.dropna(subset=["seq", "PtWinner"])
    p = p[p.seq.str.len() > 0]
    if len(p) < prima:
        import warnings
        warnings.warn(f"load_points: scartati {prima - len(p)} punti su {prima} "
                      "senza sequenza o senza vincitore")
    return p
