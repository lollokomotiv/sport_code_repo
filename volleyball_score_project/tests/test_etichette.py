"""Test della lettura del CSV esportato dalla pagina di etichettatura.

Una riga per tocco; i campi dello scambio (vincitore, fine) e il lato della nostra
squadra sono ripetuti su ogni riga, così il file basta da solo.
"""

import pytest

from lib.etichette import leggi_etichette
from lib.partita import punteggio
from lib.scambio import LONTANO, VICINO

INTESTAZIONE = "partita,set,scambio,tocco,t,lato,tipo,numero,lato_vincitore,t_fine,lato_nostro\n"


def scrivi(tmp_path, righe):
    percorso = tmp_path / "etichette.csv"
    percorso.write_text(INTESTAZIONE + "".join(r + "\n" for r in righe), encoding="utf-8")
    return percorso


def test_le_righe_di_uno_scambio_diventano_un_solo_scambio_in_ordine_di_tempo(tmp_path):
    percorso = scrivi(tmp_path, [
        "p1,1,1,2,13.4,lontano,R,,vicino,16.0,vicino",
        "p1,1,1,1,12.0,vicino,S,7,vicino,16.0,vicino",
        "p1,1,1,3,14.1,lontano,E,,vicino,16.0,vicino",
    ])

    scambi, _ = leggi_etichette(percorso)

    assert len(scambi) == 1
    assert [(t.t, t.lato, t.tipo) for t in scambi[0].tocchi] == [
        (12.0, VICINO, "S"), (13.4, LONTANO, "R"), (14.1, LONTANO, "E"),
    ]
    assert (scambi[0].set, scambi[0].indice, scambi[0].lato_vincitore) == (1, 1, VICINO)


def test_il_numero_di_maglia_vuoto_e_nessun_numero(tmp_path):
    percorso = scrivi(tmp_path, [
        "p1,1,1,1,12.0,vicino,S,7,lontano,13.0,vicino",
        "p1,1,1,2,12.9,lontano,R,,lontano,13.0,vicino",
    ])

    scambi, _ = leggi_etichette(percorso)

    assert [t.numero for t in scambi[0].tocchi] == [7, None]


def test_vincitori_diversi_nello_stesso_scambio_sono_un_errore(tmp_path):
    percorso = scrivi(tmp_path, [
        "p1,1,4,1,30.0,vicino,S,,vicino,33.0,vicino",
        "p1,1,4,2,31.0,lontano,R,,lontano,33.0,vicino",
    ])

    with pytest.raises(ValueError, match="scambio 4"):
        leggi_etichette(percorso)


def test_dal_file_al_punteggio(tmp_path):
    # Noi giochiamo nella metà lontana: vinciamo i primi due scambi, perdiamo il terzo.
    percorso = scrivi(tmp_path, [
        "p1,1,1,1,10.0,vicino,S,,lontano,12.0,lontano",
        "p1,1,1,2,11.0,lontano,R,,lontano,12.0,lontano",
        "p1,1,2,1,20.0,lontano,S,,lontano,21.0,lontano",
        "p1,1,3,1,30.0,lontano,S,,vicino,31.5,lontano",
        "p1,1,3,2,31.0,vicino,R,,vicino,31.5,lontano",
    ])

    scambi, lato_nostro = leggi_etichette(percorso)
    risultato = punteggio(scambi, lato_nostro)

    assert lato_nostro == {1: LONTANO}
    assert risultato.set == [(2, 1)]
    assert risultato.problemi == []
