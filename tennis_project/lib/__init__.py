"""Codice condiviso tra le analisi di tennis_project.

Qui sta solo ciò che serve a più di un'analisi: percorsi, download dei dati
grezzi, caricamento e normalizzazione. La logica specifica di un singolo studio
resta nella sua cartella sotto analyses/.
"""

from . import paths  # noqa: F401

__all__ = ["paths", "download", "loaders", "catalog"]
