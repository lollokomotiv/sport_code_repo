// Liste squadre per l'autocomplete nell'inserimento partite.
// Statiche e MODIFICABILI a mano: aggiorna qui a ogni stagione (promosse/retrocesse),
// poi `git push` (Vercel ridistribuisce). Nel form si può comunque digitare a mano
// una squadra non in lista (es. Champions/Coppe) — questi sono solo suggerimenti.

import type { Competition } from '@/types/round'

export const TEAMS_BY_COMPETITION: Record<Competition, string[]> = {
  // Serie A 2026-27 (20 squadre)
  serie_a: [
    'Atalanta',
    'Bologna',
    'Cagliari',
    'Como',
    'Fiorentina',
    'Frosinone',
    'Genoa',
    'Inter',
    'Juventus',
    'Lazio',
    'Lecce',
    'Milan',
    'Monza',
    'Napoli',
    'Parma',
    'Roma',
    'Sassuolo',
    'Torino',
    'Udinese',
    'Venezia',
  ],
  // Serie B 2026-27 (20 squadre)
  serie_b: [
    'Arezzo',
    'Ascoli',
    'Avellino',
    'Benevento',
    'Carrarese',
    'Catanzaro',
    'Cesena',
    'Cremonese',
    'Empoli',
    'Hellas Verona',
    'Juve Stabia',
    'Mantova',
    'Modena',
    'Padova',
    'Palermo',
    'Pisa',
    'Sampdoria',
    'Südtirol',
    'Vicenza',
    'Virtus Entella',
  ],
  // Coppe / altro: nessun suggerimento predefinito → si scrive a mano.
  champions_league: [],
  mixed: [],
}

/** Suggerimenti per una competizione (vuoto = si scrive a mano). */
export function teamsFor(competition: Competition): string[] {
  return TEAMS_BY_COMPETITION[competition] ?? []
}
