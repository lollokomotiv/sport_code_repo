// Liste squadre per l'autocomplete nell'inserimento partite.
// Statiche e MODIFICABILI a mano: aggiorna qui a ogni stagione (promosse/retrocesse),
// poi `git push` (Vercel ridistribuisce). Nel form si può comunque digitare a mano
// una squadra non in lista (es. Champions/Coppe) — questi sono solo suggerimenti.

import type { Competition } from '@/types/round'

export const TEAMS_BY_COMPETITION: Record<Competition, string[]> = {
  serie_a: [
    'Atalanta',
    'Bologna',
    'Cagliari',
    'Como',
    'Cremonese',
    'Fiorentina',
    'Genoa',
    'Hellas Verona',
    'Inter',
    'Juventus',
    'Lazio',
    'Lecce',
    'Milan',
    'Napoli',
    'Parma',
    'Pisa',
    'Roma',
    'Sassuolo',
    'Torino',
    'Udinese',
  ],
  serie_b: [
    'Avellino',
    'Bari',
    'Carrarese',
    'Catanzaro',
    'Cesena',
    'Empoli',
    'Frosinone',
    'Juve Stabia',
    'Mantova',
    'Modena',
    'Monza',
    'Padova',
    'Palermo',
    'Pescara',
    'Reggiana',
    'Sampdoria',
    'Spezia',
    'Südtirol',
    'Venezia',
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
