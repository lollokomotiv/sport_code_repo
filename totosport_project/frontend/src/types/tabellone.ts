// Voci del tabellone annuale (REGOLAMENTO §2). Tutti i campi sono opzionali
// per consentire salvataggi parziali (bozze) e PATCH parziali.
export interface TabelloneFields {
  // Serie A
  scudetto_team: string | null
  scudetto_points_guess: number | null
  relegated_a_1: string | null
  relegated_a_2: string | null
  relegated_a_3: string | null
  top_scorer_a: string | null
  top_scorer_a_goals: number | null
  // Serie B
  promoted_b_direct_1: string | null
  promoted_b_direct_2: string | null
  promoted_b_first_points: number | null
  playoff_b_1: string | null
  playoff_b_2: string | null
  playoff_b_3: string | null
  playoff_b_4: string | null
  playoff_b_5: string | null
  playoff_b_6: string | null
  promoted_b_playoff: string | null
  relegated_b_c_direct_1: string | null
  relegated_b_c_direct_2: string | null
  relegated_b_c_direct_3: string | null
  playout_b_1: string | null
  playout_b_2: string | null
  relegated_b_c_playout: string | null
  top_scorer_b: string | null
  top_scorer_b_goals: number | null
  // Coppe
  coppa_italia_winner: string | null
  champions_winner: string | null
  europa_winner: string | null
  conference_winner: string | null
}

export interface TablePredictionOut extends TabelloneFields {
  id: string
  player_id: string
  season_id: string
  mercato_penalty: number
  total_points: number | null
  points_breakdown: Record<string, number> | null
  scored_at: string | null
  submitted_at: string | null
  created_at: string
  updated_at: string
  is_modifiable: boolean
}

export type TabelloneInput = Partial<TabelloneFields>

// Chiavi dei campi nell'ordine di compilazione, raggruppate per sezione.
export type TabelloneFieldKey = keyof TabelloneFields
