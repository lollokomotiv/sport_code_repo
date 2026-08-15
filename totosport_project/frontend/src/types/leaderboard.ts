export interface LeaderboardEntry {
  rank: number
  player_id: string
  username: string
  total_points: number
  sign_points: number
  exact_points: number
  total_goals_points: number
  weekend_bonus_total: number
  tabellone_points: number
  season_bonus_total: number
}

// Classifiche "speciali" — conteggio (headline) + punti (dettaglio)
export interface SpecialRankingEntry {
  rank: number
  player_id: string
  username: string
  count: number
  points: number
}

export interface SpecialRankings {
  segni: SpecialRankingEntry[]
  pieni: SpecialRankingEntry[]
  pieni_5plus: SpecialRankingEntry[]
}
