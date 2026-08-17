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

// Classifica di una singola giornata (dettaglio punti)
export interface RoundLeaderboardEntry {
  rank: number
  player_id: string
  username: string
  round_points: number
  sign_points: number
  exact_points: number
  total_goals_points: number
  weekend_bonus: number
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
