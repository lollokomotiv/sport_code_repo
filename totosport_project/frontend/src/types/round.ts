export type Competition = 'serie_a' | 'serie_b' | 'champions_league' | 'mixed'
export type RoundStatus = 'draft' | 'open' | 'closed' | 'completed'

export interface MatchOut {
  id: string
  round_id: string
  competition: Competition
  home_team: string
  away_team: string
  requires_exact_score: boolean
  kickoff: string | null
  actual_home_goals: number | null
  actual_away_goals: number | null
  api_fixture_id: number | null
}

export interface RoundOut {
  id: string
  season_id: string
  name: string
  competition: Competition
  matchday: number | null
  deadline: string | null
  status: RoundStatus
  created_at: string
}

export interface RoundDetailOut extends RoundOut {
  matches: MatchOut[]
}
