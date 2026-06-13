import type { Competition } from '@/types/round'

export type Sign = '1' | 'X' | '2'

export interface MatchPredictionOut {
  id: string
  player_id: string
  match_id: string
  predicted_sign: Sign
  predicted_home_goals: number | null
  predicted_away_goals: number | null
  points_earned: number
  submitted_at: string | null
}

export interface RoundPredictionOut {
  id: string
  player_id: string
  round_id: string
  competition: Competition
  total_goals_guess: number
  points_earned: number
  submitted_at: string | null
}

export interface RoundPredictionsBundle {
  match_predictions: MatchPredictionOut[]
  round_predictions: RoundPredictionOut[]
}
