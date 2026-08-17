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

// F1 — schedine degli altri (visibili solo dopo la deadline)
export interface PlayerPredictions {
  player_id: string
  username: string
  submitted_at: string | null
  match_predictions: MatchPredictionOut[]
  round_predictions: RoundPredictionOut[]
}

export interface RoundPredictionsView {
  players: PlayerPredictions[]
}

// F2 — stato compilazione (admin, solo conteggi)
export interface PlayerSubmissionStatus {
  player_id: string
  username: string
  matches_predicted: number
  total_matches: number
  round_goals_count: number
  submitted_at: string | null
}

export interface RoundSubmissionStatus {
  total_matches: number
  leagues_expected: number
  players: PlayerSubmissionStatus[]
}

export interface MyRoundCompletion {
  round_id: string
  total_matches: number
  leagues_expected: number
  matches_predicted: number
  round_goals_count: number
  complete: boolean
}
