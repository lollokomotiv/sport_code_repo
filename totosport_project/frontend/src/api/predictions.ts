import api from '@/api/client'
import type {
  MatchPredictionOut,
  RoundPredictionOut,
  RoundPredictionsBundle,
} from '@/types/prediction'
import type { Competition } from '@/types/round'

export async function getMyPredictions(roundId: string): Promise<RoundPredictionsBundle> {
  const { data } = await api.get<RoundPredictionsBundle>('/predictions/me', {
    params: { round_id: roundId },
  })
  return data
}

export interface MatchPredictionInput {
  match_id: string
  predicted_sign: '1' | 'X' | '2'
  predicted_home_goals?: number
  predicted_away_goals?: number
}

export async function submitMatchPrediction(
  payload: MatchPredictionInput,
): Promise<MatchPredictionOut> {
  const { data } = await api.post<MatchPredictionOut>('/predictions/match', payload)
  return data
}

export async function submitRoundGoals(payload: {
  round_id: string
  competition: Competition
  total_goals_guess: number
}): Promise<RoundPredictionOut> {
  const { data } = await api.post<RoundPredictionOut>('/predictions/round-goals', payload)
  return data
}
