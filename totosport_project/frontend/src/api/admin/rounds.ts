import api from '@/api/client'
import type { Competition, MatchOut, RoundOut, RoundStatus } from '@/types/round'

export interface RoundCreateInput {
  name: string
  competition: Competition
  matchday?: number | null
  deadline?: string | null
}

export interface MatchCreateInput {
  home_team: string
  away_team: string
  competition: Competition
  requires_exact_score: boolean
  kickoff?: string | null
}

export async function createRound(payload: RoundCreateInput): Promise<RoundOut> {
  const { data } = await api.post<RoundOut>('/rounds', payload)
  return data
}

export async function updateRoundStatus(
  roundId: string,
  status: RoundStatus,
): Promise<RoundOut> {
  const { data } = await api.patch<RoundOut>(`/rounds/${roundId}/status`, { status })
  return data
}

export async function deleteRound(roundId: string): Promise<void> {
  await api.delete(`/rounds/${roundId}`)
}

export async function addMatch(roundId: string, payload: MatchCreateInput): Promise<MatchOut> {
  const { data } = await api.post<MatchOut>(`/rounds/${roundId}/matches`, payload)
  return data
}

export async function deleteMatch(matchId: string): Promise<void> {
  await api.delete(`/matches/${matchId}`)
}

export async function setMatchResult(
  matchId: string,
  homeGoals: number,
  awayGoals: number,
): Promise<MatchOut> {
  const { data } = await api.patch<MatchOut>(`/matches/${matchId}/result`, {
    home_goals: homeGoals,
    away_goals: awayGoals,
  })
  return data
}
