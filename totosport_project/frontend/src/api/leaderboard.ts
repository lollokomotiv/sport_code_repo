import api from '@/api/client'
import type { LeaderboardEntry, SpecialRankings } from '@/types/leaderboard'

export async function getLeaderboard(): Promise<LeaderboardEntry[]> {
  const { data } = await api.get<LeaderboardEntry[]>('/leaderboard')
  return data
}

/** Classifiche speciali: segni / pieni / pieni 5+. */
export async function getSpecialRankings(): Promise<SpecialRankings> {
  const { data } = await api.get<SpecialRankings>('/leaderboard/special')
  return data
}
