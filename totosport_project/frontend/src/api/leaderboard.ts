import api from '@/api/client'
import type { LeaderboardEntry } from '@/types/leaderboard'

export async function getLeaderboard(): Promise<LeaderboardEntry[]> {
  const { data } = await api.get<LeaderboardEntry[]>('/leaderboard')
  return data
}
