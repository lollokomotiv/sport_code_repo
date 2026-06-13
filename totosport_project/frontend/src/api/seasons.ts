import api from '@/api/client'
import type { SeasonOut } from '@/types/season'

export async function getCurrentSeason(): Promise<SeasonOut> {
  const { data } = await api.get<SeasonOut>('/seasons/current')
  return data
}
