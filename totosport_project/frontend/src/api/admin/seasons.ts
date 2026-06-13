import api from '@/api/client'
import type { SeasonOut, SeasonStatus } from '@/types/season'

export interface SeasonCreateInput {
  name: string
  tabellone_deadline?: string | null
  modification_deadline?: string | null
}

export interface SeasonBonusResult {
  bonus_signs: string[]
  bonus_exacts: string[]
  bonus_tabellone: string[]
}

export async function listSeasons(): Promise<SeasonOut[]> {
  const { data } = await api.get<SeasonOut[]>('/admin/seasons')
  return data
}

export async function createSeason(payload: SeasonCreateInput): Promise<SeasonOut> {
  const { data } = await api.post<SeasonOut>('/admin/seasons', payload)
  return data
}

export async function updateSeasonStatus(
  seasonId: string,
  status: SeasonStatus,
): Promise<SeasonOut> {
  const { data } = await api.patch<SeasonOut>(`/admin/seasons/${seasonId}/status`, { status })
  return data
}

export async function updateSeasonDeadlines(
  seasonId: string,
  payload: { tabellone_deadline?: string | null; modification_deadline?: string | null },
): Promise<SeasonOut> {
  const { data } = await api.patch<SeasonOut>(`/admin/seasons/${seasonId}`, payload)
  return data
}

export async function finalizeSeason(seasonId: string): Promise<SeasonBonusResult> {
  const { data } = await api.post<SeasonBonusResult>(`/admin/seasons/${seasonId}/finalize`)
  return data
}
