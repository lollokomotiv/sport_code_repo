import api from '@/api/client'
import type {
  SeasonOutcomeInput,
  SeasonOutcomeOut,
  TablePredictionOut,
} from '@/types/tabellone'

export async function listAllTabelloni(): Promise<TablePredictionOut[]> {
  const { data } = await api.get<TablePredictionOut[]>('/admin/tabellone')
  return data
}

/** GET /admin/season-outcome — 404 se i risultati reali non sono ancora inseriti. */
export async function getSeasonOutcome(): Promise<SeasonOutcomeOut> {
  const { data } = await api.get<SeasonOutcomeOut>('/admin/season-outcome')
  return data
}

export async function setSeasonOutcome(payload: SeasonOutcomeInput): Promise<SeasonOutcomeOut> {
  const { data } = await api.post<SeasonOutcomeOut>('/admin/season-outcome', payload)
  return data
}

export async function scoreTabellone(): Promise<TablePredictionOut[]> {
  const { data } = await api.post<TablePredictionOut[]>('/admin/tabellone/score')
  return data
}
