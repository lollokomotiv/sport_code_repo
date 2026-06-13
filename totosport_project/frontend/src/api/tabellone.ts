import api from '@/api/client'
import type { TabelloneInput, TablePredictionOut } from '@/types/tabellone'

/** GET /tabellone/me — 404 se il giocatore non ha ancora compilato. */
export async function getMyTabellone(): Promise<TablePredictionOut> {
  const { data } = await api.get<TablePredictionOut>('/tabellone/me')
  return data
}

/** POST /tabellone — compila o sovrascrive l'intero tabellone. */
export async function submitTabellone(payload: TabelloneInput): Promise<TablePredictionOut> {
  const { data } = await api.post<TablePredictionOut>('/tabellone', payload)
  return data
}

/** PATCH /tabellone/me — modifica post-mercato (solo i campi inviati). */
export async function modifyTabellone(payload: TabelloneInput): Promise<TablePredictionOut> {
  const { data } = await api.patch<TablePredictionOut>('/tabellone/me', payload)
  return data
}
