import api from '@/api/client'
import type { RoundSubmissionStatus } from '@/types/prediction'

/** F2: stato compilazione per giornata (chi ha fatto / parziale / manca). */
export async function getRoundSubmissionStatus(roundId: string): Promise<RoundSubmissionStatus> {
  const { data } = await api.get<RoundSubmissionStatus>(`/admin/predictions/${roundId}/status`)
  return data
}
