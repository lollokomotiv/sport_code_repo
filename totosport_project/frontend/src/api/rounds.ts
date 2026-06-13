import api from '@/api/client'
import type { RoundDetailOut, RoundOut } from '@/types/round'

export async function listRounds(): Promise<RoundOut[]> {
  const { data } = await api.get<RoundOut[]>('/rounds')
  return data
}

export async function getRound(id: string): Promise<RoundDetailOut> {
  const { data } = await api.get<RoundDetailOut>(`/rounds/${id}`)
  return data
}
