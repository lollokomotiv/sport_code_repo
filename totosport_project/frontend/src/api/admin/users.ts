import api from '@/api/client'
import type { UserCreateInput, UserOut } from '@/types/user'

export async function listUsers(): Promise<UserOut[]> {
  const { data } = await api.get<UserOut[]>('/admin/users')
  return data
}

export async function registerUser(payload: UserCreateInput): Promise<UserOut> {
  const { data } = await api.post<UserOut>('/auth/register', payload)
  return data
}

export async function disableUser(userId: string): Promise<void> {
  await api.delete(`/admin/users/${userId}`)
}
