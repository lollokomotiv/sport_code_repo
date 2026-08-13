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

export interface UserUpdateInput {
  username?: string
  email?: string
  password?: string
  is_active?: boolean
}

/** Modifica utente: email, password, attiva/disattiva. */
export async function updateUser(userId: string, payload: UserUpdateInput): Promise<UserOut> {
  const { data } = await api.patch<UserOut>(`/admin/users/${userId}`, payload)
  return data
}

/** Eliminazione definitiva dell'utente. */
export async function deleteUser(userId: string): Promise<void> {
  await api.delete(`/admin/users/${userId}`)
}
