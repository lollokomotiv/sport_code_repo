import type { Role } from '@/types/auth'

export interface UserOut {
  id: string
  username: string
  email: string
  role: Role
  is_active: boolean
  created_at: string
}

export interface UserCreateInput {
  username: string
  email: string
  password: string
  role?: Role
}
