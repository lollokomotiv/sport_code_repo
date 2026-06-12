export type Role = 'admin' | 'player'

export interface AuthUser {
  id: string
  username: string
  email: string
  role: Role
}

export interface TokenPair {
  access_token: string
  refresh_token: string
  token_type: string
}
