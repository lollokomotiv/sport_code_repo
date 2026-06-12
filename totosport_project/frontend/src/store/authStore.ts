import { create } from 'zustand'
import { persist } from 'zustand/middleware'

import type { AuthUser } from '@/types/auth'

interface AuthState {
  accessToken: string | null
  refreshToken: string | null
  user: AuthUser | null
  setTokens: (access: string, refresh: string) => void
  setAccessToken: (access: string) => void
  setUser: (user: AuthUser | null) => void
  logout: () => void
}

/**
 * Stato di autenticazione persistito in localStorage.
 *
 * Nota: il backend restituisce sia access che refresh token nel body (niente
 * cookie HttpOnly), quindi entrambi sono in localStorage. Accettabile per
 * un'app privata a basso rischio; da rivedere se si introdurrà il cookie.
 */
export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      accessToken: null,
      refreshToken: null,
      user: null,
      setTokens: (accessToken, refreshToken) => set({ accessToken, refreshToken }),
      setAccessToken: (accessToken) => set({ accessToken }),
      setUser: (user) => set({ user }),
      logout: () => set({ accessToken: null, refreshToken: null, user: null }),
    }),
    { name: 'totosport-auth' },
  ),
)
