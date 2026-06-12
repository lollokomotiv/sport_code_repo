import { Navigate, Outlet } from 'react-router-dom'

import { useAuthStore } from '@/store/authStore'

/** Blocca l'accesso alle aree autenticate: redirect a /login se non loggati. */
export default function ProtectedLayout() {
  const accessToken = useAuthStore((s) => s.accessToken)
  const user = useAuthStore((s) => s.user)

  if (!accessToken || !user) {
    return <Navigate to="/login" replace />
  }
  return <Outlet />
}
