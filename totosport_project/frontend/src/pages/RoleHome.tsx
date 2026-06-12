import { Navigate } from 'react-router-dom'

import { useAuthStore } from '@/store/authStore'

/** Index di "/": instrada all'area giusta in base al ruolo. */
export default function RoleHome() {
  const user = useAuthStore((s) => s.user)
  return <Navigate to={user?.role === 'admin' ? '/admin' : '/player'} replace />
}
