import { Navigate, Outlet } from 'react-router-dom'

import Navbar, { type NavItem } from '@/components/Navbar'
import { useAuthStore } from '@/store/authStore'

const items: NavItem[] = [
  { to: '/admin', label: 'Dashboard' },
  { to: '/admin/season', label: 'Stagione' },
  { to: '/admin/rounds', label: 'Giornate' },
  { to: '/admin/tabellone', label: 'Tabellone' },
  { to: '/admin/leaderboard', label: 'Classifica' },
  { to: '/admin/users', label: 'Giocatori' },
]

/** Area admin: oltre all'auth, richiede ruolo admin (altrimenti torna al player). */
export default function AdminLayout() {
  const user = useAuthStore((s) => s.user)

  if (user?.role !== 'admin') {
    return <Navigate to="/player" replace />
  }

  return (
    <div className="min-h-screen">
      <Navbar items={items} />
      <main className="mx-auto max-w-5xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}
