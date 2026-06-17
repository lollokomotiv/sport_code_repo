import { useQueryClient } from '@tanstack/react-query'
import { NavLink, useNavigate } from 'react-router-dom'

import { useAuthStore } from '@/store/authStore'

export interface NavItem {
  to: string
  label: string
}

export default function Navbar({ items }: { items: NavItem[] }) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const user = useAuthStore((s) => s.user)
  const logout = useAuthStore((s) => s.logout)

  function handleLogout() {
    logout()
    queryClient.clear() // rimuove i dati in cache del giocatore che esce
    navigate('/login', { replace: true })
  }

  return (
    <header className="border-b bg-white">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
        <div className="flex items-center gap-6">
          <span className="font-bold text-brand-600">TotoSport</span>
          <nav className="flex gap-4 text-sm">
            {items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end
                className={({ isActive }) =>
                  isActive
                    ? 'font-medium text-brand-600'
                    : 'text-neutral-600 hover:text-neutral-900'
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <span className="text-neutral-500">{user?.username}</span>
          <button onClick={handleLogout} className="text-neutral-600 hover:text-miss">
            Esci
          </button>
        </div>
      </div>
    </header>
  )
}
