import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
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
  const [open, setOpen] = useState(false)

  function handleLogout() {
    setOpen(false)
    logout()
    queryClient.clear() // rimuove i dati in cache del giocatore che esce
    navigate('/login', { replace: true })
  }

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    isActive ? 'font-medium text-brand-600' : 'text-neutral-600 hover:text-neutral-900'

  return (
    <header className="border-b bg-white">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
        <span className="font-bold text-brand-600">TotoSport</span>

        {/* Desktop: voci + utente in linea */}
        <div className="hidden items-center gap-6 md:flex">
          <nav className="flex gap-4 text-sm">
            {items.map((item) => (
              <NavLink key={item.to} to={item.to} end className={linkClass}>
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="flex items-center gap-3 text-sm">
            <span className="text-neutral-500">{user?.username}</span>
            <button onClick={handleLogout} className="text-neutral-600 hover:text-miss">
              Esci
            </button>
          </div>
        </div>

        {/* Mobile: bottone hamburger */}
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex h-11 w-11 items-center justify-center rounded-lg text-neutral-700 hover:bg-neutral-100 md:hidden"
          aria-label={open ? 'Chiudi menu' : 'Apri menu'}
          aria-expanded={open}
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            {open ? (
              <>
                <line x1="6" y1="6" x2="18" y2="18" />
                <line x1="18" y1="6" x2="6" y2="18" />
              </>
            ) : (
              <>
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </>
            )}
          </svg>
        </button>
      </div>

      {/* Mobile: menu a tendina */}
      {open && (
        <div className="border-t bg-white md:hidden">
          <nav className="flex flex-col">
            {items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  `border-b px-4 py-3 text-sm ${
                    isActive ? 'font-medium text-brand-600' : 'text-neutral-700'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="flex items-center justify-between px-4 py-3 text-sm">
            <span className="text-neutral-500">{user?.username}</span>
            <button onClick={handleLogout} className="font-medium text-miss">
              Esci
            </button>
          </div>
        </div>
      )}
    </header>
  )
}
