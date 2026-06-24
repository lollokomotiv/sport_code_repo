import { Outlet } from 'react-router-dom'

import Navbar, { type NavItem } from '@/components/Navbar'

const items: NavItem[] = [
  { to: '/player', label: 'Giornate' },
  { to: '/player/predictions', label: 'Le mie previsioni' },
  { to: '/player/leaderboard', label: 'Classifica' },
  { to: '/player/tabellone', label: 'Tabellone' },
  { to: '/player/regolamento', label: 'Regolamento' },
]

export default function PlayerLayout() {
  return (
    <div className="min-h-screen">
      <Navbar items={items} />
      <main className="mx-auto max-w-5xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}
