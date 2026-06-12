import { createBrowserRouter } from 'react-router-dom'

import ComingSoon from '@/components/ComingSoon'
import AdminLayout from '@/components/layouts/AdminLayout'
import PlayerLayout from '@/components/layouts/PlayerLayout'
import ProtectedLayout from '@/components/layouts/ProtectedLayout'
import AdminHome from '@/pages/admin/AdminHome'
import Login from '@/pages/Login'
import PlayerHome from '@/pages/player/PlayerHome'
import RoleHome from '@/pages/RoleHome'

export const router = createBrowserRouter([
  { path: '/login', element: <Login /> },
  {
    path: '/',
    element: <ProtectedLayout />,
    children: [
      { index: true, element: <RoleHome /> },
      {
        path: 'player',
        element: <PlayerLayout />,
        children: [
          { index: true, element: <PlayerHome /> },
          { path: 'predictions', element: <ComingSoon title="Le mie previsioni" /> },
          { path: 'leaderboard', element: <ComingSoon title="Classifica" /> },
          { path: 'tabellone', element: <ComingSoon title="Tabellone" /> },
        ],
      },
      {
        path: 'admin',
        element: <AdminLayout />,
        children: [
          { index: true, element: <AdminHome /> },
          { path: 'rounds', element: <ComingSoon title="Giornate" /> },
          { path: 'tabellone', element: <ComingSoon title="Tabellone" /> },
          { path: 'leaderboard', element: <ComingSoon title="Classifica" /> },
          { path: 'users', element: <ComingSoon title="Giocatori" /> },
        ],
      },
    ],
  },
])
