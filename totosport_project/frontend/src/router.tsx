import { lazy, Suspense } from 'react'
import { createBrowserRouter } from 'react-router-dom'

import LoadingSpinner from '@/components/LoadingSpinner'
import AdminLayout from '@/components/layouts/AdminLayout'
import PlayerLayout from '@/components/layouts/PlayerLayout'
import ProtectedLayout from '@/components/layouts/ProtectedLayout'
import AdminHome from '@/pages/admin/AdminHome'
import AdminTabellone from '@/pages/admin/AdminTabellone'
import Players from '@/pages/admin/Players'
import AdminRoundDetail from '@/pages/admin/RoundDetail'
import AdminRounds from '@/pages/admin/Rounds'
import Season from '@/pages/admin/Season'
import Login from '@/pages/Login'
import Leaderboard from '@/pages/player/Leaderboard'
import MyPredictions from '@/pages/player/MyPredictions'
import RoundDetail from '@/pages/player/RoundDetail'
import RoundsList from '@/pages/player/RoundsList'
import Tabellone from '@/pages/player/Tabellone'
import RoleHome from '@/pages/RoleHome'

// Lazy: react-markdown + il regolamento si scaricano solo aprendo la pagina.
const Regolamento = lazy(() => import('@/pages/Regolamento'))
const regolamentoElement = (
  <Suspense fallback={<LoadingSpinner />}>
    <Regolamento />
  </Suspense>
)

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
          { index: true, element: <RoundsList /> },
          { path: 'rounds/:id', element: <RoundDetail /> },
          { path: 'leaderboard', element: <Leaderboard /> },
          { path: 'predictions', element: <MyPredictions /> },
          { path: 'tabellone', element: <Tabellone /> },
          { path: 'regolamento', element: regolamentoElement },
        ],
      },
      {
        path: 'admin',
        element: <AdminLayout />,
        children: [
          { index: true, element: <AdminHome /> },
          { path: 'season', element: <Season /> },
          { path: 'rounds', element: <AdminRounds /> },
          { path: 'rounds/:id', element: <AdminRoundDetail /> },
          { path: 'tabellone', element: <AdminTabellone /> },
          { path: 'leaderboard', element: <Leaderboard /> },
          { path: 'users', element: <Players /> },
          { path: 'regolamento', element: regolamentoElement },
        ],
      },
    ],
  },
])
