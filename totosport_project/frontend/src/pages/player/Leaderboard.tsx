import { useQuery } from '@tanstack/react-query'

import { getLeaderboard } from '@/api/leaderboard'
import LoadingSpinner from '@/components/LoadingSpinner'
import { useAuthStore } from '@/store/authStore'

export default function Leaderboard() {
  const me = useAuthStore((s) => s.user)
  const { data, isLoading, isError } = useQuery({
    queryKey: ['leaderboard'],
    queryFn: getLeaderboard,
    refetchInterval: 30_000,
  })

  if (isLoading) return <LoadingSpinner />
  if (isError) return <p className="text-miss">Errore nel caricamento della classifica.</p>
  if (!data || data.length === 0) return <p className="text-neutral-500">Classifica ancora vuota.</p>

  return (
    <div>
      <h1 className="mb-4 text-xl font-semibold">Classifica generale</h1>
      <div className="overflow-x-auto rounded-xl border bg-white">
        <table className="w-full text-sm">
          <thead className="bg-neutral-50 text-left text-neutral-500">
            <tr>
              <th className="px-3 py-2">#</th>
              <th className="px-3 py-2">Giocatore</th>
              <th className="px-3 py-2 text-right">Totale</th>
              <th className="px-3 py-2 text-right">Segni</th>
              <th className="px-3 py-2 text-right">Pieni</th>
              <th className="px-3 py-2 text-right">Gol</th>
              <th className="px-3 py-2 text-right">Weekend</th>
              <th className="px-3 py-2 text-right">Tabellone</th>
            </tr>
          </thead>
          <tbody>
            {data.map((e) => (
              <tr
                key={e.player_id}
                className={`border-t ${e.player_id === me?.id ? 'bg-brand-50 font-medium' : ''}`}
              >
                <td className="px-3 py-2 text-neutral-500">{e.rank}</td>
                <td className="px-3 py-2">{e.username}</td>
                <td className="px-3 py-2 text-right font-semibold">{e.total_points}</td>
                <td className="px-3 py-2 text-right text-neutral-600">{e.sign_points}</td>
                <td className="px-3 py-2 text-right text-neutral-600">{e.exact_points}</td>
                <td className="px-3 py-2 text-right text-neutral-600">{e.total_goals_points}</td>
                <td className="px-3 py-2 text-right text-neutral-600">{e.weekend_bonus_total}</td>
                <td className="px-3 py-2 text-right text-neutral-600">{e.tabellone_points}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
