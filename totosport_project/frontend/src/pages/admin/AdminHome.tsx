import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'

import { listRounds } from '@/api/rounds'
import { getCurrentSeason } from '@/api/seasons'
import LoadingSpinner from '@/components/LoadingSpinner'
import RoundStatusBadge from '@/components/RoundStatusBadge'
import { useAuthStore } from '@/store/authStore'
import type { SeasonStatus } from '@/types/season'

const SEASON_LABEL: Record<SeasonStatus, string> = {
  setup: 'Setup',
  active: 'Attiva',
  mercato: 'Mercato',
  closed: 'Chiusa',
}

const QUICK_LINKS = [
  { to: '/admin/rounds', label: 'Giornate', desc: 'Crea giornate e partite, inserisci risultati' },
  { to: '/admin/season', label: 'Stagione', desc: 'Stato, deadline tabellone, finalizzazione' },
  { to: '/admin/tabellone', label: 'Tabellone', desc: 'Risultati reali e calcolo punti' },
  { to: '/admin/users', label: 'Giocatori', desc: 'Invita e gestisci i giocatori' },
]

export default function AdminHome() {
  const user = useAuthStore((s) => s.user)
  const seasonQuery = useQuery({
    queryKey: ['season', 'current'],
    queryFn: getCurrentSeason,
    retry: false,
  })
  const roundsQuery = useQuery({ queryKey: ['rounds'], queryFn: listRounds })

  const season = seasonQuery.isError ? null : (seasonQuery.data ?? null)
  const lastRound = roundsQuery.data?.[0]

  return (
    <div>
      <h1 className="mb-4 text-xl font-semibold">Dashboard — {user?.username}</h1>

      {seasonQuery.isLoading ? (
        <LoadingSpinner />
      ) : (
        <div className="mb-4 grid gap-4 sm:grid-cols-2">
          <div className="rounded-xl border bg-white p-4">
            <div className="text-sm text-neutral-500">Stagione corrente</div>
            {season ? (
              <>
                <div className="text-lg font-semibold">{season.name}</div>
                <span className="mt-1 inline-block rounded-full bg-brand-50 px-2 py-0.5 text-xs font-medium text-brand-700">
                  {SEASON_LABEL[season.status]}
                </span>
              </>
            ) : (
              <Link to="/admin/season" className="text-sm text-brand-600 hover:underline">
                Nessuna stagione — creane una →
              </Link>
            )}
          </div>

          <div className="rounded-xl border bg-white p-4">
            <div className="text-sm text-neutral-500">Ultima giornata</div>
            {lastRound ? (
              <Link
                to={`/admin/rounds/${lastRound.id}`}
                className="mt-1 flex items-center gap-2 hover:underline"
              >
                <span className="font-semibold">{lastRound.name}</span>
                <RoundStatusBadge status={lastRound.status} />
              </Link>
            ) : (
              <Link to="/admin/rounds" className="text-sm text-brand-600 hover:underline">
                Nessuna giornata — creane una →
              </Link>
            )}
          </div>
        </div>
      )}

      <h2 className="mb-2 text-sm font-medium text-neutral-700">Azioni rapide</h2>
      <div className="grid gap-3 sm:grid-cols-2">
        {QUICK_LINKS.map((l) => (
          <Link
            key={l.to}
            to={l.to}
            className="rounded-xl border bg-white p-4 transition-colors hover:border-brand-500"
          >
            <div className="font-medium">{l.label}</div>
            <div className="mt-1 text-sm text-neutral-500">{l.desc}</div>
          </Link>
        ))}
      </div>
    </div>
  )
}
