import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'

import { listRounds } from '@/api/rounds'
import CompetitionBadge from '@/components/CompetitionBadge'
import LoadingSpinner from '@/components/LoadingSpinner'
import RoundStatusBadge from '@/components/RoundStatusBadge'
import { formatDate } from '@/utils'

export default function RoundsList() {
  const { data: rounds, isLoading, isError } = useQuery({
    queryKey: ['rounds'],
    queryFn: listRounds,
  })

  if (isLoading) return <LoadingSpinner />
  if (isError) return <p className="text-miss">Errore nel caricamento delle giornate.</p>
  if (!rounds || rounds.length === 0) {
    return <p className="text-neutral-500">Nessuna giornata aperta al momento.</p>
  }

  return (
    <div>
      <h1 className="mb-4 text-xl font-semibold">Giornate</h1>
      <div className="grid gap-3">
        {rounds.map((r) => (
          <Link
            key={r.id}
            to={`/player/rounds/${r.id}`}
            className="rounded-xl border bg-white p-4 transition-colors hover:border-brand-500"
          >
            <div className="flex items-center justify-between">
              <span className="font-medium">{r.name}</span>
              <RoundStatusBadge status={r.status} />
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-3 text-sm text-neutral-500">
              <CompetitionBadge competition={r.competition} />
              {r.deadline && <span>Deadline: {formatDate(r.deadline)}</span>}
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
