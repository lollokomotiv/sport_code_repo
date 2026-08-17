import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'

import { getMyRoundsSummary } from '@/api/predictions'
import { listRounds } from '@/api/rounds'
import CompetitionBadge from '@/components/CompetitionBadge'
import LoadingSpinner from '@/components/LoadingSpinner'
import RoundStatusBadge from '@/components/RoundStatusBadge'
import type { MyRoundCompletion } from '@/types/prediction'
import { formatDate, isDeadlinePassed } from '@/utils'

export default function RoundsList() {
  const { data: rounds, isLoading, isError } = useQuery({
    queryKey: ['rounds'],
    queryFn: listRounds,
  })
  const summaryQuery = useQuery({
    queryKey: ['my-rounds-summary'],
    queryFn: getMyRoundsSummary,
  })

  const byRound: Record<string, MyRoundCompletion> = {}
  summaryQuery.data?.forEach((s) => {
    byRound[s.round_id] = s
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
        {rounds.map((r) => {
          const s = byRound[r.id]
          // Il badge di compilazione ha senso solo finché la giornata è aperta e prima della deadline
          const canStillEdit = r.status === 'open' && !isDeadlinePassed(r.deadline)
          return (
            <Link
              key={r.id}
              to={`/player/rounds/${r.id}`}
              className="rounded-xl border bg-white p-4 transition-colors hover:border-brand-500"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium">{r.name}</span>
                <RoundStatusBadge status={r.status} />
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-3 text-sm text-neutral-500">
                <CompetitionBadge competition={r.competition} />
                {r.deadline && <span>Deadline: {formatDate(r.deadline)}</span>}
                {canStillEdit && s && <CompletionBadge s={s} />}
              </div>
            </Link>
          )
        })}
      </div>
    </div>
  )
}

function CompletionBadge({ s }: { s: MyRoundCompletion }) {
  if (s.complete) {
    return (
      <span className="rounded-full bg-brand-50 px-2 py-0.5 text-xs font-medium text-brand-700">
        ✅ Completata
      </span>
    )
  }
  const done = s.matches_predicted + s.round_goals_count
  const total = s.total_matches + s.leagues_expected
  return (
    <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-800">
      ⚠️ Da completare ({done}/{total})
    </span>
  )
}
