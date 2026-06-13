import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { getMyPredictions } from '@/api/predictions'
import { getRound, listRounds } from '@/api/rounds'
import CompetitionBadge from '@/components/CompetitionBadge'
import LoadingSpinner from '@/components/LoadingSpinner'
import RoundStatusBadge from '@/components/RoundStatusBadge'
import type { MatchPredictionOut } from '@/types/prediction'
import { formatCompetition } from '@/utils'

export default function MyPredictions() {
  const roundsQuery = useQuery({ queryKey: ['rounds'], queryFn: listRounds })
  const rounds = roundsQuery.data
  const [selectedId, setSelectedId] = useState<string>('')

  // Seleziona la prima giornata appena disponibili
  useEffect(() => {
    if (!selectedId && rounds && rounds.length > 0) setSelectedId(rounds[0].id)
  }, [rounds, selectedId])

  if (roundsQuery.isLoading) return <LoadingSpinner />
  if (roundsQuery.isError) return <p className="text-miss">Errore nel caricamento.</p>
  if (!rounds || rounds.length === 0) {
    return (
      <div>
        <h1 className="mb-4 text-xl font-semibold">Le mie previsioni</h1>
        <p className="text-neutral-500">Nessuna giornata disponibile.</p>
      </div>
    )
  }

  return (
    <div>
      <h1 className="mb-4 text-xl font-semibold">Le mie previsioni</h1>

      {/* Selettore giornata */}
      <div className="mb-4 flex gap-2 overflow-x-auto pb-1">
        {rounds.map((r) => (
          <button
            key={r.id}
            onClick={() => setSelectedId(r.id)}
            className={`whitespace-nowrap rounded-full border px-3 py-1.5 text-sm transition-colors ${
              r.id === selectedId
                ? 'border-brand-600 bg-brand-50 font-medium text-brand-700'
                : 'border-neutral-200 bg-white text-neutral-600 hover:border-brand-400'
            }`}
          >
            {r.name}
          </button>
        ))}
      </div>

      {selectedId && <RoundPredictions roundId={selectedId} />}
    </div>
  )
}

function RoundPredictions({ roundId }: { roundId: string }) {
  const roundQuery = useQuery({ queryKey: ['round', roundId], queryFn: () => getRound(roundId) })
  const predQuery = useQuery({
    queryKey: ['predictions', roundId],
    queryFn: () => getMyPredictions(roundId),
  })

  if (roundQuery.isLoading || predQuery.isLoading) return <LoadingSpinner />
  if (roundQuery.isError || !roundQuery.data) return <p className="text-miss">Errore.</p>

  const round = roundQuery.data
  const preds = predQuery.data
  const completed = round.status === 'completed'

  const predByMatch: Record<string, MatchPredictionOut> = {}
  preds?.match_predictions.forEach((p) => {
    predByMatch[p.match_id] = p
  })

  const matchPoints = preds?.match_predictions.reduce((s, p) => s + p.points_earned, 0) ?? 0
  const goalPoints = preds?.round_predictions.reduce((s, p) => s + p.points_earned, 0) ?? 0
  const total = matchPoints + goalPoints

  const hasAnyPrediction = (preds?.match_predictions.length ?? 0) > 0

  return (
    <div>
      <div className="mb-3 flex items-center gap-2">
        <CompetitionBadge competition={round.competition} />
        <RoundStatusBadge status={round.status} />
        {completed && (
          <span className="ml-auto text-sm font-semibold text-brand-600">{total} pt</span>
        )}
      </div>

      {!hasAnyPrediction && (
        <p className="rounded-xl border bg-white p-4 text-sm text-neutral-500">
          Non hai inserito previsioni per questa giornata.
        </p>
      )}

      <div className="grid gap-2">
        {round.matches.map((m) => {
          const p = predByMatch[m.id]
          const hasResult = completed && m.actual_home_goals !== null
          return (
            <div
              key={m.id}
              className="flex flex-wrap items-center gap-3 rounded-xl border bg-white p-3 text-sm"
            >
              <div className="min-w-[150px] flex-1">
                <span className="font-medium">{m.home_team}</span>
                <span className="mx-1 text-neutral-400">vs</span>
                <span className="font-medium">{m.away_team}</span>
              </div>

              <div className="text-neutral-600">
                {p ? (
                  <>
                    <span className="font-semibold">{p.predicted_sign}</span>
                    {m.requires_exact_score &&
                      p.predicted_home_goals != null &&
                      p.predicted_away_goals != null && (
                        <span className="ml-2 text-neutral-500">
                          {p.predicted_home_goals}-{p.predicted_away_goals}
                        </span>
                      )}
                  </>
                ) : (
                  <span className="text-neutral-400">—</span>
                )}
              </div>

              {hasResult && (
                <div className="text-right text-xs">
                  <div className="text-neutral-500">
                    Reale: {m.actual_home_goals}-{m.actual_away_goals}
                  </div>
                  <div
                    className={
                      p && p.points_earned > 0
                        ? 'font-semibold text-brand-600'
                        : 'text-neutral-400'
                    }
                  >
                    {p ? `${p.points_earned} pt` : '—'}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Totale gol per lega */}
      {preds && preds.round_predictions.length > 0 && (
        <div className="mt-3 rounded-xl border bg-white p-3 text-sm">
          <h2 className="mb-2 text-sm font-medium text-neutral-700">Totale gol giornata</h2>
          <div className="grid gap-1">
            {preds.round_predictions.map((rp) => (
              <div key={rp.id} className="flex items-center gap-2">
                <span className="text-neutral-600">{formatCompetition(rp.competition)}:</span>
                <span className="font-semibold">{rp.total_goals_guess}</span>
                {completed && (
                  <span
                    className={
                      rp.points_earned > 0
                        ? 'ml-auto font-semibold text-brand-600'
                        : 'ml-auto text-neutral-400'
                    }
                  >
                    {rp.points_earned} pt
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
