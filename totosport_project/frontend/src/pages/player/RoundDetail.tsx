import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'

import {
  getMyPredictions,
  getRoundPredictionsView,
  submitMatchPrediction,
  submitRoundGoals,
  type MatchPredictionInput,
} from '@/api/predictions'
import { getRound } from '@/api/rounds'
import CompetitionBadge from '@/components/CompetitionBadge'
import LoadingSpinner from '@/components/LoadingSpinner'
import type { MatchPredictionOut, PlayerPredictions, Sign } from '@/types/prediction'
import type { Competition, MatchOut } from '@/types/round'
import { formatCompetition, formatDate, isDeadlinePassed } from '@/utils'

interface PredInput {
  sign: '' | Sign
  home: string
  away: string
}

const EMPTY: PredInput = { sign: '', home: '', away: '' }
const TOTAL_GOALS_LEAGUES: Competition[] = ['serie_a', 'serie_b']

export default function RoundDetail() {
  const { id = '' } = useParams()
  const queryClient = useQueryClient()

  const roundQuery = useQuery({ queryKey: ['round', id], queryFn: () => getRound(id) })
  const predQuery = useQuery({ queryKey: ['predictions', id], queryFn: () => getMyPredictions(id) })

  const [inputs, setInputs] = useState<Record<string, PredInput>>({})
  const [totals, setTotals] = useState<Record<string, string>>({})
  const [saved, setSaved] = useState(false)
  const [showOthers, setShowOthers] = useState(false)

  const round = roundQuery.data
  const preds = predQuery.data

  // F1: schedine altrui — visibili solo a finestra chiusa (deadline superata o completata).
  const windowClosed = !!round && (isDeadlinePassed(round.deadline) || round.status === 'completed')
  const othersQuery = useQuery({
    queryKey: ['round-others', id],
    queryFn: () => getRoundPredictionsView(id),
    enabled: windowClosed,
  })

  const predByMatch = useMemo(() => {
    const map: Record<string, MatchPredictionOut> = {}
    preds?.match_predictions.forEach((p) => {
      map[p.match_id] = p
    })
    return map
  }, [preds])

  const leaguesPresent = useMemo(() => {
    const set = new Set<Competition>()
    round?.matches.forEach((m) => {
      if (TOTAL_GOALS_LEAGUES.includes(m.competition)) set.add(m.competition)
    })
    return Array.from(set)
  }, [round])

  useEffect(() => {
    if (!round || !preds) return
    const initial: Record<string, PredInput> = {}
    round.matches.forEach((m) => {
      const p = preds.match_predictions.find((x) => x.match_id === m.id)
      initial[m.id] = {
        sign: p?.predicted_sign ?? '',
        home: p?.predicted_home_goals != null ? String(p.predicted_home_goals) : '',
        away: p?.predicted_away_goals != null ? String(p.predicted_away_goals) : '',
      }
    })
    const initialTotals: Record<string, string> = {}
    preds.round_predictions.forEach((rp) => {
      initialTotals[rp.competition] = String(rp.total_goals_guess)
    })
    setInputs(initial)
    setTotals(initialTotals)
  }, [round, preds])

  const mutation = useMutation({
    mutationFn: async () => {
      if (!round) return
      const matchCalls = round.matches.flatMap((m) => {
        const inp = inputs[m.id] ?? EMPTY
        if (inp.sign === '') return [] // niente segno scelto → non inviare
        const payload: MatchPredictionInput = { match_id: m.id, predicted_sign: inp.sign }
        if (m.requires_exact_score && inp.home !== '' && inp.away !== '') {
          payload.predicted_home_goals = Number(inp.home)
          payload.predicted_away_goals = Number(inp.away)
        }
        return [submitMatchPrediction(payload)]
      })
      const goalCalls = leaguesPresent
        .filter((c) => totals[c] !== undefined && totals[c] !== '')
        .map((c) =>
          submitRoundGoals({ round_id: round.id, competition: c, total_goals_guess: Number(totals[c]) }),
        )
      await Promise.all([...matchCalls, ...goalCalls])
    },
    onSuccess: () => {
      setSaved(true)
      queryClient.invalidateQueries({ queryKey: ['predictions', id] })
    },
  })

  if (roundQuery.isLoading || predQuery.isLoading) return <LoadingSpinner />
  if (roundQuery.isError || !round) return <p className="text-miss">Giornata non trovata.</p>

  const locked = round.status !== 'open' || isDeadlinePassed(round.deadline)
  const showResults = round.status === 'completed'

  function update(matchId: string, patch: Partial<PredInput>) {
    setInputs((prev) => ({ ...prev, [matchId]: { ...(prev[matchId] ?? EMPTY), ...patch } }))
    setSaved(false)
  }

  return (
    <div className="pb-24">
      <div className="mb-4">
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-semibold">{round.name}</h1>
          <CompetitionBadge competition={round.competition} />
        </div>
        {round.deadline && (
          <p className="mt-1 text-sm text-neutral-500">Deadline: {formatDate(round.deadline)}</p>
        )}
        {locked && !showResults && (
          <p className="mt-2 text-sm text-amber-700">Previsioni chiuse per questa giornata.</p>
        )}
      </div>

      {windowClosed && (
        <div className="mb-4 rounded-xl border bg-white p-4">
          <button
            onClick={() => setShowOthers((s) => !s)}
            className="flex w-full items-center justify-between text-sm font-medium text-brand-700"
          >
            <span>Schedine degli altri giocatori</span>
            <span className="text-neutral-400">{showOthers ? '▲' : '▼'}</span>
          </button>
          {showOthers && (
            <div className="mt-3">
              {othersQuery.isLoading ? (
                <LoadingSpinner />
              ) : othersQuery.isError ? (
                <p className="text-sm text-miss">Errore nel caricamento delle schedine.</p>
              ) : (
                <OthersPredictions
                  players={othersQuery.data?.players ?? []}
                  matches={round.matches}
                />
              )}
            </div>
          )}
        </div>
      )}

      {leaguesPresent.length > 0 && (
        <div className="mb-4 rounded-xl border bg-white p-4">
          <h2 className="mb-2 text-sm font-medium text-neutral-700">Totale gol giornata</h2>
          <div className="flex flex-wrap gap-4">
            {leaguesPresent.map((c) => (
              <label key={c} className="flex items-center gap-2 text-sm">
                <span className="text-neutral-600">{formatCompetition(c)}</span>
                <input
                  inputMode="numeric"
                  disabled={locked}
                  value={totals[c] ?? ''}
                  onChange={(e) => {
                    setTotals((prev) => ({ ...prev, [c]: e.target.value }))
                    setSaved(false)
                  }}
                  className="w-16 rounded-lg border px-2 py-1 text-center disabled:bg-neutral-100"
                />
              </label>
            ))}
          </div>
        </div>
      )}

      <div className="grid gap-2">
        {round.matches.map((m) => (
          <MatchRow
            key={m.id}
            match={m}
            input={inputs[m.id] ?? EMPTY}
            prediction={predByMatch[m.id]}
            locked={locked}
            showResults={showResults}
            onChange={(patch) => update(m.id, patch)}
          />
        ))}
      </div>

      {!locked && (
        <div className="fixed inset-x-0 bottom-0 border-t bg-white px-3 pt-3 pb-safe">
          <div className="mx-auto flex max-w-5xl items-center justify-end gap-3 px-4">
            {saved && <span className="text-sm text-brand-600">Salvato ✓</span>}
            {mutation.isError && <span className="text-sm text-miss">Errore nel salvataggio</span>}
            <button
              onClick={() => mutation.mutate()}
              disabled={mutation.isPending}
              className="rounded-lg bg-brand-600 px-6 py-2 font-medium text-white hover:bg-brand-700 disabled:opacity-60"
            >
              {mutation.isPending ? 'Salvataggio…' : 'Salva previsioni'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

const SIGN_OPTIONS: Sign[] = ['1', 'X', '2']

function MatchRow({
  match,
  input,
  prediction,
  locked,
  showResults,
  onChange,
}: {
  match: MatchOut
  input: PredInput
  prediction?: MatchPredictionOut
  locked: boolean
  showResults: boolean
  onChange: (patch: Partial<PredInput>) => void
}) {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border bg-white p-3">
      <div className="min-w-[160px] flex-1 text-sm">
        <span className="font-medium">{match.home_team}</span>
        <span className="mx-1 text-neutral-400">vs</span>
        <span className="font-medium">{match.away_team}</span>
        {match.requires_exact_score && (
          <span className="ml-2 rounded bg-goal/20 px-1.5 py-0.5 text-xs text-amber-700">
            risultato esatto
          </span>
        )}
      </div>

      {/* Segno: sempre */}
      <select
        disabled={locked}
        value={input.sign}
        onChange={(e) => onChange({ sign: e.target.value as '' | Sign })}
        className="rounded-lg border px-2 py-1 text-sm disabled:bg-neutral-100"
      >
        <option value="">Segno…</option>
        {SIGN_OPTIONS.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>

      {/* Risultato esatto: solo sulle partite che lo richiedono */}
      {match.requires_exact_score && (
        <div className="flex items-center gap-1">
          <input
            inputMode="numeric"
            disabled={locked}
            value={input.home}
            onChange={(e) => onChange({ home: e.target.value })}
            className="w-12 rounded-lg border px-2 py-1 text-center disabled:bg-neutral-100"
          />
          <span className="text-neutral-400">-</span>
          <input
            inputMode="numeric"
            disabled={locked}
            value={input.away}
            onChange={(e) => onChange({ away: e.target.value })}
            className="w-12 rounded-lg border px-2 py-1 text-center disabled:bg-neutral-100"
          />
        </div>
      )}

      {showResults && match.actual_home_goals !== null && (
        <div className="ml-2 text-right text-xs">
          <div className="text-neutral-500">
            Reale: {match.actual_home_goals}-{match.actual_away_goals}
          </div>
          <div
            className={
              prediction && prediction.points_earned > 0
                ? 'font-semibold text-brand-600'
                : 'text-neutral-400'
            }
          >
            {prediction ? `${prediction.points_earned} pt` : '—'}
          </div>
        </div>
      )}
    </div>
  )
}

function OthersPredictions({
  players,
  matches,
}: {
  players: PlayerPredictions[]
  matches: MatchOut[]
}) {
  if (players.length === 0) {
    return <p className="text-sm text-neutral-500">Nessun altro ha compilato la schedina.</p>
  }
  return (
    <div className="grid gap-3">
      {players.map((pl) => (
        <div key={pl.player_id} className="rounded-xl border bg-white p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="font-medium">{pl.username}</span>
            {pl.submitted_at && (
              <span className="text-xs text-neutral-400">{formatDate(pl.submitted_at)}</span>
            )}
          </div>
          <div className="grid gap-1 text-sm">
            {matches.map((m) => {
              const mp = pl.match_predictions.find((x) => x.match_id === m.id)
              return (
                <div key={m.id} className="flex items-center gap-2">
                  <span className="flex-1 text-neutral-600">
                    {m.home_team} <span className="text-neutral-400">vs</span> {m.away_team}
                  </span>
                  {mp ? (
                    <>
                      <span className="font-semibold">{mp.predicted_sign}</span>
                      {mp.predicted_home_goals != null && mp.predicted_away_goals != null && (
                        <span className="text-neutral-500">
                          {mp.predicted_home_goals}-{mp.predicted_away_goals}
                        </span>
                      )}
                    </>
                  ) : (
                    <span className="text-neutral-300">—</span>
                  )}
                </div>
              )
            })}
          </div>
          {pl.round_predictions.length > 0 && (
            <div className="mt-2 text-xs text-neutral-500">
              Totale gol:{' '}
              {pl.round_predictions
                .map((rp) => `${formatCompetition(rp.competition)} ${rp.total_goals_guess}`)
                .join(' · ')}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
