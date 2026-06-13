import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import {
  addMatch,
  deleteMatch,
  deleteRound,
  setMatchResult,
  updateRoundStatus,
  type MatchCreateInput,
} from '@/api/admin/rounds'
import { getRound } from '@/api/rounds'
import CompetitionBadge from '@/components/CompetitionBadge'
import LoadingSpinner from '@/components/LoadingSpinner'
import RoundStatusBadge from '@/components/RoundStatusBadge'
import type { Competition, MatchOut, RoundStatus } from '@/types/round'
import { errorMessage, formatDate, fromDatetimeLocal } from '@/utils'

const COMPETITIONS: Competition[] = ['serie_a', 'serie_b', 'champions_league']

const NEXT_STATUS: Partial<Record<RoundStatus, { to: RoundStatus; label: string }>> = {
  draft: { to: 'open', label: 'Apri giornata' },
  open: { to: 'closed', label: 'Chiudi giornata' },
  closed: { to: 'completed', label: 'Completa e calcola punti' },
}

export default function AdminRoundDetail() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: round, isLoading, isError } = useQuery({
    queryKey: ['round', id],
    queryFn: () => getRound(id),
  })

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ['round', id] })
    queryClient.invalidateQueries({ queryKey: ['rounds'] })
  }

  const statusMut = useMutation({
    mutationFn: (to: RoundStatus) => updateRoundStatus(id, to),
    onSuccess: invalidate,
  })
  const delRoundMut = useMutation({
    mutationFn: () => deleteRound(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['rounds'] })
      navigate('/admin/rounds')
    },
  })

  if (isLoading) return <LoadingSpinner />
  if (isError || !round) return <p className="text-miss">Giornata non trovata.</p>

  const isDraft = round.status === 'draft'
  const isClosed = round.status === 'closed'
  const next = NEXT_STATUS[round.status]

  return (
    <div>
      <button
        onClick={() => navigate('/admin/rounds')}
        className="mb-3 text-sm text-neutral-500 hover:text-neutral-800"
      >
        ← Tutte le giornate
      </button>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">{round.name}</h1>
        <CompetitionBadge competition={round.competition} />
        <RoundStatusBadge status={round.status} />
        {round.deadline && (
          <span className="text-sm text-neutral-500">Deadline: {formatDate(round.deadline)}</span>
        )}
        <div className="ml-auto flex gap-2">
          {next && (
            <button
              onClick={() => statusMut.mutate(next.to)}
              disabled={statusMut.isPending}
              className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
            >
              {next.label}
            </button>
          )}
          {isDraft && (
            <button
              onClick={() => {
                if (window.confirm('Eliminare questa giornata?')) delRoundMut.mutate()
              }}
              className="rounded-lg border border-miss px-4 py-2 text-sm font-medium text-miss hover:bg-red-50"
            >
              Elimina
            </button>
          )}
        </div>
      </div>

      {statusMut.isError && (
        <p className="mb-3 text-sm text-miss">{errorMessage(statusMut.error)}</p>
      )}

      {/* Aggiunta partite: solo in draft */}
      {isDraft && (
        <AddMatchForm
          roundId={id}
          defaultCompetition={
            round.competition === 'mixed' ? 'serie_a' : (round.competition as Competition)
          }
          onAdded={invalidate}
        />
      )}

      {/* Lista partite */}
      <h2 className="mb-2 mt-4 text-sm font-medium text-neutral-700">
        Partite ({round.matches.length})
      </h2>
      {round.matches.length === 0 ? (
        <p className="text-sm text-neutral-500">
          Nessuna partita. {isDraft ? 'Aggiungine almeno una per poter aprire la giornata.' : ''}
        </p>
      ) : (
        <div className="grid gap-2">
          {round.matches.map((m) => (
            <MatchRow
              key={m.id}
              match={m}
              canDelete={isDraft}
              canSetResult={isClosed}
              onChanged={invalidate}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function AddMatchForm({
  roundId,
  defaultCompetition,
  onAdded,
}: {
  roundId: string
  defaultCompetition: Competition
  onAdded: () => void
}) {
  const [home, setHome] = useState('')
  const [away, setAway] = useState('')
  const [competition, setCompetition] = useState<Competition>(defaultCompetition)
  const [requiresExact, setRequiresExact] = useState(false)
  const [kickoff, setKickoff] = useState('')

  const mut = useMutation({
    mutationFn: (payload: MatchCreateInput) => addMatch(roundId, payload),
    onSuccess: () => {
      onAdded()
      setHome('')
      setAway('')
      setRequiresExact(false)
      setKickoff('')
    },
  })

  function submit() {
    if (!home.trim() || !away.trim()) return
    mut.mutate({
      home_team: home.trim(),
      away_team: away.trim(),
      competition,
      requires_exact_score: requiresExact,
      kickoff: fromDatetimeLocal(kickoff),
    })
  }

  return (
    <div className="rounded-xl border bg-white p-4">
      <h2 className="mb-3 text-sm font-medium text-neutral-700">Aggiungi partita</h2>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-neutral-600">Casa</span>
          <input
            value={home}
            onChange={(e) => setHome(e.target.value)}
            className="rounded-lg border px-3 py-2"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-neutral-600">Trasferta</span>
          <input
            value={away}
            onChange={(e) => setAway(e.target.value)}
            className="rounded-lg border px-3 py-2"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-neutral-600">Competizione</span>
          <select
            value={competition}
            onChange={(e) => setCompetition(e.target.value as Competition)}
            className="rounded-lg border px-3 py-2"
          >
            {COMPETITIONS.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-neutral-600">Orario (opz.)</span>
          <input
            type="datetime-local"
            value={kickoff}
            onChange={(e) => setKickoff(e.target.value)}
            className="rounded-lg border px-3 py-2"
          />
        </label>
      </div>
      <label className="mt-3 flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={requiresExact}
          onChange={(e) => setRequiresExact(e.target.checked)}
          className="h-4 w-4"
        />
        <span className="text-neutral-700">
          Richiede il <strong>risultato esatto</strong> (oltre al segno)
        </span>
      </label>
      {mut.isError && <p className="mt-2 text-sm text-miss">{errorMessage(mut.error)}</p>}
      <div className="mt-3 flex justify-end">
        <button
          onClick={submit}
          disabled={mut.isPending || !home.trim() || !away.trim()}
          className="rounded-lg bg-brand-600 px-6 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
        >
          {mut.isPending ? 'Aggiunta…' : 'Aggiungi partita'}
        </button>
      </div>
    </div>
  )
}

function MatchRow({
  match,
  canDelete,
  canSetResult,
  onChanged,
}: {
  match: MatchOut
  canDelete: boolean
  canSetResult: boolean
  onChanged: () => void
}) {
  const [home, setHome] = useState(
    match.actual_home_goals != null ? String(match.actual_home_goals) : '',
  )
  const [away, setAway] = useState(
    match.actual_away_goals != null ? String(match.actual_away_goals) : '',
  )

  const resultMut = useMutation({
    mutationFn: () => setMatchResult(match.id, Number(home), Number(away)),
    onSuccess: onChanged,
  })
  const delMut = useMutation({
    mutationFn: () => deleteMatch(match.id),
    onSuccess: onChanged,
  })

  const hasResult = match.actual_home_goals != null

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border bg-white p-3 text-sm">
      <div className="min-w-[160px] flex-1">
        <span className="font-medium">{match.home_team}</span>
        <span className="mx-1 text-neutral-400">vs</span>
        <span className="font-medium">{match.away_team}</span>
        <span className="ml-2 text-xs text-neutral-400">{match.competition}</span>
        {match.requires_exact_score && (
          <span className="ml-2 rounded bg-goal/20 px-1.5 py-0.5 text-xs text-amber-700">
            risultato esatto
          </span>
        )}
      </div>

      {canSetResult ? (
        <div className="flex items-center gap-1">
          <input
            inputMode="numeric"
            value={home}
            onChange={(e) => setHome(e.target.value)}
            className="w-12 rounded-lg border px-2 py-1 text-center"
          />
          <span className="text-neutral-400">-</span>
          <input
            inputMode="numeric"
            value={away}
            onChange={(e) => setAway(e.target.value)}
            className="w-12 rounded-lg border px-2 py-1 text-center"
          />
          <button
            onClick={() => resultMut.mutate()}
            disabled={resultMut.isPending || home === '' || away === ''}
            className="ml-1 rounded-lg bg-brand-600 px-3 py-1 text-xs font-medium text-white hover:bg-brand-700 disabled:opacity-60"
          >
            {resultMut.isPending ? '…' : hasResult ? 'Aggiorna' : 'Salva'}
          </button>
        </div>
      ) : hasResult ? (
        <span className="font-semibold">
          {match.actual_home_goals} - {match.actual_away_goals}
        </span>
      ) : null}

      {canDelete && (
        <button
          onClick={() => delMut.mutate()}
          disabled={delMut.isPending}
          className="text-xs text-miss hover:underline"
        >
          Rimuovi
        </button>
      )}
      {resultMut.isError && (
        <span className="w-full text-xs text-miss">{errorMessage(resultMut.error)}</span>
      )}
    </div>
  )
}
