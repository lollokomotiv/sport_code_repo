import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link } from 'react-router-dom'

import { createRound, type RoundCreateInput } from '@/api/admin/rounds'
import { listRounds } from '@/api/rounds'
import CompetitionBadge from '@/components/CompetitionBadge'
import LoadingSpinner from '@/components/LoadingSpinner'
import RoundStatusBadge from '@/components/RoundStatusBadge'
import type { Competition } from '@/types/round'
import { errorMessage, formatDate, fromDatetimeLocal } from '@/utils'

const COMPETITIONS: Competition[] = ['serie_a', 'serie_b', 'champions_league', 'mixed']

export default function AdminRounds() {
  const queryClient = useQueryClient()
  const { data: rounds, isLoading, isError } = useQuery({ queryKey: ['rounds'], queryFn: listRounds })

  const [showForm, setShowForm] = useState(false)
  const [name, setName] = useState('')
  const [competition, setCompetition] = useState<Competition>('mixed')
  const [matchday, setMatchday] = useState('')
  const [deadline, setDeadline] = useState('')

  const create = useMutation({
    mutationFn: (payload: RoundCreateInput) => createRound(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['rounds'] })
      setShowForm(false)
      setName('')
      setMatchday('')
      setDeadline('')
      setCompetition('mixed')
    },
  })

  function submit() {
    if (!name.trim()) return
    create.mutate({
      name: name.trim(),
      competition,
      matchday: matchday ? Number(matchday) : null,
      deadline: fromDatetimeLocal(deadline),
    })
  }

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold">Giornate</h1>
        <button
          onClick={() => setShowForm((s) => !s)}
          className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
        >
          {showForm ? 'Annulla' : 'Nuova giornata'}
        </button>
      </div>

      {showForm && (
        <div className="mb-4 rounded-xl border bg-white p-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="flex flex-col gap-1 text-sm sm:col-span-2">
              <span className="text-neutral-600">Nome</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="es. Giornata 34"
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
              <span className="text-neutral-600">Matchday (opz.)</span>
              <input
                inputMode="numeric"
                value={matchday}
                onChange={(e) => setMatchday(e.target.value)}
                className="rounded-lg border px-3 py-2"
              />
            </label>
            <label className="flex flex-col gap-1 text-sm sm:col-span-2">
              <span className="text-neutral-600">Deadline (opz.)</span>
              <input
                type="datetime-local"
                value={deadline}
                onChange={(e) => setDeadline(e.target.value)}
                className="rounded-lg border px-3 py-2"
              />
            </label>
          </div>
          {create.isError && (
            <p className="mt-2 text-sm text-miss">{errorMessage(create.error)}</p>
          )}
          <div className="mt-3 flex justify-end">
            <button
              onClick={submit}
              disabled={create.isPending || !name.trim()}
              className="rounded-lg bg-brand-600 px-6 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
            >
              {create.isPending ? 'Creazione…' : 'Crea giornata'}
            </button>
          </div>
        </div>
      )}

      {isLoading ? (
        <LoadingSpinner />
      ) : isError ? (
        <p className="text-miss">Errore nel caricamento.</p>
      ) : !rounds || rounds.length === 0 ? (
        <p className="text-neutral-500">Nessuna giornata. Creane una con "Nuova giornata".</p>
      ) : (
        <div className="grid gap-3">
          {rounds.map((r) => (
            <Link
              key={r.id}
              to={`/admin/rounds/${r.id}`}
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
      )}
    </div>
  )
}
