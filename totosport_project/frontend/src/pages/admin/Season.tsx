import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import {
  createSeason,
  finalizeSeason,
  updateSeasonDeadlines,
  updateSeasonStatus,
} from '@/api/admin/seasons'
import { getCurrentSeason } from '@/api/seasons'
import LoadingSpinner from '@/components/LoadingSpinner'
import type { SeasonOut, SeasonStatus } from '@/types/season'
import { errorMessage, fromDatetimeLocal, toDatetimeLocal } from '@/utils'

const STATUS_FLOW: SeasonStatus[] = ['setup', 'active', 'mercato', 'closed']

const STATUS_LABEL: Record<SeasonStatus, string> = {
  setup: 'Setup',
  active: 'Attiva',
  mercato: 'Mercato',
  closed: 'Chiusa',
}

// Transizioni offerte dall'UI (il backend resta la fonte di verità).
const STATUS_ACTIONS: Record<SeasonStatus, { to: SeasonStatus; label: string }[]> = {
  setup: [{ to: 'active', label: 'Attiva stagione' }],
  active: [{ to: 'mercato', label: 'Apri finestra mercato' }],
  mercato: [{ to: 'active', label: 'Chiudi finestra mercato' }],
  closed: [],
}

export default function AdminSeason() {
  const queryClient = useQueryClient()
  const seasonQuery = useQuery({
    queryKey: ['season', 'current'],
    queryFn: getCurrentSeason,
    retry: false,
  })

  const season = seasonQuery.isError ? null : (seasonQuery.data ?? null)
  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['season', 'current'] })

  if (seasonQuery.isLoading) return <LoadingSpinner />

  return (
    <div>
      <h1 className="mb-4 text-xl font-semibold">Gestione stagione</h1>
      {season ? (
        <CurrentSeason season={season} onChanged={invalidate} />
      ) : (
        <p className="mb-4 text-neutral-500">Nessuna stagione attiva.</p>
      )}
      <CreateSeasonForm onCreated={invalidate} />
    </div>
  )
}

function CurrentSeason({ season, onChanged }: { season: SeasonOut; onChanged: () => void }) {
  const [tabDeadline, setTabDeadline] = useState('')
  const [modDeadline, setModDeadline] = useState('')

  useEffect(() => {
    setTabDeadline(toDatetimeLocal(season.tabellone_deadline))
    setModDeadline(toDatetimeLocal(season.modification_deadline))
  }, [season])

  const statusMut = useMutation({
    mutationFn: (to: SeasonStatus) => updateSeasonStatus(season.id, to),
    onSuccess: onChanged,
  })
  const deadlineMut = useMutation({
    mutationFn: () =>
      updateSeasonDeadlines(season.id, {
        tabellone_deadline: fromDatetimeLocal(tabDeadline),
        modification_deadline: fromDatetimeLocal(modDeadline),
      }),
    onSuccess: onChanged,
  })
  const finalizeMut = useMutation({
    mutationFn: () => finalizeSeason(season.id),
    onSuccess: onChanged,
  })

  return (
    <div className="mb-6 grid gap-4">
      <div className="rounded-xl border bg-white p-4">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-medium">Stagione {season.name}</h2>
          <span className="rounded-full bg-brand-50 px-2 py-0.5 text-xs font-medium text-brand-700">
            {STATUS_LABEL[season.status]}
          </span>
        </div>

        {/* Timeline stati */}
        <div className="mt-3 flex items-center gap-2 text-xs">
          {STATUS_FLOW.map((s, i) => (
            <span key={s} className="flex items-center gap-2">
              <span
                className={
                  s === season.status
                    ? 'font-semibold text-brand-600'
                    : 'text-neutral-400'
                }
              >
                {STATUS_LABEL[s]}
              </span>
              {i < STATUS_FLOW.length - 1 && <span className="text-neutral-300">→</span>}
            </span>
          ))}
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {STATUS_ACTIONS[season.status].map((a) => (
            <button
              key={a.to}
              onClick={() => statusMut.mutate(a.to)}
              disabled={statusMut.isPending}
              className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
            >
              {a.label}
            </button>
          ))}
          {season.status !== 'closed' && (
            <button
              onClick={() => {
                if (
                  window.confirm(
                    'Finalizzare la stagione? Verranno assegnati i 3 bonus di fine stagione e la stagione passerà a "Chiusa".',
                  )
                )
                  finalizeMut.mutate()
              }}
              disabled={finalizeMut.isPending}
              className="rounded-lg border border-amber-500 px-4 py-2 text-sm font-medium text-amber-700 hover:bg-amber-50 disabled:opacity-60"
            >
              Finalizza stagione
            </button>
          )}
        </div>
        {statusMut.isError && (
          <p className="mt-2 text-sm text-miss">{errorMessage(statusMut.error)}</p>
        )}
        {finalizeMut.isError && (
          <p className="mt-2 text-sm text-miss">{errorMessage(finalizeMut.error)}</p>
        )}
        {finalizeMut.isSuccess && (
          <p className="mt-2 text-sm text-brand-600">Stagione finalizzata, bonus assegnati ✓</p>
        )}
      </div>

      {/* Deadline tabellone */}
      <div className="rounded-xl border bg-white p-4">
        <h2 className="mb-3 text-sm font-medium text-neutral-700">Deadline tabellone</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-neutral-600">Deadline compilazione</span>
            <input
              type="datetime-local"
              value={tabDeadline}
              onChange={(e) => setTabDeadline(e.target.value)}
              className="rounded-lg border px-3 py-2"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-neutral-600">Deadline modifiche (mercato)</span>
            <input
              type="datetime-local"
              value={modDeadline}
              onChange={(e) => setModDeadline(e.target.value)}
              className="rounded-lg border px-3 py-2"
            />
          </label>
        </div>
        <div className="mt-3 flex items-center gap-3">
          <button
            onClick={() => deadlineMut.mutate()}
            disabled={deadlineMut.isPending}
            className="rounded-lg bg-brand-600 px-6 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
          >
            {deadlineMut.isPending ? 'Salvataggio…' : 'Salva deadline'}
          </button>
          {deadlineMut.isSuccess && <span className="text-sm text-brand-600">Salvato ✓</span>}
          {deadlineMut.isError && (
            <span className="text-sm text-miss">{errorMessage(deadlineMut.error)}</span>
          )}
        </div>
      </div>
    </div>
  )
}

function CreateSeasonForm({ onCreated }: { onCreated: () => void }) {
  const [show, setShow] = useState(false)
  const [name, setName] = useState('')

  const mut = useMutation({
    mutationFn: () => createSeason({ name: name.trim() }),
    onSuccess: () => {
      onCreated()
      setShow(false)
      setName('')
    },
  })

  if (!show) {
    return (
      <button
        onClick={() => setShow(true)}
        className="rounded-lg border border-brand-600 px-4 py-2 text-sm font-medium text-brand-700 hover:bg-brand-50"
      >
        Crea nuova stagione
      </button>
    )
  }

  return (
    <div className="rounded-xl border bg-white p-4">
      <h2 className="mb-3 text-sm font-medium text-neutral-700">Nuova stagione</h2>
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-neutral-600">Nome (es. 2026-27)</span>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="rounded-lg border px-3 py-2"
        />
      </label>
      {mut.isError && <p className="mt-2 text-sm text-miss">{errorMessage(mut.error)}</p>}
      <div className="mt-3 flex gap-2">
        <button
          onClick={() => mut.mutate()}
          disabled={mut.isPending || !name.trim()}
          className="rounded-lg bg-brand-600 px-6 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
        >
          {mut.isPending ? 'Creazione…' : 'Crea'}
        </button>
        <button
          onClick={() => setShow(false)}
          className="rounded-lg border px-4 py-2 text-sm font-medium text-neutral-600 hover:bg-neutral-50"
        >
          Annulla
        </button>
      </div>
    </div>
  )
}
