import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'

import {
  getSeasonOutcome,
  listAllTabelloni,
  scoreTabellone,
  setSeasonOutcome,
} from '@/api/admin/tabellone'
import { listUsers } from '@/api/admin/users'
import LoadingSpinner from '@/components/LoadingSpinner'
import type {
  SeasonOutcomeFields,
  SeasonOutcomeInput,
  SeasonOutcomeOut,
} from '@/types/tabellone'
import { errorMessage } from '@/utils'

type OutcomeKey = keyof SeasonOutcomeFields

interface FieldDef {
  key: OutcomeKey
  label: string
  type: 'text' | 'number' | 'bool'
}

const SECTIONS: { title: string; fields: FieldDef[] }[] = [
  {
    title: 'Serie A',
    fields: [
      { key: 'scudetto_team', label: 'Scudetto', type: 'text' },
      { key: 'scudetto_points', label: 'Punti scudetto', type: 'number' },
      { key: 'relegated_a_1', label: 'Retrocessa 1', type: 'text' },
      { key: 'relegated_a_2', label: 'Retrocessa 2', type: 'text' },
      { key: 'relegated_a_3', label: 'Retrocessa 3', type: 'text' },
      { key: 'top_scorer_a', label: 'Capocannoniere', type: 'text' },
      { key: 'top_scorer_a_goals', label: 'Gol capocannoniere', type: 'number' },
    ],
  },
  {
    title: 'Serie B',
    fields: [
      { key: 'promoted_b_direct_1', label: 'Promossa diretta 1', type: 'text' },
      { key: 'promoted_b_direct_2', label: 'Promossa diretta 2', type: 'text' },
      { key: 'promoted_b_first_points', label: 'Punti della prima', type: 'number' },
      { key: 'playoffs_held', label: 'Playoff disputati', type: 'bool' },
      { key: 'playoff_b_1', label: 'Playoff 1', type: 'text' },
      { key: 'playoff_b_2', label: 'Playoff 2', type: 'text' },
      { key: 'playoff_b_3', label: 'Playoff 3', type: 'text' },
      { key: 'playoff_b_4', label: 'Playoff 4', type: 'text' },
      { key: 'playoff_b_5', label: 'Playoff 5', type: 'text' },
      { key: 'playoff_b_6', label: 'Playoff 6', type: 'text' },
      { key: 'promoted_b_playoff', label: 'Promossa dai playoff', type: 'text' },
      { key: 'relegated_b_c_direct_1', label: 'Retrocessa diretta 1', type: 'text' },
      { key: 'relegated_b_c_direct_2', label: 'Retrocessa diretta 2', type: 'text' },
      { key: 'relegated_b_c_direct_3', label: 'Retrocessa diretta 3', type: 'text' },
      { key: 'playout_held', label: 'Playout disputati', type: 'bool' },
      { key: 'playout_b_1', label: 'Playout 1', type: 'text' },
      { key: 'playout_b_2', label: 'Playout 2', type: 'text' },
      { key: 'relegated_b_c_playout', label: 'Retrocessa dai playout', type: 'text' },
      { key: 'top_scorer_b', label: 'Capocannoniere B', type: 'text' },
      { key: 'top_scorer_b_goals', label: 'Gol capocannoniere B', type: 'number' },
    ],
  },
  {
    title: 'Coppe',
    fields: [
      { key: 'coppa_italia_winner', label: 'Coppa Italia', type: 'text' },
      { key: 'champions_winner', label: 'Champions League', type: 'text' },
      { key: 'europa_winner', label: 'Europa League', type: 'text' },
      { key: 'conference_winner', label: 'Conference League', type: 'text' },
    ],
  },
]

const ALL = SECTIONS.flatMap((s) => s.fields)
const NUMBER_KEYS = new Set(ALL.filter((f) => f.type === 'number').map((f) => f.key))
const BOOL_KEYS = new Set(ALL.filter((f) => f.type === 'bool').map((f) => f.key))

type FormState = Record<OutcomeKey, string> // bool come '' | 'true' | 'false'

function emptyForm(): FormState {
  const f = {} as FormState
  ALL.forEach((d) => {
    f[d.key] = ''
  })
  return f
}

function formFromOutcome(o: SeasonOutcomeOut): FormState {
  const f = emptyForm()
  ALL.forEach((d) => {
    const v = o[d.key]
    f[d.key] = v == null ? '' : String(v)
  })
  return f
}

function toPayload(form: FormState): SeasonOutcomeInput {
  const payload: SeasonOutcomeInput = {}
  ALL.forEach((d) => {
    const raw = form[d.key].trim()
    if (BOOL_KEYS.has(d.key)) {
      ;(payload as Record<string, unknown>)[d.key] = raw === '' ? null : raw === 'true'
    } else if (NUMBER_KEYS.has(d.key)) {
      ;(payload as Record<string, unknown>)[d.key] = raw === '' ? null : Number(raw)
    } else {
      ;(payload as Record<string, unknown>)[d.key] = raw === '' ? null : raw
    }
  })
  return payload
}

export default function AdminTabellone() {
  const queryClient = useQueryClient()

  const outcomeQuery = useQuery({
    queryKey: ['season-outcome'],
    queryFn: getSeasonOutcome,
    retry: false, // 404 = non ancora inserito
  })
  const tabelloniQuery = useQuery({ queryKey: ['all-tabelloni'], queryFn: listAllTabelloni })
  const usersQuery = useQuery({ queryKey: ['users'], queryFn: listUsers })

  const outcome = outcomeQuery.isError ? null : (outcomeQuery.data ?? null)

  const [form, setForm] = useState<FormState>(emptyForm)
  useEffect(() => {
    if (outcome) setForm(formFromOutcome(outcome))
  }, [outcome])

  const saveMut = useMutation({
    mutationFn: () => setSeasonOutcome(toPayload(form)),
    onSuccess: (data) => queryClient.setQueryData(['season-outcome'], data),
  })
  const scoreMut = useMutation({
    mutationFn: () => scoreTabellone(),
    onSuccess: (data) => {
      queryClient.setQueryData(['all-tabelloni'], data)
      queryClient.invalidateQueries({ queryKey: ['leaderboard'] })
    },
  })

  const usernameById = useMemo(() => {
    const map: Record<string, string> = {}
    usersQuery.data?.forEach((u) => {
      map[u.id] = u.username
    })
    return map
  }, [usersQuery.data])

  if (outcomeQuery.isLoading) return <LoadingSpinner />

  function update(key: OutcomeKey, value: string) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div>
      <h1 className="mb-4 text-xl font-semibold">Tabellone — risultati stagione</h1>

      {/* Riepilogo punti giocatori */}
      <div className="mb-6 overflow-x-auto rounded-xl border bg-white">
        <table className="w-full text-sm">
          <thead className="bg-neutral-50 text-left text-neutral-500">
            <tr>
              <th className="px-3 py-2">Giocatore</th>
              <th className="px-3 py-2 text-right">Penalità mercato</th>
              <th className="px-3 py-2 text-right">Punti tabellone</th>
            </tr>
          </thead>
          <tbody>
            {tabelloniQuery.data?.length ? (
              tabelloniQuery.data.map((t) => (
                <tr key={t.id} className="border-t">
                  <td className="px-3 py-2 font-medium">{usernameById[t.player_id] ?? '—'}</td>
                  <td className="px-3 py-2 text-right text-neutral-500">
                    {t.mercato_penalty ? `-${t.mercato_penalty}` : '0'}
                  </td>
                  <td className="px-3 py-2 text-right font-semibold text-brand-600">
                    {t.total_points ?? '—'}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={3} className="px-3 py-4 text-center text-neutral-500">
                  Nessun tabellone compilato.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="mb-4 flex items-center gap-3">
        <button
          onClick={() => scoreMut.mutate()}
          disabled={scoreMut.isPending || !outcome}
          className="rounded-lg border border-brand-600 px-4 py-2 text-sm font-medium text-brand-700 hover:bg-brand-50 disabled:opacity-60"
          title={outcome ? '' : 'Inserisci prima i risultati reali'}
        >
          {scoreMut.isPending ? 'Calcolo…' : 'Calcola punti tabellone'}
        </button>
        {scoreMut.isError && <span className="text-sm text-miss">{errorMessage(scoreMut.error)}</span>}
        {scoreMut.isSuccess && <span className="text-sm text-brand-600">Punti aggiornati ✓</span>}
      </div>

      {/* Form risultati reali */}
      <h2 className="mb-2 text-sm font-medium text-neutral-700">Risultati reali</h2>
      <div className="grid gap-4 pb-2">
        {SECTIONS.map((section) => (
          <div key={section.title} className="rounded-xl border bg-white p-4">
            <h3 className="mb-3 text-sm font-semibold text-neutral-700">{section.title}</h3>
            <div className="grid gap-3 sm:grid-cols-2">
              {section.fields.map((def) => (
                <label key={def.key} className="flex flex-col gap-1 text-sm">
                  <span className="text-neutral-600">{def.label}</span>
                  {def.type === 'bool' ? (
                    <select
                      value={form[def.key]}
                      onChange={(e) => update(def.key, e.target.value)}
                      className="rounded-lg border px-3 py-2"
                    >
                      <option value="">—</option>
                      <option value="true">Sì</option>
                      <option value="false">No</option>
                    </select>
                  ) : (
                    <input
                      type={def.type === 'number' ? 'number' : 'text'}
                      inputMode={def.type === 'number' ? 'numeric' : undefined}
                      value={form[def.key]}
                      onChange={(e) => update(def.key, e.target.value)}
                      className="rounded-lg border px-3 py-2"
                    />
                  )}
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="sticky bottom-0 -mx-4 flex items-center justify-end gap-3 border-t bg-white px-4 py-3">
        {saveMut.isSuccess && <span className="text-sm text-brand-600">Salvato ✓</span>}
        {saveMut.isError && <span className="text-sm text-miss">{errorMessage(saveMut.error)}</span>}
        <button
          onClick={() => saveMut.mutate()}
          disabled={saveMut.isPending}
          className="rounded-lg bg-brand-600 px-6 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
        >
          {saveMut.isPending ? 'Salvataggio…' : 'Salva risultati'}
        </button>
      </div>
    </div>
  )
}
