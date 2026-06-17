import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { isAxiosError } from 'axios'
import { useEffect, useMemo, useState } from 'react'

import { getCurrentSeason } from '@/api/seasons'
import { getMyTabellone, modifyTabellone, submitTabellone } from '@/api/tabellone'
import LoadingSpinner from '@/components/LoadingSpinner'
import type {
  TabelloneFieldKey,
  TabelloneInput,
  TablePredictionOut,
} from '@/types/tabellone'
import { formatDate, isDeadlinePassed } from '@/utils'

interface FieldDef {
  key: TabelloneFieldKey
  label: string
  type: 'text' | 'number'
}

interface Section {
  title: string
  fields: FieldDef[]
}

const SECTIONS: Section[] = [
  {
    title: 'Serie A',
    fields: [
      { key: 'scudetto_team', label: 'Scudetto', type: 'text' },
      { key: 'scudetto_points_guess', label: 'Punti dello scudetto', type: 'number' },
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

const ALL_FIELDS = SECTIONS.flatMap((s) => s.fields)
const NUMBER_KEYS = new Set(ALL_FIELDS.filter((f) => f.type === 'number').map((f) => f.key))
const MERCATO_PENALTY = 5
const LATE_COMPILE_PENALTY = 30

type FormState = Record<TabelloneFieldKey, string>

function emptyForm(): FormState {
  const f = {} as FormState
  ALL_FIELDS.forEach((def) => {
    f[def.key] = ''
  })
  return f
}

function formFromTabellone(t: TablePredictionOut): FormState {
  const f = emptyForm()
  ALL_FIELDS.forEach((def) => {
    const v = t[def.key]
    f[def.key] = v == null ? '' : String(v)
  })
  return f
}

/** Converte il form in payload: '' → null; campi numerici → number. */
function toPayload(form: FormState, only?: TabelloneFieldKey[]): TabelloneInput {
  const keys = only ?? ALL_FIELDS.map((d) => d.key)
  const payload: TabelloneInput = {}
  keys.forEach((k) => {
    const raw = form[k].trim()
    if (NUMBER_KEYS.has(k)) {
      ;(payload as Record<string, unknown>)[k] = raw === '' ? null : Number(raw)
    } else {
      ;(payload as Record<string, unknown>)[k] = raw === '' ? null : raw
    }
  })
  return payload
}

export default function Tabellone() {
  const queryClient = useQueryClient()
  const seasonQuery = useQuery({ queryKey: ['season', 'current'], queryFn: getCurrentSeason })

  const tabelloneQuery = useQuery({
    queryKey: ['tabellone', 'me'],
    queryFn: getMyTabellone,
    retry: false, // 404 = non ancora compilato, non riprovare
  })

  const [form, setForm] = useState<FormState>(emptyForm)
  const [baseline, setBaseline] = useState<FormState | null>(null) // valori salvati (per il mercato)
  const [isEditing, setIsEditing] = useState(false)

  const tabellone = tabelloneQuery.isError ? null : (tabelloneQuery.data ?? null)

  useEffect(() => {
    if (tabellone) {
      const f = formFromTabellone(tabellone)
      setForm(f)
      setBaseline(f)
    }
  }, [tabellone])

  const season = seasonQuery.data
  const status = season?.status
  const isMercato = status === 'mercato'
  // La deadline che conta dipende dallo stato: compilazione vs finestra mercato.
  const activeDeadline = isMercato
    ? (season?.modification_deadline ?? null)
    : (season?.tabellone_deadline ?? null)
  const deadlinePassed = isDeadlinePassed(activeDeadline)

  // Quando si può scrivere: in mercato fino alla modification_deadline;
  // in setup/active solo prima della tabellone_deadline.
  const editable =
    status === 'setup' ||
    (status === 'active' && !deadlinePassed) ||
    (isMercato && !deadlinePassed)

  const changedKeys = useMemo(() => {
    if (!baseline) return []
    return ALL_FIELDS.map((d) => d.key).filter((k) => form[k].trim() !== baseline[k].trim())
  }, [form, baseline])

  const mutation = useMutation({
    mutationFn: async () => {
      // Modifica (PATCH) solo se esiste già un tabellone in mercato; altrimenti
      // compilazione (POST): normale in setup/active, tardiva (-30) in mercato.
      if (isMercato && tabellone) {
        return modifyTabellone(toPayload(form, changedKeys))
      }
      return submitTabellone(toPayload(form))
    },
    onSuccess: (data) => {
      queryClient.setQueryData(['tabellone', 'me'], data)
      const f = formFromTabellone(data)
      setForm(f)
      setBaseline(f)
      setIsEditing(false) // torna alla vista riepilogo
    },
  })

  if (seasonQuery.isLoading || tabelloneQuery.isLoading) return <LoadingSpinner />

  if (seasonQuery.isError || !season) {
    return (
      <div>
        <h1 className="mb-4 text-xl font-semibold">Tabellone</h1>
        <p className="text-neutral-500">Nessuna stagione attiva al momento.</p>
      </div>
    )
  }

  // Mostra il form quando si sta modificando, oppure in prima compilazione
  // (nessun tabellone ancora salvato ma finestra aperta).
  const showForm = isEditing || (editable && !tabellone)

  function update(key: TabelloneFieldKey, value: string) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  function startEditing() {
    if (baseline) setForm(baseline) // riparti dai valori salvati
    mutation.reset()
    setIsEditing(true)
  }

  function cancelEditing() {
    if (baseline) setForm(baseline)
    setIsEditing(false)
  }

  function onSave() {
    if (isMercato && !tabellone) {
      // Compilazione tardiva: penalità fissa una tantum.
      const ok = window.confirm(
        'Stai compilando il tabellone in ritardo (a mercato aperto).\n\n' +
          `Questo comporta una penalità una tantum di -${LATE_COMPILE_PENALTY} pt.\n\nConfermi?`,
      )
      if (!ok) return
    } else if (isMercato) {
      if (changedKeys.length === 0) return
      const ok = window.confirm(
        'Stai salvando modifiche al tabellone.\n\n' +
          `Ogni voce diversa dal tabellone originale costa -${MERCATO_PENALTY} pt (una sola volta) ` +
          'e dimezza il guadagno massimo su quella voce. Riportare una voce al valore originale ' +
          'annulla la sua penalità.\n\nConfermi?',
      )
      if (!ok) return
    }
    mutation.mutate()
  }

  const errMsg = mutation.isError
    ? isAxiosError(mutation.error) && mutation.error.response?.data?.detail
      ? String(mutation.error.response.data.detail)
      : 'Errore nel salvataggio.'
    : null

  return (
    <div className={showForm ? 'pb-24' : ''}>
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Tabellone {season.name}</h1>
          {activeDeadline && editable && (
            <p className="mt-1 text-sm text-neutral-500">
              {isMercato ? 'Modificabile fino al' : 'Compilazione/modifica entro il'}{' '}
              {formatDate(activeDeadline)}
            </p>
          )}
        </div>
        {/* Tasto Modifica: solo in vista riepilogo, con tabellone salvato e finestra aperta */}
        {!showForm && tabellone && editable && (
          <button
            onClick={startEditing}
            className="shrink-0 rounded-lg border border-brand-600 px-4 py-2 text-sm font-medium text-brand-700 hover:bg-brand-50"
          >
            Modifica
          </button>
        )}
      </div>

      {/* Banner di stato — modifica di un tabellone esistente */}
      {isMercato && editable && tabellone && (
        <div className="mb-4 rounded-xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800">
          ⚠️ <strong>Finestra di mercato aperta</strong>
          {season.modification_deadline && <> fino al {formatDate(season.modification_deadline)}</>}.
          Ogni voce modificata costa <strong>-{MERCATO_PENALTY} pt</strong> immediati e dimezza il
          guadagno massimo su quella voce. Penalità già accumulata:{' '}
          <strong>{tabellone.mercato_penalty ?? 0} pt</strong>.
        </div>
      )}
      {/* Banner di stato — compilazione tardiva (nessun tabellone ancora) */}
      {isMercato && editable && !tabellone && (
        <div className="mb-4 rounded-xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800">
          ⚠️ <strong>Compilazione tardiva</strong>: non hai compilato il tabellone entro la
          deadline. Puoi farlo ora
          {season.modification_deadline && <> (fino al {formatDate(season.modification_deadline)})</>}
          , ma comporta una <strong>penalità una tantum di -{LATE_COMPILE_PENALTY} pt</strong>.
        </div>
      )}
      {!editable && (
        <div className="mb-4 rounded-xl border bg-neutral-50 p-3 text-sm text-neutral-600">
          {tabellone
            ? 'Compilazione chiusa: il tabellone non è più modificabile.'
            : 'Compilazione chiusa: non hai un tabellone per questa stagione.'}
        </div>
      )}

      {/* Punteggio (se già assegnato) */}
      {tabellone?.total_points != null && (
        <div className="mb-4 rounded-xl border bg-white p-3">
          <div className="text-sm text-neutral-500">Punti tabellone</div>
          <div className="text-2xl font-semibold text-brand-600">{tabellone.total_points} pt</div>
        </div>
      )}

      {showForm ? (
        <TabelloneForm
          form={form}
          tabellone={tabellone}
          isMercato={isMercato}
          changedKeys={changedKeys}
          onChange={update}
        />
      ) : tabellone ? (
        <TabelloneSummary tabellone={tabellone} form={baseline ?? form} />
      ) : null}

      {/* Barra salvataggio sticky — solo in modalità form */}
      {showForm && (
        <div className="fixed inset-x-0 bottom-0 border-t bg-white p-3">
          <div className="mx-auto flex max-w-5xl items-center justify-end gap-3 px-4">
            {errMsg && <span className="text-sm text-miss">{errMsg}</span>}
            {isMercato && tabellone && changedKeys.length > 0 && (
              <span className="text-sm text-amber-700">
                {changedKeys.length} voce/i da salvare · penalità ricalcolata sull'originale
              </span>
            )}
            {isMercato && !tabellone && (
              <span className="text-sm text-amber-700">
                Compilazione tardiva · -{LATE_COMPILE_PENALTY} pt una tantum
              </span>
            )}
            {tabellone && (
              <button
                onClick={cancelEditing}
                disabled={mutation.isPending}
                className="rounded-lg border px-4 py-2 text-sm font-medium text-neutral-600 hover:bg-neutral-50 disabled:opacity-60"
              >
                Annulla
              </button>
            )}
            <button
              onClick={onSave}
              disabled={mutation.isPending || (isMercato && !!tabellone && changedKeys.length === 0)}
              className="rounded-lg bg-brand-600 px-6 py-2 font-medium text-white hover:bg-brand-700 disabled:opacity-60"
            >
              {mutation.isPending
                ? 'Salvataggio…'
                : isMercato && tabellone
                  ? 'Salva modifiche'
                  : 'Salva tabellone'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Form di compilazione/modifica ──────────────────────────────────────────────

function TabelloneForm({
  form,
  tabellone,
  isMercato,
  changedKeys,
  onChange,
}: {
  form: FormState
  tabellone: TablePredictionOut | null
  isMercato: boolean
  changedKeys: TabelloneFieldKey[]
  onChange: (key: TabelloneFieldKey, value: string) => void
}) {
  return (
    <div className="grid gap-4">
      {SECTIONS.map((section) => (
        <div key={section.title} className="rounded-xl border bg-white p-4">
          <h2 className="mb-3 text-sm font-semibold text-neutral-700">{section.title}</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {section.fields.map((def) => {
              const isChanged = isMercato && changedKeys.includes(def.key)
              const breakdown = tabellone?.points_breakdown?.[def.key]
              return (
                <label key={def.key} className="flex flex-col gap-1 text-sm">
                  <span className="flex items-center gap-1 text-neutral-600">
                    {def.label}
                    {isChanged && <span className="text-xs text-amber-600">• modificato</span>}
                    {breakdown != null && (
                      <span
                        className={`ml-auto text-xs ${
                          breakdown > 0 ? 'font-semibold text-brand-600' : 'text-neutral-400'
                        }`}
                      >
                        {breakdown} pt
                      </span>
                    )}
                  </span>
                  <input
                    type={def.type === 'number' ? 'number' : 'text'}
                    inputMode={def.type === 'number' ? 'numeric' : undefined}
                    value={form[def.key]}
                    onChange={(e) => onChange(def.key, e.target.value)}
                    className={`rounded-lg border px-3 py-2 ${isChanged ? 'border-amber-400' : ''}`}
                  />
                </label>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

// ─── Vista riepilogo (sola lettura) ──────────────────────────────────────────────

function TabelloneSummary({
  tabellone,
  form,
}: {
  tabellone: TablePredictionOut
  form: FormState
}) {
  return (
    <div className="grid gap-4">
      {SECTIONS.map((section) => (
        <div key={section.title} className="overflow-hidden rounded-xl border bg-white">
          <h2 className="border-b bg-neutral-50 px-4 py-2 text-sm font-semibold text-neutral-700">
            {section.title}
          </h2>
          <dl className="divide-y">
            {section.fields.map((def) => {
              const value = form[def.key]?.trim()
              const breakdown = tabellone.points_breakdown?.[def.key]
              return (
                <div key={def.key} className="flex items-center gap-3 px-4 py-2 text-sm">
                  <dt className="w-1/2 text-neutral-500">{def.label}</dt>
                  <dd className="flex-1 font-medium">
                    {value ? value : <span className="text-neutral-300">—</span>}
                  </dd>
                  {breakdown != null && (
                    <span
                      className={`text-xs ${
                        breakdown > 0 ? 'font-semibold text-brand-600' : 'text-neutral-400'
                      }`}
                    >
                      {breakdown} pt
                    </span>
                  )}
                </div>
              )
            })}
          </dl>
        </div>
      ))}
    </div>
  )
}
