export function deriveSign(home: number, away: number): '1' | 'X' | '2' {
  if (home > away) return '1'
  if (home === away) return 'X'
  return '2'
}

export function formatDate(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString('it-IT', {
    weekday: 'short',
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const COMPETITION_LABELS: Record<string, string> = {
  serie_a: 'Serie A',
  serie_b: 'Serie B',
  champions_league: 'Champions League',
  mixed: 'Mista',
}

export function formatCompetition(comp: string): string {
  return COMPETITION_LABELS[comp] ?? comp
}

export function isDeadlinePassed(deadline: string | null): boolean {
  if (!deadline) return false
  return new Date(deadline).getTime() < Date.now()
}

/** ISO → valore per <input type="datetime-local"> (ora locale, "YYYY-MM-DDTHH:mm"). */
export function toDatetimeLocal(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

/** Valore di <input type="datetime-local"> → ISO (UTC) per il backend; '' → null. */
export function fromDatetimeLocal(value: string): string | null {
  if (!value) return null
  return new Date(value).toISOString()
}

/** Estrae un messaggio leggibile da un errore di rete/axios. */
export function errorMessage(err: unknown, fallback = 'Si è verificato un errore.'): string {
  if (err && typeof err === 'object' && 'response' in err) {
    const detail = (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail
    if (typeof detail === 'string') return detail
  }
  return fallback
}
