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
