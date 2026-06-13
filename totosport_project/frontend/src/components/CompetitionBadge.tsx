import { formatCompetition } from '@/utils'
import type { Competition } from '@/types/round'

const STYLES: Record<Competition, string> = {
  serie_a: 'bg-blue-100 text-blue-800',
  serie_b: 'bg-amber-100 text-amber-800',
  champions_league: 'bg-indigo-100 text-indigo-800',
  mixed: 'bg-neutral-200 text-neutral-700',
}

export default function CompetitionBadge({ competition }: { competition: Competition }) {
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${STYLES[competition]}`}>
      {formatCompetition(competition)}
    </span>
  )
}
