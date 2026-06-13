import type { RoundStatus } from '@/types/round'

const STYLES: Record<RoundStatus, string> = {
  draft: 'bg-neutral-200 text-neutral-700',
  open: 'bg-brand-100 text-brand-700',
  closed: 'bg-amber-100 text-amber-800',
  completed: 'bg-neutral-800 text-white',
}

const LABELS: Record<RoundStatus, string> = {
  draft: 'Bozza',
  open: 'Aperta',
  closed: 'Chiusa',
  completed: 'Conclusa',
}

export default function RoundStatusBadge({ status }: { status: RoundStatus }) {
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${STYLES[status]}`}>
      {LABELS[status]}
    </span>
  )
}
