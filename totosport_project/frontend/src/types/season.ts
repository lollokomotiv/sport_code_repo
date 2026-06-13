export type SeasonStatus = 'setup' | 'active' | 'mercato' | 'closed'

export interface SeasonOut {
  id: string
  name: string
  status: SeasonStatus
  tabellone_deadline: string | null
  modification_deadline: string | null
  created_at: string
}
