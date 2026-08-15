import { useQuery } from '@tanstack/react-query'
import { isAxiosError } from 'axios'
import { useState } from 'react'

import { getLeaderboard, getSpecialRankings } from '@/api/leaderboard'
import LoadingSpinner from '@/components/LoadingSpinner'
import { useAuthStore } from '@/store/authStore'
import type { LeaderboardEntry, SpecialRankingEntry } from '@/types/leaderboard'

type Tab = 'generale' | 'speciali'

export default function Leaderboard() {
  const me = useAuthStore((s) => s.user)
  const [tab, setTab] = useState<Tab>('generale')

  const generalQuery = useQuery({
    queryKey: ['leaderboard'],
    queryFn: getLeaderboard,
    refetchInterval: 30_000,
  })
  const specialQuery = useQuery({
    queryKey: ['leaderboard-special'],
    queryFn: getSpecialRankings,
    enabled: tab === 'speciali',
    refetchInterval: 30_000,
  })

  const noSeason =
    generalQuery.isError &&
    isAxiosError(generalQuery.error) &&
    generalQuery.error.response?.status === 404

  return (
    <div>
      <h1 className="mb-4 text-xl font-semibold">Classifica</h1>

      <div className="mb-4 flex gap-2">
        {(['generale', 'speciali'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`rounded-full border px-3 py-1.5 text-sm transition-colors ${
              tab === t
                ? 'border-brand-600 bg-brand-50 font-medium text-brand-700'
                : 'border-neutral-200 bg-white text-neutral-600 hover:border-brand-400'
            }`}
          >
            {t === 'generale' ? 'Generale' : 'Speciali'}
          </button>
        ))}
      </div>

      {noSeason ? (
        <p className="text-neutral-500">
          Nessuna stagione attiva: la classifica sarà disponibile quando l'admin crea una stagione.
        </p>
      ) : tab === 'generale' ? (
        generalQuery.isLoading ? (
          <LoadingSpinner />
        ) : generalQuery.isError ? (
          <p className="text-miss">Errore nel caricamento della classifica.</p>
        ) : !generalQuery.data || generalQuery.data.length === 0 ? (
          <p className="text-neutral-500">Classifica ancora vuota.</p>
        ) : (
          <GeneralTable rows={generalQuery.data} meId={me?.id} />
        )
      ) : specialQuery.isLoading ? (
        <LoadingSpinner />
      ) : specialQuery.isError ? (
        <p className="text-miss">Errore nel caricamento delle classifiche speciali.</p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-3">
          <MiniRanking title="Segni" entries={specialQuery.data?.segni ?? []} meId={me?.id} />
          <MiniRanking title="Pieni" entries={specialQuery.data?.pieni ?? []} meId={me?.id} />
          <MiniRanking
            title="Pieni 5+ gol"
            entries={specialQuery.data?.pieni_5plus ?? []}
            meId={me?.id}
          />
        </div>
      )}
    </div>
  )
}

function GeneralTable({ rows, meId }: { rows: LeaderboardEntry[]; meId?: string }) {
  return (
    <div className="overflow-x-auto rounded-xl border bg-white">
      <table className="w-full text-sm">
        <thead className="bg-neutral-50 text-left text-neutral-500">
          <tr>
            <th className="px-3 py-2">#</th>
            <th className="px-3 py-2">Giocatore</th>
            <th className="px-3 py-2 text-right">Totale</th>
            <th className="px-3 py-2 text-right">Segni</th>
            <th className="px-3 py-2 text-right">Pieni (pt)</th>
            <th className="px-3 py-2 text-right">Gol</th>
            <th className="px-3 py-2 text-right">Weekend</th>
            <th className="px-3 py-2 text-right">Tabellone</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((e) => (
            <tr
              key={e.player_id}
              className={`border-t ${e.player_id === meId ? 'bg-brand-50 font-medium' : ''}`}
            >
              <td className="px-3 py-2 text-neutral-500">{e.rank}</td>
              <td className="px-3 py-2">{e.username}</td>
              <td className="px-3 py-2 text-right font-semibold">{e.total_points}</td>
              <td className="px-3 py-2 text-right text-neutral-600">{e.sign_points}</td>
              <td className="px-3 py-2 text-right text-neutral-600">{e.exact_points}</td>
              <td className="px-3 py-2 text-right text-neutral-600">{e.total_goals_points}</td>
              <td className="px-3 py-2 text-right text-neutral-600">{e.weekend_bonus_total}</td>
              <td className="px-3 py-2 text-right text-neutral-600">{e.tabellone_points}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function MiniRanking({
  title,
  entries,
  meId,
}: {
  title: string
  entries: SpecialRankingEntry[]
  meId?: string
}) {
  return (
    <div className="overflow-hidden rounded-xl border bg-white">
      <h3 className="border-b bg-neutral-50 px-3 py-2 text-sm font-medium text-neutral-700">
        {title}
      </h3>
      {entries.length === 0 ? (
        <p className="px-3 py-3 text-sm text-neutral-400">Ancora nessun dato.</p>
      ) : (
        <ul className="divide-y">
          {entries.map((e) => (
            <li
              key={e.player_id}
              className={`flex items-center gap-3 px-3 py-2 text-sm ${
                e.player_id === meId ? 'bg-brand-50 font-medium' : ''
              }`}
            >
              <span className="w-5 text-neutral-500">{e.rank}</span>
              <span className="flex-1">{e.username}</span>
              <span className="font-semibold">{e.count}</span>
              <span className="text-xs text-neutral-400">{e.points} pt</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
