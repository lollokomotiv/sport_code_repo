import { useState } from 'react'

import type { PlayerPredictions } from '@/types/prediction'
import type { MatchOut } from '@/types/round'
import { formatCompetition, formatDate } from '@/utils'

/** Elenco delle schedine dei giocatori, ognuna un sotto-accordion (chiuso di default). */
export default function OthersSlips({
  players,
  matches,
}: {
  players: PlayerPredictions[]
  matches: MatchOut[]
}) {
  if (players.length === 0) {
    return <p className="text-sm text-neutral-500">Nessuno ha compilato la schedina.</p>
  }
  return (
    <div className="grid gap-2">
      {players.map((pl) => (
        <PlayerSlip key={pl.player_id} player={pl} matches={matches} />
      ))}
    </div>
  )
}

function PlayerSlip({ player, matches }: { player: PlayerPredictions; matches: MatchOut[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="overflow-hidden rounded-xl border bg-white">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-sm"
      >
        <span className="font-medium">{player.username}</span>
        <span className="flex items-center gap-2">
          {player.submitted_at && (
            <span className="text-xs text-neutral-400">{formatDate(player.submitted_at)}</span>
          )}
          <span className="text-neutral-400">{open ? '▲' : '▼'}</span>
        </span>
      </button>
      {open && (
        <div className="border-t px-3 py-2">
          <div className="grid gap-1 text-sm">
            {matches.map((m) => {
              const mp = player.match_predictions.find((x) => x.match_id === m.id)
              return (
                <div key={m.id} className="flex items-center gap-2">
                  <span className="flex-1 text-neutral-600">
                    {m.home_team} <span className="text-neutral-400">vs</span> {m.away_team}
                  </span>
                  {mp ? (
                    <>
                      <span className="font-semibold">{mp.predicted_sign}</span>
                      {mp.predicted_home_goals != null && mp.predicted_away_goals != null && (
                        <span className="text-neutral-500">
                          {mp.predicted_home_goals}-{mp.predicted_away_goals}
                        </span>
                      )}
                    </>
                  ) : (
                    <span className="text-neutral-300">—</span>
                  )}
                </div>
              )
            })}
          </div>
          {player.round_predictions.length > 0 && (
            <div className="mt-2 text-xs text-neutral-500">
              Totale gol:{' '}
              {player.round_predictions
                .map((rp) => `${formatCompetition(rp.competition)} ${rp.total_goals_guess}`)
                .join(' · ')}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
