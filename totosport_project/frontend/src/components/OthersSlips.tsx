import { useState } from 'react'

import type { PlayerPredictions } from '@/types/prediction'
import type { Competition, MatchOut } from '@/types/round'
import { deriveSign, formatCompetition, formatDate, groupByCompetition } from '@/utils'

/**
 * Elenco delle schedine dei giocatori, ognuna un sotto-accordion (chiuso di default).
 * Con `withResults` mostra il risultato reale a fianco e colora le previsioni
 * (verde = giusto, rosso = sbagliato) — usato nel tab "Dettaglio" (giornate completate).
 */
export default function OthersSlips({
  players,
  matches,
  withResults = false,
}: {
  players: PlayerPredictions[]
  matches: MatchOut[]
  withResults?: boolean
}) {
  if (players.length === 0) {
    return <p className="text-sm text-neutral-500">Nessuno ha compilato la schedina.</p>
  }
  return (
    <div className="grid gap-2">
      {players.map((pl) => (
        <PlayerSlip key={pl.player_id} player={pl} matches={matches} withResults={withResults} />
      ))}
    </div>
  )
}

function PlayerSlip({
  player,
  matches,
  withResults,
}: {
  player: PlayerPredictions
  matches: MatchOut[]
  withResults: boolean
}) {
  const [open, setOpen] = useState(false)

  // Totale gol reale per lega (per colorare il pronostico "totale gol")
  function actualLeagueTotal(comp: Competition): number | null {
    const ms = matches.filter(
      (m) => m.competition === comp && m.actual_home_goals != null && m.actual_away_goals != null,
    )
    if (ms.length === 0) return null
    return ms.reduce((s, m) => s + (m.actual_home_goals! + m.actual_away_goals!), 0)
  }

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
            {groupByCompetition(matches).map((group) => (
              <div key={group.competition} className="grid gap-1">
                {groupByCompetition(matches).length > 1 && (
                  <div className="mt-1 text-xs font-semibold uppercase tracking-wide text-neutral-400">
                    {formatCompetition(group.competition)}
                  </div>
                )}
                {group.items.map((m) => {
              const mp = player.match_predictions.find((x) => x.match_id === m.id)
              const hasResult = m.actual_home_goals != null && m.actual_away_goals != null
              const showRes = withResults && hasResult
              const actualSign = hasResult
                ? deriveSign(m.actual_home_goals!, m.actual_away_goals!)
                : null
              const signOk = mp != null && actualSign != null && mp.predicted_sign === actualSign
              const exactOk =
                mp != null &&
                hasResult &&
                mp.predicted_home_goals != null &&
                mp.predicted_away_goals != null &&
                mp.predicted_home_goals === m.actual_home_goals &&
                mp.predicted_away_goals === m.actual_away_goals
              const color = (ok: boolean) =>
                showRes ? (ok ? 'text-brand-600' : 'text-miss') : ''
              return (
                <div key={m.id} className="flex items-center gap-2">
                  <span className="flex-1 text-neutral-600">
                    {m.home_team} <span className="text-neutral-400">vs</span> {m.away_team}
                  </span>
                  {mp ? (
                    <>
                      <span className={`font-semibold ${color(signOk)}`}>{mp.predicted_sign}</span>
                      {mp.predicted_home_goals != null && mp.predicted_away_goals != null && (
                        <span className={color(exactOk)}>
                          {mp.predicted_home_goals}-{mp.predicted_away_goals}
                        </span>
                      )}
                    </>
                  ) : (
                    <span className="text-neutral-300">—</span>
                  )}
                  {showRes && (
                    <span className="text-xs text-neutral-400">
                      · reale {m.actual_home_goals}-{m.actual_away_goals}
                    </span>
                  )}
                </div>
              )
                })}
              </div>
            ))}
          </div>
          {player.round_predictions.length > 0 && (
            <div className="mt-2 text-xs text-neutral-500">
              Totale gol:{' '}
              {player.round_predictions.map((rp, i) => {
                const actual = actualLeagueTotal(rp.competition)
                const ok = actual != null && rp.total_goals_guess === actual
                return (
                  <span key={rp.id}>
                    {i > 0 && ' · '}
                    {formatCompetition(rp.competition)}{' '}
                    <span
                      className={
                        withResults && actual != null
                          ? ok
                            ? 'font-medium text-brand-600'
                            : 'font-medium text-miss'
                          : ''
                      }
                    >
                      {rp.total_goals_guess}
                    </span>
                    {withResults && actual != null && (
                      <span className="text-neutral-400"> (reale {actual})</span>
                    )}
                  </span>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
