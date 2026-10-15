# Volleyball Score

Given the video of an indoor volleyball match, reconstruct the sequence of rallies and
which team won each one. The running score of every set should come out of the video,
not out of a manual scorer. How each rally ended (ace, kill, block, error) is the
second step.

> Status: **setup.** No code yet. The approach below is a plan, not a result.

---

## Scope of the first version

- **Output:** the rallies of a match, each with its start time in the video and the
  team that won it, and the score of every set.
- **Workflow:** the pipeline runs locally after the match; a review screen lets a person
  fix the rallies the pipeline got wrong. Success means reviewing is much faster than
  scouting the match by hand, not zero errors.
- **Rules:** FIPAV, best of five sets, sets to 25, deciding set to 15, two-point margin.
- **Not in scope:** live scoring, touch-by-touch scouting, uploads by the team.

---

## Data

Matches of an amateur team (FIPAV Serie D), one fixed camera, high behind an end line,
filmed under a [recording protocol](docs/protocollo-ripresa.md). The videos show real
people and are **not public**: they live in `data/raw/` and are never committed.

| Set | Role |
|---|---|
| Older videos (few, low resolution, uncontrolled setup) | first test of the ball tracker, worst case |
| League match 1, labeled by hand | development and threshold tuning |
| League match 2, labeled by hand | final measurement only, not looked at during development |
| Public matches with a similar view | demo images and extra checks, never measurements |

Public images of this project are court diagrams (ball track, rally timeline) or clips
with faces blurred: the team agreed to be filmed, opponents and referees did not.

---

## Planned approach

1. **Ground truth first.** Label every rally of the two league matches by hand: start and
   end time, serving team, point winner. The label schema already has a field for how the
   rally ended, filled where possible, so the same labels serve the second step.
2. **Rally segmentation from the ball.** A pre-trained ball tracker
   ([fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference),
   VballNet models) gives the ball trajectory and splits the video into rallies. No
   training at this stage.
3. **Point winner from the serve.** Under rally scoring the team that wins a rally serves
   the next one, so the winner of rally *n* is the side that serves rally *n+1*. Spotting
   the serving side is easier than judging where the ball lands. The last rally of a set
   needs no extra signal: a set only ends on a point by the team that closes it.
4. **Consistency checks.** Every reconstructed set score must be a legal score, and the
   number of rallies must equal the points played. The set scores written down at the
   venue are a free check on the result.
5. **How the rally ended** comes next, on top of the same review screen.

Prior work and how each repository fits: [docs/prior-work.md](docs/prior-work.md).

---

## Next steps

Tracked as GitHub issues under
[#10](https://github.com/lollokomotiv/sport_code_repo/issues/10). The main open risk:
the sample video comes from the gym's fixed camera at **15 fps**, and the ball tracker
has not been tested at that frame rate yet. The scoreboard is in frame but unreadable
at that quality.
