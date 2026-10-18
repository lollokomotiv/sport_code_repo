# Volleyball Score

Given the video of an indoor volleyball match, reconstruct every rally and every touch:
which team won each rally, the score of every set, and the type of each touch (serve,
reception, set, attack, block, dig, free ball). Points and errors by type follow from
the last touch of each rally.

> Status: **setup.** No code yet. The approach below is a plan, not a result.

---

## Scope of the first version

- **Output:** for every match, the rallies with the team that won each one and the score
  of every set; every touch with its time, court side and type; points and errors by
  type; serves attributed to players through the rotation.
- **Not evaluated:** the quality of a touch (DataVolley's `#` `+` `!` `-` `/` `=`).
  Judging it, and recognising players automatically, are research problems this project
  does not take on. Player numbers on other touches can be typed in during review.
- **Workflow:** the pipeline runs locally after the match; a review screen lets a person
  fix what the pipeline got wrong. Success means reviewing is much faster than scouting
  the match by hand, not zero errors.
- **Rules:** FIPAV, best of five sets, sets to 25, deciding set to 15, two-point margin.
- **Not in scope:** touch evaluations, automatic player recognition, live scoring,
  uploads by the team.

---

## Data

Matches of an amateur team (FIPAV Serie D), filmed with an iPhone at 1080p and 60 fps
from the same spot as the gym's fixed camera, high behind an end line, under a
[recording protocol](docs/protocollo-ripresa.md). The videos show real
people and are **not public**: they live in `data/raw/` and are never committed.

| Set | Role |
|---|---|
| Sample video (gym camera, 15 fps, 48 minutes) | first test of the ball tracker; 10 minutes labeled to try the labeling page, not used as ground truth |
| League match 1, labeled by hand | development and threshold tuning |
| League match 2, labeled by hand | final measurement only, not looked at during development |
| Public matches with a similar view | demo images and extra checks, never measurements |

Public images of this project are court diagrams (ball track, rally timeline) or clips
with faces blurred.

---

## Planned approach

1. **Ground truth first.** Label every touch of the two league matches by hand, for both
   teams: time, court side, type, and the winner of each rally. The labels are checked
   by the same rules the pipeline uses.
2. **Ball tracking.** A pre-trained ball tracker
   ([fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference),
   VballNet models) gives the ball trajectory. Rallies and touches are found from it:
   breaks in the flight path, and which side of the net they happen on. No training at
   this stage.
3. **Touch types from the rules of the game.** The first touch is the serve, the
   receiving team's first touch is the reception, then set and attack; after an attack,
   a touch at the net on the other side is a block, otherwise a dig. Computer vision
   only has to find when and on which side each touch happens.
4. **Score and server.** The team that wins a rally serves the next one. With the
   starting lineups and substitutions from the scoresheet, the rotation tells which
   player served. Every set score must be legal and the number of rallies must equal
   the points played.
5. **Review screen.** The labeling page loads the pipeline's proposals; a person fixes
   them and types player numbers where wanted.

Prior work and how each repository fits: [docs/prior-work.md](docs/prior-work.md).

---

## Next steps

Tracked as GitHub issues under
[#10](https://github.com/lollokomotiv/sport_code_repo/issues/10). The main open risk is
touch detection: whether the ball tracker finds when and where each touch happens,
especially on the far half of the court and at the net.
