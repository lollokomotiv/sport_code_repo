# Volleyball Score

Given the video of an indoor volleyball match, reconstruct the sequence of rallies and
the outcome of each one: which team won the point and how the rally ended. The running
score of every set should come out of the video, not out of a manual scorer.

> Status: **setup.** No code yet. The approach below is a plan, not a result.

---

## Data

A few matches of an amateur team, single fixed camera, low resolution. The videos show
real people and are **not public**: they live in `data/raw/` and are never committed,
nor are frames, crops or clips taken from them.

---

## Planned approach

1. **Ground truth first.** Label every rally of the sample videos by hand: start and end
   time, serving team, point winner, how the rally ended. Every later number is measured
   against these labels, with one match kept aside for the final measurement.
2. **Rally segmentation from the ball.** A pre-trained ball tracker
   ([fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference),
   VballNet models) gives the ball trajectory and splits the video into rallies. No
   training at this stage.
3. **Point winner from the serve.** Under rally scoring the team that wins a rally serves
   the next one, so the winner of rally *n* is the side that serves rally *n+1*. Spotting
   the serving side is easier than judging where the ball lands. The last rally of a set
   needs another signal.
4. **How the rally ended** (ace, service error, attack kill, block, attack error) comes
   later, with a review screen where a person confirms or corrects each rally.

Prior work and how each repository fits: [docs/prior-work.md](docs/prior-work.md).

---

## Open questions

- [ ] What "score" means for the first version: point winner only, or also how the
      rally ended, or touch-by-touch scouting (DataVolley style)?
- [ ] Does the ball tracker work on these videos at all? Detection rate on one match
      decides the rest of the plan.
- [ ] Is the scoreboard visible in the footage? It would be a free check on the
      reconstructed score.
- [ ] Who uses the app and how: batch report after the match, or a review interface?
