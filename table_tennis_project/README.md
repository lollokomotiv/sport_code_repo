# Table Tennis Match Analysis

Given the video of an amateur singles match, filmed from a tripod in a corner of the
hall (a diagonal view), find every stroke of both players, classify its side and type,
work out who won each point, and turn it into a report a player can use.

> Status: **design done, no code yet.** The decisions below come from a design review;
> the numbers that will judge them do not exist yet.

---

## Why

Amateur players film their matches and upload them to YouTube. Commercial tools
exist: OSAI classifies stroke types and tracks the ball, but from a dedicated, centred
side camera, and its development has stalled; Stupa focuses on ball placement and on
federations. What is missing is a fine-grained stroke breakdown from the video an
amateur actually has: "your backhand topspin ends in an error 40% of the time".

No public dataset covers this: the closest ones (OpenTTGames and its stroke-type
extension, TTStroke-21) use a fixed, centred side camera and non-commercial licences.
The dataset will be built here.

## Scope of the first version

For each match: sets, score after every point, server. For each rally: start, end,
point winner, let. For each stroke:

| Field | Values |
|---|---|
| impact | the instant the ball touches the racket |
| player | near or far (relative to the camera) |
| side | forehand, backhand |
| type | serve, topspin, flick, push, block, smash, other (with optional sub-type: lob, drop shot, chop block, defence) |
| outcome | in play, error, winner |

Out of scope for now: doubles, player identification, error kind (net/out), forced vs
unforced errors, ball placement, direction and speed. Everything that needs the ball
waits for a later phase.

## Approach

- **Outcome is derived, not learned.** Every stroke but the last one in a rally is in
  play; the last one is an error if its player lost the point, a winner if they won it.
- **The point winner is read from the flip scoreboard**, which is almost always in shot
  in amateur side views: the user boxes it once, and digits are recognised in a fixed
  crop. The rules of the game (11 points, win by 2, serve every 2 points) catch
  inconsistencies and send them to review.
- **A stroke is anchored on its impact**, a single frame, rather than on ambiguous
  start/end boundaries.
- **Impacts come from pose and audio.** Off-the-shelf pose estimation (RTMPose) needs no
  training; the racket-ball "tock" covers short strokes (blocks, pushes) where the
  wrist barely moves. A go/no-go threshold is fixed before measuring.
- **Strokes are classified** from pose sequences around each impact; left-handed
  players are mirrored.
- **A short manual setup per video** (table corners, scoreboard box, near player,
  handedness, first server) replaces recognition problems that would each be a project.
- **One tool for labelling and review**: the pipeline pre-fills the labeller, the user
  corrects. The MVP succeeds if reviewing a match takes at most a third of the time
  needed to label it from scratch.

## Data and validation

- Training data: amateur singles from YouTube in the diagonal view, split by channel so
  that the same player never ends up on both sides of the split. Highlight reels are
  used only for stroke classification and rare classes.
- Final measurement: matches filmed with a fixed protocol (phone, 1080p, 60 fps,
  tripod in the corner), at least one never opened during development.
- Every metric is reported with its denominator, separately for the near and far
  player and for 30 and 60 fps footage.

Videos show real people: they are not versioned, and the public video list holds no
titles or names.

## Phases

| Phase | Content |
|---|---|
| 0 | Recording protocol, video list, YouTube search script |
| 1 | Labelling tool, two pilot matches labelled |
| 2a | Impact detection (pose + audio) — the first risk |
| 2b | Scoreboard reading (in parallel) |
| 3 | Stroke classification |
| 4 | Rallies and outcomes |
| 5 | End-to-end pipeline, review, report, final measurement |

See [FEASIBILITY.md](FEASIBILITY.md) for the risk analysis, [CONTEXT.md](CONTEXT.md)
for the vocabulary and [docs/adr/](docs/adr/) for the decisions.
