# Feasibility & Key Challenges

Honest assessment of the table tennis action detection project. Bottom line: it is
**feasible**, but it is close to a research-grade effort, not a weekend YOLO fine-tune.
The hard part is **not the model** — it is the data and one of the two sub-tasks.

---

## The two tasks have very different difficulty

- **A) Recognize the stroke type** → hard but tractable.
- **B) Detect whether the shot was a mistake** → much harder; it depends on the ball and
  its trajectory.

Treat them as two phases, not one requirement.

---

## Biggest challenges (in order of risk)

1. **Annotation is the real bottleneck.** Fine-grained stroke labeling needs thousands of
   labeled shots, with temporal boundaries (when does a topspin start/end?) that are often
   ambiguous. It requires an expert annotator — an advantage for quality, but it makes the
   single domain expert the single point of failure. Manual labeling is extremely slow.
   This, not the architecture, is what kills most of these projects.

2. **The ball.** Tiny (a few pixels), very fast, motion-blurred, sometimes invisible
   between frames. And the ball is exactly the signal needed for task B (net, out, double
   bounce). Tracking it from 25-30 fps YouTube video is unreliable. Dedicated approaches
   exist (TrackNet-style heatmaps) but remain fragile.

3. **Defining "wrong shot" objectively from video alone.** An error is usually a
   *consequence* (the point ends): attributing it to a specific shot and deciding whether
   it was net/out/forced requires trajectory + bounces + rules logic, from a single camera
   (and the far player is small and occluded). This is the riskiest piece.

4. **YouTube footage heterogeneity.** Different angles, zoom, replays, on-screen graphics,
   broadcast cuts, frame rates, tables and lighting. A model trained on one setup does not
   generalize to another (domain shift). At 30 fps a fast stroke lasts 2-4 frames;
   high-speed footage would help a lot but is rarely available.

5. **Temporal segmentation.** A rally is continuous: you must *localize* stroke instances
   in time, not just classify a pre-cut clip. Temporal action detection is harder than
   classification and needs frame-accurate boundary labels.

6. **Domain details a player knows well:** handedness, near vs far player (mirrored and
   different in size), strokes that differ mainly by **racket angle and contact** — often
   occluded by the body/hand and visible for only a few frames. Class imbalance (many
   topspins/pushes, few smashes/lobs; errors rarer than valid shots).

7. **Legal/ToS** considerations for scraping YouTube (minor but real).

---

## The good news / how to de-risk

- **Domain expertise is the asset.** Defining a **closed, sensible taxonomy** (start with
  5-6 classes: serve, forehand topspin, backhand topspin, push, block, smash) and the
  **error criterion** is an expert decision — the project's competitive advantage.
- **Use pose (player skeleton) as an intermediate representation** for task A: off-the-shelf
  pose models work decently and are more robust than raw pixels for classifying stroke
  *type* (most table-tennis stroke-recognition literature does this). This partly sidesteps
  the ball problem — **but it does not help task B.**
- **Narrow the scope at first:** a single consistent camera angle + a small taxonomy +
  start with **classification of pre-cut clips** (easier) before temporal detection.
- **Decouple task B:** initially derive the "error" *weakly* from rally structure (the last
  shot before the point stops) instead of a fine visual judgment.
- **Prior work to study** (verify the details): **TTNet / OpenTTGames dataset** (ball
  detection + bounce/net event spotting in real time) and the **MediaEval table tennis
  stroke classification task (TTStroke-21-style dataset)**. Caveat: those datasets use a
  **fixed, controlled camera** — exactly the difference from arbitrary YouTube video, which
  is challenge #4.

---

## In one line

Feasible **if you start narrow**: an MVP of *stroke-type classification* on fixed-camera
clips, 5-6 classes, using pose features. Reliable "wrong shot" detection comes later and
depends on being able to track the ball — keep it as phase 2, not an initial requirement.

---

## Suggested next step (on paper, no code yet)

Write down two things together, since they unlock everything else:
1. The **stroke taxonomy** (the closed list of action classes).
2. The **operational definition of an error** (what objectively counts as a wrong shot).
