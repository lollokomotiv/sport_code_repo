# Table Tennis Action Detection

A computer vision project to **automatically recognize the actions in a table tennis
(ping pong) match** from video, classifying, for each shot:

1. **The type of action/stroke** (e.g. serve, forehand, backhand, topspin, push, smash, block…)
2. **Whether the shot was executed correctly or was a mistake** (error vs valid shot)

The end goal is a model able to analyze a match and produce an annotated sequence of
actions, useful for statistics, technical analysis and review.

---

## Planned approach

The envisioned pipeline has four stages:

### 1. Data collection — scraping videos from YouTube
- Download table tennis matches from YouTube (preferring stable footage, side/top camera
  angle, good quality).
- Normalize the videos (resolution, frame rate) and cut them into clips per action/point.
- **Note**: respect YouTube's terms of service and the copyright of the content; use the
  videos for research/training only, not for redistribution.

### 2. Video annotation (tagging)
This is the most delicate stage and drives the quality of the model. To be defined:
- **What to annotate**: bounding boxes (ball/players/racket), player keypoints, or
  temporal labels per segment (start/end of each stroke).
- **Stroke taxonomy**: a closed, unambiguous list of action types.
- **Definition of "wrong shot"**: an objective criterion (ball into the net, out,
  double bounce…) to label errors.
- **Tooling**: e.g. CVAT, Label Studio or Roboflow for video labeling.

### 3. Training the detection model
- An object/action detection model (e.g. the YOLO family for frame-by-frame detection,
  or *temporal action detection* architectures if the temporal dimension is needed).
- Possible combination: spatial detection (where the shot happens) + temporal
  classification (which stroke it is, and whether it is valid or not).

### 4. Inference and output
- Given a video, return the sequence of actions with timestamps, stroke type and
  outcome (valid/wrong).

---

## Open questions (to decide before starting)

- [ ] **Task granularity**: per-frame detection, clip classification, or temporal action
      detection over the whole rally?
- [ ] **Annotation schema** and the final stroke taxonomy.
- [ ] **Operational criterion** to distinguish a valid shot from an error.
- [ ] **Data volume** required and the annotation strategy (manual vs assisted).
- [ ] **Starting model** (pre-trained) and the evaluation metric (mAP, per-class accuracy…).
- [ ] **Legal aspects** of scraping and of using the videos.

---

## Proposed structure (tentative)

```
table_tennis_project/
├── data/              # raw videos, clips, annotations (NOT versioned)
├── scraping/          # scripts to download/normalize videos
├── annotation/        # labeling config and tooling, dataset export
├── training/          # training code, model configs
├── inference/         # scripts to analyze new videos
└── notebooks/         # data exploration and prototypes
```

> Status: **initial idea.** This README fixes the goal and the approach; the structure
> and technical choices will be defined as the project progresses.
