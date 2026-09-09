# Emotion Engine — Overview and Evaluation Report

Dataset: `eval_data/emotion_aug_v3_3level/` (28 utterances, 112 emotion labels)

## 1. How the emotion engine works

The emotion engine (`emotion_engine.py`) tracks the emotional state of an
agent (the patient or a relative) in the ED world-model simulation.

**State.** Four emotion dimensions — fear, sadness, overwhelm, shame — each at
one of three ordinal levels: `absent` < `present` < `extreme`. Every encounter
starts deterministically at all-absent; only dialogue moves emotions.

**Input (one call per received utterance).**
- listener role and speaker role
- the listener's current four-emotion state
- clinical context/scenario (who the listener is, how they are doing clinically)
- recent dialogue (context only)
- the single utterance to score

**Output.** The listener's next emotion state — a level (`absent` /
`present` / `extreme`) for each of the four emotions after hearing the
utterance （— plus an `evidence` phrase quoted from the utterance and a short
`reasoning` string, returned as one JSON object.）

**Judging rules.** The prompt embeds shared emotion definitions and
transition rules: default to no change; move at most one level per utterance;
acknowledgment is not relief; extreme is a terminal, absorbing crisis level.

**Role in the ED world model.** The TurnLoop calls the engine (through
`EmotionEngineAdapter`) once per turn for each verbal message addressed to the
tracked agent; messages spoken to someone else, and self-utterances, never
move the listener's state. The resulting patient emotion is fed into agent
observations (it conditions how the patient speaks) and into termination:
if any emotion reaches `extreme`, the encounter ends
(`patient_emotion_extreme`). Extreme never de-escalates.

**Engine model:** Claude Opus 4.8, temperature 0.

## 2. Engine predictions vs. augmentation target labels

The 28 test utterances were produced by rewriting utterances from full
simulations toward target emotion/level cells; the generating model also wrote
a draft next-state label for every item.

- **Utterance augmentation / draft-label model:** Claude Sonnet 4.6
  (per the dataset build records)
- **Emotion engine model:** Claude Opus 4.8

| Metric | Engine vs draft labels |
|---|---|
| Exact accuracy | 83.9% |
| MAE (0–2) | 0.161 |
| QWK | 0.779 |
| Change precision / recall / F1 | 0.48 / 0.65 / 0.55 |
| Direction accuracy | 100% |

How to read these values:

- **Exact accuracy** — share of labels predicted at exactly the right level.
  Higher is better; 83.9% is good.
- **MAE** — average size of the level error on the 0–2 ordinal scale. Lower is
  better; 0.161 means errors are rare and never more than one level.
- **QWK** — chance-corrected agreement (0 = chance, 1 = perfect; big misses
  penalized more). Higher is better; 0.779 is substantial agreement.
- **Change precision / recall** — of the changes the engine predicted, how
  many the draft labels endorse (precision), and of the labeled changes, how
  many the engine caught (recall). Higher is better; 0.48 precision looks
  weak but is an artifact (see below), 0.65 recall is decent.
- **Direction accuracy** — when both sides agree something changed, does the
  direction (up/down) match. Higher is better; 100% is perfect.

The low change precision is expected and is not evidence against the engine:
draft labels mark only the intended target change and under-label side
effects, so engine-predicted side-dimension changes count as false positives
here even when human annotators endorse them.

## 3. Engine predictions vs. human gold (3 annotators)

Gold = majority vote of 3 independent human annotators (112/112 labels
covered; 64 labels unanimous, 48 decided 2:1).

| Metric | Engine vs human gold |
|---|---|
| Exact accuracy | **83.0%** |
| MAE (0–2) | 0.170 |
| QWK | 0.759 |
| Macro F1 (3 levels) | 0.756 |
| Change precision / recall / F1 | 0.74 / 0.57 / 0.64 |
| Direction accuracy | 100% |

How to read these values:

- **Exact accuracy** (higher = better) — 83.0% is good: inside the human
  annotator range (78.6–93.8%).
- **MAE** (lower = better) — 0.170 is good: errors are uncommon and always a
  single level.
- **QWK** (higher = better) — 0.759 is substantial, chance-corrected
  agreement.
- **Macro F1** (higher = better) — 0.756 is good overall; it is pulled down
  by the small, hard `extreme` class (F1 0.57), not by absent/present.
- **Change precision 0.74** (higher = better) — good: three quarters of the
  changes the engine predicts are human-endorsed. **Recall 0.57** is
  moderate: the engine is deliberately conservative and misses some changes
  humans saw — the safer direction for a termination-linked signal.
- **Direction accuracy** (higher = better) — 100% is perfect: the engine
  never moves an emotion the wrong way.

For reference, the three annotators individually score 78.6%, 84.8%, and
93.8% against the same gold — the engine sits inside the human range. On the
64 unanimous labels the engine scores **96.9%** (62/64).

**Confusion matrix (human gold → engine prediction):**

| Gold \ Engine | absent | present | extreme |
|---|---|---|---|
| **absent** (43) | **39** | 4 | 0 |
| **present** (61) | 9 | **50** | 2 |
| **extreme** (8) | 0 | 4 | **4** |

**Per emotion (exact):** fear 89.3%, overwhelm 85.7%, sadness 82.1%,
shame 75.0%.

All 19 errors are single-level; none cross two levels and no predicted change
is in the wrong direction. Errors concentrate on the extreme boundary
(2 false alarms, 4 misses — all on labels where the annotators themselves
split 2:1) and on absent↔present jitter for sadness and shame side triggers.


