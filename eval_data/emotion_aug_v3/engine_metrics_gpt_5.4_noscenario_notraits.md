# GPT-5.4 Emotion Engine Evaluation 

**File:** `engine_metrics_gpt_5.4_noscenario_notraits.json`

**Overall verdict:** the engine is **qualitatively reliable but quantitatively
over-sensitive** — it never gets the direction of an emotional change wrong and
its errors are small, but it moves emotions more often than humans would, and
its exact agreement sits just below the weakest human annotator.

## Setup

GPT-5.4 runs the emotion engine's judging prompt over 28 augmented utterances.
For each item it predicts the listener's next emotion state (fear, sadness,
overwhelm, shame; four levels: `not at all < low < high < extreme`) from the
**current emotion state + the single utterance only** — the scenario and
listener traits are withheld (`_noscenario_notraits`), so the engine sees
exactly the same information as the human annotators did.

emotion level:  not at all-negative high-positive 
state transition: delete current state-low; keep same 


Predictions are scored against two references:

1. **Human gold** — majority vote of 3 independent annotators (106 of 112
   labels; 6 unresolved ties excluded).
2. **Draft labels** — the labels the generating model assigned when writing
   each utterance.

## Results vs. human gold (the main comparison)

| Metric | Value | Good or bad? | What it means |
|---|---|---|---|
| Exact accuracy | **70.8%** | ⚠️ **Moderate — below human level.** Individual humans score 76–91% (mean 84%) against the same gold, so the engine trails the weakest annotator by ~6 points. | Predicted level matches the human gold exactly. |
| MAE | **0.30** | ✅ **Good.** Average miss is under one-third of a level (0 would be perfect; 1.0 would mean off by a full level on average). | Error magnitude on the 0–3 ordinal scale. |
| Change precision | **0.58** | ❌ **Weak — the engine's main flaw.** 42% of the changes it predicts are moves humans would not make: it **over-triggers**. | Of the changes the engine predicted, how many humans endorsed. |
| Change recall | **0.86** | ✅ **Good.** The engine almost never misses a change humans saw. | Of the changes humans saw, how many the engine caught. |
| Change F1 | 0.69 | ⚠️ **Moderate.** Dragged down by the weak precision, not by missed detections. | Harmonic mean of precision and recall. |
| Direction accuracy | **100%** | ✅ **Excellent — cannot be better.** Every disagreement with humans is about magnitude, never about direction. | When both agree something changed, does the direction (up/down) match. |
| Target-dimension accuracy | 77.8% | ✅ **Good.** On the emotion each utterance was engineered to move, the engine is close to human level. | Accuracy on the intended-change dimension. |
| Side-dimension accuracy | 68.4% | ⚠️ **Weak.** The 9-point gap below the target dimension shows where the over-triggering happens: unnecessary moves on emotions the utterance did not address. | Accuracy on the other three dimensions (tests restraint). |
| Quadratic weighted kappa | **0.87** | ✅ **Very good.** Above the 0.8 "almost perfect" convention; far beyond chance (0). | Chance-corrected ordinal agreement; big misses are penalized quadratically. |
**Per emotion:**

| Emotion | Exact | Good or bad? |
|---|---|---|
| shame | 89% | ✅ Strong |
| sadness | 84% | ✅ Strong |
| fear | 56% | ❌ Weak — threshold placement on escalations disagrees with humans |
| overwhelm | 54% | ❌ Weakest — but note humans also agree least with each other here (lowest inter-annotator kappa), so part of this is noise in the gold itself |

## Results vs. draft labels (secondary)

Exact 66.1%, QWK 0.87, change P/R = 0.38/0.91. The very low precision here is
**not** evidence against the engine: draft labels only mark the intended target
change and under-label side effects, so many engine-predicted side-dimension
changes count as "false positives" against drafts even when humans endorse
them.

## Takeaways

1. **Qualitatively correct, quantitatively over-sensitive.** Direction is
   always right and errors are small (MAE 0.30), but the engine changes
   emotion levels more often than the human panel does (precision 0.58 vs.
   recall 0.86).
2. **The failure mode is restraint, not detection.** Fixing it means
   strengthening the "default to no change" behavior in the judging prompt,
   not making the engine more sensitive.
3. **Overwhelm is the weakest construct** for both the engine and the human
   annotators — its definition needs tightening before further calibration.


Human Gold \ GPT    not at all    low    high    extreme
  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━  ━━━━━━  ━━━━━━━━━
   not at all                  39      4       0          0
  ──────────────────  ────────────  ─────  ──────  ─────────
   low                          3     11       3          0
  ──────────────────  ────────────  ─────  ──────  ─────────
   high                         1      4      17         16
  ──────────────────  ────────────  ─────  ──────  ─────────
   extreme                      0      0       0          8