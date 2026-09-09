# Emotion Engine Evaluation — Claude Opus 4.8 (3-level scale, revised extreme rule)

**Files:** `engine_predictions_claude_opus_4_8.json` / `engine_metrics_claude_opus_4_8.json`

## Setup

28 augmented ED utterances, 112 emotion labels (4 emotions × 28 items). The
engine predicts the listener's next emotion state (`absent` / `present` /
`extreme`) from the current state + the utterance + scenario summary +
listener traits, at temperature 0. Gold = majority vote of 3 independent human
annotators (112/112 coverage on the 3-level scale — no unresolved ties).

The prompt is the redesigned 3-level rule set, including the rewritten
extreme rule: *extreme describes the listener's breakdown, not the severity of
the news; a concrete protective plan usually buffers extreme, unless the
utterance conveys imminent death, catastrophic loss, or devastating
humiliation in a way that makes breakdown likely.*


Metric                     Result
  ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━
   Human unanimous cells    64 / 112
  ───────────────────────  ──────────
   Exact Accuracy              96.9%
  ───────────────────────  ──────────
   Within-1 Accuracy          100.0%
  ───────────────────────  ──────────
   MAE                         0.031
  ───────────────────────  ──────────
   QWK                         0.940

   Human/claude   absent  present  extreme
  absent          36       1        0
  present          0      26        1
  extreme          0       0        0

  Per Emotion

   Emotion        n     Exact      MAE      QWK
  ━━━━━━━━━━━  ━━━━━  ━━━━━━━━  ━━━━━━━  ━━━━━━━
   fear          18     88.9%    0.111    0.719
  ───────────  ─────  ────────  ───────  ───────
   sadness       18    100.0%    0.000    1.000
  ───────────  ─────  ────────  ───────  ───────
   overwhelm     14    100.0%    0.000    1.000
  ───────────  ─────  ────────  ───────  ───────
   shame         14    100.0%    0.000    1.000

## Overall metrics (n = 112)

| Metric | Value | Notes |
|---|---|---|
| Exact accuracy | **83.0%** | Individual annotators score 78.6–93.8% vs the same gold (mean 85.7%) — the engine sits inside the human range |
| Macro F1 (over 3 levels) | **0.756** | absent 0.86 / present 0.84 / extreme 0.57 |
| MAE (0–2 ordinal) | 0.170 | Errors are almost always a single level |
| QWK | 0.759 | Substantial chance-corrected agreement |
| Change P / R / F1 | 0.74 / 0.57 / 0.64 | Of predicted changes, 74% are human-endorsed |
| Direction accuracy | **100%** | No predicted change ever moves in the wrong direction |

**Per-level P/R/F1:**

| Level | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| absent | 0.81 | 0.91 | 0.86 | 43 |
| present | 0.86 | 0.82 | 0.84 | 61 |
| extreme | 0.67 | 0.50 | 0.57 | 8 |

Extreme remains the weakest class and fully explains the macro-F1 gap:
2 false alarms (idx8 fear, idx16 overwhelm) and 4 misses (idx24/25 overwhelm,
idx27/28 shame).

## Per-emotion analysis (n = 28 each)

| Emotion | Exact | MAE | What the errors are |
|---|---|---|---|
| **fear** | **89.3%** | 0.107 | Best dimension. 3 errors: one extreme false alarm (idx8, an intubation-now utterance humans unanimously kept at present), one spurious onset, one spurious drop. The clinical-threat-escalation rule is well calibrated. |
| **overwhelm** | 85.7% | 0.143 | The biggest winner of the redesign (GPT-5.4 scored 50%). 4 errors: 2 missed extremes (idx24/25 — the "information-avalanche collapse" form), 1 extreme false alarm (idx16), 1 spurious drop. |
| **sadness** | 82.1% | 0.179 | 5 errors, all absent↔present in both directions — boundary jitter on side-dimension loss cues (e.g. mention of "the baby"), never magnitude. No extreme labels exist for sadness in this set. |
| **shame** | **75.0%** | 0.250 | Now the weakest dimension. 7 errors: 5 absent↔present (4 of them the engine *dropping* shame that humans kept — it under-holds shame after blame utterances), plus both missed extremes (idx27/28, public humiliation during helplessness). The "neutral disclosure ≠ shame" rule works; shame *persistence* and its extreme form need attention. |

## Unanimous-labels subset (all 3 annotators agree, n = 64)

Inter-annotator agreement on the 3-level scale: all three annotators chose the
same level on **64 of 112 labels (57.1%)**; the remaining **48 labels (42.9%)**
were decided by a 2:1 majority vote. (Per emotion, unanimous counts are:
fear 18/28, sadness 18/28, overwhelm 14/28, shame 14/28.)

| | Full gold (112) | Unanimous only (64) | Contested 2:1 (48) |
|---|---|---|---|
| Engine exact | 83.0% | **96.9% (62/64)** | 64.6% (31/48) |
| Annotators (reference) | 78.6–93.8% | 100% by construction | 50.0 / 64.6 / 85.4% |

On labels where the ground truth is uncontroversial the engine is
near-ceiling: sadness 18/18, overwhelm 14/14, shame 14/14, fear 16/18. The
only two misses are fear on idx8 (extreme false alarm) and idx12 (spurious
onset). Note that the unanimous subset contains **no extreme labels**
(37 absent + 27 present) — the annotators never unanimously agreed on
extreme, so this subset cannot test the termination boundary. On contested
labels the engine (64.6%) exactly matches the median human annotator.

## Takeaways

1. **The engine is at human-annotator level** on this benchmark: inside the
   human accuracy range overall, and 96.9% wherever humans actually agree.
2. **Residual error is concentrated in two places:** the extreme boundary
   (6 of 19 errors; false alarms and misses now roughly balanced, which is
   the safe trade-off for a termination trigger) and **shame persistence**
   (the engine releases shame too readily and under-detects its
   public-humiliation extreme form).
3. **Further prompt tuning has hit diminishing returns.** Of the 19 errors,
   17 fall on labels where the annotators themselves split 2:1, and all 8
   extreme gold labels are 2:1 votes. Adjudicating the 48 contested labels
   (especially the extreme ones) would raise the measurable ceiling for every
   model evaluated on this set.
