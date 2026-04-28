# EMSim Cardiology — Rule vs LLM Engine Comparison

## Setup

- **Dataset:** Cardiology only (132 transition pairs across 30 cases). Cardiology is the only category in which `mechanism.rhythm` is fully labeled
- **Engines:**
  - **Rule:** `rule_engine.engine:RuleEngine` — physiology engine with hand-authored pathology / intervention / drug rules.
  - **LLM:** `emsim_llm_engine:LLMEngine` — `gpt-4o-mini`, zero-shot, response-cached. 
- **Eval changes this session:**
  - Two independent channels surfaced separately: vitals (6 vitals × within-tol + direction-match) and rhythm (7-enum exact match).
  - New metrics: `vitals_pass_count` (0–6 fully-passing vitals per pair) and `vitals_partial_score` (continuous 0–1, fraction of 12 underlying checks passed)

## Headline numbers

| Metric | Rule | LLM | Δ |
|---|---|---|---|
| **Vitals strict (6/6 pass)** | 18/132 = **13.6%** | 9/132 = **6.8%** | rule +6.8 pp |
| **Vitals ≥5/6 pass** | 38/132 = 28.8% | 39/132 = **29.5%** | LLM +1 pair |
| **Vitals ≥4/6 pass** | 68/132 = **51.5%** | 61/132 = 46.2% | rule +5.3 pp |
| **Vitals partial score** | **0.660** | 0.642 | rule +0.018 |
| **Rhythm pair pass** | 93/128 = 72.7% | 91/125 = **72.8%** | tie |

The choice of cutoff materially changes who "wins":
- **Strict 6/6 favors rule** (rule 2× LLM).
- **≥5/6 ties** (LLM marginally ahead).
- **≥4/6 and partial favor rule** by a small margin.



## Per-vital breakdown

| Vital | Within-tol | | Dir-match | | MAE | |
|---|---:|---:|---:|---:|---:|---:|
|  | Rule | LLM | Rule | LLM | Rule | LLM |
| HR | **58.3%** | 53.8% | **50.0%** | 46.2% | **25.94** | 27.36 |
| BP_sys | 52.3% | **56.1%** | 40.9% | **46.2%** | 26.09 | **25.66** |
| BP_dia | 65.9% | **68.2%** | **51.5%** | 44.7% | 15.54 | **15.24** |
| RR | **74.2%** | 71.2% | **72.0%** | 68.2% | **4.34** | 4.86 |
| O2Sat | 64.4% | **65.2%** | **63.6%** | 62.9% | 13.93 | **13.64** |
| T | **100%** | 94.7% | **98.5%** | 93.2% | **0.00** | 0.11 |

Patterns:
- **Rule is stronger on HR, RR, T** (the "easy" vitals where engine can pass-through or apply tight mapping). T is degenerate: rule never moves T in Cardiology, GT also rarely moves it.
- **LLM is stronger on BP within-tol and dir-match** (BP_sys/BP_dia magnitudes), suggesting LLM's clinical priors give better BP deltas than rule's mostly-identity behavior on non-arrest pairs.
- **Direction-match is uniformly low (40–50%) on HR/BP** for both engines — the engine is often within tolerance numerically but moves vitals in the wrong direction (or holds stable when GT moved).

## Rhythm channel 

| Engine | Rhythm pass | Skipped (GT-null or pred-null) |
|---|---|---|
| Rule | 93/128 = **72.7%** | 4 |
| LLM | 91/125 = **72.8%** | 7 (4 GT-null + 3 LLM-null) |

- **The two engines hit the same accuracy** on the rhythm channel (~72.8%).

  

## Why strict 6/6 is misleading

The engine evaluation uses two thresholds per vital:
- Tolerance band (e.g. HR ±10 bpm, BP_sys ±15 mmHg, O2Sat ±3%) — calibrated to absorb author quantization noise.
- Direction dead-zone (HR ±3, BP ±5, O2Sat ±1) — much tighter, designed to catch wrong-direction predictions.

The dead-zone is deliberately tight (Phase 4 docs explicitly note "any drift rule that pushes a vital past the dead-zone in a direction the author didn't script costs a dir_match"). The combination of 6 independent vitals × 2 checks per vital means a pair only has to lose one of 12 checks to fail strict 6/6.

For LLM specifically, this is punishing: LLM tends to predict reasonable BP/HR magnitudes but occasionally misses by the dead-zone on one vital. Rule's frequent identity behavior happens to pass dead-zone checks on stable cases.

The newly-added `vitals_partial_score` and the `vitals_pass_count` distribution capture this shape — they're the primary signals to track during engine iteration. Strict 6/6 should be retained as a backward-compatible reference (matches Phase 4–6 baseline numbers) but is too coarse to drive incremental improvement.
