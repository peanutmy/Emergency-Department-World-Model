# EMSim Evaluation Harness

Step-wise evaluation of physiology engines against the 808 authored transition pairs.

## Files

- `emsim_engine.py` — engine interface (`PhysiologyEngine` protocol) + `IdentityEngine` baseline
- `emsim_eval.py`   — harness: loads pairs, runs engine, scores, reports

Drop both files in `D:\wmed\emsim\` alongside `transitions/`.

## Quick start

```bash
# Baseline (identity: after = before)
python emsim_eval.py --transitions transitions

# Your engine (put class in rule_engine.py)
python emsim_eval.py --transitions transitions --engine rule_engine:RuleEngine

# Export per-pair CSV for deeper analysis
python emsim_eval.py --transitions transitions --export results.csv
```

## Plugging in your engine

Implement the `.step()` method:

```python
# rule_engine.py
class RuleEngine:
    def step(self, before: dict, actions: list[dict], duration_s: float) -> dict:
        # ... your rules over hidden state ...
        return after  # complete {vitals, interventions, drugs, mechanism} dict
```

Run with `--engine rule_engine:RuleEngine`. No changes to the harness.

## What gets scored

Per pair, every predicted `after` is compared against the authored `after` on:

**Vitals (6)** — absolute-error tolerance + direction match

| Vital  | Tolerance | Dead-zone |
|--------|-----------|-----------|
| HR     | ±10       | ±3        |
| BP_sys | ±15       | ±5        |
| BP_dia | ±15       | ±5        |
| RR     | ±4        | ±1        |
| O2Sat  | ±3        | ±1        |
| T      | ±0.3      | ±0.1      |

Direction = up / stable / down based on `after − before` vs the dead-zone. Direction can
fail while tolerance passes — this catches engines that are close but move the wrong way.

**Intervention flags** — 9 booleans (exact) + 2 categorical (exact) + 7 numeric (small tol)

**Mechanism** — pathology `name` + `severity` (exact)

**Drugs** — sorted name list (exact; dose/route not scored in v1)

## Metrics reported

- **Strict pair pass**: all vitals in tolerance AND all directions match AND all flags match AND mechanism match AND drugs match
- **Partial score** ∈ [0, 1]: fraction of per-dimension checks passed on a pair
- **Per-vital MAE, within-tolerance rate, direction-match rate**
- **Per-category pass rate**
- **Pure-wait vs action-pair breakdown** — isolates pathology drift from action effects
- **Top-N worst offenders** — ranked by tolerance-normalized total vital error, with before→predicted vs actual printed inline

## Iteration workflow

1. Run baseline → note per-vital MAE and worst offenders (sanity check)
2. Implement hidden-state layer + vitals mapping in `rule_engine.py`
3. Add rules one module at a time: pathology drift → single interventions → composite / drug effects
4. After each addition: re-run, check partial score ↑ and per-vital MAE ↓
5. Use `--export results.csv` + pivot in Excel when bulk-diagnosing

## Notes on tolerance

Tolerances are calibrated to absorb author-quantization noise (scenarios round to nearest
5 or 10). Tighten after reaching a stable pass rate — e.g. HR ±10 → ±5 once drift is dialed
in. Direction dead-zones should stay small; they catch the "wrong-way" failure mode.
