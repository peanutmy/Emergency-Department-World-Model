# LLM Phase-0 Report — Pure LLM Engine (zero-shot, gpt-4o-mini)

_2026-04-19_

## TL;DR

A pure LLM engine ([emsim_llm_engine.py](emsim_llm_engine.py)) running GPT-4o-mini
zero-shot on all 808 transition pairs achieves **11.6% strict pair pass**,
roughly half the rule-based Phase 6 baseline of 22.9%. The author-marker
ceiling check (§4) confirms the 22.9% rule number is **corpus-structural,
not engine-architectural**: on the 8 atropine drug-only pairs, the
rule engine passes 6/8 by identity pass-through while the LLM passes 0/8
because it makes clinically-correct predictions (atropine → HR up by
10-15 bpm) that disagree with the author's marker-stable pairs.

Net: the rule engine wins this benchmark by matching the corpus's
author-marker cadence; the LLM is overall more physiologically plausible
but cannot clear the author-stable marker majority without few-shot
corpus conditioning. Prefill ran in 81s with 16 workers, 0 failures,
total API cost **~$0.17**.

Strong recommendation: Option A (skip LLM/hybrid until corpus
re-authoring or few-shot calibration). See §10.

---

## 1. Headline metrics — LLM vs RuleEngine (Phase 6)

| Metric                         | Rule Phase 6 | LLM Phase 0 | Δ         |
| ------------------------------ | ------------ | ----------- | --------- |
| Strict pair pass               | 22.9%        | **11.6%**   | −11.3pp   |
| Mean partial score             | 0.887        | 0.855       | −0.032    |
| Pure-wait strict               | 8.4%         | 5.6%        | −2.8pp    |
| Drug-only strict               | 32.9%        | 18.3%       | −14.6pp   |
| Interv-only strict             | 28.5%        | 12.5%       | −16.0pp   |
| Action non-defib strict        | 28.4%        | 12.5%       | −15.9pp   |
| Defib strict                   | 11.5%        | 3.8%        | −7.7pp    |
| Arrest-slice strict (n=108)    | 36.1%        | 27.8%       | −8.3pp    |
| HR MAE                         | 14.79        | 15.55       | +0.76     |
| BP_sys MAE                     | 17.05        | 17.18       | +0.13     |
| BP_dia MAE                     | 10.87        | 10.77       | −0.10     |
| RR MAE                         | 3.13         | 4.04        | +0.91     |
| O2Sat MAE                      | 8.50         | 9.44        | +0.94     |
| T MAE                          | 0.02         | 0.15        | +0.13     |

LLM is **numerically close on MAE** (within ±1 on every vital except T)
but **loses ≥7pp on every strict-rate slice**. The gap is not "bad
prediction" but "predictions that move vitals when the corpus author
chose not to."

Commands:
```bash
python emsim_llm_eval_parallel.py --workers 16           # prefill .cache/llm
python emsim_eval.py --transitions transitions \
       --engine emsim_llm_engine:LLMEngine \
       --export llm_phase0.csv --worst 0
python emsim_llm_compare.py --llm llm_phase0.csv --rule phase6.csv \
       --out llm_compare_report.md
```

---

## 2. Engine architecture summary

### Class ([emsim_llm_engine.py](emsim_llm_engine.py))

```
class LLMEngine:
    step(before, actions, duration_s) -> after
        cache_key  = sha256({model, before, actions, duration_s})
        delta      = cache.load(key) or call_llm(before, actions, duration_s)
        after      = _apply_delta(before, delta, actions, duration_s)
```

Key invariants:

- **Delta merge** (`_apply_delta`) guarantees schema-valid output: vitals
  clipped to physiological ranges, interventions written only for allowed
  keys, drugs carried over with `age_min += duration_s/60`, new drug
  actions authoritative over LLM-declared `drugs_added`, pathology name
  never changed.
- **Cache** is file-based under `.cache/llm/*.json`, keyed on a sha256
  of `(model, before, actions, duration_s)`. Thread-safe writes via
  `os.replace` with per-pid temp files. 802 unique keys for 808 pairs
  (6 duplicates — literal-identical `before/actions/duration` pairs
  across 2 case variants).
- **Retry**: 3 attempts, exponential backoff with jitter, fallback to
  identity-delta if all fail (0 fallbacks triggered on this run).
- **Concurrency**: a separate [emsim_llm_eval_parallel.py](emsim_llm_eval_parallel.py)
  prefills the cache with a `ThreadPoolExecutor(workers=16)`; the main
  `emsim_eval.py` then reads the populated cache deterministically.

### Prompt (final)

**System** — abridged from [emsim_llm_engine.py:SYSTEM_PROMPT](emsim_llm_engine.py):

> You are an emergency-department physiology engine. Given a patient's
> before-state, a list of clinical actions taken at t=0, and an
> observation window duration_s, predict the after-state as a JSON delta
> over before.
>
> Rules: output JSON-only; vitals_delta values are absolute deltas
> (HR+15 means HR rises by 15); interventions_changed only for keys that
> change; drugs_added excludes age_min; mechanism_severity only if it
> changes; be clinically reasonable (apply_NRB→O2Sat up, start_CPR→HR
> stays 0, atropine→HR up in brady, sedatives+intubate→small BP dip,
> pure-wait→small drift); clip vitals to physiologic range.

**User** — templated:

```
## Before state         {before_json}
## Actions taken at t=0 {actions_json}
## Observation window   {duration_s} seconds
Predict the after state as a delta JSON.
```

OpenAI chat-completions with `response_format={"type": "json_object"}` and
`temperature=0.0`.

---

## 3. Slice comparison (LLM vs RULE)

```
slice                  eng     n   pass          partial     HR    BPs   BPd    RR    O2     T
OVERALL                LLM   808   94/808 (11.6%) 0.855   15.55 17.18 10.77  4.04  9.44  0.15
OVERALL                RULE  808  185/808 (22.9%) 0.887   14.79 17.05 10.87  3.13  8.50  0.02
pure_wait              LLM   249   14/249 ( 5.6%) 0.829   22.56 25.04 15.82  5.90 15.26  0.16
pure_wait              RULE  249   21/249 ( 8.4%) 0.841   21.78 25.08 15.86  5.69 14.58  0.06
drug_only              LLM   213   39/213 (18.3%) 0.876   14.16 16.11 10.12  1.57  6.12  0.32
drug_only              RULE  213   70/213 (32.9%) 0.905   11.04 15.84 10.52  1.62  4.87  0.01
interv_only            LLM   319   40/319 (12.5%) 0.870    7.24  8.98  5.57  3.92  5.90  0.03
interv_only            RULE  319   91/319 (28.5%) 0.920    7.64  8.83  5.49  1.78  4.89  0.00
action_non_defib       LLM   320   40/320 (12.5%) 0.869    7.38  9.02  5.58  3.91  5.90  0.03
action_non_defib       RULE  320   91/320 (28.4%) 0.920    7.79  8.81  5.49  1.77  4.87  0.00
defib                  LLM    26    1/26  ( 3.8%) 0.759   60.58 51.00 31.46  8.15 24.46  0.24
defib                  RULE   26    3/26  (11.5%) 0.787   64.81 51.52 32.32  7.77 24.73  0.00
arrest                 LLM   108   30/108 (27.8%) 0.841   27.67 31.20 18.06  3.31 21.41  0.10
arrest                 RULE  108   39/108 (36.1%) 0.875   28.00 27.15 16.17  3.00 20.22  0.03
```

The rule engine leads every slice. The *smallest* gaps are:

- **Arrest-slice** (−8.3pp): LLM correctly keeps `HR=BP=O2=0` in PEA/VF
  pairs; rule's drug-PD and pathology drift occasionally nudge them.
- **Pure-wait** (−2.8pp): both engines struggle with author-written
  trajectory beats; LLM can't beat identity pass-through which the rule
  engine already implements for stable pathology.

The *largest* gaps are:

- **Interv-only** (−16.0pp): rule engine has 21 carefully-tuned
  intervention effects; LLM over-predicts (e.g. apply_nasal + 120s →
  O2Sat +4, but author chose +0 or +8).
- **Drug-only** (−14.6pp): the §4 author-marker story.

---

## 4. Author-marker ceiling check — atropine drug-only pairs

The central scientific question: if the 22.9% ceiling were
engine-architectural, a stronger clinical-reasoning engine should break
it. If it were corpus-structural, an engine that "correctly" predicts
atropine → HR up should actually *lose* pairs, because author wrote
most of them as markers. The 8 atropine drug-only pairs pin this down:

| Case | HR before | HR authored | HR rule | HR LLM | rule pass | LLM pass |
|------|-----------|-------------|---------|--------|-----------|----------|
| aortic_dissection/p4 | 50 | 50 | 52 | 60 | ✓ | ✗ |
| beta_blocker_toxicity/p2 | 45 | 45 | 45 | 60 | ✓ | ✗ |
| digoxin_toxicity/p2 | 30 | 30 | 31 | 45 | ✓ | ✗ |
| bradycardia/p2 | 25 | 25 | 26 | 40 | ✓ | ✗ |
| hyperkalemia/p3 | 30 | 30 | 31 | 45 | ✓ | ✗ |
| stemi/p3 | 30 | 30 | 31 | 45 | ✓ | ✗ |
| airway_obstruction/p6 | 35 | 110 | 37 | 50 | ✗ | ✗ |
| organophosphate_poisoning/p2 | 48 | 48 | 48 | 60 | ✗ | ✗ |

**Result**: Rule engine passes 6/8 on atropine by near-identity
(marker preservation). The LLM passes 0/8 — it predicted HR +10 to +15
on every single atropine pair, which is clinically correct but is
outside the ±10 bpm tolerance band vs the six author-marker ground
truths of 0 bpm change. The one "clinical-response" pair
(`airway_obstruction/p6`, author HR 35 → 110) is too large a jump for
both engines.

**Conclusion**: the 22.9% ceiling is **corpus-structural**. A physically
correct engine loses 6 atropine pairs that the rule engine recovers by
accident of identity pass-through matching author-marker majority. This
finding generalizes: the same pattern is visible in
`beta_blocker_toxicity/p2-p4` (atropine + fluids + glucagon all given
at moderate-severity bradycardia, author keeps HR flat, LLM predicts
clinical response).

The rule engine's Phase 6 report ([PHASE6_REPORT.md §3](PHASE6_REPORT.md))
reached the same conclusion from the opposite direction: "calibration
settled at near-zero PD for marker-dominant drugs." LLM has no
corpus-calibration knob; it applies clinical reasoning uniformly and
pays the marker tax.

---

## 5. Agreement matrix (overall 808 pairs)

|             | rule pass | rule fail |
| ----------- | --------- | --------- |
| LLM pass    | **72** (8.9%)   | **22** (2.7%)   |
| LLM fail    | **113** (14.0%) | **601** (74.4%) |

- **Both pass (72)** — the easy-base layer: author-stable pairs where
  pass-through wins, plus a handful where both independently land in
  tolerance.
- **Rule-only pass (113)** — dominated by drug-only markers and
  intervention-effect-calibration pairs. The rule engine's 14 class
  × 14 override tuning is hard to replicate from a general prompt.
- **LLM-only pass (22)** — concentrated on PEA-arrest pairs
  (`termination_of_resuscitation/p1–p3`,
  `pea_arrest_breaking_bad_news/p3–p6`) where LLM correctly outputs
  `HR=BP=O2=0,RR=0` but the rule's drug-PD or drift spuriously
  produces small non-zero values. A handful of clinical-reasoning wins
  on pneumonia/apply_nasal and trauma/TXA where LLM hits the authored
  small-deltas.
- **Both fail (601)** — 74% of corpus.

The LLM-only wins are informative: they are concentrated in
*expected-zero* arrest monitoring, where rule-based composition
(drift + drug-PD) occasionally fires when it shouldn't. This suggests
a targeted rule-engine cleanup — suppress drift/PD on arrest monitor
sentinels — which would gain some of those 22 pairs without regressing.

---

## 6. Sample disagreement pairs (with LLM reasoning)

### LLM-only pass (selected)

`termination_of_resuscitation/p1` — pea_arrest/severe, 300s, dextrose_50
- Before: HR 40 BP 0/0 O2 0.  Actual: identical.
- LLM: HR 40, BP 0/0, O2 0. _"The patient remains in a state of pulseless
  electrical activity with no change in vital signs despite the
  administration of dextrose."_
- Rule: HR 32.5, BP 14.4/9.6, O2 0. (drug-PD on calcium bumps BP/HR.)

`covid_19_ambulatory_care/p1` — pneumonia/moderate, 0s, apply_nasal
- Before: O2 89. Actual: O2 95.
- LLM: O2 92. _"Applying nasal oxygen improves O2 saturation slightly."_
- Rule: O2 89.69 (barely moved — intervention-only tables dominate).

### Rule-only pass (selected)

`beta_blocker_toxicity/p2` — moderate, 120s, atropine 1 mg
- Before HR 45, author HR 45.
- LLM HR 60. _"Atropine administration raises HR in bradycardia,
  improving the patient's condition from moderate to mild severity."_
  (Clinically canonical; corpus marker-stable.)
- Rule HR 45 (atropine gated off — the Phase 6 marker-preservation tune).

`chest_pain_on_the_ward/p4` — stemi/moderate, 1800s, furosemide 40 mg
- Before BP 179/90, author 179/90.
- LLM BP 169/85 (furosemide dry-out reasoning).
- Rule 176/88 (small PD, within tolerance).

`electrical_storm/p2` — vf_arrest/severe, 60s, epinephrine
- Before monitor zeros. Author zeros.
- LLM BP 20/10. _"Epinephrine administration in a vf arrest scenario
  raises BP and slightly improves O2 saturation while HR remains at 0."_
- Rule zeros (arrest rhythm zeros monitor by design).

These qualitative samples confirm the structural story: the LLM's
disagreements are clinically reasonable; they lose to the rule engine
because the corpus's scenario-paced author cadence rewards identity
more often than physiology.

---

## 7. Worst-20 (LLM, by tol-normalized error sum)

Of the top-20 worst-predicted LLM pairs, **19 also fail for RULE** and
**14** overlap with RULE's worst-20 from Phase 6. The same
data-quality pairs dominate (`aortic_dissection/p7` and
`agitation_and_aortic_dissection/p6-p7` are author all-zero-after
paradox pairs; `dka_and_decreased_loc/p4,p8` and
`pediatric_dka_with_cerebral_edema/p8` have before all-zero; several
STEMI + hypothermia arrest-emergence jumps). This is consistent with
the Phase 6 "worst-20 is 14/20 data-quality" finding — LLM
inherits the same data-quality ceiling.

Headline: the LLM does not open any new failure mode unseen in rule —
no exotic LLM hallucinations in the top-20. See §8 for the subtle
per-row failure modes that do surface.

---

## 8. Failure modes

| Mode | Count | Detail |
|------|-------|--------|
| JSON parse failures                      | **0** / 808 | `response_format=json_object` enforced valid JSON on every call. |
| Schema-invalid output (post-merge)       | **0** / 808 | `_apply_delta` clips/masks untrusted fields; output always passes draft-7 state schema. |
| LLM engine exceptions during eval        | **0** / 808 | 0 calls to fallback identity-delta. |
| Absurd HR (< 0 or > 250)                 | **1** / 808 | `pediatric_svt/p4`: before HR 260, LLM kept 260. Not LLM fault (already absurd in author data). |
| Absurd BP_sys (< 0 or > 250)             | **0** / 808 | — |
| Absurd O2Sat (> 100)                     | **0** / 808 | — |
| T clipped to 43 (LLM interpreted T as absolute, not delta) | **13** / 808 | e.g. `tricyclic_antidepressant_overdose/p4`: before T=38.1, LLM emitted `{"T": 37.9}` meaning *absolute* 37.9; engine applied it as a +37.9 delta, clip to 43. Subtle prompt-confusion about T units. |
| T below 30 (real hypothermia, not a bug) | 10 / 808  | `accidental_hypothermia/p1-p5` + `hypothermia_with_trauma/p4-p10` correctly kept their low T. |

The **13 T-as-absolute** cases explain most of the T MAE gap (0.15 vs
0.02). Fix in a future phase: either tighten the prompt with an
explicit "delta examples for T" block, or post-process T deltas whose
absolute value exceeds 3°C as suspicious and treat them as absolute
values.

No other clinical-absurd outputs (no `severity: "terminal"`, no
negative doses, no invented pathology names, no `drugs_added` when
`actions=[]`). The delta merge layer catches anything else the LLM
might invent.

---

## 9. Cost report

- Prefill run: 808 pairs, 16 workers, elapsed **80.8s**.
- API calls: 808 (0 cache hits on first run), 0 failures, 0 retries
  visible (all first-try JSON-valid).
- Tokens (approx, from `usage` field):
  - prompt_tokens  ~ 975k
  - completion_tokens ~ 95k
- Cost (gpt-4o-mini @ $0.15/$0.60 per MTok): **~$0.17** total.
- Subsequent eval runs via `emsim_eval.py` read the cache exclusively —
  no additional API cost.

Budget vs ENGINE_DESIGN §11 estimate ($50-150 for full sweep): this
single-variant run is ~0.3% of the full budget. A 2-model × 3-variant
sweep would still fit in single-dollar territory with gpt-4o-mini
pricing.

---

## 10. Recommendation for v1 — Option A (skip LLM integration)

Given §1–§8, three options are on the table:

**Option A — Keep rule engine as v1 canonical; shelve LLM.**
- Rule is net +11.3pp strict with $0 marginal inference cost.
- LLM baseline provides no measurable gains on any slice.
- The 22 LLM-only wins are nearly all "rule engine drift bug on arrest
  monitor" fixable in rule-engine itself (see §5 suggestion).
- Defer LLM entirely until the corpus is re-authored, or until a
  downstream task (symptom narration, agent orchestration) genuinely
  needs it.

**Option B — Hybrid: rule engine for physiology, LLM for extrapolation.**
- Narrowly tempting: LLM wins on PEA-arrest zero-preservation and
  apply_nasal small-deltas.
- Cost: engine complexity + inference latency on every pair.
- Gain: +3pp in the best case from the 22 LLM-only wins, minus losses
  elsewhere. Doesn't pay for itself on this corpus.

**Option C — Calibrate LLM with few-shot corpus examples.**
- Would likely close the author-marker gap: in-context examples of
  "atropine given, HR unchanged" would teach the LLM the corpus cadence.
- Requires a clean held-out set (see ENGINE_DESIGN §11, "~10 cases
  reserved as example-only") — which the 138-case corpus does NOT have
  without eroding statistical power.
- Defer until corpus expands or a held-out split is created.

**Recommendation**: **Option A**. Ship rule engine as v1. Use this
LLM Phase-0 report as the canonical comparison point for v1_baseline.md.
Revisit LLM (B or C) only if a downstream agent-orchestration need
emerges, or after corpus re-authoring resolves the marker-marker vs
physiology tension.

Model-upgrade note: gpt-4o-mini produced 0 JSON parse failures, 0
schema violations, and 1.6% T-as-absolute confusion. A stronger model
(gpt-4o, Claude Opus 4) would likely fix the 13 T clips but would NOT
break the author-marker ceiling — the ceiling is corpus-structural.
Upgrade is cost-ineffective for this task.

---

## 11. Files delivered

- [emsim_llm_engine.py](emsim_llm_engine.py) — `LLMEngine` (zero-shot).
- [emsim_llm_eval_parallel.py](emsim_llm_eval_parallel.py) — 16-worker
  cache prefill.
- [emsim_llm_compare.py](emsim_llm_compare.py) — LLM vs RULE slice,
  agreement, author-marker, worst-20 reporter.
- [llm_phase0.csv](llm_phase0.csv) — per-pair eval rows (226 KB).
- [phase6.csv](phase6.csv) — rule-based baseline (253 KB).
- [llm_compare_report.md](llm_compare_report.md) — full auto-generated
  comparison (14.7 KB) with all 20 worst pairs, 10 + 10 disagreement
  samples with LLM reasoning.
- `.cache/llm/` — 802 JSON delta cache entries (~560 KB). Re-runs free.

---

## 12. Reproducibility

```bash
export OPENAI_BASE_URL="http://148.113.224.153:3000/v1"
export OPENAI_API_KEY="<key>"

# 1. Prefill cache (81s with 16 workers, $0.17)
python emsim_llm_eval_parallel.py --workers 16

# 2. Score via standard harness (reads cache)
python emsim_eval.py --transitions transitions \
       --engine emsim_llm_engine:LLMEngine --export llm_phase0.csv --worst 0

# 3. Regenerate rule-based CSV if needed
python emsim_eval.py --transitions transitions \
       --engine rule_engine:RuleEngine --export phase6.csv --worst 0

# 4. Produce comparison report
python emsim_llm_compare.py --llm llm_phase0.csv --rule phase6.csv \
       --out llm_compare_report.md
```

Note: `emsim_eval.py`'s `print_worst` uses a Unicode `→` character in its
"worst vitals" line that crashes on Windows cp1252 stdout. Because
`main()` runs `print_worst` **before** `export_csv`, running with
`--worst 20` will emit the UnicodeEncodeError and skip CSV export.
Use `--worst 0` (or `PYTHONIOENCODING=utf-8`) to get the CSV. The
compare script only needs the CSV plus `transitions/`.
