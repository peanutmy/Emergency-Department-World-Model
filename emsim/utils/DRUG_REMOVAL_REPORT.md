# Corpus `state.drugs[]` Removal — Migration Report

_Date: 2026-04-19_

## TL;DR

Removed the corpus-level `state.drugs[]` field (plus `drug_entry.age_min`) from
the 808-pair transitions set and from the engine's I/O contract. Drugs now live
**only in `pair.actions[]`** as bolus events. Within-pair PD still runs on
`HiddenState.active_drug_effects`; cross-pair persistence is deferred to L2
Session. 41 known data-debt bugs (36 silent drug-disappearances + 5 phantom
drug-appearances + frozen `age_min`) are eliminated "by construction" because
the field no longer exists. Rule engine eval is flat to slightly better
(overall strict **22.8% → 23.0%, +0.2pp**); round-trip unchanged at **97.0%**.
Nothing committed — working tree only, for user review.

## Baseline comparison

All pre/post numbers from `rule_engine:RuleEngine` on the 808-pair corpus.
Pre = `pre_drug_removal.csv`; post = `post_drug_removal.csv`.

### Slice-level strict pass rate

| Slice             | Pre             | Post            | Δ    |
|-------------------|-----------------|-----------------|------|
| overall           | 184/808 = 22.8% | 186/808 = 23.0% | +0.2pp |
| pure_wait         |  20/249 =  8.0% |  20/249 =  8.0% |  0.0pp |
| drug_only         |  70/212 = 33.0% |  72/212 = 34.0% | +1.0pp |
| interv_only       |  94/345 = 27.2% |  94/345 = 27.2% |  0.0pp |
| mixed_interv_drug |   0/  2 =  0.0% |   0/  2 =  0.0% |  0.0pp |
| action_non_defib  | 161/533 = 30.2% | 163/533 = 30.6% | +0.4pp |
| defib             |   3/ 26 = 11.5% |   3/ 26 = 11.5% |  0.0pp |
| arrest            |  24/ 50 = 48.0% |  24/ 50 = 48.0% |  0.0pp |

All shifts within the ±1pp constraint. Most notable: drug_only up 1.0pp. The
mechanism: previously `decode_state` re-enqueued every carried drug via
`enqueue_carried_drug` with `_prev_contrib` pre-seeded, so subsequent
`drug_lib.tick` calls applied only the marginal contribution. That path was a
no-op in aggregate (Phase 6 analysis already showed ≈ 0 contribution) but
introduced small per-pair noise; removing it slightly cleans up a handful of
drug-only pairs around the tolerance boundary.

Partial-score comparison is not apples-to-apples because the eval harness
dropped `drugs_match` from the per-pair check list (n_checks reduced by 1).
For reference: pre partial = 0.887 (over 29 checks per pair); post = 0.885
(over 28 checks). Raw correctness is flat.

### Per-vital MAE (all 808 pairs)

| Vital  | Pre MAE | Post MAE | Δ      |
|--------|---------|----------|--------|
| HR     | 14.846  | 14.813   | −0.033 |
| BP_sys | 17.054  | 17.016   | −0.038 |
| BP_dia | 10.873  | 10.831   | −0.042 |
| RR     |  3.133  |  3.131   | −0.002 |
| O2Sat  |  8.502  |  8.502   |  0.000 |
| T      |  0.022  |  0.022   |  0.000 |

All six MAEs either identical or microscopically better.

## Data migration stats

Script: `_migrate_drop_drugs.py` (one-shot, idempotent, root dir; delete after
user review).

| Metric                        | Count |
|-------------------------------|-------|
| Cases migrated                | 138   |
| Pairs touched                 | 808   |
| `drugs[]` entries removed from `initial_state` | 9     |
| `drugs[]` entries removed from `before` / `after` | 1193  |
| **Total drug entries removed**| **1202** |

All 138 files schema-valid against the updated `transitions/schema.json` after
migration (verified via `jsonschema.validate` over all files).

## File-by-file changes

### Schema & data
- [`transitions/schema.json`](transitions/schema.json) — deleted `drug_entry`
  definition; removed `"drugs"` from `state.required` and from
  `state.properties`; added description note on the new drug-is-action-only
  model.
- `transitions/<category>/*.json` × 138 — `drugs: [...]` removed from
  `initial_state` + every `pair.before` / `pair.after`. No action list
  modifications.

### Engine
- [`rule_engine/io.py`](rule_engine/io.py) — `decode_state` no longer iterates
  `before.get("drugs", [])` or calls `enqueue_carried_drug`; `encode_state`
  dropped the `drug_actions` kwarg and the `_merge_drugs` helper (deleted);
  output dict no longer contains `drugs`.
- [`rule_engine/drug_lib.py`](rule_engine/drug_lib.py) — deleted
  `enqueue_carried_drug` function; updated module/tick docstrings to describe
  the new L1-stateless / L2-persistent split; simplified `_enqueue_effects`
  (removed unused `pre_apply` parameter and the pre-aged seeding branch).
  `DRUG_CLASSES` / `DRUG_TO_CLASS` / `DRUG_STANDARD_DOSE` / `DRUG_OVERRIDES` /
  `_gate_scale` / `start_drug` / `tick` / `EMSIM_DISABLE_DRUG_PD` are
  unchanged.
- [`rule_engine/engine.py`](rule_engine/engine.py) — `RuleEngine.step` no
  longer collects `drug_actions` or forwards them to `encode_state`.
- [`rule_engine/hidden_state.py`](rule_engine/hidden_state.py) — updated
  `DrugEffect` docstring to drop the "pre-aged carried-over via
  `state.drugs[i].age_min`" path (no behavior change).

### Eval harness
- [`emsim_eval.py`](emsim_eval.py) — removed the `drugs_match` check and the
  corresponding sorted-name list comparison; adjusted `n_checks` from 29→28;
  removed `drugs_match` from CSV columns. Aggregation logic unchanged.

### LLM engine
- [`emsim_llm_engine.py`](emsim_llm_engine.py) — `SYSTEM_PROMPT` drops the
  `drugs_added` field description and adds one rule clarifying that drug
  actions are inputs only (no echo in output); `_apply_delta` removes the
  `before.drugs` carry-over block and the `drugs_added` merge, and no longer
  sets `after["drugs"]`. The `duration_s` parameter is retained in the
  `_apply_delta` signature (private method; LLM prompt still consumes
  `duration_s`). Failure stub no longer includes `drugs_added: []`.

### Documentation
- [`transitions/EXTRACTION_GUIDE.md`](transitions/EXTRACTION_GUIDE.md) — §4
  rewritten (drug duration_s guidance only; removed age_min / cumulative /
  carry-over / data-gap subsections); §7 trailing "Drug age_min" paragraph
  deleted; §9 step 4 no longer mentions drug update; PK table heading tweaked.
- [`ENGINE_DESIGN.md`](ENGINE_DESIGN.md) — architecture diagram updated
  (state dict no longer advertises "drugs"); §3 var 12 description refreshed
  (L1 within-pair, L2 cross-pair); §5 drug section opening rewritten to
  describe Phase 6 current state + L1/L2 split; §6 integrator sample encode
  call no longer passes `drug_actions`; §6.1 `encode_state` contract updated;
  §6.5 EngineSession stub drops `self._drugs`; §7 Phase 1 heading/target
  refreshed; §10 removed stale "age_min field plumbed" caveat; §11 LLM JSON
  example dropped `drugs_added`.
- [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — added "2026-04-19 corpus
  drugs removal" banner at top; §Completed schema description rewritten;
  Core-semantics drug-tracking row rewritten; stateless step description
  updated; removed the data-migration bullet from Deferred and added L2
  Session in its place.

### Tooling added to working tree (for user review, may delete)
- `_migrate_drop_drugs.py` — one-shot migration script, idempotent.
- `_slice_eval.py` — computes slice-level strict/partial stats from eval CSV
  (used to generate the tables above); could graduate into `emsim_eval.py`
  proper as a follow-up.
- `pre_drug_removal.csv`, `post_drug_removal.csv` — full per-pair eval
  exports.
- `pre_drug_removal_summary.txt`, `post_drug_removal_summary.txt`,
  `pre_drug_removal_slices.txt`, `post_drug_removal_slices.txt`,
  `roundtrip_post_drug_removal.txt`, `_migrate_stats.txt` — baseline logs.

## Round-trip sanity

`python -m rule_engine._roundtrip_test transitions` after the migration:

- **784 / 808 pairs (97.0%)** pass all 6 vitals within `tolerance/2`.
- Matches Phase 6 baseline (97.0%) byte-for-byte — no new decode/encode
  regressions introduced.
- Sample failure pairs (24 total) are the same class of known pairs as before
  (hyperkalemia × 4, svt × 4, airway_obstruction × 4, lvad_thrombosis × 3,
  pea_arrest × 3, bronchiolitis × 3, ...), all pre-existing pathology-model
  edge cases unrelated to drug state.

## Unexpected regressions

None. The only slice that moved >0.5pp was `drug_only` (+1.0pp, improvement).
No slice regressed >1pp (constraint from the task spec). Per-vital MAEs were
all flat-to-microscopically-better.

## LLM cache invalidation notice

**The LLM eval was not re-run.** The LLM engine's cache key
(`emsim_llm_engine.LLMEngine._cache_key`) is `sha256(model, before, actions,
duration_s)`. The `before` dict changed shape (no `drugs` key), so **every one
of the 808 cached entries will miss on next `emsim_llm_eval_parallel.py` run**
and a full re-prefill is needed.

Prior LLM prefill cost (`LLM_PHASE0_REPORT.md` §token accounting): ~$0.17
total, ~80 s wall-clock on the parallel harness with gpt-4o-mini.
Re-prefilling is small-dollar but not free.

**Decision point for user**: keep the stale cache directory (`.cache/llm/`) as
is, delete it, or leave it and re-run when next doing LLM comparison work.
This report leaves the cache untouched.

## Suggested follow-ups (out of scope; not done)

- Integrate `_slice_eval.py` functionality into `emsim_eval.py` proper as a
  `--slices` flag so future reports don't need the sidecar script.
- Delete `_migrate_drop_drugs.py` once the user confirms the migration is
  final (idempotent, safe to keep or drop).
- Re-run LLM prefill and regenerate `llm_compare_report.md` against the new
  corpus — separate task; also a good moment to see how the LLM engine scores
  without the `drugs_added` / carry-over complication.
- L1 noise cleanup that surfaced while searching for drug references
  (harmless but might-as-well): `DRUG_OVERRIDES` unit mismatch warnings for
  `insulin_regular U/hr`, `epinephrine mcg/kg/min`, `norepinephrine mcg/min`,
  `procainamide mg`, `epinephrine mg/hr`, `norepinephrine mcg` — these are
  extraction-side dose-unit inconsistencies (the engine silently drops
  dose-scaling on mismatch, so they were noise-only). Proper fix is a
  data-side pass to normalize units in `pair.actions[]`, unrelated to this
  task.
- L2 Session implementation — the natural next step now that L1 has a clean
  per-step drug lifecycle. Spec already in `ENGINE_DESIGN.md §6.5`; no code
  change needed in L1 to support it (just wrap and persist `self._h`).
- Separately track and batch-merge the 11 high-priority + ~20
  medium-priority Layer 1 data-quality pairs (PEEP / stopCPR / severity /
  intubate+BVM merges) from `PROJECT_STATUS.md §Known Issues` — orthogonal to
  this task, previously deferred.

## Next steps recommendation

1. User reviews the diff, eyeballs `DRUG_REMOVAL_REPORT.md` + baseline CSVs.
2. If accepted: commit as one coherent change ("remove corpus drugs[] field;
   41 data-debt bugs retired by construction"). Delete the sidecar tooling
   (`_migrate_drop_drugs.py`, intermediate txt files) in the same commit or
   in a follow-up.
3. Decide on LLM cache: wipe vs keep-stale vs re-prefill now.
4. Prioritize L2 Session (ENGINE_DESIGN §6.5) vs Layer-1 data-quality
   clean-up (PROJECT_STATUS §Known Issues) as the next work package.
