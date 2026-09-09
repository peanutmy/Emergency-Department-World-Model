# Paper Outline — Emotion Engine Section

Working structure for the emotion-engine part of the ED world-model paper.
Each subsection lists: what to claim, what evidence/files back it, and
suggested figures/tables. File paths refer to this repo.

---

## 1. Role in the System (Why an Emotion Engine)

**Claim to make:** a clinically faithful ED simulation cannot model dialogue
consequences with physiology alone; what clinicians *say* changes what
patients *can do* next. The emotion engine is the psychological state channel
of the world model, parallel to the physiology engine.

Points to cover:

- **Motivation from the domain.** ED communication research shows delivery of
  bad news, blame, and information overload change patient cooperation,
  comprehension, and outcomes. A world model that only tracks vitals treats
  "your potassium is 7" and "you're going to be fine" as equivalent inputs.
- **Emotion as world-model state, not agent flavor.** The engine is not part
  of the patient LLM's persona prompt. It is a separate, auditable state
  variable in `GlobalState.psych_state` (`ed_world_model/state/global_state.py`),
  updated by an independent judge model. This separation is the key design
  decision to argue for: (a) it makes emotion *consistent* (the patient agent
  cannot quietly forget it is terrified), (b) it makes emotion *evaluable*
  (Section 3), (c) it prevents self-serving drift (an agent cannot talk itself
  into a convenient emotional state — self-utterances never move own emotion).
- **Three downstream consumers** (this is the "role in the whole system"):
  1. **Agent behavior**: patient/relative observations include the current
     emotion vector; their utterance generation is conditioned on it
     (`ed_world_model/orchestration/observation_builder.py`,
     `ed_world_model/agents/prompting.py`).
  2. **Termination**: any emotion reaching `extreme` ends the encounter with
     reason `patient_emotion_extreme` (`ed_world_model/state/termination.py`).
     This gives the simulation a *psychological failure mode* symmetric to
     physiological death — a clinician can "lose" the encounter by
     communication alone.
  3. **Trajectory logging / evaluation of clinician policies**: emotion
     trajectories are recorded per turn, enabling downstream scoring of
     communication quality, not just clinical correctness.
- **Driver definition** (state the transition semantics precisely): emotion
  changes *only* in response to an utterance received by the agent
  (recipient-gated); speaker identity matters; self-utterances and
  overheard/other-directed messages never move state
  (`ed_world_model/adapters/emotion_engine_adapter.py`).

Suggested figure: system diagram with physiology engine and emotion engine as
the two state-transition channels feeding the turn loop; arrows into agent
observations and termination check.

---

## 2. What the Emotion Engine Does (Formal Description)

**Present it as a state-transition function, not "a prompt".**

- **State space.** 4 dimensions × 3 ordinal levels:
  (fear, sadness, overwhelm, shame) ∈ {absent < present < extreme}⁴.
  Justify the four dimensions (ED-salient, distinguishable triggers: threat /
  loss / processing overload / dignity+blame) and the deliberately coarse
  3-level ordinal scale (LLM judges are unreliable on fine-grained intensity;
  3 levels are human-labelable with high agreement — cite the M14y milestone
  decision, `docs/milestones/M14y_three_level_emotion_schema_handoff.md`).
- **Initialization.** Deterministic all-`absent` opening state; no LLM call
  (`init_emotion`, `emotion_engine.py`). Rationale: opening emotion inferred
  from vitals/scenario was unfalsifiable and double-counted the scenario; a
  neutral baseline makes every level movement attributable to a specific
  utterance (cite `M14aa_initial_emotion_absent_handoff.md`).
- **Transition function.** One LLM call per received utterance:
  `step(state, utterance, speaker, clinical_context, dialogue) → state'`
  returning the full 4-vector plus `evidence` (quoted trigger phrase) and
  `reasoning` (per-changed-emotion explanation). Inputs and their purposes:
  clinical context constrains reactions (e.g. an obtunded child cannot react);
  dialogue digest scores the utterance in-context, not in a vacuum.
- **Hard invariants** (enforced by rules and/or code — a table works well):
  1. Recipient-gating (code: adapter skips non-recipient messages).
  2. Move ≤ 1 level per utterance; sole exception: rare absent→extreme jump
     when a single utterance carries a terminal crisis cue (rules).
  3. `extreme` is terminal and absorbing — never de-escalates; code-level
     clamp guarantees it even if the judge model misbehaves
     (`step_emotion` in `emotion_engine.py`).
  4. Stickiness: no change without a concrete trigger; unaddressed emotions
     keep their level (rules).
  5. Fail-safe: any engine/LLM error leaves state unchanged rather than
     corrupting the simulation (adapter `last_error` path).
- **The 6 reachable transitions** per emotion: absent→{absent,present,extreme},
  present→{absent,present,extreme}. `extreme` is never a source. This
  transition taxonomy is what the evaluation grid is built on (Section 3).
- **Multi-message turns**: within a turn, utterances are applied sequentially;
  later utterances see earlier ones in the dialogue digest (adapter loop).
- **Engine model**: Claude Opus 4.8, temperature 0; model selected by the
  benchmark in Section 4 (say this explicitly — model choice was empirical,
  not default).

Suggested figure: state machine diagram for one emotion dimension (3 nodes,
labeled edges, extreme as absorbing/terminal); example turn trace showing an
utterance, evidence, reasoning, and the vector update.

---

## 3. Evaluation Dataset Construction (Simulation-Grounded Augmentation)

This is a methodological contribution — give it its own subsection rather
than burying it under "performance". The problem: no public dataset labels
*per-utterance emotion transitions* in ED dialogue, and naturally occurring
simulation logs are heavily imbalanced (mostly absent→absent; extreme almost
never occurs — it terminates the encounter).

Pipeline (`tools/build_emotion_augmentation_dataset.py`):

1. **Real simulations across all 49 scenarios** (10 specialties), full system:
   agents + hybrid physiology + emotion engine. States, traits, and utterances
   in the eval items are *real simulation states*, not invented vignettes.
2. **Candidate extraction**: every utterance addressed to patient/relative,
   paired with the listener's true pre-utterance emotion vector (from the
   previous turn), traits, and scenario summary.
3. **Cell-controlled augmentation**: grid = 4 emotions × 6 valid transitions
   × 5 items = 120. For each cell, a candidate whose *real* current level
   matches the transition's source level is sampled and an LLM rewrites the
   utterance to induce exactly the target transition. Emphasize: source
   levels are filtered from real states, never asserted; only the utterance
   is synthetic.
4. **Automatic validity checks** (reject + resample): target reached in the
   draft label; no illegal multi-level moves on side emotions (absent→extreme
   allowed only for the primary target); banned generic-urgency openings
   ("I need to be straight with you…") so items test *content*, not template
   cues; speech-act preservation instruction.
5. **Blind human labeling**: labelers see only listener, speaker,
   current_state, and the utterance (`human_label_queue.jsonl`; the target
   transition is withheld). Gold = majority vote of 3 independent annotators.

Key honesty points for the paper:
- Same rule text (`STEP_EMOTION_RULES`) is shared verbatim between the engine
  prompt and the item-generation prompt (single source of truth, enforced by
  `tests/test_emotion_rules_consistency.py`) — generation and judging are
  calibrated to the same semantics, and the paper should disclose this.
- Draft labels from the generator are *silver*; human majority vote is the
  gold reference. Report both comparisons but rest claims on the human one.
- Class balance is by construction (coverage report per cell), which is why
  extreme transitions are testable at all.

Suggested table: the 4×6 grid with item counts (from
`eval_data/emotion_aug0821/coverage_report.json`).

---

## 4. Performance (Metrics, Results, Why These Metrics)

**Existing results to build on** (28-item / 112-label set,
`eval_data/emotion_aug_v3_3level/emotion_engine_report.md`); re-run the same
protocol on the new 120-item / 480-label set (`eval_data/emotion_aug0821/`)
with `tools/evaluate_emotion_engine.py` once human labels are merged.

**Metric suite and the argument for each** (all computed per label =
item × emotion):

| Metric | Why it is the right metric here |
|---|---|
| Exact accuracy | Interpretable headline; but insufficient alone because the label distribution is skewed toward "no change". |
| MAE on the 0–2 ordinal scale | Levels are *ordinal*; being off by two is worse than off by one. Accuracy can't see that. |
| Quadratic Weighted Kappa | Chance-corrected ordinal agreement; the standard for ordinal-rating agreement, penalizes big misses quadratically. |
| Change precision / recall / F1 | The clinically meaningful events are *transitions*, which are rare; a judge predicting "no change" everywhere scores high accuracy but zero recall. Precision matters because false escalations can wrongly terminate an encounter. |
| Direction accuracy | Given both sides agree something moved, did it move the right way — separates calibration errors from sign errors. |
| Target- vs side-dimension accuracy | Detects tunnel vision: does the engine keep uninvolved emotions stable while moving the targeted one. |
| Per-emotion breakdown | Shame/sadness rely on subtler cues than fear; report the spread, not just the mean. |
| Human-vs-human agreement range | The ceiling: engine accuracy is only interpretable relative to annotator agreement (prior run: annotators 78.6–93.8% vs gold; engine 83.0% — inside the human range). |

**Comparisons to report:**
1. **Engine vs human gold** (primary claim). Prior numbers: 83.0% exact,
   MAE 0.170, QWK 0.759, change P/R 0.74/0.57, direction 100%, 96.9% on
   unanimous labels; all errors single-level. Include the 3×3 confusion
   matrix and the finding that errors concentrate on the extreme boundary —
   exactly where humans also split 2:1.
2. **Engine vs draft (target) labels** (consistency of the pipeline; explain
   the change-precision artifact: draft labels under-mark side effects).
3. **Model comparison / selection**: Opus 4.8 vs GPT-5.x on the same items
   (metrics files exist in `eval_data/emotion_aug_v3_3level/`); this
   justifies the deployed engine model.
4. **Ablations** (the eval script already supports them):
   `--no-scenario` / `--no-traits` — information-parity variants that measure
   how much clinical context and traits contribute; this doubles as evidence
   that the engine actually *uses* context rather than keyword-matching.
5. Optional baselines to strengthen the section: majority-class
   ("never change") baseline; rules-free prompt (definitions only) to isolate
   the contribution of the transition rules.

**Interpretation to argue:** conservative-by-design (recall < precision) is
the correct trade-off for a termination-linked signal — a false `extreme`
kills an encounter, a missed change usually just delays a trajectory.

---

## 5. Prompt Design Logic

Frame this as *encoding domain theory as verifiable constraints*, not prompt
craft. Organize by rule, each with its failure mode it was added to fix
(milestone docs in `docs/milestones/M14*` give the actual history — the
paper can honestly say rules were iterated against observed failures):

1. **Shared definitions block** (`EMOTION_DEFINITIONS_TEXT`): four emotions
   defined by *triggers*, not synonyms — fear=threat, sadness=loss,
   overwhelm=processing overload, shame=judgment/dignity. Distinctions the
   judge must make (fear vs sadness: danger vs loss) are stated explicitly.
2. **Default-to-no-change / stickiness**: fixes emotion flicker and
   spontaneous decay (distress persisting until actually addressed).
3. **Clinical-threat escalation with valence judgment**: serious bad news
   *must* raise fear at least to present (tone doesn't cancel it), but the
   same clinical word can be good news ("we will NOT need to intubate") —
   forces meaning-level, not keyword-level, judgment
   (M14j/M14k: escalation + threat floor).
4. **Move-at-most-one-level + rare absent→extreme jump**: bounds per-utterance
   dynamics; the two-level jump is explicitly marked RARE and requires a
   terminal crisis cue in the utterance itself.
5. **Acknowledgment ≠ relief; questions are not soothing**: empathy without
   resolution holds level; prevents the judge from rewarding validation
   phrases — de-escalation requires the trigger to be actually addressed.
6. **Extreme = listener breakdown, not news severity**: terminal semantics
   with a behavioral threshold (unable to keep participating); protects the
   termination channel from overtriggering (M14z).
7. **Speaker identity matters**: same words land differently from clinician
   vs family.
8. **Structured output contract**: full 4-vector + evidence + per-change
   reasoning; plus tolerant JSON coercion and the code-level extreme clamp as
   defense-in-depth (M14p).
9. **Single source of truth**: the rules block is imported verbatim by the
   dataset builder and its consistency is unit-tested — prompt drift between
   generation and judging is structurally impossible.

Suggested figure/box: the rules block itself (abridged) with annotations
mapping each rule to its metric consequence (e.g. stickiness → change
precision; one-level rule → zero two-level errors observed).

---

## 6. Limitations & Future Work (reviewers will ask)

- Judge-model dependence: engine quality is bounded by the underlying LLM;
  temperature-0 single call, no self-consistency ensemble.
- Eval utterances are LLM-rewritten (though state/context-grounded); fully
  natural held-out dialogue evaluation is future work.
- Shared rules between generator and judge: disclose, and note the human-gold
  comparison is the guard against circularity.
- 3-level scale trades granularity for labelability; no decay/time dynamics
  between utterances; no cross-emotion coupling model.
- Single tracked patient (+relative) emotion; clinician/nurse carry no
  emotional state by design.
- Human gold from 3 annotators on 28 items so far; the 120-item set scales
  this ~4× (report new annotator count / agreement when available).

---

## 7. Numbers & Assets Checklist

Already available:
- 28-item results + confusion matrix + human range:
  `eval_data/emotion_aug_v3_3level/` (report md + metrics jsons).
- Model-comparison metrics (Opus 4.8, GPT-5.4/5.5 runs).
- 120-item balanced dataset: `eval_data/emotion_aug0821/`
  (augmented_drafts.jsonl, coverage_report.json, human_label_queue.jsonl).

To produce for the paper:
- [ ] Human labels for the 120-item set → `human_labels_merged.json`.
- [ ] `tools/evaluate_emotion_engine.py --provider anthropic
      --model claude-opus-4-8 --input-dir eval_data/emotion_aug0821`
      (+ `--no-scenario --no-traits` ablation runs).
- [ ] Per-transition-type accuracy table (4×6 grid) — new analysis, small
      script over predictions + gold.
- [ ] Inter-annotator agreement (pairwise + vs majority) on the new set.
- [ ] Optional baselines: never-change; definitions-only prompt.
- [ ] Example trajectory figure from a full run (emotion vector over turns
      with the triggering utterances).
