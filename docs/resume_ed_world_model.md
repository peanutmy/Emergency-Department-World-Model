# ED World Model — Resume Material (Full System)

Resume-ready summary of the technical skills demonstrated by the Emergency
Department (ED) World Model project — the complete platform: expert-validated
clinical dataset, physiology engine, multi-agent simulation, emotion modeling,
and evaluation pipelines. Pick bullets per target role; all numbers are real
and defensible in interviews.

---

## Project Title

**Emergency Department World Model — Expert-Grounded Multi-Agent Clinical Simulation Platform**

*A full-stack "world model" of emergency-department care: a hand-curated,
physician-sourced and faculty-validated clinical dataset (49 cases, 235
vital-sign transition pairs) grounds a physiology engine; LLM-driven agents
(clinician, nurse, patient, family) play out encounters over structured state
with partial observability; an ordinal emotion engine tracks psychological
state; and every learned component is validated against human judgment through
purpose-built evaluation pipelines.*

---

## What Was Built (system inventory)

| Layer | Components |
|---|---|
| **Clinical dataset (original)** | 49 ER cases across 10 specialties digitized from physician-contributed real case reports (PDF sources), reviewed and corrected with medical-school faculty; 235 structured vital-sign transition pairs (before-state → typed clinical action → labeled after-state); versioned JSON schema; PDF-grounded curation of chief complaints, symptoms, baseline vitals, and lab turnaround times |
| **Physiology engine** | Hybrid physiology adapter predicting vitals/features responses to treatments, grounded in the expert-validated transition pairs |
| **Simulation core** | Turn orchestrator; agent activation policy; deterministic turn loop with verbal-action commit pipeline; termination logic (terminal emotional states, clinical endpoints) |
| **State & world model** | Pydantic strict-schema global state (ground-truth state, patient state, known facts, psychological state, runtime state); event sourcing; pending-question tracking; diagnostic test bank with realistic turnaround delays; medication administration record (MAR) |
| **Agents** | Two-stage clinician (structured clinical decision → natural-language utterance); verbal-only nurse with bedside-event slots; two-stage patient with disclosure rules and hidden history; family-member agent with realistic knowledge boundaries; scripted stubs for deterministic testing |
| **Partial observability** | Role-specific observation builders — each agent sees only what its role realistically could (hidden history, unreleased lab results, monitor data access-controlled per role) |
| **Action system** | Typed action registry (treatment families, kind hints, parameter templates); action validator with parameter normalization; duplicate-order guards; nurse shadow-execution of physician orders |
| **Emotion engine** | Per-agent 4-dimension ordinal emotion state (fear, sadness, overwhelm, shame × 4 levels) driven by an LLM transition function with explicit rules: persistence, one-level movement with shock exceptions, terminal crisis level, addressed-listener filtering |
| **Clinical realism** | Consciousness modeling (sedation/intubation suppress perception, with correct same-turn event ordering); preverbal-child dialogue redirection to guardians; communication-style policies producing realistic non-idealized clinician speech (calibrated urgency, blunt disclosure, post-stabilization confrontation of non-adherence) |
| **Data & eval pipelines** | (a) Labeled utterance database: three-stage seed → augment → LLM-judge pipeline with acceptance gating over a 16-cell emotion×level grid; (b) emotion-transition benchmark: reachability-constrained augmentation, 3-annotator human labeling, majority-vote gold; (c) LLM-as-judge evaluation harness with information-parity ablations and from-scratch ordinal metrics |
| **LLM infrastructure** | Unified multi-provider clients (Anthropic / OpenAI, relay-compatible) with throttling, retry/backoff, rate-limit handling, robust JSON extraction, self-repair retries; resumable batch pipelines |
| **Process** | 53 documented milestone handoffs (M0–M14u) recording design decisions, patches, and regressions across the project's lifetime; 760+ tests |

---

## Resume Bullets

### Dataset Construction & Domain Grounding

- Built an original expert-validated clinical dataset: digitized 49 real
  emergency-department cases (10 specialties) from physician-contributed case
  reports into a versioned JSON schema, and curated 235 vital-sign transition
  pairs (before-state → typed clinical action → labeled after-state) reviewed
  and corrected with medical-school faculty
- Designed the transition-pair schema bridging free-text clinical narratives
  and machine-usable supervision: typed actions (family / kind / parameters),
  structured vitals and clinical features (rhythm, airway status, GCS), and
  per-pair evaluation targets
- Grounded simulation timing in clinical reality: per-test lab turnaround
  delays, PDF-sourced baseline vitals and presenting symptoms per case

### System / Architecture

- Architected a multi-agent LLM simulation platform for emergency-medicine
  encounters: 4 role-specialized agents over a deterministic turn
  orchestrator, Pydantic strict-schema world state, typed action registry with
  validation, and event sourcing; 760+ passing unit/integration tests
- Built a POMDP-structured interactive environment — typed state/action
  spaces, role-specific partial observations, stochastic transition functions
  (physiology + emotion), and terminal conditions — suitable as a training and
  evaluation environment for clinical decision-making agents
- Designed role-based partial observability so each agent only accesses
  information its real-world counterpart could (hidden patient history,
  unreleased lab results), preventing cross-agent information leakage
- Modeled clinical-realism mechanics: consciousness suppression under
  sedation/intubation with correct same-turn event ordering, dialogue
  redirection for non-verbal patients (pediatric/intubated) to family members,
  and a persistent medication administration record that eliminated LLM
  treatment amnesia across long episodes
- Closed the physiology loop: physician orders validated against a typed
  action registry, shadow-executed by a nurse agent, applied to vitals via a
  transition engine grounded in the expert-validated pairs, and fed back into
  agent observations with realistic lab turnaround delays

### LLM Engineering

- Engineered production-grade LLM output reliability: strict JSON-schema
  parsing with markdown-fence tolerance, automatic self-repair retries
  (re-prompting with the parse error and expected schema), and empty-response
  recovery — cut agent output failure from ~70% of turns to near zero after a
  model migration
- Built unified multi-provider LLM infrastructure (Anthropic Claude, OpenAI
  GPT, relay-compatible) with request throttling, retry/backoff, rate-limit
  and model-unavailability handling, and resumable batch pipelines
- Designed a two-stage generation architecture separating structured decisions
  from natural-language surface realization, enabling independent validation
  of clinical logic and dialogue quality
- Authored and iterated behavior-policy prompts producing realistic,
  non-idealized clinical communication (calibrated urgency, blunt disclosure,
  post-stabilization confrontation), maintained as single-source-of-truth
  shared constants locked by regression tests
- Built an LLM emotion-transition engine over ordinal state with explicit
  transition rules (persistence, bounded movement, terminal states), validated
  against a human-annotated benchmark

### Evaluation & Data Pipelines

- Built a three-stage labeled-utterance pipeline (LLM seed generation →
  paraphrase augmentation → independent LLM-judge rating with acceptance
  gating) achieving guaranteed coverage of a 16-cell emotion×level grid
- Built an emotion-transition benchmark end-to-end: reachability-constrained
  LLM augmentation, 3-annotator human labeling with bilingual guides and
  evidence capture, format validation and auto-repair, inter-annotator
  agreement analysis (Fleiss' κ), and majority-vote gold with adjudication
- Implemented ordinal evaluation metrics from scratch — quadratic weighted
  kappa, ordinal MAE, change-detection precision/recall/F1, direction
  accuracy, target-vs-side-dimension breakdown — and established human-ceiling
  baselines from annotator-vs-gold agreement
- Ran controlled model comparisons (GPT vs. Claude judges under identical
  inputs) and information-parity ablations (withholding scenario/traits) to
  separate prompt-driven, model-driven, and information-driven effects
- Demonstrated near-human engine performance (QWK 0.87, 100% direction
  accuracy, exact agreement within 2–6 points of the weakest human annotator)
  and localized the dominant failure mode (premature terminal-level
  escalation: 16 of 22 false positives) via systematic error clustering

### Software Engineering & Process

- Test-driven development across the stack: 760+ pytest cases including
  regression locks on prompt content, parser edge cases, and orchestration
  invariants; every bug fix shipped with a reproducing test
- Debugged multi-layer LLM system failures end-to-end from production run
  logs: silent-turn detection → drop-event forensics → raw-output capture →
  root-cause isolation (concurrent token-budget truncation and output-format
  drift)
- Reproducible experiment design: seeded randomization for scenario/trait
  sampling, deterministic reruns, versioned dataset directories with coverage
  reports
- Maintained engineering continuity across a long-running solo project via 53
  documented milestone handoffs recording design decisions, patches, and
  regressions

---

## Skills Keywords (for the skills section)

> **LLM / AI Engineering**: multi-agent systems, LLM orchestration, prompt
> engineering, structured output & JSON schema enforcement, LLM-as-judge,
> synthetic data augmentation, model benchmarking & migration (Anthropic
> Claude API, OpenAI API), agent memory design, POMDP environment design,
> partial observability
>
> **Data & Evaluation**: expert-validated dataset construction, clinical data
> curation and schema design, human annotation pipelines, inter-annotator
> agreement (Fleiss' κ, quadratic weighted kappa), ordinal metrics, ablation
> studies, error analysis, human-ceiling baselining
>
> **Engineering**: Python, Pydantic, pytest, JSONL data pipelines, rate-limit
> and retry infrastructure, resumable batch jobs, argparse CLI tooling,
> event-sourced state management, reproducible experiments, Git

---

## Quantified Achievements (interview-ready numbers)

| Claim | Number |
|---|---|
| Expert-validated clinical dataset | 49 real ER cases, 10 specialties, 235 vital-transition pairs, faculty-reviewed |
| Test suite | 760+ passing tests |
| Documented development milestones | 53 handoff documents (M0–M14u) |
| Agent output failure after migration fix | ~70% of turns → near zero |
| Emotion engine vs. human gold | QWK 0.87, direction accuracy 100% |
| Human annotation | 3 annotators, majority gold on 106/112 labels, Fleiss' κ analysis |
| Engine vs. human ceiling | within 2–6 points of weakest annotator (74.5% vs. 76.4% exact) |
| Error localization | 16/22 false positives traced to one calibration issue |

---

## Two-Minute Interview Stories (prepare these)

1. **The dataset story** — how do you ground an LLM simulation in clinical
   reality? Collected physician-contributed real ER case reports → designed a
   transition-pair schema (state, typed action, labeled next state) → digitized
   49 cases / 235 pairs → review loop with medical-school faculty → the pairs
   drive both physiology prediction and evaluation. Shows real-world data
   curation, schema design, and expert-in-the-loop validation — rare and
   valuable for domain-AI roles.
2. **The silent-agents debugging story** — after a model migration, most agent
   turns went silent. Silent-turn detection → drop events showed 100%
   strict-JSON parse failures → added raw-output capture → found two
   independent causes (markdown fences + thinking-budget truncation) → fixed
   with fence-tolerant parsing, self-repair retries, and budget adjustment.
   Shows systematic debugging of nondeterministic LLM systems.
3. **The evaluation-design story** — how do you evaluate an LLM emotion model
   with no ground truth? Constraint-filtered synthetic data → 3 human
   annotators with deliberately unconstrained judgment (avoiding circularity
   with the engine's own rules) → majority gold + agreement analysis →
   information-parity ablation → metrics that defeat the 77%-no-change class
   imbalance (change-detection P/R alongside QWK).
4. **The realism-policy story** — LLM agents were "too polite" to generate
   shame/sadness data; designed communication policies (confrontation of
   non-adherence after stabilization) grounded in scenario facts, producing
   the first organic shame-state transitions in simulation.
