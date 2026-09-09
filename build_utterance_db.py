#!/usr/bin/env python
"""Build a labelled test utterance database for the emotion model.

Pipeline (per (emotion, level) target cell):
  1. SEED      - LLM generates an utterance a patient/relative with given traits
                 would say in a given scenario while expressing the target
                 emotion at the target level.
  2. AUGMENT   - LLM paraphrases each seed into variants that preserve the same
                 expressed emotion but vary wording / length / register.
  3. JUDGE     - an LLM judge reads each candidate and rates all FOUR emotions
                 (fear, sadness, overwhelm, shame) on the 4-level scale. The
                 judge's full vector becomes the row's label. A candidate is
                 ACCEPTED only if the judge reads the target emotion at the
                 target level (so labels are trustworthy).

Coverage: every accepted row counts toward all four (dimension, level) cells of
its judged vector. The loop keeps generating until each of the 12 cells has at
least --per-cell accepted utterances.

Reuses generate_utterances.py's relay client (throttle + retry + robust JSON),
so it honours AGENT_LLM_MODEL / AGENT_LLM_MIN_INTERVAL and OPENAI/ANTHROPIC env.

Run (resumable - appends to and re-reads the output file):
    AGENT_LLM_MODEL=claude-sonnet-4-6 python build_utterance_db.py \
        --provider anthropic --per-cell 6 --augment 3 --output utterance_db.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_utterances as gu
from ed_world_model.agents.agent_traits import AgentTraitLibrary
from ed_world_model.scenario_loader import load_scenario
from emotion_engine import EMOTION_LEVELS, EMOTION_NAMES

# Concise emotion definitions (kept consistent with emotion_engine prompts).
EMOTION_DEFS = {
    "fear": "worry, alarm, or panic about health, safety, survival, or the safety of someone closely tied to the patient.",
    "sadness": "loss, grief, helplessness, hopelessness, sorrow, or anticipated personally devastating loss.",
    "overwhelm": "acute emotional or cognitive overload from too much information, urgency, complexity, procedures, or distress.",
    "shame": "feeling exposed, blamed, judged, humiliated, flawed, personally at fault, or stripped of dignity.",
}
LEVELS = list(EMOTION_LEVELS)            # absent < present < extreme
EMOTIONS = list(EMOTION_NAMES)           # fear, sadness, overwhelm, shame
CELLS = [(e, l) for e in EMOTIONS for l in LEVELS]   # 12 cells


# ---------------------------------------------------------------------------
# Scenario / trait sampling
# ---------------------------------------------------------------------------

def _scenarios() -> list[Path]:
    files = sorted((ROOT / "transitions").glob("*/*.json"))
    return [f for f in files if f.name != "schema.json"]


def _scenario_context(path: Path, rng: random.Random):
    """Return (scenario_id, description, patient_traits, relative_traits)."""
    gs = load_scenario(path, trait_seed=rng.randint(0, 10_000))
    desc = gs.truth_state.scenario_description or path.stem
    return (
        f"{path.parent.name}/{path.stem}",
        desc.strip(),
        dict(gs.agent_profiles.patient.traits),
        dict(gs.agent_profiles.relative.traits),
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _seed_prompt(role, traits, scenario_id, desc, emotion, level) -> str:
    if level == "absent":
        target = (
            f"You feel NO {emotion} at all ({EMOTION_DEFS[emotion]}). Your line "
            f"should be neutral with respect to {emotion} - it must NOT express "
            f"{emotion}. You may feel other things or be matter-of-fact."
        )
    else:
        target = (
            f"You are expressing {emotion.upper()} at intensity '{level}'. "
            f"{emotion} = {EMOTION_DEFS[emotion]} "
            f"'present' = active to any non-crisis degree; "
            f"'extreme' = crisis-level panic, collapse, shutdown, unbearable "
            f"humiliation, or catastrophic grief. Make the "
            f"{emotion} read clearly at '{level}', no stronger and no weaker."
        )
    return (
        f"You are the {role} in an Emergency Department simulation.\n"
        f"Scenario: {desc}\n"
        f"Your traits (use them to shape tone/word choice, not facts): {traits}\n\n"
        f"{target}\n\n"
        "Say ONE short, natural spoken line (1-2 sentences) you would actually "
        "say out loud in this moment. Do not narrate or describe feelings in the "
        "third person; just speak. Do not invent specific clinical facts.\n\n"
        'Return JSON only: {"utterance": "..."}'
    )


def _augment_prompt(role, traits, emotion, level, utterance) -> str:
    constraint = (
        f"keep it NEUTRAL with respect to {emotion} (no {emotion} expressed)"
        if level == "absent"
        else f"keep the SAME {emotion} intensity ('{level}') - no stronger, no weaker"
    )
    return (
        f"Here is a spoken line from the {role} (traits: {traits}):\n"
        f'"{utterance}"\n\n'
        f"Write 3 alternative ways the same person could say essentially the same "
        f"thing in this moment. Vary wording, length, and phrasing, but {constraint}, "
        f"and stay consistent with the traits. Each must be a natural spoken line, "
        f"no narration.\n\n"
        'Return JSON only: {"variants": ["...", "...", "..."]}'
    )


def _judge_prompt(role, utterance) -> str:
    defs = "\n".join(f"  - {e}: {EMOTION_DEFS[e]}" for e in EMOTIONS)
    return (
        f"Read this spoken line from a {role} in an Emergency Department and rate "
        f"the intensity of EACH emotion the SPEAKER is expressing.\n\n"
        f"Emotions:\n{defs}\n"
        f"Levels (ordinal): absent < present < extreme. "
        f"'absent' = genuinely not present; 'present' = active but non-crisis.\n\n"
        f'Line: "{utterance}"\n\n'
        'Return JSON only with all four keys: '
        '{"fear": "...", "sadness": "...", "overwhelm": "...", "shame": "..."}'
    )


# ---------------------------------------------------------------------------
# DB build
# ---------------------------------------------------------------------------

def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _coverage(rows: list[dict]) -> dict:
    counts = {(e, l): 0 for (e, l) in CELLS}
    for r in rows:
        for e in EMOTIONS:
            counts[(e, r["emotion_state"][e])] += 1
    return counts


def _valid_state(state) -> bool:
    return isinstance(state, dict) and all(
        state.get(e) in LEVELS for e in EMOTIONS
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["openai", "anthropic"], default="anthropic")
    p.add_argument("--per-cell", type=int, default=6, help="min utterances per cell")
    p.add_argument("--augment", type=int, default=3, help="variants per seed")
    p.add_argument("--speakers", default="patient,relative",
                   help="comma list: patient,relative")
    p.add_argument("--no-validate", action="store_true",
                   help="skip the LLM judge; label = generation target (noisier)")
    p.add_argument("--max-rounds", type=int, default=200,
                   help="safety cap on generation rounds")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="utterance_db.jsonl")
    args = p.parse_args(argv)

    model = gu._model_for(args.provider)
    llm = gu._make_json_call(args.provider, model)
    rng = random.Random(args.seed)
    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    scenarios = _scenarios()
    out_path = ROOT / args.output

    rows = _load_existing(out_path)
    next_id = len(rows)
    print(f"Provider: {args.provider} | model: {model} | speakers: {speakers}")
    print(f"Target: each of {len(CELLS)} cells >= {args.per_cell} "
          f"(resuming from {len(rows)} existing rows)")
    print("=" * 72)

    def cell_done(counts):
        return all(counts[c] >= args.per_cell for c in CELLS)

    fout = out_path.open("a", encoding="utf-8")

    rounds = 0
    while not cell_done(_coverage(rows)) and rounds < args.max_rounds:
        rounds += 1
        counts = _coverage(rows)
        # target the most under-filled cell
        emotion, level = min(CELLS, key=lambda c: counts[c])
        role = rng.choice(speakers)
        scen_path = rng.choice(scenarios)
        sid, desc, ptraits, rtraits = _scenario_context(scen_path, rng)
        traits = ptraits if role == "patient" else rtraits

        # 1. seed
        try:
            seed = llm(_seed_prompt(role, traits, sid, desc, emotion, level))
            seed_utt = str(seed.get("utterance", "")).strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  [seed error] {str(exc)[:80]}")
            continue
        if not seed_utt:
            continue

        # 2. augment
        candidates = [seed_utt]
        if args.augment > 0:
            try:
                aug = llm(_augment_prompt(role, traits, emotion, level, seed_utt))
                candidates += [str(v).strip() for v in aug.get("variants", []) if str(v).strip()]
            except Exception as exc:  # noqa: BLE001
                print(f"  [augment error] {str(exc)[:80]}")

        # 3. judge + accept
        accepted = 0
        for utt in candidates:
            if args.no_validate:
                state = {e: (level if e == emotion else "absent") for e in EMOTIONS}
            else:
                try:
                    state = llm(_judge_prompt(role, utt))
                except Exception as exc:  # noqa: BLE001
                    print(f"  [judge error] {str(exc)[:80]}")
                    continue
                if not _valid_state(state):
                    continue
                state = {e: state[e] for e in EMOTIONS}
                if state[emotion] != level:   # primary target must match
                    continue
            row = {
                "id": f"utt_{next_id:05d}",
                "scenario_id": sid,
                "speaker_role": role,
                "speaker_traits": traits,
                "target_emotion": emotion,
                "target_level": level,
                "emotion_state": state,
                "utterance": utt,
                "method": "seed" if utt == seed_utt else "augmented",
            }
            rows.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            next_id += 1
            accepted += 1

        c = _coverage(rows)
        filled = sum(1 for cc in CELLS if c[cc] >= args.per_cell)
        print(f"[round {rounds}] target {emotion}/{level} role={role} "
              f"accepted={accepted} | cells filled {filled}/{len(CELLS)}")

    fout.close()
    counts = _coverage(rows)
    print("=" * 72)
    print("Final coverage (utterances per cell):")
    for e in EMOTIONS:
        print("  " + e.ljust(10) + " ".join(
            f"{l}={counts[(e, l)]}" for l in LEVELS))
    print(f"Total rows: {len(rows)}  ->  {out_path}")
    if not cell_done(counts):
        print("WARNING: hit --max-rounds before every cell reached the target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
