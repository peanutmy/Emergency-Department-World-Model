#!/usr/bin/env python
"""Build silver emotion-transition utterance drafts from full simulations.

This script implements the first three steps of the emotion-engine evaluation
pipeline:

1. Run full simulations from ``transitions/``.
2. Randomly extract patient/relative-directed source utterance candidates.
3. Use an LLM to rewrite each candidate toward a target emotion/transition
   cell (4 emotions x 6 valid transitions; e.g. fear/absent->extreme).

The output is *not* gold labels. Human reviewers should edit/fill
``human_label_next_state`` later before evaluation.

Example:

    env AGENT_LLM_MODEL="claude-sonnet-4-6" \\
    ANTHROPIC_THINKING_TYPE=adaptive \\
    ANTHROPIC_THINKING_EFFORT=max \\
    ANTHROPIC_THINKING_DISPLAY=omitted \\
    python tools/build_emotion_augmentation_dataset.py \\
      --provider anthropic \\
      --all-scenarios \\
      --turns 15 \\
      --target-min-per-cell 5 \\
      --output-dir eval_data/emotion_aug
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_utterances as gu
from emotion_engine import (
    EMOTION_DEFINITIONS_TEXT,
    EMOTION_LEVELS,
    EMOTION_NAMES,
    STEP_EMOTION_RULES,
)


LISTENERS = {"patient", "relative"}

# The grid is emotion x TRANSITION (from_level -> to_level). These six are
# exactly the transitions the engine can produce: 'extreme' is terminal, so it
# is never a source state (the encounter has ended; not even de-escalation is
# possible), and the only multi-level move is the rare absent->extreme jump,
# reserved for a single utterance carrying a terminal crisis cue. Both
# escalations into extreme trigger termination.
VALID_TRANSITIONS: tuple[tuple[str, str], ...] = (
    ("absent", "absent"),
    ("absent", "present"),
    ("absent", "extreme"),
    ("present", "absent"),
    ("present", "present"),
    ("present", "extreme"),
)
GENERIC_EMOTIONAL_OPENING_PREFIXES = (
    "i need to be straight with you",
    "i need to be honest with you",
    "i have to be straight with you",
    "i have to be honest with you",
    "i need you to listen to me",
    "i need you to hear me",
    "i need you to focus on me",
    "listen to me very carefully",
    "i won't sugarcoat",
    "i am not going to sugarcoat",
    "i'm not going to sugarcoat",
    "let me be very clear",
)


def _safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value[:90] or "item"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _scenario_files(root: Path, explicit: list[str] | None) -> list[Path]:
    if explicit:
        files = [(ROOT / item).resolve() if not Path(item).is_absolute() else Path(item) for item in explicit]
    else:
        if not root.is_absolute():
            root = ROOT / root
        files = sorted(root.glob("*/*.json"))
    return [path for path in files if path.exists() and path.name != "schema.json"]


def _parse_int_csv(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _looks_like_scenario_path(value: str) -> bool:
    if len(value) > 240:
        return False
    return value.endswith(".json") or "/" in value or "\\" in value


def _scenario_summary(scenario_value: str | None) -> str:
    if not scenario_value:
        return ""
    if not _looks_like_scenario_path(scenario_value):
        return str(scenario_value).strip()

    path = Path(scenario_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return str(scenario_value)

    data = _read_json(path)
    if data.get("scenario_description"):
        return str(data["scenario_description"]).strip()

    context = data.get("case_context") or {}
    pieces: list[str] = []
    demographics = context.get("demographics") or {}
    if demographics:
        age = demographics.get("age")
        sex = demographics.get("sex")
        if age or sex:
            pieces.append(" ".join(str(v) for v in (age, sex) if v))
    for key in ("chief_complaint", "symptoms", "history"):
        value = context.get(key)
        if isinstance(value, list):
            pieces.append(", ".join(str(item) for item in value[:8]))
        elif value:
            pieces.append(str(value))
    return "; ".join(piece for piece in pieces if piece) or path.stem


def _emotion_vector(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, str] = {}
    for emotion in EMOTION_NAMES:
        level = value.get(emotion)
        if level not in EMOTION_LEVELS:
            return None
        result[emotion] = level
    return result


def _has_generic_emotional_opening(utterance: str) -> bool:
    opening = utterance.strip().lower().replace("’", "'")
    opening = re.sub(
        r"^(sir|ma'am|maam|madam|mr\.?|mrs\.?|ms\.?)\s*[,—-]?\s*",
        "",
        opening,
    )
    return any(opening.startswith(prefix) for prefix in GENERIC_EMOTIONAL_OPENING_PREFIXES)


def _run_simulations(args: argparse.Namespace, rng: random.Random) -> list[Path]:
    runs_dir = Path(args.output_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.run:
        return [
            (ROOT / item).resolve() if not Path(item).is_absolute() else Path(item)
            for item in args.run
        ]

    if args.skip_simulations:
        return sorted(runs_dir.glob(args.run_glob))

    scenarios = _scenario_files(Path(args.scenario_root), args.scenario)
    if not scenarios:
        raise RuntimeError("No scenario files found.")

    seed_pool = _parse_int_csv(args.trait_seeds)
    run_paths: list[Path] = []
    if args.all_scenarios:
        run_plan = list(enumerate(scenarios))
    else:
        run_plan = [(index, rng.choice(scenarios)) for index in range(args.num_runs)]

    for index, scenario in run_plan:
        trait_seed = seed_pool[index % len(seed_pool)] if seed_pool else rng.randint(1, 999_999)
        try:
            scenario_name = str(scenario.relative_to(ROOT))
        except ValueError:
            scenario_name = scenario.stem
        stem = f"run_{index:03d}_{_safe_id(scenario_name)}_seed{trait_seed}"
        output_path = runs_dir / f"{stem}.json"
        run_paths.append(output_path)

        if output_path.exists() and not args.force_rerun:
            print(f"[simulation] skip existing {output_path}")
            continue

        env = os.environ.copy()
        if args.model:
            env["AGENT_LLM_MODEL"] = args.model

        cmd = [
            sys.executable,
            str(ROOT / "generate_utterances.py"),
            "--provider",
            args.provider,
            "--emotion",
            "--turns",
            str(args.turns),
            "--trait-seed",
            str(trait_seed),
            "--physiology-mode",
            args.physiology_mode,
            "--scenario",
            str(scenario),
            "--output",
            str(output_path),
        ]
        print(f"[simulation] {scenario} seed={trait_seed} -> {output_path}")
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        try:
            payload = _read_json(output_path)
            payload["scenario_path"] = str(scenario.relative_to(ROOT))
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - run output is still usable
            print(f"[simulation] warning: could not add scenario_path: {exc}")

    return run_paths


def _extract_candidates_from_run(
    run_path: Path,
    *,
    rng: random.Random,
    candidates_per_run: int,
) -> list[dict[str, Any]]:
    data = _read_json(run_path)
    turns = data.get("turns") or []
    scenario_path = data.get("scenario_path")
    scenario_summary = _scenario_summary(scenario_path or data.get("scenario"))
    run_candidates: list[dict[str, Any]] = []

    for turn_index in range(1, len(turns)):
        turn = turns[turn_index]
        previous_turn = turns[turn_index - 1]
        for message_index, message in enumerate(turn.get("messages") or []):
            listener = message.get("recipient")
            speaker = message.get("speaker")
            content = str(message.get("content") or "").strip()
            if listener not in LISTENERS or not speaker or speaker == listener or not content:
                continue

            current_state = _emotion_vector(previous_turn.get(f"{listener}_emotion"))
            if current_state is None:
                continue

            traits = data.get(f"{listener}_traits") or {}
            candidate_id = (
                f"{_safe_id(run_path.stem)}_t{turn.get('turn', turn_index)}_"
                f"m{message_index}_{listener}"
            )
            run_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "source_run": str(run_path),
                    "source_turn": turn.get("turn", turn_index),
                    "scenario_path": scenario_path,
                    "scenario_summary": scenario_summary,
                    "listener": listener,
                    "speaker": speaker,
                    "traits": traits,
                    "current_state": current_state,
                    "original_utterance": content,
                }
            )

    rng.shuffle(run_candidates)
    if candidates_per_run > 0:
        run_candidates = run_candidates[:candidates_per_run]
    return run_candidates


def _extract_candidates(
    run_paths: list[Path],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for run_path in run_paths:
        if not run_path.exists():
            print(f"[candidates] missing run file, skip {run_path}")
            continue
        candidates.extend(
            _extract_candidates_from_run(
                run_path,
                rng=rng,
                candidates_per_run=args.candidates_per_run,
            )
        )
    rng.shuffle(candidates)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    return candidates


def _augmentation_prompt(
    candidate: dict[str, Any],
    *,
    target_emotion: str,
    target_level: str,
) -> str:
    current_level = candidate["current_state"][target_emotion]
    current_index = EMOTION_LEVELS.index(current_level)
    target_index = EMOTION_LEVELS.index(target_level)

    if current_level == "absent" and target_level == "extreme":
        target_instruction = (
            f"The listener's current {target_emotion} is 'absent'. Write the "
            f"utterance so it JUMPS {target_emotion} two levels straight to "
            "'extreme'. NOTE: this two-level jump is the engine's only allowed "
            "multi-level move and it is RARE - it happens only in extremely "
            "severe cases, not for ordinary bad news. The single utterance "
            "itself must carry a terminal crisis cue (imminent death, "
            "catastrophic loss, total loss of control, or devastating "
            "humiliation/blame) strong enough that breakdown is plausible "
            "with no prior emotional buildup."
        )
    elif target_index > current_index:
        target_instruction = (
            f"The listener's current {target_emotion} is '{current_level}'. "
            f"Write the utterance so it RAISES {target_emotion} by exactly one "
            f"level to '{target_level}': it needs a concrete new trigger of "
            "exactly that size - strong enough to move one level, not strong "
            "enough to move two."
        )
    elif target_index < current_index:
        target_instruction = (
            f"The listener's current {target_emotion} is '{current_level}'. "
            f"Write the utterance so it LOWERS {target_emotion} by exactly one "
            f"level to '{target_level}': it must concretely resolve or address "
            "the trigger (a resolved threat, a concrete plan, real relief). "
            "Naming, validating, or empathizing with the emotion alone is NOT "
            "relief and would leave the level unchanged."
        )
    else:
        if current_level == "absent":
            target_instruction = (
                f"The listener's current {target_emotion} is already "
                f"'absent'. Write the utterance so it KEEPS "
                f"{target_emotion} at exactly 'absent': do not introduce a "
                "new concrete threat, loss, processing burden, blame, "
                "humiliation, or dignity cue."
            )
        else:
            target_instruction = (
                f"The listener's current {target_emotion} is already "
                f"'{current_level}'. Write the utterance so it KEEPS "
                f"{target_emotion} at exactly '{target_level}': do not add a "
                "new trigger big enough to raise it, and do not fully resolve "
                "the existing trigger."
            )
    if target_level == "absent" and current_level != "absent":
        target_instruction += (
            " The end state must be genuine absence: the trigger is fully "
            "resolved or gone, not merely reduced."
        )
    if target_level == "extreme":
        target_instruction += (
            " 'extreme' is a terminal crisis level describing the listener's "
            "BREAKDOWN, not merely severe news: the utterance must plausibly "
            "leave the listener unable to keep participating in the encounter "
            "(panic attack, collapse, shutdown, unbearable humiliation, "
            "catastrophic grief). Because extreme ends the simulation, do not "
            "use it for strong but cooperative distress. Convey imminent "
            "death, catastrophic loss, total loss of control, or devastating "
            "humiliation/blame in a way that would plausibly make breakdown "
            "likely, grounded in this scenario. A protective plan usually "
            "buffers extreme, but it does not automatically block extreme."
        )

    return f"""\
You are creating one single-utterance test item for an Emergency Department \
emotion transition engine.

The test item will be judged only from:
scenario_summary + listener traits + current_state + ONE utterance.
There is no prior dialogue context.

The judging engine scores items using these definitions of the four emotions:
{EMOTION_DEFINITIONS_TEXT}

The judging engine applies the following rules; your draft label must respect \
them:
{STEP_EMOTION_RULES}

Rewrite the original utterance into a realistic utterance from the same speaker \
to the same listener. Keep it grounded in the scenario. Do not invent unsupported \
diagnoses, outcomes, family facts, medications, completed procedures, successful \
treatment responses, or test results. You may make the wording more emotionally \
salient when clinically plausible.

Do not increase emotion by simply adding a generic urgency preamble. Avoid \
templated openings such as "I need to be straight with you", "I need you to \
listen to me very carefully", "I need you to hear me", "I need you to focus on \
me", or "I won't sugarcoat this". The target emotion should come from the \
specific clinical content, disclosure, question, reassurance, loss, complexity, \
or dignity/judgment cue in the utterance. Preserve the original utterance's \
clinical speech act when possible: if it was a question, update, instruction, \
or reassurance, the rewrite should still feel like that kind of ED utterance.

Primary target:
- primary_target_emotion: {target_emotion}
- primary_target_level: {target_level}
- {target_instruction}
- The primary target is the main purpose of the rewrite. Other emotions may \
change if the same utterance would naturally change them; label those secondary \
effects in draft_label_next_state, but do not make them the main purpose.

Scenario:
{candidate["scenario_summary"]}

Listener: {candidate["listener"]}
Speaker: {candidate["speaker"]}
Listener traits:
{json.dumps(candidate["traits"], ensure_ascii=False)}

Current emotion state:
{json.dumps(candidate["current_state"], ensure_ascii=False)}

Original utterance:
{candidate["original_utterance"]}

If the scenario, listener, or state seems inconsistent or unusual, do NOT ask \
questions and do NOT add commentary - still return only your best-effort JSON \
object.

Return JSON only:
{{
  "augmented_utterance": "...",
  "draft_label_next_state": {{
    "fear": "absent|present|extreme",
    "sadness": "absent|present|extreme",
    "overwhelm": "absent|present|extreme",
    "shame": "absent|present|extreme"
  }},
  "rationale": "one short sentence explaining why the primary target emotion reaches the target level",
  "tags": ["..."]
}}
"""


def _valid_draft(
    result: Any,
    *,
    current_state: dict[str, str],
    target_emotion: str,
    target_level: str,
) -> bool:
    if not isinstance(result, dict):
        return False
    augmented_utterance = str(result.get("augmented_utterance") or "").strip()
    if not augmented_utterance:
        return False
    if _has_generic_emotional_opening(augmented_utterance):
        return False
    state = _emotion_vector(result.get("draft_label_next_state"))
    if state is None:
        return False
    if state[target_emotion] != target_level:
        return False
    # Engine semantics: one utterance moves any emotion at most one level.
    # Sole exception: the rare absent->extreme jump, and only for the primary
    # target emotion - a secondary emotion hitting extreme would terminate the
    # encounter on its own and hijack the item.
    for emotion in EMOTION_NAMES:
        moved = abs(
            EMOTION_LEVELS.index(state[emotion])
            - EMOTION_LEVELS.index(current_state[emotion])
        )
        if moved > 1:
            is_primary_extreme_jump = (
                emotion == target_emotion
                and current_state[emotion] == "absent"
                and state[emotion] == "extreme"
            )
            if not is_primary_extreme_jump:
                return False
    return True


def _augment_candidates(
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        raise RuntimeError("No source candidates available for augmentation.")

    model = args.model or gu._model_for(args.provider)
    llm = gu._make_json_call(args.provider, model)
    drafts: list[dict[str, Any]] = []
    target_cells = [
        (emotion, from_level, to_level)
        for emotion in EMOTION_NAMES
        for from_level, to_level in VALID_TRANSITIONS
    ]
    if getattr(args, "cells", None):
        requested = []
        for spec in str(args.cells).split(","):
            spec = spec.strip()
            if not spec:
                continue
            emotion, _, transition = spec.partition("/")
            from_level, _, to_level = transition.partition("->")
            if (emotion, from_level, to_level) not in target_cells:
                raise SystemExit(
                    f"--cells: unknown cell '{spec}' "
                    "(use e.g. 'fear/absent->extreme')"
                )
            requested.append((emotion, from_level, to_level))
        target_cells = requested
    eligible_counts: dict[str, int] = {}
    skipped_cells: list[dict[str, Any]] = []

    for target_emotion, from_level, target_level in target_cells:
        cell_key = f"{target_emotion}/{from_level}->{target_level}"
        eligible = [
            candidate
            for candidate in candidates
            if candidate["current_state"][target_emotion] == from_level
        ]
        eligible_counts[cell_key] = len(eligible)
        if not eligible:
            # The run pool has no candidate in this cell's source state. Skip
            # honestly instead of forcing the transition on a random candidate.
            print(f"[augment] SKIP {cell_key}: no eligible candidates")
            skipped_cells.append(
                {"emotion": target_emotion, "from_level": from_level,
                 "level": target_level, "reason": "no_eligible_candidates"}
            )
            continue
        print(f"[augment] {cell_key}: {len(eligible)} eligible candidates")

        for cell_index in range(args.target_min_per_cell):
            if args.max_augmented_items > 0 and len(drafts) >= args.max_augmented_items:
                return drafts, {
                    "eligible_counts": eligible_counts,
                    "skipped_cells": skipped_cells,
                }

            candidate: dict[str, Any] | None = None
            result: dict[str, Any] | None = None
            for attempt in range(args.augment_attempts):
                # Fresh draw each attempt so one bad source cannot burn the item.
                attempt_candidate = rng.choice(eligible)
                try:
                    response = llm(
                        _augmentation_prompt(
                            attempt_candidate,
                            target_emotion=target_emotion,
                            target_level=target_level,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[augment] LLM error {cell_key} "
                        f"attempt {attempt + 1}: {str(exc)[:120]}"
                    )
                    continue
                if _valid_draft(
                    response,
                    current_state=attempt_candidate["current_state"],
                    target_emotion=target_emotion,
                    target_level=target_level,
                ):
                    candidate = attempt_candidate
                    result = response
                    break
                print(
                    f"[augment] invalid draft {cell_key} "
                    f"attempt {attempt + 1}: "
                    f"{json.dumps(response, ensure_ascii=False)[:200]}"
                )

            if result is None or candidate is None:
                print(f"[augment] failed cell item {cell_key}")
                continue

            item_id = (
                f"{target_emotion}_{from_level}_to_{target_level}_"
                f"{cell_index:03d}_{len(drafts):04d}"
            )
            drafts.append(
                {
                    "item_id": item_id,
                    "source_candidate_id": candidate["candidate_id"],
                    "source_run": candidate["source_run"],
                    "source_turn": candidate["source_turn"],
                    "scenario_path": candidate["scenario_path"],
                    "scenario_summary": candidate["scenario_summary"],
                    "listener": candidate["listener"],
                    "speaker": candidate["speaker"],
                    "traits": candidate["traits"],
                    "current_state": candidate["current_state"],
                    "original_utterance": candidate["original_utterance"],
                    "augmented_utterance": str(result["augmented_utterance"]).strip(),
                    "target_emotion": target_emotion,
                    "target_level": target_level,
                    "transition": f"{from_level}->{target_level}",
                    "draft_label_next_state": result["draft_label_next_state"],
                    "rationale": str(result.get("rationale") or "").strip(),
                    "tags": result.get("tags") if isinstance(result.get("tags"), list) else [],
                    "human_label_next_state": None,
                    "status": "needs_human_label",
                }
            )
            print(
                f"[augment] {cell_key} "
                f"{cell_index + 1}/{args.target_min_per_cell}"
            )
    return drafts, {
        "eligible_counts": eligible_counts,
        "skipped_cells": skipped_cells,
    }


def _coverage_report(
    candidates: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    *,
    target_min_per_cell: int,
    augment_notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transition_keys = [f"{f}->{t}" for f, t in VALID_TRANSITIONS]
    target_counts: dict[str, dict[str, int]] = {
        emotion: {key: 0 for key in transition_keys} for emotion in EMOTION_NAMES
    }
    for draft in drafts:
        transition = draft.get("transition") or (
            f"{draft['current_state'][draft['target_emotion']]}->"
            f"{draft['target_level']}"
        )
        if transition in target_counts.get(draft["target_emotion"], {}):
            target_counts[draft["target_emotion"]][transition] += 1

    undercovered = []
    for emotion in EMOTION_NAMES:
        for transition in transition_keys:
            count = target_counts[emotion][transition]
            if count < target_min_per_cell:
                undercovered.append(
                    {
                        "emotion": emotion,
                        "transition": transition,
                        "count": count,
                        "needed": target_min_per_cell - count,
                    }
                )

    current_level_counts: dict[str, dict[str, int]] = {
        emotion: {level: 0 for level in EMOTION_LEVELS} for emotion in EMOTION_NAMES
    }
    for candidate in candidates:
        for emotion in EMOTION_NAMES:
            current_level_counts[emotion][candidate["current_state"][emotion]] += 1

    return {
        "target_min_per_cell": target_min_per_cell,
        "total_source_candidates": len(candidates),
        "total_augmented_drafts": len(drafts),
        "coverage_by_target": target_counts,
        "undercovered": undercovered,
        "source_candidate_current_levels": current_level_counts,
        "eligible_counts": (augment_notes or {}).get("eligible_counts", {}),
        "skipped_cells": (augment_notes or {}).get("skipped_cells", []),
        "source_candidates_by_listener": dict(Counter(c["listener"] for c in candidates)),
        "augmented_drafts_by_listener": dict(Counter(d["listener"] for d in drafts)),
        "source_candidates_by_scenario": dict(Counter(c["scenario_path"] for c in candidates)),
    }


def _human_label_rows(drafts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a concise human-labeling view of augmented drafts."""

    rows: list[dict[str, Any]] = []
    empty_label = {emotion: "" for emotion in EMOTION_NAMES}
    for draft in drafts:
        rows.append(
            {
                "scenario": draft.get("scenario_summary", ""),
                "listener": draft["listener"],
                "speaker": draft["speaker"],
                "traits": draft["traits"],
                "current_state": draft["current_state"],
                "augmented_utterance": draft["augmented_utterance"],
                "human_label_next_state": dict(empty_label),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="anthropic")
    parser.add_argument("--model", default=None, help="Overrides AGENT_LLM_MODEL.")
    parser.add_argument("--scenario-root", default="transitions")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Specific scenario path. Repeatable. If omitted, scenarios are sampled randomly.",
    )
    parser.add_argument(
        "--run",
        action="append",
        help=(
            "Existing full simulation JSON to extract from. Repeatable. "
            "If provided, new simulations are not run."
        ),
    )
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help=(
            "Run each discovered scenario once, in sorted order. If --scenario is "
            "provided, run each explicit scenario once. Overrides --num-runs."
        ),
    )
    parser.add_argument("--turns", type=int, default=15)
    parser.add_argument("--trait-seeds", default=None, help="Comma list, e.g. 6,10,17.")
    parser.add_argument("--physiology-mode", default="hybrid")
    parser.add_argument("--output-dir", default="eval_data/emotion_aug")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--skip-simulations", action="store_true")
    parser.add_argument("--run-glob", default="*.json")
    parser.add_argument("--candidates-per-run", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--skip-augmentation", action="store_true")
    parser.add_argument(
        "--cells",
        default=None,
        help=(
            "Comma list of emotion/transition cells to generate, e.g. "
            "'fear/absent->extreme,shame/present->extreme'. Default: the "
            "full 4x6 grid."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Merge newly generated drafts into the existing "
            "augmented_drafts.jsonl instead of overwriting it. The label "
            "queue and coverage report are rebuilt from the merged set."
        ),
    )
    parser.add_argument("--target-min-per-cell", type=int, default=5)
    parser.add_argument("--augment-attempts", type=int, default=3)
    parser.add_argument(
        "--max-augmented-items",
        type=int,
        default=0,
        help="Debug cap. 0 means no cap.",
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_paths = _run_simulations(args, rng)
    candidates = _extract_candidates(run_paths, args, rng)

    candidates_path = output_dir / "source_candidates.jsonl"
    _write_jsonl(candidates_path, candidates)
    print(f"[write] {len(candidates)} source candidates -> {candidates_path}")

    drafts: list[dict[str, Any]]
    augment_notes: dict[str, Any] = {}
    drafts_path = output_dir / "augmented_drafts.jsonl"
    if args.skip_augmentation:
        drafts = _read_jsonl(drafts_path)
    else:
        drafts, augment_notes = _augment_candidates(candidates, args, rng)
        if args.append:
            existing = _read_jsonl(drafts_path)
            offset = len(existing)
            for index, draft in enumerate(drafts):
                draft["item_id"] = f"{draft['item_id']}_a{offset + index:04d}"
            print(
                f"[append] merging {len(drafts)} new drafts into "
                f"{len(existing)} existing"
            )
            drafts = existing + drafts
        _write_jsonl(drafts_path, drafts)
        print(f"[write] {len(drafts)} augmented drafts -> {drafts_path}")

    human_label_path = output_dir / "human_label_queue.jsonl"
    _write_jsonl(human_label_path, _human_label_rows(drafts))
    print(f"[write] {len(drafts)} human-label rows -> {human_label_path}")

    report = _coverage_report(
        candidates,
        drafts,
        target_min_per_cell=args.target_min_per_cell,
        augment_notes=augment_notes,
    )
    report_path = output_dir / "coverage_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[write] coverage report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
