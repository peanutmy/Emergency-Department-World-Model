"""The emotion/level/transition rules must be textually identical between the
emotion engine's judging prompt and the augmentation dataset builder's
generation prompt. STEP_EMOTION_RULES is the single source of truth."""
from __future__ import annotations

import json
from pathlib import Path
import random
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_engine import (
    EMOTION_DEFINITIONS_TEXT,
    EMOTION_NAMES,
    STEP_EMOTION_RULES,
    STEP_PROMPT_TEMPLATE,
)
from tools.build_emotion_augmentation_dataset import (
    VALID_TRANSITIONS,
    _augment_candidates,
    _augmentation_prompt,
    _run_simulations,
    _valid_draft,
)


def _candidate() -> dict:
    return {
        "scenario_summary": "A test scenario.",
        "listener": "patient",
        "speaker": "clinician",
        "traits": {},
        "current_state": {
            "fear": "absent",
            "sadness": "absent",
            "overwhelm": "absent",
            "shame": "absent",
        },
        "original_utterance": "We are starting treatment now.",
    }


def test_step_prompt_embeds_shared_rules_verbatim() -> None:
    assert STEP_EMOTION_RULES in STEP_PROMPT_TEMPLATE


def test_augmentation_prompt_embeds_shared_rules_verbatim() -> None:
    prompt = _augmentation_prompt(
        _candidate(), target_emotion="fear", target_level="present"
    )
    assert STEP_EMOTION_RULES in prompt


def test_step_prompt_embeds_shared_definitions_verbatim() -> None:
    assert EMOTION_DEFINITIONS_TEXT in STEP_PROMPT_TEMPLATE


def test_augmentation_prompt_embeds_shared_definitions_verbatim() -> None:
    prompt = _augmentation_prompt(
        _candidate(), target_emotion="fear", target_level="present"
    )
    assert EMOTION_DEFINITIONS_TEXT in prompt


def test_rules_are_format_safe() -> None:
    # The augmentation prompt embeds the rules inside an f-string and the
    # engine embeds them inside a .format() template - braces would break both.
    assert "{" not in STEP_EMOTION_RULES
    assert "}" not in STEP_EMOTION_RULES


def test_old_local_definitions_removed() -> None:
    prompt = _augmentation_prompt(
        _candidate(), target_emotion="fear", target_level="present"
    )
    assert "Emotion definitions:" not in prompt
    assert "Transition rules the engine enforces" not in prompt


def test_transition_grid_covers_exactly_engine_reachable_transitions() -> None:
    assert set(VALID_TRANSITIONS) == {
        ("absent", "absent"),
        ("absent", "present"),
        ("absent", "extreme"),
        ("present", "absent"),
        ("present", "present"),
        ("present", "extreme"),
    }
    # 'extreme' is terminal: it must never appear as a source state.
    assert all(from_level != "extreme" for from_level, _ in VALID_TRANSITIONS)


def test_absent_to_extreme_prompt_notes_rarity() -> None:
    prompt = _augmentation_prompt(
        _candidate(), target_emotion="sadness", target_level="extreme"
    )
    assert "two levels" in prompt
    assert "RARE" in prompt


def test_valid_draft_allows_absent_to_extreme_for_primary_target_only() -> None:
    current = {emotion: "absent" for emotion in EMOTION_NAMES}
    draft = {
        "augmented_utterance": "Despite everything we did, your son died a few minutes ago.",
        "draft_label_next_state": {
            "fear": "absent",
            "sadness": "extreme",
            "overwhelm": "absent",
            "shame": "absent",
        },
    }
    assert _valid_draft(
        draft, current_state=current, target_emotion="sadness", target_level="extreme"
    )
    # The same two-level jump on a non-target emotion stays invalid.
    assert not _valid_draft(
        draft, current_state=current, target_emotion="fear", target_level="absent"
    )


def test_absent_to_absent_prompt_does_not_reference_existing_trigger() -> None:
    prompt = _augmentation_prompt(
        _candidate(), target_emotion="fear", target_level="absent"
    )
    primary_target = prompt.split("Primary target:", 1)[1].split("Scenario:", 1)[0]
    assert "existing trigger" not in primary_target
    assert "do not introduce a new concrete threat" in primary_target


def test_all_scenarios_runs_each_explicit_scenario_once(tmp_path, monkeypatch) -> None:
    scenario_a = tmp_path / "A.json"
    scenario_b = tmp_path / "B.json"
    scenario_a.write_text(json.dumps({}), encoding="utf-8")
    scenario_b.write_text(json.dumps({}), encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(cmd, *, cwd, env, check):  # noqa: ANN001
        calls.append(list(cmd))
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.write_text(
            json.dumps({"scenario": "test", "turns": []}),
            encoding="utf-8",
        )

    import tools.build_emotion_augmentation_dataset as builder

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        run=None,
        skip_simulations=False,
        scenario_root="transitions",
        scenario=[str(scenario_a), str(scenario_b)],
        all_scenarios=True,
        num_runs=999,
        trait_seeds="11,22",
        model=None,
        provider="openai",
        turns=15,
        physiology_mode="hybrid",
        force_rerun=False,
    )

    run_paths = _run_simulations(args, random.Random(1))

    assert len(run_paths) == 2
    assert len(calls) == 2
    assert [Path(call[call.index("--scenario") + 1]) for call in calls] == [
        scenario_a,
        scenario_b,
    ]
    assert [call[call.index("--trait-seed") + 1] for call in calls] == ["11", "22"]


def test_transition_cells_fill_target_from_matching_sources(monkeypatch) -> None:
    import tools.build_emotion_augmentation_dataset as builder

    def _source(index: int, fear_level: str) -> dict:
        candidate = _candidate()
        candidate["current_state"]["fear"] = fear_level
        candidate.update(
            candidate_id=f"c{index}",
            source_run="run.json",
            source_turn=1,
            scenario_path="transitions/x.json",
        )
        return candidate

    candidates = [_source(i, "absent") for i in range(8)] + [
        _source(8 + i, "present") for i in range(4)
    ]

    def fake_llm(prompt: str) -> dict:
        state_line = prompt.split("Current emotion state:\n", 1)[1].splitlines()[0]
        label = json.loads(state_line)
        label["fear"] = "extreme"
        return {
            "augmented_utterance": "Your heart is failing and we are out of options.",
            "draft_label_next_state": label,
            "rationale": "r",
            "tags": [],
        }

    monkeypatch.setattr(builder.gu, "_make_json_call", lambda provider, model: fake_llm)
    args = SimpleNamespace(
        model="m",
        provider="openai",
        cells="fear/absent->extreme,fear/present->extreme",
        target_min_per_cell=5,
        augment_attempts=2,
        max_augmented_items=0,
    )

    drafts, _notes = _augment_candidates(candidates, args, random.Random(3))

    assert len(drafts) == 10
    jumps = [d for d in drafts if d["transition"] == "absent->extreme"]
    steps = [d for d in drafts if d["transition"] == "present->extreme"]
    assert len(jumps) == 5 and len(steps) == 5
    # Every item's source candidate really was in the cell's from-state.
    assert all(d["current_state"]["fear"] == "absent" for d in jumps)
    assert all(d["current_state"]["fear"] == "present" for d in steps)
