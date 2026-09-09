#!/usr/bin/env python
"""End-to-end emotion-engine trajectory runner.

Pipeline (the thing we are testing):

    scenario JSON
        -> ScenarioLoader  (random agent traits + levels)
        -> initial vitals
        -> init_emotion(patient)                    # deterministic all-absent baseline
        -> for each time tick:
               step physiology  -> new vitals       # deterministic stand-in
               pick an utterance (from a NON-patient speaker)
               step_emotion(patient, utterance)      # LLM update
        -> print/save the full trajectory

The *patient* is the listener whose emotion we track: the emotion engine only
lets an utterance from ANOTHER agent move emotion (self-utterance is rejected),
so the patient's state evolves in response to clinician/nurse/relative speech.

Modes
-----
--emotion real   (default) uses emotion_engine._call_llm_json -> Anthropic API.
                 Requires ANTHROPIC_API_KEY (and optionally ANTHROPIC_BASE_URL /
                 EMOTION_ENGINE_MODEL). This is the real engine under test.
--emotion fake   uses a deterministic keyword-based llm_call so you can verify
                 the whole pipeline wiring WITHOUT any API key.

Usage
-----
    # No key needed - proves the wiring end to end:
    python examples/run_emotion_trajectory.py \
        --scenario "transitions/Cardiology/Nightmares Case 7 Hyperkalemia.json" \
        --emotion fake

    # Real emotion engine (needs ANTHROPIC_API_KEY):
    export ANTHROPIC_API_KEY=sk-...
    python examples/run_emotion_trajectory.py \
        --scenario "transitions/Cardiology/Nightmares Case 7 Hyperkalemia.json" \
        --emotion real --trait-seed 7
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import emotion_engine as ee
from ed_world_model.agents.agent_traits import AgentTraitLibrary
from ed_world_model.scenario_loader import ScenarioLoader
from ed_world_model.state.global_state import GlobalState, Vitals


# ---------------------------------------------------------------------------
# Utterance script (one driver utterance per tick).
#
# Each entry pairs a clinical action (drives the vitals stepper) with an
# utterance spoken by a NON-patient agent (drives the patient's emotion). Swap
# this list out for real-agent output when you want generated dialogue.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tick:
    speaker: str          # who speaks the utterance (not "patient")
    utterance: str        # the line scored by the emotion engine
    action: str           # "none" | "oxygen" | "treatment" -> physiology stepper


DEFAULT_SCRIPT: list[Tick] = [
    Tick("clinician",
         "Your blood test shows a dangerously high potassium level, and it is "
         "affecting your heart rhythm. We need to act quickly.",
         action="none"),
    Tick("nurse",
         "I'm placing an oxygen mask over your nose and mouth now - you'll feel "
         "some air flow and a bit of pressure.",
         action="oxygen"),
    Tick("clinician",
         "Your heart rate is very slow right now. If it doesn't pick up, we may "
         "have to place a temporary wire to pace your heart.",
         action="none"),
    Tick("clinician",
         "I know this is a lot. Here is exactly what we're doing: medicine to "
         "protect your heart, then medicine to bring the potassium down. We are "
         "right here with you the whole time.",
         action="treatment"),
    Tick("clinician",
         "The medicine is working - your potassium is coming down and your heart "
         "rate is recovering. You are going to be okay.",
         action="treatment"),
]


# ---------------------------------------------------------------------------
# Deterministic physiology stand-in (no LLM).
#
# This is NOT the hybrid physiology engine - it's a transparent stepper so the
# trajectory runs without an API key. It nudges vitals toward normal when a
# treatment/oxygen action is applied and lets them drift when untreated.
# ---------------------------------------------------------------------------

def seed_vitals(scenario_vitals: Vitals) -> dict[str, float]:
    """Use scenario baseline vitals, filling any gaps with a hypoxic/brady set."""
    base = scenario_vitals.model_dump()
    defaults = {"HR": 38.0, "BP_sys": 92.0, "BP_dia": 58.0,
                "RR": 24.0, "O2Sat": 87.0, "T": 36.8}
    return {k: (base.get(k) if base.get(k) is not None else v)
            for k, v in defaults.items()}


def step_vitals(vitals: dict[str, float], action: str) -> dict[str, float]:
    """Advance vitals one tick based on the action. Returns a new dict."""
    v = dict(vitals)
    if action in ("oxygen", "treatment"):
        v["O2Sat"] = min(99.0, v["O2Sat"] + 4)
    if action == "treatment":
        v["HR"] = min(80.0, v["HR"] + 8)      # bradycardia recovering
        v["RR"] = max(16.0, v["RR"] - 2)
    if action == "none":
        v["O2Sat"] = max(80.0, v["O2Sat"] - 1)
        v["RR"] = min(34.0, v["RR"] + 1)
    return {k: round(x, 1) for k, x in v.items()}


# ---------------------------------------------------------------------------
# Trait enrichment: turn {trait: level} into {trait: {level, definition}} so the
# emotion engine's init prompt gets the full trait definition, not just a label.
# ---------------------------------------------------------------------------

def enrich_traits(role: str, traits: dict[str, Any],
                  library: AgentTraitLibrary) -> dict[str, Any]:
    defs = {t.name: t.definition for t in library.traits_for(role)}
    return {
        name: {"level": level, "definition": defs.get(name, "")}
        for name, level in traits.items()
    }


# ---------------------------------------------------------------------------
# Fake emotion llm_call: deterministic, key-free wiring smoke test.
# ---------------------------------------------------------------------------

_ORDER = ["absent", "present", "extreme"]


def _bump(level: str, steps: int) -> str:
    i = max(0, min(len(_ORDER) - 1, _ORDER.index(level) + steps))
    return _ORDER[i]


def fake_llm_call(prompt: str, *, model: str = "fake") -> dict[str, Any]:
    """Stand-in for emotion_engine._call_llm_json - no network, deterministic."""
    if '"initial_state"' in prompt:  # init prompt
        return {
            "initial_state": {"fear": "present", "sadness": "present",
                              "overwhelm": "present", "shame": "absent"},
            "rationale": "[fake] Acute, frightening cardiac emergency drives fear; "
                         "some sadness/overwhelm; no shame trigger.",
        }
    # step prompt: read current levels back out and nudge by keyword.
    def cur(name: str) -> str:
        for line in prompt.splitlines():
            if f"{name}=" in line and "Listener's CURRENT" not in line:
                token = line.split(f"{name}=", 1)[1].split(",")[0].strip()
                if token in _ORDER:
                    return token
        return "absent"

    utt = prompt.rsplit("Utterance (score this one):", 1)[-1].lower()
    fear, sad, over, shame = cur("fear"), cur("sadness"), cur("overwhelm"), cur("shame")
    evidence, notes = "no salient cue", "no change"
    if any(w in utt for w in ("working", "okay", "recovering", "coming down")):
        fear, evidence, notes = _bump(fear, -1), "reassuring resolution", "fear eased"
    elif any(w in utt for w in ("dangerous", "slow", "wire", "pace", "quickly", "high potassium")):
        fear, evidence, notes = _bump(fear, +1), "threat escalation", "fear rose"
    elif any(w in utt for w in ("lot", "exactly what", "step", "with you")):
        over, evidence, notes = _bump(over, -1), "concrete plan offered", "overwhelm eased"
    return {
        "new_state": {"fear": fear, "sadness": sad, "overwhelm": over, "shame": shame},
        "evidence": f"[fake] {evidence}",
        "reasoning": f"[fake] {notes}; others held.",
    }


# ---------------------------------------------------------------------------
# Trajectory record
# ---------------------------------------------------------------------------

@dataclass
class TickRecord:
    tick: int
    action: str
    vitals: dict[str, float]
    speaker: str
    utterance: str
    emotion_before: dict[str, str]
    emotion_after: dict[str, str]
    evidence: str | None
    reasoning: str | None


@dataclass
class TrajectoryRecord:
    scenario: str | None
    patient_traits: dict[str, Any]
    initial_vitals: dict[str, float]
    init_rationale: str | None
    initial_emotion: dict[str, str]
    ticks: list[TickRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "patient_traits": self.patient_traits,
            "initial_vitals": self.initial_vitals,
            "init_rationale": self.init_rationale,
            "initial_emotion": self.initial_emotion,
            "ticks": [vars(t) for t in self.ticks],
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(scenario_path: str, *, emotion_mode: str, trait_seed: int | None,
        script: list[Tick]) -> TrajectoryRecord:
    library = AgentTraitLibrary()
    loader = ScenarioLoader(randomize_traits=True, trait_seed=trait_seed,
                            trait_library=library)
    state: GlobalState = loader.load(scenario_path)

    scenario_text = state.truth_state.scenario_description or ""
    patient_traits = dict(state.agent_profiles.patient.traits)
    enriched = enrich_traits("patient", patient_traits, library)

    vitals = seed_vitals(state.patient_state.vitals)

    llm_call = fake_llm_call if emotion_mode == "fake" else ee._call_llm_json

    # ---- deterministic baseline: initialize the patient's emotion ----
    traj = ee.init_emotion(
        agent_role="patient",
        scenario=scenario_text,
        initial_vitals=vitals,
        agent_traits=enriched,
        llm_call=llm_call,
    )

    record = TrajectoryRecord(
        scenario=state.truth_state.scenario_description,
        patient_traits=patient_traits,
        initial_vitals=vitals,
        init_rationale=traj.init_rationale,
        initial_emotion=traj.state.summary(),
    )

    clinical_context = _clinical_context(state)
    dialogue_lines: list[str] = []

    # ---- per-tick: step vitals, then step emotion on the utterance ----
    for i, tick in enumerate(script, start=1):
        vitals = step_vitals(vitals, tick.action)
        emotion_before = traj.state.summary()

        dialogue_digest = "\n".join(dialogue_lines[-6:])
        traj = ee.step_emotion(
            traj,
            utterance=tick.utterance,
            speaker_role=tick.speaker,
            clinical_context=f"{clinical_context}\nCurrent vitals: "
                             f"{_fmt_vitals(vitals)}",
            dialogue=dialogue_digest,
            llm_call=llm_call,
        )
        dialogue_lines.append(f"{tick.speaker}: {tick.utterance}")

        record.ticks.append(TickRecord(
            tick=i,
            action=tick.action,
            vitals=vitals,
            speaker=tick.speaker,
            utterance=tick.utterance,
            emotion_before=emotion_before,
            emotion_after=traj.state.summary(),
            evidence=traj.last_evidence,
            reasoning=traj.last_reasoning,
        ))

    return record


def _clinical_context(state: GlobalState) -> str:
    demo = state.truth_state.demographics
    parts = []
    if demo.age is not None:
        parts.append(f"{demo.age}yo")
    if demo.sex:
        parts.append(demo.sex)
    who = " ".join(parts) or "patient"
    cc = state.truth_state.patient_internal_state.chief_complaint or ""
    return f"Listener is the patient ({who}). {cc}".strip()


def _fmt_vitals(v: dict[str, float]) -> str:
    return ", ".join(f"{k}={x}" for k, x in v.items())


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(record: TrajectoryRecord) -> str:
    lines = [
        f"Scenario: {record.scenario}",
        "",
        f"Patient traits: {record.patient_traits}",
        f"Initial vitals: {_fmt_vitals(record.initial_vitals)}",
        f"Initial emotion: {record.initial_emotion}",
        f"  rationale: {record.init_rationale}",
        "",
        "Trajectory",
        "==========",
    ]
    for t in record.ticks:
        changed = {k: f"{t.emotion_before[k]}->{v}"
                   for k, v in t.emotion_after.items()
                   if t.emotion_before[k] != v} or "(no change)"
        lines += [
            f"[tick {t.tick}] action={t.action}  vitals: {_fmt_vitals(t.vitals)}",
            f"  {t.speaker}: {t.utterance}",
            f"  emotion: {changed}",
            f"  now: {t.emotion_after}",
            f"  evidence: {t.evidence}",
            f"  reasoning: {t.reasoning}",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON.")
    parser.add_argument("--emotion", choices=["real", "fake"], default="real",
                        help="real = Anthropic emotion engine (needs ANTHROPIC_API_KEY); "
                             "fake = deterministic key-free wiring test.")
    parser.add_argument("--trait-seed", type=int, default=None,
                        help="Seed for reproducible random agent traits.")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Emit JSON instead of readable text.")
    parser.add_argument("--output", default=None,
                        help="Optional path to write the trajectory JSON.")
    args = parser.parse_args(argv)

    record = run(args.scenario, emotion_mode=args.emotion,
                 trait_seed=args.trait_seed, script=DEFAULT_SCRIPT)

    if args.output:
        Path(args.output).write_text(json.dumps(record.as_dict(), indent=2),
                                     encoding="utf-8")
    if args.json_output:
        print(json.dumps(record.as_dict(), indent=2))
    else:
        print(render(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
