"""Controlled fake-agent demo runner for v1.3.1 scenarios."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

from ed_world_model.actions.registry import ActionFamily, ActionRegistry, KindHint
from ed_world_model.adapters.noop_emotion import NoopEmotionEngine
from ed_world_model.agents.clinician import ClinicianAgent
from ed_world_model.agents.llm_client import (
    DEFAULT_CLINICIAN_DECISION_RESPONSE,
    DEFAULT_PATIENT_DECISION_RESPONSE,
    DEFAULT_VERBAL_ONLY_RESPONSE,
    FakeLLMClient,
    OpenAICompatibleLLMClient,
)
from ed_world_model.agents.nurse import NurseAgent
from ed_world_model.agents.patient import PatientAgent
from ed_world_model.agents.relative import RelativeAgent
from ed_world_model.orchestration.orchestrator import CLINICIAN, NURSE, PATIENT, RELATIVE
from ed_world_model.orchestration.runner import (
    IntegrationRunner,
    RecordingStubPhysiologyAdapter,
    TrajectoryTurn,
)
from ed_world_model.scenario_loader import ScenarioLoader
from ed_world_model.state.global_state import GlobalState
from ed_world_model.trajectory import TrajectoryLogger


AGENT_MODE_FAKE = "fake"
AGENT_MODE_REAL = "real"
PHYSIOLOGY_MODE_FAKE = "fake"
PHYSIOLOGY_MODE_HYBRID = "hybrid"
FAKE_MODE = AGENT_MODE_FAKE

LLMClientFactory = Callable[[], Any]
HybridPhysiologyAdapter: Any | None = None


@dataclass(frozen=True)
class DemoTurn:
    """Readable and JSON-serializable record for one demo turn."""

    turn_index: int
    turn_index_after: int
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    active_agents: list[str]
    messages: list[dict[str, Any]]
    events: list[dict[str, Any]]
    clinician_action: dict[str, Any] | None
    diagnostic_orders_created: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_results_released: list[dict[str, Any]] = field(default_factory=list)
    nurse_shadow_execution: list[dict[str, Any]] = field(default_factory=list)
    nurse_bedside_slots: list[dict[str, Any]] = field(default_factory=list)
    physiology_action_kind_hint: str | None = None
    physiology_action: dict[str, Any] | None = None
    agent_verbal_decisions: list[dict[str, Any]] = field(default_factory=list)
    patient_state_after: dict[str, Any] = field(default_factory=dict)
    validation_drops: list[dict[str, Any]] = field(default_factory=list)
    parser_errors: list[dict[str, Any]] = field(default_factory=list)
    terminated: bool = False
    termination_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "turn_index_after": self.turn_index_after,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "active_agents": self.active_agents,
            "messages": self.messages,
            "events": self.events,
            "clinician_action": self.clinician_action,
            "diagnostic_orders_created": self.diagnostic_orders_created,
            "diagnostic_results_released": self.diagnostic_results_released,
            "nurse_shadow_execution": self.nurse_shadow_execution,
            "nurse_bedside_slots": self.nurse_bedside_slots,
            "physiology_action_kind_hint": self.physiology_action_kind_hint,
            "physiology_action": self.physiology_action,
            "agent_verbal_decisions": self.agent_verbal_decisions,
            "patient_state_after": self.patient_state_after,
            "validation_drops": self.validation_drops,
            "parser_errors": self.parser_errors,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
        }


@dataclass(frozen=True)
class DemoResult:
    """Complete demo trajectory plus final public runtime state."""

    scenario_identifier: str | None
    agent_mode: str
    physiology_mode: str
    requested_turns: int
    turns: list[DemoTurn]
    final_patient_state: dict[str, Any]
    final_known_facts: dict[str, Any]
    final_turn_index: int

    def as_dict(self) -> dict[str, Any]:
        turns = [turn.as_dict() for turn in self.turns]
        return {
            "scenario_identifier": self.scenario_identifier,
            "agent_mode": self.agent_mode,
            "physiology_mode": self.physiology_mode,
            "requested_turns": self.requested_turns,
            "turn_count": len(turns),
            "turns": turns,
            "messages": [
                message
                for turn in turns
                for message in turn["messages"]
            ],
            "events": [
                event
                for turn in turns
                for event in turn["events"]
            ],
            "final_patient_state": self.final_patient_state,
            "final_known_facts": self.final_known_facts,
            "final_turn_index": self.final_turn_index,
        }


def run_demo_scenario(
    scenario_path: str | Path,
    *,
    turns: int = 5,
    mode: str | None = None,
    agent_mode: str = AGENT_MODE_FAKE,
    physiology_mode: str = PHYSIOLOGY_MODE_FAKE,
    scenario_loader: ScenarioLoader | None = None,
    llm_client_factory: LLMClientFactory | None = None,
    physiology_llm_client_factory: LLMClientFactory | None = None,
) -> DemoResult:
    """Load a scenario and run a v1.3.1 demo trajectory."""

    if mode is not None:
        if mode != AGENT_MODE_FAKE:
            raise ValueError(
                f"Unsupported legacy demo mode {mode!r}; use --agent-mode instead."
            )
        if agent_mode != AGENT_MODE_FAKE:
            raise ValueError("Use either mode or agent_mode, not both.")
        agent_mode = mode
    _validate_agent_mode(agent_mode)
    _validate_physiology_mode(physiology_mode)
    if turns < 0:
        raise ValueError("turns must be non-negative.")

    path = Path(scenario_path)
    loader = scenario_loader or ScenarioLoader()
    state = loader.load(path, max_turns=turns)
    runner = IntegrationRunner(
        state,
        agents=build_demo_agents(
            state,
            agent_mode=agent_mode,
            llm_client_factory=llm_client_factory,
        ),
        physiology_adapter=build_demo_physiology_adapter(
            physiology_mode=physiology_mode,
            llm_client_factory=physiology_llm_client_factory
            or llm_client_factory,
        ),
        emotion_engine=NoopEmotionEngine(),
    )

    demo_turns: list[DemoTurn] = []
    for _ in range(turns):
        trajectory_turn = runner.run_turn()
        demo_turns.append(_build_demo_turn(trajectory_turn, runner.state))
        if trajectory_turn.terminated:
            break

    return DemoResult(
        scenario_identifier=_scenario_identifier(path),
        agent_mode=agent_mode,
        physiology_mode=physiology_mode,
        requested_turns=turns,
        turns=demo_turns,
        final_patient_state=runner.state.patient_state.model_dump(),
        final_known_facts=runner.state.known_facts.model_dump(),
        final_turn_index=runner.state.runtime_state.turn_index,
    )


def build_demo_agents(
    state: GlobalState,
    *,
    agent_mode: str = AGENT_MODE_FAKE,
    llm_client_factory: LLMClientFactory | None = None,
) -> dict[str, Any]:
    """Build fake or real LLM-backed agents for the demo."""

    _validate_agent_mode(agent_mode)
    if agent_mode == AGENT_MODE_FAKE:
        return build_fake_demo_agents(state)
    return build_real_demo_agents(
        state,
        llm_client_factory=llm_client_factory,
    )


def build_fake_demo_agents(state: GlobalState) -> dict[str, Any]:
    """Build deterministic fake LLM-backed agents for the controlled demo."""

    first_test_name = (
        state.truth_state.test_bank[0].name
        if state.truth_state.test_bank
        else None
    )
    clinician_script = [
        _initial_clinician_decision(first_test_name),
        _initial_clinician_verbal(),
        DEFAULT_CLINICIAN_DECISION_RESPONSE,
        _oxygen_support_decision(verbal=True),
        _oxygen_support_verbal(),
        _oxygen_support_decision(verbal=False),
    ]

    return {
        CLINICIAN: ClinicianAgent(
            FakeLLMClient(
                clinician_script,
                default_response=DEFAULT_CLINICIAN_DECISION_RESPONSE,
            ),
            profile=state.agent_profiles.clinician.model_dump(),
        ),
        NURSE: NurseAgent(
            FakeLLMClient(
                [_nurse_result_report_response(), _nurse_bedside_response()],
                default_response=DEFAULT_VERBAL_ONLY_RESPONSE,
            ),
            profile=state.agent_profiles.nurse.model_dump(),
        ),
        PATIENT: PatientAgent(
            FakeLLMClient(
                [_patient_required_decision(), _patient_required_response()],
                default_response=DEFAULT_PATIENT_DECISION_RESPONSE,
            ),
            profile=state.agent_profiles.patient.model_dump(),
        ),
        RELATIVE: RelativeAgent(
            FakeLLMClient(
                [],
                default_response=DEFAULT_VERBAL_ONLY_RESPONSE,
            ),
            profile=state.agent_profiles.relative.model_dump(),
        ),
    }


def build_real_demo_agents(
    state: GlobalState,
    *,
    llm_client_factory: LLMClientFactory | None = None,
) -> dict[str, Any]:
    """Build real-client-backed agents while preserving parser boundaries."""

    llm_client = _make_real_llm_client(llm_client_factory)
    return {
        CLINICIAN: ClinicianAgent(
            llm_client,
            profile=state.agent_profiles.clinician.model_dump(),
        ),
        NURSE: NurseAgent(
            llm_client,
            profile=state.agent_profiles.nurse.model_dump(),
        ),
        PATIENT: PatientAgent(
            llm_client,
            profile=state.agent_profiles.patient.model_dump(),
        ),
        RELATIVE: RelativeAgent(
            llm_client,
            profile=state.agent_profiles.relative.model_dump(),
        ),
    }


def build_demo_physiology_adapter(
    *,
    physiology_mode: str = PHYSIOLOGY_MODE_FAKE,
    llm_client_factory: LLMClientFactory | None = None,
) -> Any:
    """Build fake or Hybrid physiology for the demo runner."""

    _validate_physiology_mode(physiology_mode)
    if physiology_mode == PHYSIOLOGY_MODE_FAKE:
        return RecordingStubPhysiologyAdapter(output=_fake_physiology_output)

    llm_client = _make_real_llm_client(llm_client_factory)
    adapter_class = _hybrid_physiology_adapter_class()
    return adapter_class(llm_client=llm_client)


def render_readable_trajectory(result: DemoResult) -> str:
    """Render a compact, readable per-turn trajectory."""

    lines = [
        f"Scenario: {result.scenario_identifier or 'unknown'}",
        f"Final turn_index: {result.final_turn_index}",
        "",
    ]
    for turn in result.turns:
        lines.extend(
            [
                f"Turn {turn.turn_index}",
                f"  turn_index: {turn.turn_index}",
                f"  active_agents: {_format_list(turn.active_agents)}",
                "  committed messages:",
            ]
        )
        lines.extend(_format_messages(turn.messages, indent="    "))
        lines.extend(
            [
                f"  clinician_action: {_format_clinician_action(turn.clinician_action)}",
                "  diagnostic_orders_created:",
            ]
        )
        lines.extend(_format_diagnostic_orders(turn.diagnostic_orders_created))
        lines.append("  diagnostic_results_released:")
        lines.extend(_format_diagnostic_results(turn.diagnostic_results_released))
        lines.append("  nurse_shadow_execution:")
        lines.extend(_format_nurse_shadow(turn.nurse_shadow_execution))
        lines.append("  nurse_bedside_verbal_slots:")
        lines.extend(_format_nurse_bedside_slots(turn.nurse_bedside_slots))
        lines.append(
            "  physiology_action_kind_hint: "
            f"{turn.physiology_action_kind_hint or 'none'}"
        )
        lines.append(
            "  physiology_action: "
            f"{_format_physiology_action(turn.physiology_action)}"
        )
        lines.append(
            "  patient_state_after: "
            f"{_format_patient_state(turn.patient_state_after)}"
        )
        lines.append("  validation_drops / parser_errors:")
        lines.extend(_format_validation_drops(turn.validation_drops))
        lines.append("  events:")
        lines.extend(_format_events(turn.events))
        if turn.terminated:
            lines.append(f"  termination_reason: {turn.termination_reason}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_demo_json(result: DemoResult) -> str:
    return json.dumps(result.as_dict(), indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a v1.3.1 scenario demo."
    )
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON.")
    parser.add_argument("--turns", type=int, default=5, help="Number of turns to run.")
    parser.add_argument(
        "--mode",
        default=None,
        choices=[FAKE_MODE],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--agent-mode",
        default=AGENT_MODE_FAKE,
        choices=[AGENT_MODE_FAKE, AGENT_MODE_REAL],
        help="Agent backend mode. Defaults to deterministic fake agents.",
    )
    parser.add_argument(
        "--physiology-mode",
        default=PHYSIOLOGY_MODE_FAKE,
        choices=[PHYSIOLOGY_MODE_FAKE, PHYSIOLOGY_MODE_HYBRID],
        help="Physiology backend mode. Defaults to deterministic fake physiology.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print JSON instead of readable text.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for trajectory.json and summary.json.",
    )
    args = parser.parse_args(argv)

    result = run_demo_scenario(
        args.scenario,
        turns=args.turns,
        mode=args.mode,
        agent_mode=args.agent_mode,
        physiology_mode=args.physiology_mode,
    )
    if args.output_dir is not None:
        TrajectoryLogger(args.output_dir).save_all(result)
    if args.json_output:
        print(render_demo_json(result))
    else:
        print(render_readable_trajectory(result), end="")
    return 0


def _validate_agent_mode(agent_mode: str) -> None:
    if agent_mode not in {AGENT_MODE_FAKE, AGENT_MODE_REAL}:
        raise ValueError(
            "agent_mode must be 'fake' or 'real'; "
            f"received {agent_mode!r}."
        )


def _validate_physiology_mode(physiology_mode: str) -> None:
    if physiology_mode not in {PHYSIOLOGY_MODE_FAKE, PHYSIOLOGY_MODE_HYBRID}:
        raise ValueError(
            "physiology_mode must be 'fake' or 'hybrid'; "
            f"received {physiology_mode!r}."
        )


def _make_real_llm_client(
    llm_client_factory: LLMClientFactory | None = None,
) -> Any:
    if llm_client_factory is not None:
        return llm_client_factory()
    return OpenAICompatibleLLMClient()


def _hybrid_physiology_adapter_class() -> Any:
    global HybridPhysiologyAdapter
    if HybridPhysiologyAdapter is None:
        from ed_world_model.adapters.hybrid_physiology_adapter import (
            HybridPhysiologyAdapter as adapter_class,
        )

        HybridPhysiologyAdapter = adapter_class
    return HybridPhysiologyAdapter


def _initial_clinician_decision(test_name: str | None) -> dict[str, Any]:
    action = None
    if test_name is not None:
        action = {"type": "diagnostic_order", "test_name": test_name}
    return {
        "action": action,
        "verbal_decision": {
            "speaker": CLINICIAN,
            "should_speak": True,
            "target": PATIENT,
            "intent": "ask how the patient is feeling",
            "reasoning_summary": "The patient can provide focused symptom information.",
            "key_points": ["Ask how the patient is feeling right now."],
            "forbidden_points": ["Do not mention additional tests or treatments."],
            "requires_response": True,
        },
    }


def _initial_clinician_verbal() -> dict[str, Any]:
    return {
        "verbal_action": {
            "speaker": CLINICIAN,
            "target": PATIENT,
            "content": "Can you tell me how you are feeling right now?",
            "requires_response": True,
        },
    }


def _oxygen_support_decision(*, verbal: bool = True) -> dict[str, Any]:
    verbal_decision = {
        "speaker": CLINICIAN,
        "should_speak": verbal,
        "target": PATIENT if verbal else None,
        "intent": (
            "explain oxygen support"
            if verbal
            else "stay silent while oxygen support continues"
        ),
        "reasoning_summary": (
            "Oxygen support is the selected structured action."
            if verbal
            else "The treatment action can proceed without another verbal message."
        ),
        "key_points": (
            ["Oxygen support is starting.", "The team will keep reassessing."]
            if verbal
            else []
        ),
        "forbidden_points": ["Do not mention additional tests or treatments."],
        "requires_response": False,
    }
    return {
        "action": {
            "type": "medical_treatment_order",
            "family": ActionFamily.RESPIRATORY_SUPPORT,
            "kind_hint": KindHint.OXYGEN_SUPPORT,
            "params": {
                "oxygen_device": "NRB",
                "FiO2": 1.0,
            },
        },
        "verbal_decision": verbal_decision,
    }


def _oxygen_support_verbal() -> dict[str, Any]:
    return {
        "verbal_action": {
            "speaker": CLINICIAN,
            "target": PATIENT,
            "content": "I am starting oxygen support while we keep reassessing.",
            "requires_response": False,
        },
    }


def _patient_required_decision() -> dict[str, Any]:
    return {
        "speaker": PATIENT,
        "should_speak": True,
        "target": CLINICIAN,
        "intent": "answer the clinician's symptom question",
        "reasoning_summary": "The clinician asked the patient how they are feeling.",
        "key_points": ["The patient feels short of breath and uncomfortable."],
        "forbidden_points": ["Do not add new symptoms."],
        "requires_response": False,
    }


def _patient_required_response() -> dict[str, Any]:
    return {
        "verbal_action": {
            "speaker": PATIENT,
            "target": CLINICIAN,
            "content": "I feel short of breath and uncomfortable.",
            "requires_response": False,
        }
    }


def _nurse_result_report_response() -> dict[str, Any]:
    return {
        "verbal_action": {
            "speaker": NURSE,
            "target": CLINICIAN,
            "content": "The first diagnostic result is back and available.",
            "requires_response": False,
        }
    }


def _nurse_bedside_response() -> dict[str, Any]:
    return {
        "verbal_action": {
            "speaker": NURSE,
            "target": PATIENT,
            "content": "I am placing the oxygen mask now.",
            "requires_response": False,
        }
    }


def _fake_physiology_output(
    global_state: GlobalState,
    engine_facing_action: Mapping[str, Any],
) -> dict[str, Any]:
    vitals = global_state.patient_state.vitals.model_dump()
    features: dict[str, Any] = {}
    kind_hint = engine_facing_action.get("kind_hint")
    params = engine_facing_action.get("params") or {}

    if kind_hint == KindHint.OXYGEN_SUPPORT:
        if "oxygen_device" in params:
            features["oxygen_device"] = params.get("oxygen_device")
        if "FiO2" in params:
            features["FiO2"] = params.get("FiO2")
        o2_sat = vitals.get("O2Sat")
        if isinstance(o2_sat, int | float) and o2_sat < 95:
            vitals["O2Sat"] = min(95, o2_sat + 5)
    elif kind_hint == KindHint.NO_ACTION:
        o2_sat = vitals.get("O2Sat")
        if isinstance(o2_sat, int | float) and o2_sat < 90:
            vitals["O2Sat"] = o2_sat - 1

    output: dict[str, Any] = {"vitals": vitals}
    if features:
        output["features"] = features
    return output


def _build_demo_turn(
    trajectory_turn: TrajectoryTurn,
    state: GlobalState,
) -> DemoTurn:
    events = list(trajectory_turn.events)
    validation_drops = [
        event for event in events if event.get("type") == "validation_drop"
    ]
    parser_errors = [
        event
        for event in validation_drops
        if event.get("payload", {}).get("item") == "agent_proposal"
    ]
    return DemoTurn(
        turn_index=trajectory_turn.turn_index_before,
        turn_index_after=trajectory_turn.turn_index_after,
        state_before=deepcopy(trajectory_turn.state_before),
        state_after=deepcopy(trajectory_turn.state_after),
        active_agents=list(trajectory_turn.active_agents),
        messages=list(trajectory_turn.committed_messages),
        events=events,
        clinician_action=_clinician_action(trajectory_turn),
        diagnostic_orders_created=[
            event.get("payload", {})
            for event in events
            if event.get("type") == "diagnostic_order_created"
        ],
        diagnostic_results_released=list(trajectory_turn.released_diagnostics),
        nurse_shadow_execution=[
            event.get("payload", {})
            for event in events
            if event.get("type") == "nurse_shadow_execution"
        ],
        nurse_bedside_slots=[
            event.get("payload", {})
            for event in events
            if event.get("type") == "nurse_bedside_verbal_slot"
        ],
        physiology_action_kind_hint=trajectory_turn.physiology_action_kind_hint,
        physiology_action=deepcopy(trajectory_turn.physiology_action),
        agent_verbal_decisions=deepcopy(trajectory_turn.agent_verbal_decisions),
        patient_state_after=deepcopy(
            trajectory_turn.state_after.get(
                "patient_state",
                state.patient_state.model_dump(),
            )
        ),
        validation_drops=validation_drops,
        parser_errors=parser_errors,
        terminated=trajectory_turn.terminated,
        termination_reason=trajectory_turn.termination_reason,
    )


def _clinician_action(trajectory_turn: TrajectoryTurn) -> dict[str, Any] | None:
    for validation_result in trajectory_turn.validation_results:
        if not validation_result.get("ok"):
            continue
        action_type = validation_result.get("action_type")
        if action_type is None:
            return None
        normalized_action = validation_result.get("normalized_action") or {}
        action: dict[str, Any] = {
            "type": action_type,
            "action_type": action_type,
        }
        if action_type == "diagnostic_order":
            test_name = normalized_action.get("test_name")
            if test_name is not None:
                action["test_name"] = test_name
            return action
        if action_type == "medical_treatment_order":
            kind_hint = normalized_action.get("kind_hint")
            if kind_hint is not None:
                action["kind_hint"] = kind_hint
                family = _family_for_kind_hint(kind_hint)
                if family is not None:
                    action["family"] = family
            action["params"] = deepcopy(normalized_action.get("params"))
            return action
        return {
            "type": action_type,
            "action_type": action_type,
        }
    return None


def _family_for_kind_hint(kind_hint: Any) -> str | None:
    if not isinstance(kind_hint, str):
        return None
    registry = ActionRegistry()
    if not registry.is_known_kind_hint(kind_hint):
        return None
    return registry.get_definition(kind_hint).family


def _scenario_identifier(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return path.stem or None
    if isinstance(data, dict):
        for key in ("scenario_id", "case_id", "id", "name", "title"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
    return path.stem or None


def _format_list(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def _format_messages(messages: list[dict[str, Any]], *, indent: str) -> list[str]:
    if not messages:
        return [f"{indent}none"]
    lines = []
    for message in messages:
        speaker = message.get("speaker", "unknown")
        recipient = message.get("recipient") or "public"
        content = message.get("content", "")
        lines.append(f"{indent}{speaker} -> {recipient}: {content}")
    return lines


def _format_clinician_action(action: dict[str, Any] | None) -> str:
    if action is None:
        return "none"
    action_type = action.get("type") or action.get("action_type")
    parts = [f"type={action_type}"]
    family = action.get("family")
    if family is not None:
        parts.append(f"family={family}")
    kind_hint = action.get("kind_hint")
    if kind_hint is not None:
        parts.append(f"kind_hint={kind_hint}")
    if "params" in action:
        parts.append(f"params={_format_compact_value(action.get('params'))}")
    test_name = action.get("test_name")
    if test_name is not None:
        parts.append(f"test_name={test_name}")
    if parts:
        return " ".join(parts)
    normalized = action.get("normalized_action") or {}
    if action_type == "diagnostic_order":
        return f"diagnostic_order {normalized.get('test_name')}"
    if action_type == "medical_treatment_order":
        return f"medical_treatment_order {normalized.get('kind_hint')}"
    return str(action_type)


def _format_physiology_action(action: dict[str, Any] | None) -> str:
    if action is None:
        return "none"
    parts = []
    if action.get("kind_hint") is not None:
        parts.append(f"kind_hint={action.get('kind_hint')}")
    if "raw_text" in action:
        parts.append(f"raw_text={_format_compact_value(action.get('raw_text'))}")
    if "params" in action:
        parts.append(f"params={_format_compact_value(action.get('params'))}")
    return " ".join(parts) if parts else _format_compact_value(action)


def _format_diagnostic_orders(orders: list[dict[str, Any]]) -> list[str]:
    if not orders:
        return ["    none"]
    return [
        "    "
        f"{order.get('test_name')} "
        f"ordered_at_turn={order.get('ordered_at_turn')} "
        f"ready_at_turn={order.get('ready_at_turn')}"
        for order in orders
    ]


def _format_diagnostic_results(results: list[dict[str, Any]]) -> list[str]:
    if not results:
        return ["    none"]
    return [
        f"    {result.get('name')}: {result.get('result')}"
        for result in results
    ]


def _format_nurse_shadow(events: list[dict[str, Any]]) -> list[str]:
    if not events:
        return ["    none"]
    lines = []
    for event in events:
        action = event.get("action") or {}
        lines.append(
            "    "
            f"ordered_by={event.get('ordered_by')} "
            f"executed_by={event.get('executed_by')} "
            f"execution_mode={event.get('execution_mode')} "
            f"kind_hint={action.get('kind_hint')}"
        )
    return lines


def _format_nurse_bedside_slots(slots: list[dict[str, Any]]) -> list[str]:
    if not slots:
        return ["    none"]
    return [
        "    "
        f"triggered={slot.get('triggered')} "
        f"spoke={slot.get('spoke')}"
        for slot in slots
    ]


def _format_patient_state(patient_state: dict[str, Any]) -> str:
    vitals = _format_mapping(patient_state.get("vitals", {}))
    features = _format_mapping(patient_state.get("features", {}))
    flags = _format_mapping(patient_state.get("status_flags", {}))
    return f"vitals[{vitals}]; features[{features}]; status_flags[{flags}]"


def _format_mapping(value: Any) -> str:
    if not isinstance(value, dict):
        return "none"
    visible = {
        key: item
        for key, item in value.items()
        if item is not None
    }
    if not visible:
        return "none"
    return ", ".join(f"{key}={item}" for key, item in visible.items())


def _format_compact_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _format_validation_drops(drops: list[dict[str, Any]]) -> list[str]:
    if not drops:
        return ["    none"]
    lines = []
    for drop in drops:
        payload = drop.get("payload", {})
        errors = payload.get("errors") or []
        error_text = "; ".join(str(error) for error in errors)
        error_type = payload.get("error_type")
        label = f"{payload.get('agent')} {payload.get('item')}"
        if error_type:
            label = f"{label} {error_type}"
        lines.append(f"    {label}: {error_text}")
    return lines


def _format_events(events: list[dict[str, Any]]) -> list[str]:
    if not events:
        return ["    none"]
    lines = []
    for event in events:
        event_type = event.get("type")
        payload = event.get("payload", {})
        if event_type == "physiology_engine_call":
            lines.append(f"    {event_type} kind_hint={payload.get('kind_hint')}")
        elif event_type == "nurse_bedside_verbal_slot":
            lines.append(
                "    "
                f"{event_type} triggered={payload.get('triggered')} "
                f"spoke={payload.get('spoke')}"
            )
        elif event_type == "diagnostic_release":
            lines.append(
                "    "
                f"{event_type} test_name={payload.get('test_name')} "
                f"released_at_turn={payload.get('released_at_turn')}"
            )
        else:
            lines.append(f"    {event_type}")
    return lines


__all__ = [
    "DemoResult",
    "DemoTurn",
    "FAKE_MODE",
    "build_fake_demo_agents",
    "main",
    "render_demo_json",
    "render_readable_trajectory",
    "run_demo_scenario",
]
