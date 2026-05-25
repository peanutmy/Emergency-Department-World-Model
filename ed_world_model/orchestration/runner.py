"""Integration runner for deterministic v1.3.1 trajectories."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
from typing import Any

from ed_world_model.actions.registry import ActionRegistry
from ed_world_model.actions.validator import ActionValidator
from ed_world_model.adapters.noop_emotion import NoopEmotionEngine
from ed_world_model.agents.stubs import (
    ScriptedClinicianAgent,
    ScriptedNurseAgent,
    ScriptedPatientAgent,
    ScriptedProposal,
    ScriptedRelativeAgent,
    SilentAgent,
)
from ed_world_model.orchestration.observation_builder import ObservationBuilder
from ed_world_model.orchestration.orchestrator import (
    CLINICIAN,
    NURSE,
    PATIENT,
    RELATIVE,
    Orchestrator,
)
from ed_world_model.orchestration.turn_loop import TurnLoop, TurnResult
from ed_world_model.scenario_loader import ScenarioLoader
from ed_world_model.state.global_state import GlobalState
from ed_world_model.state.state_manager import StateManager


StateSource = GlobalState | str | Path | Mapping[str, Any] | None
PhysiologyOutputFactory = Callable[[GlobalState, Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class TrajectoryTurn:
    """Compact debug record for one attempted turn."""

    turn_index_before: int
    turn_index_after: int
    completed: bool
    terminated: bool
    active_agents: list[str] = field(default_factory=list)
    committed_messages: list[dict[str, Any]] = field(default_factory=list)
    validation_results: list[dict[str, Any]] = field(default_factory=list)
    physiology_called: bool = False
    physiology_action_kind_hint: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    released_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecordingStubPhysiologyAdapter:
    """Deterministic physiology adapter for tests and demos.

    The default output is a no-op vitals update. Calls are recorded with the
    released diagnostic results visible through ``known_facts`` only.
    """

    def __init__(
        self,
        output: Mapping[str, Any] | PhysiologyOutputFactory | None = None,
    ) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    def predict(
        self,
        global_state: GlobalState,
        engine_facing_action: Mapping[str, Any],
    ) -> dict[str, Any]:
        action = deepcopy(dict(engine_facing_action))
        self.calls.append(
            {
                "turn_index": global_state.runtime_state.turn_index,
                "action": action,
                "known_results": [
                    result.model_dump()
                    for result in global_state.known_facts.available_results
                ],
            }
        )
        if callable(self.output):
            return dict(self.output(global_state, action))
        if self.output is not None:
            return deepcopy(dict(self.output))
        return {"vitals": global_state.patient_state.vitals.model_dump()}


_DEFAULT_LLM_RESPONSES_BY_ROLE = {
    CLINICIAN: {"verbal_action": None, "action": None},
    NURSE: {"verbal_action": None},
    PATIENT: {"verbal_action": None},
    RELATIVE: {"verbal_action": None},
}


class ScriptedLLMCallable:
    """Callable fake LLM that returns scripted strict-JSON responses."""

    def __init__(
        self,
        responses: Iterable[str | Mapping[str, Any]],
        *,
        role: str = CLINICIAN,
        default_response: str | Mapping[str, Any] | None = None,
    ) -> None:
        if role not in _DEFAULT_LLM_RESPONSES_BY_ROLE:
            raise ValueError(f"Unsupported scripted LLM role: {role!r}.")
        self.role = role
        self._responses = list(responses)
        self._default_response = (
            _DEFAULT_LLM_RESPONSES_BY_ROLE[role]
            if default_response is None
            else default_response
        )
        self.prompts: list[str] = []
        self._index = 0

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._index >= len(self._responses):
            response = self._default_response
            if isinstance(response, str):
                return response
            return json.dumps(response)
        response = self._responses[self._index]
        self._index += 1
        if isinstance(response, str):
            return response
        return json.dumps(response)


class IntegrationRunner:
    """Run a multi-turn trajectory with injected deterministic components.

    A provided ``GlobalState`` is shared with the runner and mutated through
    ``StateManager`` during turn execution.
    """

    def __init__(
        self,
        source: StateSource = None,
        *,
        agents: Mapping[str, Any] | None = None,
        physiology_adapter: Any | None = None,
        emotion_engine: Any | None = None,
        orchestrator: Orchestrator | None = None,
        observation_builder: ObservationBuilder | None = None,
        action_validator: ActionValidator | None = None,
        scenario_loader: ScenarioLoader | None = None,
        max_turns: int | None = None,
    ) -> None:
        state = _load_state(
            source,
            scenario_loader=scenario_loader,
            max_turns=max_turns,
        )
        self.state_manager = StateManager(state)
        self.physiology_adapter = (
            physiology_adapter
            if physiology_adapter is not None
            else RecordingStubPhysiologyAdapter()
        )
        self.emotion_engine = emotion_engine or NoopEmotionEngine()
        self.agents = dict(agents or default_scripted_agents())
        self.turn_loop = TurnLoop(
            state_manager=self.state_manager,
            physiology_adapter=self.physiology_adapter,
            agents=self.agents,
            orchestrator=orchestrator or Orchestrator(),
            observation_builder=observation_builder or ObservationBuilder(),
            action_validator=action_validator or ActionValidator(ActionRegistry()),
            emotion_engine=self.emotion_engine,
        )

    @property
    def state(self) -> GlobalState:
        return self.state_manager.state

    def run_turn(
        self,
        *,
        explicitly_selected_agents: set[str] | None = None,
    ) -> TrajectoryTurn:
        turn_index_before = self.state.runtime_state.turn_index
        result = self.turn_loop.run_turn(
            explicitly_selected_agents=explicitly_selected_agents
        )
        return _trajectory_turn(
            result,
            turn_index_before=turn_index_before,
            turn_index_after=self.state.runtime_state.turn_index,
        )

    def run(
        self,
        turns: int,
        *,
        explicitly_selected_agents_by_turn: Sequence[set[str] | None] | None = None,
    ) -> list[TrajectoryTurn]:
        if turns < 0:
            raise ValueError("turns must be non-negative.")

        trajectory: list[TrajectoryTurn] = []
        for index in range(turns):
            explicit_agents = None
            if explicitly_selected_agents_by_turn is not None:
                if index < len(explicitly_selected_agents_by_turn):
                    explicit_agents = explicitly_selected_agents_by_turn[index]
            turn = self.run_turn(explicitly_selected_agents=explicit_agents)
            trajectory.append(turn)
            if turn.terminated:
                break
        return trajectory


def run_trajectory(
    source: StateSource = None,
    *,
    turns: int = 1,
    agents: Mapping[str, Any] | None = None,
    physiology_adapter: Any | None = None,
    emotion_engine: Any | None = None,
    max_turns: int | None = None,
    explicitly_selected_agents_by_turn: Sequence[set[str] | None] | None = None,
) -> list[TrajectoryTurn]:
    runner = IntegrationRunner(
        source,
        agents=agents,
        physiology_adapter=physiology_adapter,
        emotion_engine=emotion_engine,
        max_turns=max_turns,
    )
    return runner.run(
        turns,
        explicitly_selected_agents_by_turn=explicitly_selected_agents_by_turn,
    )


def default_scripted_agents() -> dict[str, Any]:
    return {
        CLINICIAN: SilentAgent(),
        NURSE: SilentAgent(),
        PATIENT: SilentAgent(),
        RELATIVE: SilentAgent(),
    }


def scripted_agents(
    *,
    clinician: Iterable[ScriptedProposal] | None = None,
    nurse: Iterable[ScriptedProposal] | None = None,
    patient: Iterable[ScriptedProposal] | None = None,
    relative: Iterable[ScriptedProposal] | None = None,
) -> dict[str, Any]:
    return {
        CLINICIAN: ScriptedClinicianAgent(clinician or []),
        NURSE: ScriptedNurseAgent(nurse or []),
        PATIENT: ScriptedPatientAgent(patient or []),
        RELATIVE: ScriptedRelativeAgent(relative or []),
    }


def _load_state(
    source: StateSource,
    *,
    scenario_loader: ScenarioLoader | None,
    max_turns: int | None,
) -> GlobalState:
    if source is None:
        state = GlobalState()
        if max_turns is not None:
            state = state.model_copy(deep=True)
            state.runtime_state.max_turns = max_turns
        return state
    if isinstance(source, GlobalState):
        if max_turns is not None:
            raise ValueError(
                "max_turns is only applied when loading a scenario path or "
                "default state; set it on the provided GlobalState instead."
            )
        return source
    loader = scenario_loader or ScenarioLoader()
    return loader.load(source, max_turns=max_turns)


def _trajectory_turn(
    result: TurnResult,
    *,
    turn_index_before: int,
    turn_index_after: int,
) -> TrajectoryTurn:
    return TrajectoryTurn(
        turn_index_before=turn_index_before,
        turn_index_after=turn_index_after,
        completed=result.completed,
        terminated=result.terminated,
        active_agents=list(result.active_agents),
        committed_messages=deepcopy(result.committed_messages),
        validation_results=[_dump(item) for item in result.validation_results],
        physiology_called=result.physiology_called,
        physiology_action_kind_hint=result.physiology_action_kind_hint,
        events=deepcopy(result.events),
        released_diagnostics=deepcopy(result.released_diagnostics),
        termination_reason=result.termination_reason,
    )


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    return deepcopy(value)


__all__ = [
    "IntegrationRunner",
    "RecordingStubPhysiologyAdapter",
    "ScriptedLLMCallable",
    "TrajectoryTurn",
    "default_scripted_agents",
    "run_trajectory",
    "scripted_agents",
]
