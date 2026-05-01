"""Deterministic no-LLM simulation orchestrator."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..actions import (
    ResponseMode,
    ResponseOpportunity,
    ResponseOpportunityQueue,
    apply_response_to_discovered_memory,
    status_for_response_mode,
)
from ..evaluation import RoundLogger
from ..memory import (
    ConversationMemory,
    DiscoveredClinicalMemory,
    GroundTruthMemory,
    PatientPrivateMemory,
    RelativePrivateMemory,
)
from ..observation import ObservationGateway
from ..world import ClinicalOrder, PendingEvent, WorkflowAdvanceResult, WorkflowEngine
from .activation_policy import AgentActivationPolicy
from .sim_mode import DECLARE_CODE, EXIT_CODE, SimMode, SimModeManager, get_round_dt_s


@dataclass
class VerbalAction:
    """Structured clinician speech action.

    ``is_question`` and ``expects_response`` are explicit flags. The
    orchestrator does not infer questions from punctuation or parse content.
    """

    speaker: str = "clinician"
    target: str | None = None
    content: str = ""
    is_question: bool = False
    expects_response: bool = False
    addressed_to: str | None = None
    question_type: str = "clarification"
    requiredness: str = "expected"
    sensitivity: str = "low"
    urgency: str = "routine"
    expected_slots: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.speaker = str(self.speaker)
        if self.target is not None:
            self.target = str(self.target)
        self.content = str(self.content)
        self.is_question = bool(self.is_question)
        self.expects_response = bool(self.expects_response)
        if self.addressed_to is not None:
            self.addressed_to = str(self.addressed_to)
        self.expected_slots = deepcopy(self.expected_slots)
        self.metadata = deepcopy(self.metadata)
        if (self.is_question or self.expects_response) and not (
            self.target or self.addressed_to
        ):
            raise ValueError(
                "verbal questions or response requests require target or addressed_to"
            )

    @property
    def response_target(self) -> str:
        response_target = self.addressed_to or self.target
        if not response_target:
            raise ValueError(
                "verbal questions or response requests require target or addressed_to"
            )
        return response_target


@dataclass
class ClinicianTurn:
    """Structured no-LLM clinician input for one simulation round."""

    verbal_action: VerbalAction | dict[str, Any] | None = None
    information_actions: list[dict[str, Any]] = field(default_factory=list)
    order_requests: list[ClinicalOrder | dict[str, Any]] = field(default_factory=list)
    meta_action: Any | None = None
    meta_actions: list[Any] = field(default_factory=list)
    wait_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.verbal_action = _coerce_verbal_action(self.verbal_action)
        self.information_actions = [
            deepcopy(action) for action in self.information_actions
        ]
        self.order_requests = [deepcopy(order) for order in self.order_requests]
        self.meta_actions = deepcopy(self.meta_actions)
        if self.wait_s is not None:
            self.wait_s = float(self.wait_s)
            if self.wait_s < 0:
                raise ValueError("wait_s must be non-negative")
        self.metadata = deepcopy(self.metadata)


@dataclass
class RoundResult:
    """Outputs from one deterministic no-LLM orchestrator round.

    ``agent_calls`` is an M5 placeholder and remains empty in the M4 no-LLM
    orchestrator.
    """

    now_s: float
    sim_mode: SimMode
    next_sim_mode: SimMode
    dt_s: float
    clinician_turn: ClinicianTurn
    submitted_orders: list[ClinicalOrder]
    workflow_result: WorkflowAdvanceResult
    applied_emsim_actions: list[dict[str, Any]]
    engine_result: Any
    fired_events: list[PendingEvent]
    observations: dict[str, Any]
    memory_deltas: dict[str, Any] = field(default_factory=dict)
    response_opportunities_created: list[ResponseOpportunity] = field(default_factory=list)
    response_opportunities_resolved: list[ResponseOpportunity] = field(default_factory=list)
    activation_recommendations: dict[str, bool] = field(default_factory=dict)
    agent_calls: list[str] = field(default_factory=list)
    log_entry: Any | None = None


class SimulationOrchestrator:
    """Deterministic ED simulation loop with no parser, agents, or LLM calls."""

    def __init__(
        self,
        engine_session: Any,
        workflow_engine: WorkflowEngine | None = None,
        *,
        conversation_memory: ConversationMemory | None = None,
        discovered_memory: DiscoveredClinicalMemory | None = None,
        ground_truth_memory: GroundTruthMemory | None = None,
        patient_private_memory: PatientPrivateMemory | None = None,
        relative_private_memory: RelativePrivateMemory | None = None,
        response_opportunities: ResponseOpportunityQueue | None = None,
        observation_gateway: ObservationGateway | None = None,
        sim_mode_manager: SimModeManager | None = None,
        activation_policy: AgentActivationPolicy | None = None,
        round_logger: Any | None = None,
        observation_roles: tuple[str, ...] = (
            "clinician",
            "nurse",
            "patient",
            "relative",
        ),
    ):
        self.engine_session = engine_session
        self.now_s = self._engine_now_s()
        self.workflow_engine = workflow_engine or WorkflowEngine(now_s=self.now_s)
        self._validate_workflow_time_alignment()

        self.conversation_memory = (
            conversation_memory
            or getattr(observation_gateway, "conversation_memory", None)
            or ConversationMemory()
        )
        self.discovered_memory = (
            discovered_memory
            or getattr(observation_gateway, "discovered_memory", None)
            or DiscoveredClinicalMemory()
        )
        self.ground_truth_memory = (
            ground_truth_memory
            or getattr(observation_gateway, "ground_truth_memory", None)
            or GroundTruthMemory()
        )
        self.patient_private_memory = (
            patient_private_memory
            or getattr(observation_gateway, "patient_private_memory", None)
            or PatientPrivateMemory()
        )
        self.relative_private_memory = (
            relative_private_memory
            or getattr(observation_gateway, "relative_private_memory", None)
            or RelativePrivateMemory()
        )
        self.response_opportunities = (
            response_opportunities
            or getattr(observation_gateway, "response_opportunities", None)
            or ResponseOpportunityQueue()
        )
        self.observation_gateway = observation_gateway or ObservationGateway(
            engine_session=self.engine_session,
            workflow_engine=self.workflow_engine,
            ground_truth_memory=self.ground_truth_memory,
            discovered_memory=self.discovered_memory,
            conversation_memory=self.conversation_memory,
            patient_private_memory=self.patient_private_memory,
            relative_private_memory=self.relative_private_memory,
            response_opportunities=self.response_opportunities,
        )
        self._sync_gateway()

        self.sim_mode_manager = sim_mode_manager or SimModeManager()
        self.activation_policy = activation_policy or AgentActivationPolicy()
        self.round_logger = round_logger or RoundLogger()
        self.observation_roles = tuple(observation_roles)
        self.workflow_events: list[PendingEvent] = []
        self.physiology_events: list[Any] = []
        self.applied_emsim_actions: list[dict[str, Any]] = []
        self._pending_resolutions: list[ResponseOpportunity] = []
        self._last_clinician_agent_activation_s: float | None = None
        self._last_relative_agent_activation_s: float | None = None

    def run_round(
        self,
        clinician_turn: ClinicianTurn | Mapping[str, Any] | None = None,
    ) -> RoundResult:
        """Run one structured clinician turn plus deterministic world advance."""

        turn = _coerce_clinician_turn(clinician_turn)
        self._sync_workflow_time()

        meta_actions = _turn_meta_actions(turn)
        resolved_opportunities = list(self._pending_resolutions)
        self._pending_resolutions.clear()
        memory_deltas: dict[str, Any] = {
            "conversation_turns_added": [],
            "information_actions": [],
            "test_results_added": {},
        }

        response_created = self._process_verbal_action(turn, memory_deltas)
        self._process_information_actions(turn, memory_deltas)

        submitted_orders = [
            self.workflow_engine.submit_order(order)
            for order in turn.order_requests
        ]

        sim_mode = self.sim_mode_manager.update(
            vitals=self.engine_session.observe_vitals(),
            clinician_meta_events=meta_actions,
        )
        dt_s = self._round_dt_s(turn, sim_mode)

        workflow_result = self.workflow_engine.advance(dt_s)
        applied_actions = self._apply_workflow_emsim_actions(workflow_result)
        engine_result = self.engine_session.advance(dt_s)

        self.now_s = self._engine_now_s()
        fired_events = list(workflow_result.events)
        self.workflow_events.extend(fired_events)
        self.physiology_events.extend(getattr(engine_result, "events", []))
        self._process_fired_events(fired_events, memory_deltas)

        next_sim_mode = self.sim_mode_manager.current_mode
        observations = self._build_observations()
        activation_recommendations = self._activation_recommendations(
            next_sim_mode,
            observations,
            fired_events,
            getattr(engine_result, "events", []),
        )

        result = RoundResult(
            now_s=self.now_s,
            sim_mode=sim_mode,
            next_sim_mode=next_sim_mode,
            dt_s=dt_s,
            clinician_turn=turn,
            submitted_orders=submitted_orders,
            workflow_result=workflow_result,
            applied_emsim_actions=applied_actions,
            engine_result=engine_result,
            fired_events=fired_events,
            observations=observations,
            memory_deltas=memory_deltas,
            response_opportunities_created=response_created,
            response_opportunities_resolved=resolved_opportunities,
            activation_recommendations=activation_recommendations,
            agent_calls=[],
        )
        result.log_entry = self.round_logger.log_round(result)
        return result

    def apply_hardcoded_response(
        self,
        opportunity_id: str,
        response_mode: ResponseMode | str,
        slots: dict[str, Any] | None = None,
        *,
        content: str | None = None,
        speaker: str | None = None,
        target: str = "clinician",
    ) -> ResponseOpportunity:
        """Resolve one response opportunity using structured deterministic slots."""

        opportunity = self.response_opportunities.opportunities[opportunity_id]
        apply_response_to_discovered_memory(
            opportunity,
            response_mode,
            slots,
            self.discovered_memory,
        )
        resolved = self.response_opportunities.mark_resolved(
            opportunity_id,
            status_for_response_mode(response_mode),
        )
        self._pending_resolutions.append(resolved)
        if content is not None:
            self.conversation_memory.add_turn(
                speaker=speaker or opportunity.addressed_to,
                target=target,
                content=content,
                time_s=self.now_s,
                metadata={
                    "response_opportunity_id": opportunity_id,
                    "response_mode": _enum_value(response_mode),
                },
            )
        return resolved

    def resolve_response_opportunity(
        self,
        opportunity_id: str,
        response_mode: ResponseMode | str,
        slots: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ResponseOpportunity:
        return self.apply_hardcoded_response(
            opportunity_id,
            response_mode,
            slots,
            **kwargs,
        )

    def report_current_nurse_observation(self) -> Any:
        """Return the structured nurse view; no natural-language report is built."""

        return self.observation_gateway.for_nurse()

    def _process_verbal_action(
        self,
        turn: ClinicianTurn,
        memory_deltas: dict[str, Any],
    ) -> list[ResponseOpportunity]:
        verbal = turn.verbal_action
        if verbal is None:
            return []

        conversation_turn = self.conversation_memory.add_turn(
            speaker=verbal.speaker,
            target=verbal.target,
            content=verbal.content,
            time_s=self.now_s,
            metadata=verbal.metadata,
        )
        memory_deltas["conversation_turns_added"].append(conversation_turn)

        if not (verbal.is_question or verbal.expects_response):
            return []

        opportunity = self.response_opportunities.add(
            ResponseOpportunity(
                asked_by=verbal.speaker,
                addressed_to=verbal.response_target,
                question_text=verbal.content,
                question_type=verbal.question_type,
                requiredness=verbal.requiredness,
                sensitivity=verbal.sensitivity,
                urgency=verbal.urgency,
                expected_slots=verbal.expected_slots,
                created_at_s=self.now_s,
                metadata=verbal.metadata,
            )
        )
        return [opportunity]

    def _process_information_actions(
        self,
        turn: ClinicianTurn,
        memory_deltas: dict[str, Any],
    ) -> None:
        for action in turn.information_actions:
            action_type = _action_type(action)
            if action_type in {"record_vitals", "observe_vitals", "measure_vitals"}:
                snapshot = {
                    "time_s": self.now_s,
                    "vitals": self.engine_session.observe_vitals(),
                }
                self.discovered_memory.add_vitals(snapshot)
                memory_deltas["information_actions"].append(
                    {"action_type": action_type, "vitals_added": snapshot}
                )
            elif action_type in {"structured_memory_update", "discover_facts"}:
                slots = deepcopy(action.get("slots", action.get("memory_update", {})))
                opportunity = ResponseOpportunity(expected_slots=[])
                apply_response_to_discovered_memory(
                    opportunity,
                    ResponseMode.DIRECT_ANSWER,
                    slots,
                    self.discovered_memory,
                )
                memory_deltas["information_actions"].append(
                    {"action_type": action_type, "slots_applied": slots}
                )
            else:
                memory_deltas["information_actions"].append(deepcopy(action))

    def _process_fired_events(
        self,
        fired_events: list[PendingEvent],
        memory_deltas: dict[str, Any],
    ) -> None:
        for event in fired_events:
            if event.event_type != "lab_result_ready":
                continue

            test_name = str(event.payload.get("test_name", ""))
            if not test_name:
                continue
            result_payload = deepcopy(event.payload.get("result", {}))
            self.discovered_memory.add_test_result(test_name, result_payload)
            memory_deltas["test_results_added"][test_name] = result_payload

            source_order_id = event.source_order_id
            if source_order_id is None:
                continue
            if self.workflow_engine.order_manager.orders.get(source_order_id) is None:
                continue
            # WorkflowEngine.advance owns lab order completion; do not update
            # order status again when the orchestrator consumes the payload.

    def _apply_workflow_emsim_actions(
        self,
        workflow_result: WorkflowAdvanceResult,
    ) -> list[dict[str, Any]]:
        applied: list[dict[str, Any]] = []
        for action in workflow_result.emsim_actions:
            action_copy = deepcopy(action)
            self.engine_session.apply_emsim_action(action_copy)
            applied.append(action_copy)
            self.applied_emsim_actions.append(deepcopy(action_copy))
        return applied

    def _build_observations(self) -> dict[str, Any]:
        self._sync_gateway()
        return {
            role: self.observation_gateway.for_role(role)
            for role in self.observation_roles
        }

    def _activation_recommendations(
        self,
        mode: SimMode,
        observations: dict[str, Any],
        workflow_events: list[PendingEvent],
        physiology_events: list[Any],
    ) -> dict[str, bool]:
        nurse_report_pending = any(
            getattr(event, "event_type", "") == "lab_result_ready"
            for event in workflow_events
        )
        major_event = bool(workflow_events or physiology_events)
        clinician = self.activation_policy.should_call_clinician_agent(
            mode,
            now_s=self.now_s,
            last_clinician_act_s=self._last_clinician_agent_activation_s,
            pending_decision_event=False,
            rhythm_check_due=False,
            nurse_report_pending=nurse_report_pending,
        )
        if clinician:
            self._last_clinician_agent_activation_s = self.now_s

        nurse = self.activation_policy.should_call_nurse_agent(
            mode,
            event=workflow_events[0] if workflow_events else None,
            nurse_report_pending=nurse_report_pending,
            major_clinical_change=bool(physiology_events),
        )
        patient_observation = observations.get("patient")
        subjective_state = getattr(patient_observation, "subjective_state", None)
        patient = self.activation_policy.should_call_patient_agent(
            mode,
            speech_capacity=subjective_state,
            has_response_opportunity=bool(
                self.response_opportunities.list_open("patient")
            ),
        )
        relative = self.activation_policy.should_call_relative_agent(
            mode,
            now_s=self.now_s,
            last_relative_act_s=self._last_relative_agent_activation_s,
            major_event=major_event,
        )
        if relative:
            self._last_relative_agent_activation_s = self.now_s

        return {
            "clinician_agent": clinician,
            "nurse_agent": nurse,
            "patient_agent": patient,
            "relative_agent": relative,
        }

    def _round_dt_s(self, turn: ClinicianTurn, sim_mode: SimMode) -> float:
        if turn.wait_s is not None:
            return float(turn.wait_s)
        return float(get_round_dt_s(sim_mode))

    def _sync_gateway(self) -> None:
        self.observation_gateway.engine_session = self.engine_session
        self.observation_gateway.workflow_engine = self.workflow_engine
        self.observation_gateway.ground_truth_memory = self.ground_truth_memory
        self.observation_gateway.discovered_memory = self.discovered_memory
        self.observation_gateway.conversation_memory = self.conversation_memory
        self.observation_gateway.patient_private_memory = self.patient_private_memory
        self.observation_gateway.relative_private_memory = self.relative_private_memory
        self.observation_gateway.response_opportunities = self.response_opportunities

    def _sync_workflow_time(self) -> None:
        self._validate_workflow_time_alignment()

    def _validate_workflow_time_alignment(self) -> None:
        workflow_now_s = float(getattr(self.workflow_engine, "now_s", self.now_s))
        if abs(workflow_now_s - self.now_s) > 1e-6:
            raise ValueError(
                "workflow_engine.now_s must match EngineSession time at "
                "SimulationOrchestrator construction/run_round; got "
                f"workflow_engine.now_s={workflow_now_s}, engine now_s={self.now_s}"
            )

    def _engine_now_s(self) -> float:
        observation = self.engine_session.observe()
        if isinstance(observation, Mapping) and "time_s" in observation:
            return float(observation["time_s"])
        return float(getattr(self.engine_session, "now_s", 0.0))


def _coerce_clinician_turn(
    clinician_turn: ClinicianTurn | Mapping[str, Any] | None,
) -> ClinicianTurn:
    if clinician_turn is None:
        return ClinicianTurn()
    if isinstance(clinician_turn, ClinicianTurn):
        return clinician_turn
    if isinstance(clinician_turn, Mapping):
        return ClinicianTurn(**deepcopy(dict(clinician_turn)))
    raise TypeError("clinician_turn must be a ClinicianTurn, mapping, or None")


def _coerce_verbal_action(
    verbal_action: VerbalAction | Mapping[str, Any] | None,
) -> VerbalAction | None:
    if verbal_action is None:
        return None
    if isinstance(verbal_action, VerbalAction):
        return verbal_action
    if isinstance(verbal_action, Mapping):
        return VerbalAction(**deepcopy(dict(verbal_action)))
    raise TypeError("verbal_action must be a VerbalAction, mapping, or None")


def _turn_meta_actions(turn: ClinicianTurn) -> list[Any]:
    actions: list[Any] = []
    if turn.meta_action is not None:
        if isinstance(turn.meta_action, list):
            actions.extend(turn.meta_action)
        else:
            actions.append(turn.meta_action)
    actions.extend(turn.meta_actions)
    return actions


def _action_type(action: Mapping[str, Any]) -> str:
    for key in ("action_type", "type", "kind", "name"):
        if key in action:
            return str(action[key]).strip().lower()
    return ""


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


__all__ = [
    "ClinicianTurn",
    "RoundResult",
    "SimulationOrchestrator",
    "VerbalAction",
    "DECLARE_CODE",
    "EXIT_CODE",
]
