"""Injected one-turn coordinator for v1.3.1 stub-agent runs."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ed_world_model.actions.registry import ActionRegistry, KindHint
from ed_world_model.actions.validator import ActionValidator, ValidationResult
from ed_world_model.adapters.noop_emotion import NoopEmotionEngine
from ed_world_model.agents.schemas import AgentProposal, VerbalAction
from ed_world_model.agents.stubs import SilentAgent
from ed_world_model.orchestration.observation_builder import ObservationBuilder
from ed_world_model.orchestration.orchestrator import (
    CLINICIAN,
    NURSE,
    Orchestrator,
    OrchestratorDecision,
)
from ed_world_model.state.global_state import Event, Message
from ed_world_model.state.state_manager import StateManager
from ed_world_model.state.termination import is_terminated, termination_reason


NO_ACTION_ENGINE_ACTION = {
    "raw_text": None,
    "kind_hint": KindHint.NO_ACTION,
    "params": {"elapsed_min": 1},
}


@dataclass(frozen=True)
class TurnResult:
    completed: bool
    terminated: bool
    active_agents: list[str] = field(default_factory=list)
    committed_messages: list[dict[str, Any]] = field(default_factory=list)
    validation_results: list[ValidationResult] = field(default_factory=list)
    physiology_called: bool = False
    physiology_action_kind_hint: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    released_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str | None = None


class TurnLoop:
    """Run one v1.3.1 turn with injected runtime components."""

    def __init__(
        self,
        *,
        state_manager: StateManager,
        physiology_adapter: Any,
        agents: Mapping[str, Any] | None = None,
        orchestrator: Orchestrator | None = None,
        observation_builder: ObservationBuilder | None = None,
        action_validator: ActionValidator | None = None,
        emotion_engine: Any | None = None,
    ) -> None:
        self.state_manager = state_manager
        self.orchestrator = orchestrator or Orchestrator()
        self.observation_builder = observation_builder or ObservationBuilder()
        self.action_validator = action_validator or ActionValidator(ActionRegistry())
        self.physiology_adapter = physiology_adapter
        self.emotion_engine = emotion_engine or NoopEmotionEngine()
        self.agents = dict(agents or {})
        self._default_silent_agent = SilentAgent()

    def run_turn(
        self,
        *,
        explicitly_selected_agents: set[str] | None = None,
    ) -> TurnResult:
        state = self.state_manager.state
        current_turn = state.runtime_state.turn_index
        released = self.state_manager.release_ready_diagnostic_results(
            current_turn=current_turn
        )

        if is_terminated(self.state_manager.state):
            return TurnResult(
                completed=False,
                terminated=True,
                released_diagnostics=[_dump(result) for result in released],
                events=_dump_current_events(self.state_manager),
                termination_reason=termination_reason(self.state_manager.state),
            )

        decision = self.orchestrator.select_active_agents(
            self.state_manager.state,
            explicitly_selected_agents=explicitly_selected_agents,
        )
        observations = self.observation_builder.build_for(
            decision.active_agents,
            self.state_manager.state,
        )
        proposals = self._generate_proposals(decision, observations)

        validation_results: list[ValidationResult] = []
        valid_action_type: str | None = None
        valid_action: dict[str, Any] | None = None
        clinician_proposal = proposals.get(CLINICIAN, AgentProposal())
        validation_result = self.action_validator.validate_clinician_proposal(
            clinician_proposal.model_dump(exclude_none=True),
            self.state_manager.state,
        )
        validation_results.append(validation_result)
        if validation_result.ok:
            valid_action_type = validation_result.action_type
            valid_action = validation_result.normalized_action
        else:
            self._record_validation_drop(
                agent=CLINICIAN,
                errors=validation_result.errors,
            )

        treatment_action: dict[str, Any] | None = None
        deferred_bedside_nurse_proposal: AgentProposal | None = None
        primary_active_agents = list(decision.active_agents)
        if valid_action_type == "medical_treatment_order" and valid_action is not None:
            treatment_action = valid_action
            nurse_proposal = proposals.get(NURSE)
            if nurse_proposal is not None and nurse_proposal.verbal_action is not None:
                deferred_bedside_nurse_proposal = nurse_proposal
                primary_active_agents = [
                    agent for agent in primary_active_agents if agent != NURSE
                ]

        committed_messages = self._commit_primary_verbal_actions(
            primary_active_agents,
            proposals,
        )

        if valid_action_type == "diagnostic_order" and valid_action is not None:
            self.state_manager.create_pending_diagnostic_result(
                valid_action["test_name"],
                current_turn=current_turn,
            )
        elif treatment_action is not None:
            self._record_nurse_shadow_execution(treatment_action)
            committed_messages.extend(
                self._run_nurse_bedside_verbal_slot(
                    treatment_action,
                    nurse_proposal=deferred_bedside_nurse_proposal,
                )
            )

        if committed_messages:
            emotion_output = self.emotion_engine.predict(
                self.state_manager.state.psych_state.patient_emotion,
                conversation_input=committed_messages,
                recent_messages=[
                    _dump(message)
                    for message in self.state_manager.state.runtime_state.messages
                ],
                patient_profile_context=_dump(
                    self.state_manager.state.agent_profiles.patient
                ),
            )
            self.state_manager.update_patient_emotion(
                emotion_output,
                record_event=True,
            )

        physiology_action = (
            treatment_action
            if treatment_action is not None
            else deepcopy(NO_ACTION_ENGINE_ACTION)
        )
        physiology_output = self.physiology_adapter.predict(
            self.state_manager.state,
            physiology_action,
        )
        self.state_manager.record_event(
            Event(
                type="physiology_engine_call",
                turn_index=current_turn,
                payload={"kind_hint": physiology_action.get("kind_hint")},
            )
        )
        self._apply_physiology_output(physiology_output)

        events = _dump_current_events(self.state_manager)
        self.state_manager.advance_turn()

        return TurnResult(
            completed=True,
            terminated=False,
            active_agents=list(decision.active_agents),
            committed_messages=committed_messages,
            validation_results=validation_results,
            physiology_called=True,
            physiology_action_kind_hint=physiology_action.get("kind_hint"),
            events=events,
            released_diagnostics=[_dump(result) for result in released],
            termination_reason=None,
        )

    def _generate_proposals(
        self,
        decision: OrchestratorDecision,
        observations: dict[str, dict[str, Any]],
    ) -> dict[str, AgentProposal]:
        proposals: dict[str, AgentProposal] = {}
        for agent_name in decision.active_agents:
            agent = self.agents.get(agent_name, self._default_silent_agent)
            try:
                proposal = AgentProposal.model_validate(
                    agent.generate(observations.get(agent_name, {}))
                )
            except ValidationError as exc:
                self._record_validation_drop(
                    agent=agent_name,
                    errors=[str(exc)],
                    item="agent_proposal",
                )
                proposal = AgentProposal()
            if agent_name != CLINICIAN and proposal.action is not None:
                self._record_validation_drop(
                    agent=agent_name,
                    errors=[f"{agent_name} proposals cannot include actions."],
                )
                proposal = AgentProposal(verbal_action=proposal.verbal_action)
            proposals[agent_name] = proposal
        return proposals

    def _commit_primary_verbal_actions(
        self,
        active_agents: list[str],
        proposals: dict[str, AgentProposal],
    ) -> list[dict[str, Any]]:
        committed_messages: list[dict[str, Any]] = []
        for agent_name in active_agents:
            verbal_action = proposals.get(agent_name, AgentProposal()).verbal_action
            if verbal_action is None:
                continue
            message = self._commit_verbal_action(verbal_action)
            committed_messages.append(_dump(message))
        return committed_messages

    def _commit_verbal_action(
        self,
        verbal_action: VerbalAction,
        *,
        default_recipient: str | None = None,
    ) -> Message:
        recipient = verbal_action.message_recipient or default_recipient
        message = self.state_manager.add_message(
            Message(
                speaker=verbal_action.speaker,
                recipient=recipient,
                content=verbal_action.content,
                turn_index=self.state_manager.state.runtime_state.turn_index,
            )
        )
        self._call_pending_question_hooks(verbal_action)
        return message

    def _call_pending_question_hooks(self, verbal_action: VerbalAction) -> None:
        target_agent = verbal_action.message_recipient
        if target_agent and verbal_action.requires_response:
            create_pending_question = getattr(
                self.state_manager,
                "create_pending_question",
                None,
            )
            if callable(create_pending_question):
                create_pending_question(
                    source_agent=verbal_action.speaker,
                    target_agent=target_agent,
                    question_text=verbal_action.content,
                )

        resolve_pending_questions = getattr(
            self.state_manager,
            "resolve_pending_questions_for_agent",
            None,
        )
        if callable(resolve_pending_questions):
            resolve_pending_questions(verbal_action.speaker)

    def _record_validation_drop(
        self,
        *,
        agent: str,
        errors: list[str],
        item: str = "action",
    ) -> None:
        self.state_manager.record_event(
            Event(
                type="validation_drop",
                turn_index=self.state_manager.state.runtime_state.turn_index,
                payload={"agent": agent, "item": item, "errors": list(errors)},
            )
        )

    def _record_nurse_shadow_execution(self, treatment_action: dict[str, Any]) -> None:
        self.state_manager.record_event(
            Event(
                type="nurse_shadow_execution",
                turn_index=self.state_manager.state.runtime_state.turn_index,
                payload={
                    "ordered_by": CLINICIAN,
                    "executed_by": NURSE,
                    "execution_mode": "shadow_execution",
                    "action": treatment_action,
                },
            )
        )

    def _run_nurse_bedside_verbal_slot(
        self,
        treatment_action: dict[str, Any],
        *,
        nurse_proposal: AgentProposal | None = None,
    ) -> list[dict[str, Any]]:
        nurse_agent = self.agents.get(NURSE)
        committed_messages: list[dict[str, Any]] = []
        proposal = nurse_proposal
        if proposal is None and nurse_agent is not None:
            observation = {
                "role": NURSE,
                "current_bedside_event_if_any": {
                    "type": "nurse_shadow_execution",
                    "action": treatment_action,
                },
            }
            try:
                proposal = AgentProposal.model_validate(
                    nurse_agent.generate(observation)
                )
            except ValidationError as exc:
                self._record_validation_drop(
                    agent=NURSE,
                    errors=[str(exc)],
                    item="agent_proposal",
                )
                proposal = AgentProposal()
        if proposal is not None:
            if proposal.action is not None:
                self._record_validation_drop(
                    agent=NURSE,
                    errors=["nurse bedside slot cannot include actions."],
                )
            if proposal.verbal_action is not None:
                message = self._commit_verbal_action(
                    proposal.verbal_action,
                    default_recipient="patient",
                )
                committed_messages.append(_dump(message))

        self.state_manager.record_event(
            Event(
                type="nurse_bedside_verbal_slot",
                turn_index=self.state_manager.state.runtime_state.turn_index,
                payload={
                    "triggered": True,
                    "spoke": bool(committed_messages),
                    "visible_to": "public",
                },
            )
        )
        return committed_messages

    def _apply_physiology_output(self, physiology_output: Mapping[str, Any]) -> None:
        update_kwargs = {
            key: physiology_output.get(key)
            for key in ("vitals", "features", "status_flags")
            if physiology_output.get(key) is not None
        }
        if not update_kwargs:
            self.state_manager.record_event(
                Event(
                    type="physiology_update_skipped",
                    turn_index=self.state_manager.state.runtime_state.turn_index,
                    payload={"reason": "empty_physiology_output"},
                )
            )
            return
        self.state_manager.apply_physiology_update(
            record_event=True,
            **update_kwargs,
        )


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _dump_current_events(state_manager: StateManager) -> list[dict[str, Any]]:
    return [
        _dump(event)
        for event in state_manager.state.runtime_state.current_turn_events
    ]


__all__ = ["NO_ACTION_ENGINE_ACTION", "TurnLoop", "TurnResult"]
