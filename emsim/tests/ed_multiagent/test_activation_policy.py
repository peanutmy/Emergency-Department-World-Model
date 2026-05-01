from __future__ import annotations

from pathlib import Path
import sys


EMSIM_ROOT = Path(__file__).resolve().parents[2]
if str(EMSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(EMSIM_ROOT))

from ed_multiagent.orchestrator import (
    AgentActivationPolicy,
    SimMode,
    should_call_clinician_agent,
    should_call_nurse_agent,
    should_call_patient_agent,
    should_call_relative_agent,
)


def test_patient_agent_skipped_in_code_when_unable_without_response_opportunity():
    policy = AgentActivationPolicy()

    should_call = policy.should_call_patient_agent(
        SimMode.CODE,
        speech_capacity="unable",
        response_opportunity=None,
    )

    assert should_call is False


def test_patient_agent_called_when_response_opportunity_present():
    assert (
        should_call_patient_agent(
            SimMode.CODE,
            speech_capacity="unable",
            response_opportunity=object(),
        )
        is True
    )


def test_patient_agent_called_in_stable_and_urgent_regardless_of_speech_capacity():
    assert (
        should_call_patient_agent(SimMode.STABLE, speech_capacity="unable")
        is True
    )
    assert (
        should_call_patient_agent(SimMode.URGENT, speech_capacity="unable")
        is True
    )


def test_relative_agent_is_throttled_in_code_mode():
    policy = AgentActivationPolicy()

    assert policy.should_call_relative_agent(SimMode.CODE, 5, 0) is False
    assert policy.should_call_relative_agent(SimMode.CODE, 29, 0) is False
    assert policy.should_call_relative_agent(SimMode.CODE, 30, 0) is True


def test_relative_agent_called_immediately_on_major_event():
    assert (
        should_call_relative_agent(
            SimMode.CODE,
            now_s=5,
            last_relative_act_s=0,
            major_event=True,
        )
        is True
    )


def test_relative_agent_not_throttled_in_non_code_modes():
    assert should_call_relative_agent(SimMode.STABLE, now_s=0, last_relative_act_s=0)
    assert should_call_relative_agent(SimMode.URGENT, now_s=0, last_relative_act_s=0)


def test_nurse_agent_called_for_report_clarification_or_warning():
    assert should_call_nurse_agent(SimMode.CODE, nurse_report_pending=True) is True
    assert should_call_nurse_agent(SimMode.CODE, needs_clarification=True) is True
    assert should_call_nurse_agent(SimMode.CODE, warning_pending=True) is True


def test_nurse_agent_can_be_skipped_when_no_verbalization_needed():
    policy = AgentActivationPolicy()

    assert policy.should_call_nurse_agent(SimMode.CODE) is False


def test_nurse_llm_policy_does_not_control_deterministic_task_execution():
    policy = AgentActivationPolicy()

    assert policy.should_call_nurse_agent(SimMode.CODE) is False
    assert (
        policy.should_call_nurse_agent(SimMode.CODE, nurse_report_pending=True)
        is True
    )


def test_clinician_agent_not_called_every_5_seconds_in_code():
    policy = AgentActivationPolicy()

    should_call = policy.should_call_clinician_agent(
        SimMode.CODE,
        now_s=5,
        last_clinician_act_s=0,
        pending_decision_event=False,
        rhythm_check_due=False,
        nurse_report_pending=False,
    )

    assert should_call is False


def test_clinician_agent_called_for_rhythm_check_due():
    assert (
        should_call_clinician_agent(
            SimMode.CODE,
            now_s=5,
            last_clinician_act_s=0,
            pending_decision_event=False,
            rhythm_check_due=True,
            nurse_report_pending=False,
        )
        is True
    )


def test_clinician_agent_called_for_pending_decision_event():
    assert (
        should_call_clinician_agent(
            SimMode.CODE,
            now_s=5,
            last_clinician_act_s=0,
            pending_decision_event=True,
            rhythm_check_due=False,
            nurse_report_pending=False,
        )
        is True
    )


def test_clinician_agent_called_for_nurse_report_pending():
    assert (
        should_call_clinician_agent(
            SimMode.CODE,
            now_s=5,
            last_clinician_act_s=0,
            pending_decision_event=False,
            rhythm_check_due=False,
            nurse_report_pending=True,
        )
        is True
    )


def test_clinician_agent_called_for_medication_window_due():
    assert (
        should_call_clinician_agent(
            SimMode.CODE,
            now_s=5,
            last_clinician_act_s=0,
            pending_decision_event=False,
            rhythm_check_due=False,
            nurse_report_pending=False,
            medication_window_due=True,
        )
        is True
    )


def test_clinician_agent_called_every_120_seconds_in_code_without_event():
    policy = AgentActivationPolicy()

    assert (
        policy.should_call_clinician_agent(
            SimMode.CODE,
            now_s=119,
            last_clinician_act_s=0,
            pending_decision_event=False,
            rhythm_check_due=False,
            nurse_report_pending=False,
        )
        is False
    )
    assert (
        policy.should_call_clinician_agent(
            SimMode.CODE,
            now_s=120,
            last_clinician_act_s=0,
            pending_decision_event=False,
            rhythm_check_due=False,
            nurse_report_pending=False,
        )
        is True
    )


def test_clinician_agent_called_every_30_seconds_in_urgent_without_event():
    policy = AgentActivationPolicy()

    assert (
        policy.should_call_clinician_agent(
            SimMode.URGENT,
            now_s=29,
            last_clinician_act_s=0,
            pending_decision_event=False,
            rhythm_check_due=False,
            nurse_report_pending=False,
        )
        is False
    )
    assert (
        policy.should_call_clinician_agent(
            SimMode.URGENT,
            now_s=30,
            last_clinician_act_s=0,
            pending_decision_event=False,
            rhythm_check_due=False,
            nurse_report_pending=False,
        )
        is True
    )


def test_clinician_agent_may_act_every_round_in_stable_mode():
    assert (
        should_call_clinician_agent(
            SimMode.STABLE,
            now_s=1,
            last_clinician_act_s=0,
            pending_decision_event=False,
            rhythm_check_due=False,
            nurse_report_pending=False,
        )
        is True
    )
