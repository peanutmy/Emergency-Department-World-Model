from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

from ed_world_model.agents.llm_client import FakeLLMClient
from ed_world_model.agents.stubs import ScriptedPatientAgent, SilentAgent
from ed_world_model.facts.llm_extractor import LLMFactExtractor
from ed_world_model.orchestration.runner import IntegrationRunner
from ed_world_model.state.global_state import GlobalState
from ed_world_model.trajectory import TrajectoryLogger
import ed_world_model.trajectory.logger as logger_module


class FailingFactLLMClient:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("provider failure for sk-test-secret")


def _sample_trajectory() -> dict:
    return {
        "scenario_identifier": "case-123",
        "agent_mode": "fake",
        "physiology_mode": "fake",
        "fact_extractor_mode": "none",
        "requested_turns": 1,
        "final_turn_index": 1,
        "turns": [
            {
                "turn_index": 0,
                "turn_index_after": 1,
                "state_before": {
                    "patient_state": {
                        "vitals": {"HR": 110.0, "O2Sat": 88.0},
                        "features": {"oxygen_device": None},
                        "status_flags": {
                            "is_alive": True,
                            "can_speak": True,
                            "is_conscious": True,
                        },
                    },
                    "known_facts": {
                        "known_history": [],
                        "known_allergies": [],
                        "known_medications": [],
                        "known_symptoms": [
                            {
                                "name": "dyspnea",
                                "status": "present",
                                "onset": None,
                                "severity": None,
                                "source_texts": ["I feel short of breath."],
                            }
                        ],
                        "available_results": [],
                    },
                    "truth_state": {"hidden": "not for trajectory snapshots"},
                    "test_bank": [{"name": "ECG", "result": "hidden result"}],
                },
                "active_agents": ["clinician"],
                "messages": [
                    {
                        "speaker": "clinician",
                        "recipient": "patient",
                        "content": "I am starting oxygen.",
                        "turn_index": 0,
                    }
                ],
                "clinician_action": {
                    "type": "medical_treatment_order",
                    "action_type": "medical_treatment_order",
                    "family": "respiratory_support",
                    "kind_hint": "oxygen_support",
                    "params": {"oxygen_device": "NRB", "FiO2": 1.0},
                    "normalized_action": {
                        "raw_text": None,
                        "kind_hint": "oxygen_support",
                        "params": {"oxygen_device": "NRB", "FiO2": 1.0},
                    },
                },
                "diagnostic_orders_created": [
                    {
                        "test_name": "ECG",
                        "ordered_at_turn": 0,
                        "ready_at_turn": 1,
                    }
                ],
                "diagnostic_results_released": [
                    {"name": "ECG", "result": "Sinus tachycardia."}
                ],
                "nurse_shadow_execution": [
                    {
                        "ordered_by": "clinician",
                        "executed_by": "nurse",
                        "execution_mode": "shadow_execution",
                    }
                ],
                "nurse_bedside_slots": [
                    {"triggered": True, "spoke": True, "visible_to": "public"}
                ],
                "physiology_action": {
                    "kind_hint": "oxygen_support",
                    "raw_text": None,
                    "params": {"oxygen_device": "NRB", "FiO2": 1.0},
                },
                "state_after": {
                    "patient_state": {
                        "vitals": {"HR": 110.0, "O2Sat": 93.0},
                        "features": {"oxygen_device": "NRB", "FiO2": 1.0},
                        "status_flags": {
                            "is_alive": True,
                            "can_speak": True,
                            "is_conscious": True,
                        },
                    },
                    "known_facts": {
                        "known_history": [],
                        "known_allergies": [],
                        "known_medications": [],
                        "known_symptoms": [
                            {
                                "name": "dyspnea",
                                "status": "present",
                                "onset": None,
                                "severity": None,
                                "source_texts": ["I feel short of breath."],
                            }
                        ],
                        "available_results": [
                            {"name": "ECG", "result": "Sinus tachycardia."}
                        ],
                    },
                    "truth_state": {"hidden": "not for trajectory snapshots"},
                    "test_bank": [{"name": "ECG", "result": "hidden result"}],
                },
                "validation_drops": [
                    {
                        "type": "validation_drop",
                        "turn_index": 0,
                        "payload": {
                            "agent": "clinician",
                            "item": "action",
                            "errors": ["invalid duplicate order"],
                        },
                    }
                ],
                "parser_errors": [
                    {
                        "type": "validation_drop",
                        "turn_index": 0,
                        "payload": {
                            "agent": "patient",
                            "item": "agent_proposal",
                            "errors": ["not json"],
                            "error_type": "PatientParserError",
                        },
                    }
                ],
                "events": [
                    {
                        "type": "validation_drop",
                        "turn_index": 0,
                        "payload": {
                            "agent": "clinician",
                            "item": "action",
                            "errors": ["invalid duplicate order"],
                        },
                    }
                ],
            }
        ],
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_trajectory_logger_creates_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "trajectory"

    TrajectoryLogger(output_dir)

    assert output_dir.is_dir()


def test_save_json_writes_valid_json_with_required_turn_fields(tmp_path: Path) -> None:
    path = TrajectoryLogger(tmp_path).save_json(_sample_trajectory())

    payload = _read_json(path)
    turn = payload["turns"][0]
    assert payload["scenario_identifier"] == "case-123"
    assert payload["agent_mode"] == "fake"
    assert payload["physiology_mode"] == "fake"
    assert payload["fact_extractor_mode"] == "none"
    assert payload["turn_count"] == 1
    assert payload["final_turn_index"] == 1
    assert turn["state_before"]["patient_state"]["vitals"]["O2Sat"] == 88.0
    assert turn["state_after"]["patient_state"]["vitals"]["O2Sat"] == 93.0
    assert turn["state_before"]["known_facts"]["known_symptoms"] == [
        {
            "name": "dyspnea",
            "status": "present",
            "onset": None,
            "severity": None,
            "source_texts": ["I feel short of breath."],
        }
    ]
    assert turn["state_after"]["known_facts"]["available_results"] == [
        {"name": "ECG", "result": "Sinus tachycardia."}
    ]
    assert turn["committed_messages"][0]["speaker"] == "clinician"
    assert turn["clinician_action"]["params"]["FiO2"] == 1.0
    assert "normalized_action" not in turn["clinician_action"]
    assert turn["physiology_action"]["raw_text"] is None
    assert turn["physiology_action"]["params"]["oxygen_device"] == "NRB"
    assert turn["validation_drops"][0]["payload"]["errors"] == [
        "invalid duplicate order"
    ]
    assert turn["parser_errors"][0]["payload"]["error_type"] == "PatientParserError"


def test_trajectory_known_facts_do_not_add_raw_statement_stores(tmp_path: Path) -> None:
    path = TrajectoryLogger(tmp_path).save_json(_sample_trajectory())

    payload = _read_json(path)
    known_facts = payload["turns"][0]["state_before"]["known_facts"]
    assert "known_patient_statements" not in known_facts
    assert "known_relative_statements" not in known_facts
    assert "structured_symptoms" not in known_facts
    assert "structured_history" not in known_facts


def test_trajectory_preserves_fact_extraction_error_events(tmp_path: Path) -> None:
    trajectory = _sample_trajectory()
    trajectory["turns"][0]["events"].append(
        {
            "type": "fact_extraction_error",
            "turn_index": 0,
            "payload": {
                "speaker": "patient",
                "error_type": "RuntimeError",
                "error_message": "extractor failed",
            },
        }
    )

    path = TrajectoryLogger(tmp_path).save_json(trajectory)

    payload = _read_json(path)
    assert any(
        event["type"] == "fact_extraction_error"
        for event in payload["turns"][0]["events"]
    )


def test_trajectory_json_includes_known_facts_after_llm_fact_extractor(
    tmp_path: Path,
) -> None:
    patient_statement = "I feel short of breath."
    llm_client = FakeLLMClient(
        [
            {
                "symptoms": [
                    {
                        "name": "shortness of breath",
                        "status": "present",
                        "onset": None,
                        "severity": None,
                        "source_texts": [patient_statement],
                    }
                ]
            }
        ]
    )
    runner = IntegrationRunner(
        GlobalState(runtime_state={"required_response_agents": ["patient"]}),
        agents={
            "clinician": SilentAgent(),
            "patient": ScriptedPatientAgent([patient_statement]),
        },
        fact_extractor=LLMFactExtractor(llm_client),
    )

    turn = runner.run_turn()
    path = TrajectoryLogger(tmp_path).save_json([turn])

    payload = _read_json(path)
    assert payload["turns"][0]["state_after"]["known_facts"]["known_symptoms"] == [
        {
            "name": "shortness of breath",
            "status": "present",
            "onset": None,
            "severity": None,
            "source_texts": [patient_statement],
        }
    ]
    known_facts = payload["turns"][0]["state_after"]["known_facts"]
    assert "known_patient_statements" not in known_facts


def test_fact_extraction_error_redacts_api_key_in_events_and_trajectory_json(
    tmp_path: Path,
) -> None:
    runner = IntegrationRunner(
        GlobalState(runtime_state={"required_response_agents": ["patient"]}),
        agents={
            "clinician": SilentAgent(),
            "patient": ScriptedPatientAgent(["I feel short of breath."]),
        },
        fact_extractor=LLMFactExtractor(FailingFactLLMClient()),
    )

    turn = runner.run_turn()

    assert turn.completed is True
    error_events = [
        event for event in turn.events if event["type"] == "fact_extraction_error"
    ]
    assert error_events
    assert "sk-test-secret" not in str(error_events[0]["payload"])
    assert "[REDACTED_API_KEY]" in error_events[0]["payload"]["error_message"]

    path = TrajectoryLogger(tmp_path).save_json([turn])
    payload_text = path.read_text(encoding="utf-8")
    assert "sk-test-secret" not in payload_text
    assert "[REDACTED_API_KEY]" in payload_text


def test_save_summary_writes_valid_summary_json(tmp_path: Path) -> None:
    path = TrajectoryLogger(tmp_path).save_summary(
        {
            "scenario_identifier": "case-123",
            "agent_mode": "fake",
            "physiology_mode": "fake",
            "requested_turns": 3,
            "final_turn_index": 2,
            "turns_recorded": 2,
            "total_validation_drops": 1,
            "total_parser_errors": 1,
        }
    )

    payload = _read_json(path)
    assert payload["scenario_identifier"] == "case-123"
    assert payload["requested_turns"] == 3
    assert payload["turns_recorded"] == 2
    assert payload["total_validation_drops"] == 1
    assert payload["total_parser_errors"] == 1
    assert payload["output_timestamp"]


def test_save_all_writes_only_json_outputs_and_no_markdown(tmp_path: Path) -> None:
    paths = TrajectoryLogger(tmp_path).save_all(_sample_trajectory())

    assert paths == {
        "trajectory": tmp_path / "trajectory.json",
        "summary": tmp_path / "summary.json",
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "summary.json",
        "trajectory.json",
    ]
    assert not (tmp_path / "trajectory.md").exists()


def test_trajectory_snapshots_do_not_include_truth_state_or_test_bank(
    tmp_path: Path,
) -> None:
    TrajectoryLogger(tmp_path).save_all(_sample_trajectory())

    turn = _read_json(tmp_path / "trajectory.json")["turns"][0]
    assert "truth_state" not in turn["state_before"]
    assert "test_bank" not in turn["state_before"]
    assert "truth_state" not in turn["state_after"]
    assert "test_bank" not in turn["state_after"]


def test_trajectory_clinician_action_omits_normalized_action(tmp_path: Path) -> None:
    TrajectoryLogger(tmp_path).save_all(_sample_trajectory())

    turn = _read_json(tmp_path / "trajectory.json")["turns"][0]
    assert "normalized_action" not in turn["clinician_action"]


def test_trajectory_clinician_action_contains_params_for_treatment(
    tmp_path: Path,
) -> None:
    TrajectoryLogger(tmp_path).save_all(_sample_trajectory())

    turn = _read_json(tmp_path / "trajectory.json")["turns"][0]
    assert turn["clinician_action"] == {
        "type": "medical_treatment_order",
        "action_type": "medical_treatment_order",
        "family": "respiratory_support",
        "kind_hint": "oxygen_support",
        "params": {"oxygen_device": "NRB", "FiO2": 1.0},
    }


def test_trajectory_physiology_action_keeps_raw_text_kind_hint_and_params(
    tmp_path: Path,
) -> None:
    TrajectoryLogger(tmp_path).save_all(_sample_trajectory())

    turn = _read_json(tmp_path / "trajectory.json")["turns"][0]
    assert turn["physiology_action"] == {
        "kind_hint": "oxygen_support",
        "raw_text": None,
        "params": {"oxygen_device": "NRB", "FiO2": 1.0},
    }


def test_trajectory_diagnostic_order_clinician_action_contains_test_name(
    tmp_path: Path,
) -> None:
    trajectory = _sample_trajectory()
    trajectory["turns"][0]["clinician_action"] = {
        "type": "diagnostic_order",
        "action_type": "diagnostic_order",
        "test_name": "ECG",
        "normalized_action": {
            "type": "diagnostic_order",
            "test_name": "ECG",
        },
    }
    TrajectoryLogger(tmp_path).save_all(trajectory)

    turn = _read_json(tmp_path / "trajectory.json")["turns"][0]
    assert turn["clinician_action"] == {
        "type": "diagnostic_order",
        "action_type": "diagnostic_order",
        "test_name": "ECG",
    }


def test_logger_does_not_mutate_input_trajectory(tmp_path: Path) -> None:
    trajectory = _sample_trajectory()
    before = deepcopy(trajectory)

    TrajectoryLogger(tmp_path).save_all(trajectory)

    assert trajectory == before


def test_logger_redacts_api_key_metadata(tmp_path: Path) -> None:
    TrajectoryLogger(tmp_path).save_all(
        _sample_trajectory(),
        metadata={"OPENAI_API_KEY": "sk-test-secret"},
    )

    payload_text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    trajectory_text = (tmp_path / "trajectory.json").read_text(encoding="utf-8")
    assert "sk-test-secret" not in payload_text
    assert "sk-test-secret" not in trajectory_text


def test_logger_has_no_runtime_component_dependencies() -> None:
    source = inspect.getsource(logger_module)

    assert "StateManager" not in source
    assert "ActionValidator" not in source
    assert "transition_engines" not in source
