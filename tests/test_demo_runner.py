from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.demo as demo
from ed_world_model.orchestration.orchestrator import NURSE, PATIENT, RELATIVE
from ed_world_model.scenario_loader import ScenarioLoader


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "scenario_loader_minimal.json"
EXAMPLE_SCRIPT = ROOT / "examples" / "run_demo_scenario.py"


def _fixture_copy(tmp_path: Path) -> Path:
    path = tmp_path / "demo_scenario.json"
    path.write_text(FIXTURE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _low_o2_fixture_copy(tmp_path: Path) -> Path:
    path = tmp_path / "low_o2_demo_scenario.json"
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    payload["case_context"]["baseline_vitals"]["O2Sat"] = 88
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_demo_runner_loads_minimal_scenario_and_runs_without_api(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    scenario_path = _fixture_copy(tmp_path)
    before = scenario_path.read_text(encoding="utf-8")

    result = demo.run_demo_scenario(scenario_path, turns=3)

    assert scenario_path.read_text(encoding="utf-8") == before
    assert result.scenario_identifier == "demo_scenario"
    assert result.final_turn_index == 3
    assert len(result.turns) == 3
    assert result.turns[0].active_agents == ["clinician"]
    assert result.turns[0].physiology_action_kind_hint == "no_action"
    assert result.turns[2].physiology_action_kind_hint == "oxygen_support"
    assert result.final_patient_state["vitals"]["HR"] == 104.0
    assert result.final_patient_state["features"]["oxygen_device"] == "NRB"
    assert result.final_patient_state["features"]["FiO2"] == 1.0


def test_readable_output_contains_required_trajectory_fields(tmp_path: Path) -> None:
    result = demo.run_demo_scenario(_fixture_copy(tmp_path), turns=4)

    text = demo.render_readable_trajectory(result)

    assert "Turn 0" in text
    assert "turn_index: 0" in text
    assert "active_agents: clinician" in text
    assert "committed messages:" in text
    assert "clinician -> patient:" in text
    assert "clinician_action: diagnostic_order Initial ECG" in text
    assert "diagnostic_orders_created:" in text
    assert "diagnostic_results_released:" in text
    assert "The first diagnostic result is back and available." in text
    assert "nurse_shadow_execution:" in text
    assert "nurse_bedside_verbal_slots:" in text
    assert "spoke=True" in text
    assert "spoke=False" in text
    assert "physiology_action_kind_hint: no_action" in text
    assert "patient_state_after:" in text
    assert "validation_drops / parser_errors:" in text
    assert "events:" in text
    assert "diagnostic_order_created" in text


def test_json_output_is_valid_and_contains_expected_sections(tmp_path: Path) -> None:
    result = demo.run_demo_scenario(_fixture_copy(tmp_path), turns=4)

    payload = json.loads(demo.render_demo_json(result))

    assert payload["scenario_identifier"] == "demo_scenario"
    assert len(payload["turns"]) == 4
    assert payload["messages"]
    assert payload["events"]
    assert payload["turns"][2]["physiology_action_kind_hint"] == "oxygen_support"
    assert payload["turns"][2]["nurse_bedside_slots"] == [
        {"triggered": True, "spoke": True, "visible_to": "public"}
    ]
    assert payload["turns"][3]["nurse_bedside_slots"] == [
        {"triggered": True, "spoke": False, "visible_to": "public"}
    ]
    assert payload["final_patient_state"]["features"]["rhythm"] == "sinus tachycardia"
    assert payload["final_known_facts"]["available_results"] == [
        {"name": "Initial ECG", "result": "Sinus tachycardia without STEMI."}
    ]


def test_demo_runner_uses_scenario_loader(tmp_path: Path) -> None:
    class SpyLoader(ScenarioLoader):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, object]] = []

        def load(self, source, *, max_turns=None):
            self.calls.append({"source": source, "max_turns": max_turns})
            return super().load(source, max_turns=max_turns)

    loader = SpyLoader()
    scenario_path = _fixture_copy(tmp_path)

    demo.run_demo_scenario(scenario_path, turns=2, scenario_loader=loader)

    assert loader.calls == [{"source": scenario_path, "max_turns": 2}]


def test_demo_code_does_not_import_transition_engines() -> None:
    demo_source = inspect.getsource(demo)
    script_source = EXAMPLE_SCRIPT.read_text(encoding="utf-8")

    assert "transition_engines" not in demo_source
    assert "transition_engines" not in script_source


def test_diagnostic_release_and_treatment_events_appear_after_turnaround(
    tmp_path: Path,
) -> None:
    result = demo.run_demo_scenario(_fixture_copy(tmp_path), turns=3)

    assert result.turns[0].diagnostic_orders_created == [
        {
            "test_name": "Initial ECG",
            "ordered_at_turn": 0,
            "ready_at_turn": 1,
            "turnaround_turns": 1,
        }
    ]
    assert result.turns[1].diagnostic_results_released == [
        {"name": "Initial ECG", "result": "Sinus tachycardia without STEMI."}
    ]
    assert result.turns[2].clinician_action == {
        "action_type": "medical_treatment_order",
        "normalized_action": {
            "raw_text": None,
            "kind_hint": "oxygen_support",
            "params": {
                "oxygen_device": "NRB",
                "FiO2": 1.0,
                "PEEP_used": None,
                "PEEP_cmH2O": None,
            },
        },
    }
    assert result.turns[2].nurse_shadow_execution
    assert result.turns[2].physiology_action_kind_hint == "oxygen_support"


def test_demo_exercises_required_patient_response_and_nurse_result_reporting(
    tmp_path: Path,
) -> None:
    result = demo.run_demo_scenario(_fixture_copy(tmp_path), turns=3)

    assert any(
        event["type"] == "pending_question_created"
        for event in result.turns[0].events
    )
    assert result.turns[1].active_agents == ["clinician", "nurse", "patient"]
    assert [
        message["speaker"] for message in result.turns[1].messages
    ] == ["nurse", "patient"]
    assert "result is back" in result.turns[1].messages[0]["content"]
    assert "short of breath" in result.turns[1].messages[1]["content"]
    assert any(
        event["type"] == "pending_question_resolved"
        for event in result.turns[1].events
    )


def test_nurse_bedside_slot_reports_spoken_and_silent(tmp_path: Path) -> None:
    result = demo.run_demo_scenario(_fixture_copy(tmp_path), turns=4)

    assert result.turns[2].nurse_bedside_slots == [
        {"triggered": True, "spoke": True, "visible_to": "public"}
    ]
    assert result.turns[2].messages[-1] == {
        "speaker": "nurse",
        "content": "I am placing the oxygen mask now.",
        "turn_index": 2,
        "recipient": "patient",
    }
    assert result.turns[3].nurse_bedside_slots == [
        {"triggered": True, "spoke": False, "visible_to": "public"}
    ]


def test_fake_physiology_updates_no_action_and_oxygen_effects(
    tmp_path: Path,
) -> None:
    result = demo.run_demo_scenario(_low_o2_fixture_copy(tmp_path), turns=4)

    assert result.turns[0].patient_state_after["vitals"]["O2Sat"] == 87.0
    assert result.turns[1].patient_state_after["vitals"]["O2Sat"] == 86.0
    assert result.turns[2].patient_state_after["vitals"]["O2Sat"] == 91.0
    assert result.turns[3].patient_state_after["vitals"]["O2Sat"] == 95.0
    assert result.final_patient_state["features"]["oxygen_device"] == "NRB"
    assert result.final_patient_state["features"]["FiO2"] == 1.0


def test_patient_nurse_relative_demo_agents_are_verbal_only(tmp_path: Path) -> None:
    state = ScenarioLoader().load(_fixture_copy(tmp_path), max_turns=1)
    agents = demo.build_fake_demo_agents(state)

    for role in (NURSE, PATIENT, RELATIVE):
        proposal = agents[role].generate({})
        assert proposal.action is None


def test_cli_entrypoint_prints_readable_and_json(tmp_path: Path, capsys) -> None:
    scenario_path = _fixture_copy(tmp_path)

    assert demo.main(["--scenario", str(scenario_path), "--turns", "1"]) == 0
    readable = capsys.readouterr().out
    assert "Scenario: demo_scenario" in readable
    assert "Turn 0" in readable

    assert (
        demo.main(["--scenario", str(scenario_path), "--turns", "1", "--json"])
        == 0
    )
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["scenario_identifier"] == "demo_scenario"
    assert parsed["turns"][0]["active_agents"] == ["clinician"]
