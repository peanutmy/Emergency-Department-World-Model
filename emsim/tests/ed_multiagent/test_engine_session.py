from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


EMSIM_ROOT = Path(__file__).resolve().parents[2]
if str(EMSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(EMSIM_ROOT))

from ed_multiagent.emsim_adapter import EngineAdvanceResult, EngineSession


FORBIDDEN_OBSERVATION_KEYS = {
    "rhythm",
    "pathology",
    "mechanism",
    "interventions",
    "hidden_state",
    "active_drug_effects",
}


def _load_case(relative_path: str) -> dict:
    with (EMSIM_ROOT / relative_path).open(encoding="utf-8") as f:
        return json.load(f)


def _base_state() -> dict:
    return {
        "vitals": {
            "HR": 120,
            "BP_sys": 92,
            "BP_dia": 55,
            "RR": 28,
            "O2Sat": 88,
            "T": 37.0,
        },
        "interventions": {
            "airway": False,
            "O2_device": None,
            "PEEP": 0,
            "FiO2": 0.21,
            "vent_rate": None,
            "vent_TV_ml": None,
            "intubated": False,
            "CPR_active": False,
            "defib_last_J": None,
            "pacing_active": False,
            "pacing_rate": None,
            "fluids_rate_ml_hr": 0,
            "fluid_type": None,
            "warming_active": False,
            "cooling_active": False,
            "needle_decompression": False,
            "chest_tube": False,
            "pericardiocentesis": False,
        },
        "mechanism": {
            "pathology": {
                "name": "septic_shock",
                "severity": "severe",
            }
        },
    }


def _arrest_prone_brady_state() -> dict:
    state = _base_state()
    state["vitals"].update({"HR": 20, "BP_sys": 40, "BP_dia": 20})
    state["mechanism"]["pathology"] = {
        "name": "bradycardia",
        "severity": "severe",
    }
    return state


def _vf_arrest_state() -> dict:
    state = _base_state()
    state["vitals"].update(
        {
            "HR": 0,
            "BP_sys": 0,
            "BP_dia": 0,
            "RR": 0,
            "O2Sat": 70,
        }
    )
    state["interventions"]["CPR_active"] = True
    state["mechanism"]["pathology"] = {
        "name": "vf_arrest",
        "severity": "severe",
    }
    return state


def _forbidden_keys_in(value) -> set[str]:
    if isinstance(value, dict):
        found = set(value) & FORBIDDEN_OBSERVATION_KEYS
        for nested in value.values():
            found.update(_forbidden_keys_in(nested))
        return found
    if isinstance(value, (list, tuple)):
        found: set[str] = set()
        for nested in value:
            found.update(_forbidden_keys_in(nested))
        return found
    return set()


def test_constructs_from_scenario_initial_state_and_observes_vitals_only():
    case = _load_case("transitions/Respiratory/Nightmares Case 2 Pneumonia.json")
    session = EngineSession(case["initial_state"])

    observation = session.observe(agent_id="clinician")

    assert observation == {
        "time_s": 0.0,
        "vitals": case["initial_state"]["vitals"],
    }
    assert "mechanism" not in observation
    assert "interventions" not in observation
    assert "hidden_state" not in observation
    assert "active_drug_effects" not in observation


def test_repeated_advance_persists_state_and_time():
    session = EngineSession(_base_state())
    initial = session.observe_vitals()

    first = session.advance(30)
    second = session.advance(30)

    assert isinstance(first, EngineAdvanceResult)
    assert session.now_s == 60.0
    assert second.state == session.current_state
    assert second.state["vitals"] != initial


def test_can_run_thirty_physiology_only_rounds_and_record_vitals_drift():
    case = _load_case("transitions/Endocrine/Adrenal Crisis.json")
    session = EngineSession(case["initial_state"])

    vitals_log = [session.observe_vitals()]
    for _ in range(30):
        result = session.advance(30)
        vitals_log.append(result.state["vitals"])

    assert session.now_s == 900.0
    assert len(vitals_log) == 31
    assert all(set(vitals) == {"HR", "BP_sys", "BP_dia", "RR", "O2Sat", "T"} for vitals in vitals_log)
    assert len({tuple(vitals.items()) for vitals in vitals_log}) > 1


def test_advance_zero_returns_no_events():
    session = EngineSession(_base_state())

    result = session.advance(0)

    assert result.events == []
    assert session.now_s == 0.0


def test_advance_negative_raises_value_error():
    session = EngineSession(_base_state())

    with pytest.raises(ValueError):
        session.advance(-1)


def test_arrest_event_fires_when_advancing_into_arrest_state():
    session = EngineSession(_arrest_prone_brady_state())

    result = session.advance(600)
    kinds = [event.kind for event in result.events]

    assert "arrest" in kinds
    assert result.state["mechanism"]["rhythm"] == "asystole"
    assert all(kind in {"arrest", "rosc", "rhythm_change"} for kind in kinds)


def test_rhythm_change_event_fires_when_advancing_to_distinct_rhythm():
    session = EngineSession(_arrest_prone_brady_state())

    result = session.advance(600)
    rhythm_events = [
        event for event in result.events if event.kind == "rhythm_change"
    ]

    assert rhythm_events
    assert rhythm_events[0].payload == {
        "from": "bradycardia",
        "to": "asystole",
    }


def test_rosc_event_fires_when_action_leaves_arrest_state():
    session = EngineSession(_vf_arrest_state())

    session.apply_emsim_action(
        {
            "type": "intervention",
            "name": "defibrillate",
            "value": 200,
        }
    )
    result = session.advance(1)
    kinds = [event.kind for event in result.events]

    assert "rosc" in kinds
    assert "rhythm_change" in kinds
    assert result.state["mechanism"]["rhythm"] == "sinus"


def test_non_arrest_rhythm_change_event_fires_when_action_changes_rhythm():
    state = _base_state()
    state["vitals"].update({"HR": 40, "BP_sys": 78, "BP_dia": 42})
    state["mechanism"]["pathology"] = {
        "name": "bradycardia",
        "severity": "severe",
    }
    session = EngineSession(state)

    session.apply_emsim_action(
        {
            "type": "intervention",
            "name": "start_pacing",
            "value": 80,
        }
    )
    result = session.advance(1)

    assert [event.kind for event in result.events] == ["rhythm_change"]
    assert result.events[0].payload == {"from": "bradycardia", "to": "sinus"}


def test_apply_nrb_uses_emsim_intervention_rules():
    session = EngineSession(_base_state())
    before = session.observe_vitals()

    session.apply_emsim_action({"type": "intervention", "name": "apply_NRB"})

    after_state = session.current_state
    after = after_state["vitals"]
    assert after_state["interventions"]["O2_device"] == "NRB"
    assert after_state["interventions"]["FiO2"] == pytest.approx(1.0)
    assert after["O2Sat"] > before["O2Sat"]


def test_observation_surfaces_omit_forbidden_keys_at_any_depth():
    session = EngineSession(_base_state())
    session.apply_emsim_action({"type": "intervention", "name": "apply_NRB"})
    session.advance(30)

    assert _forbidden_keys_in(session.observe(agent_id="clinician")) == set()
    assert _forbidden_keys_in(session.observe_vitals()) == set()


def test_mutating_observed_vitals_does_not_change_session_state():
    session = EngineSession(_base_state())
    observation = session.observe()

    observation["vitals"]["HR"] = 999

    assert session.observe_vitals()["HR"] != 999


def test_mutating_current_state_does_not_change_session_state():
    session = EngineSession(_base_state())
    exposed_state = session.current_state

    exposed_state["vitals"]["HR"] = 999
    exposed_state["mechanism"]["pathology"]["name"] = "mutated"
    exposed_state["interventions"]["FiO2"] = 1.0

    current_state = session.current_state
    assert current_state["vitals"]["HR"] != 999
    assert current_state["mechanism"]["pathology"]["name"] != "mutated"
    assert current_state["interventions"]["FiO2"] != 1.0


def test_mutating_advance_result_state_does_not_change_session_state():
    session = EngineSession(_base_state())
    result = session.advance(30)

    result.state["vitals"]["HR"] = 999
    result.state["mechanism"]["pathology"]["name"] = "mutated"

    current_state = session.current_state
    assert current_state["vitals"]["HR"] != 999
    assert current_state["mechanism"]["pathology"]["name"] != "mutated"


def test_give_fluids_persists_as_continuous_emsim_intervention():
    session = EngineSession(_base_state())
    before = session.current_state

    session.apply_emsim_action(
        {
            "type": "intervention",
            "name": "give_fluids",
            "value": 1000,
        }
    )
    after_action = session.current_state
    session.advance(300)
    after_advance = session.current_state

    assert after_action["interventions"]["fluids_rate_ml_hr"] == 1000
    assert after_action["interventions"]["fluid_type"] == "crystalloid"
    assert after_advance["vitals"]["BP_sys"] >= before["vitals"]["BP_sys"]


def test_drug_action_persists_across_advances_without_exposing_hidden_state():
    session = EngineSession(_base_state())

    session.apply_emsim_action(
        {
            "type": "drug",
            "name": "epinephrine",
            "dose": 1,
            "unit": "mg",
            "route": "IV",
        }
    )
    after_action_observation = session.observe("clinician")
    session.advance(30)
    mid = session.observe_vitals()
    session.advance(30)
    late = session.observe_vitals()

    assert set(after_action_observation) == {"time_s", "vitals"}
    assert "hidden_state" not in after_action_observation
    assert "active_drug_effects" not in after_action_observation
    assert late["BP_sys"] > mid["BP_sys"]
