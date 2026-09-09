from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ed_world_model.state.global_state import GlobalState, KnownFacts
from ed_world_model.state.state_manager import StateManager


def test_known_facts_has_optional_chief_complaint() -> None:
    facts = KnownFacts(chief_complaint="shortness of breath")

    assert KnownFacts().chief_complaint is None
    assert facts.chief_complaint == "shortness of breath"


def test_state_manager_releases_chief_complaint_without_reading_hidden_truth() -> None:
    state = GlobalState(
        truth_state={
            "patient_internal_state": {
                "chief_complaint": "hidden shortness of breath",
                "hidden_history": ["diabetes"],
            }
        }
    )
    truth_before = state.truth_state.model_dump()
    manager = StateManager(state)

    manager.update_known_facts(chief_complaint="shortness of breath")

    assert manager.state.known_facts.chief_complaint == "shortness of breath"
    assert manager.state.known_facts.known_history == []
    assert manager.state.truth_state.model_dump() == truth_before
