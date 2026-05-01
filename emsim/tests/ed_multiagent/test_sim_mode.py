from __future__ import annotations

from pathlib import Path
import sys

import pytest


EMSIM_ROOT = Path(__file__).resolve().parents[2]
if str(EMSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(EMSIM_ROOT))

from ed_multiagent.orchestrator import (
    DECLARE_CODE,
    EXIT_CODE,
    SimMode,
    SimModeManager,
    get_round_dt_s,
)


HR_FIXTURE = [70, 90, 120, 178, 182, 179, 183, 178, 181, 80]

EXPECTED_HR_FIXTURE_MODES = [
    SimMode.STABLE,
    SimMode.STABLE,
    SimMode.STABLE,
    SimMode.STABLE,
    SimMode.URGENT,
    SimMode.URGENT,
    SimMode.URGENT,
    SimMode.URGENT,
    SimMode.URGENT,
    SimMode.URGENT,
]


def _vitals(
    *,
    hr: float = 80,
    bp_sys: float = 120,
    o2_sat: float = 98,
) -> dict:
    return {
        "HR": hr,
        "BP_sys": bp_sys,
        "BP_dia": 70,
        "RR": 16,
        "O2Sat": o2_sat,
        "T": 37.0,
    }


def test_sim_mode_starts_as_stable():
    manager = SimModeManager()

    assert manager.current_mode == SimMode.STABLE


def test_stable_vitals_keep_mode_stable():
    manager = SimModeManager()

    mode = manager.update(vitals=_vitals(), rhythm="sinus")

    assert mode == SimMode.STABLE
    assert manager.current_mode == SimMode.STABLE


def test_hr_above_180_upgrades_immediately_to_urgent():
    manager = SimModeManager()

    assert manager.update(vitals=_vitals(hr=181), rhythm="sinus") == SimMode.URGENT


def test_o2sat_below_85_upgrades_immediately_to_code():
    manager = SimModeManager()

    assert manager.update(vitals=_vitals(o2_sat=84), rhythm="sinus") == SimMode.CODE


@pytest.mark.parametrize("rhythm", ["VF", "PEA", "asystole"])
def test_arrest_rhythm_upgrades_immediately_to_code(rhythm: str):
    manager = SimModeManager()

    assert manager.update(vitals=_vitals(), rhythm=rhythm) == SimMode.CODE


def test_physiology_arrest_event_upgrades_immediately_to_code():
    manager = SimModeManager()

    mode = manager.update(
        vitals=_vitals(),
        rhythm="sinus",
        physiology_events=[{"kind": "arrest", "payload": {"rhythm": "asystole"}}],
    )

    assert mode == SimMode.CODE


def test_declare_code_upgrades_immediately_to_code():
    manager = SimModeManager()

    mode = manager.update(
        vitals=_vitals(),
        rhythm="sinus",
        clinician_meta_events=[DECLARE_CODE],
    )

    assert mode == SimMode.CODE


def test_clinician_meta_event_via_dict_action_type():
    manager = SimModeManager()

    mode = manager.update(
        vitals=_vitals(),
        rhythm="sinus",
        clinician_event={"action_type": "declare_code"},
    )

    assert mode == SimMode.CODE


def test_code_does_not_automatically_downgrade_when_vital_improves():
    manager = SimModeManager()
    assert manager.update(vitals=_vitals(o2_sat=80), rhythm="sinus") == SimMode.CODE

    assert manager.update(vitals=_vitals(o2_sat=99), rhythm="sinus") == SimMode.CODE


def test_exit_code_without_rosc_does_not_downgrade_code():
    manager = SimModeManager()
    manager.update(vitals=_vitals(o2_sat=80), rhythm="sinus")

    mode = manager.update(
        vitals=_vitals(o2_sat=99),
        rhythm="sinus",
        clinician_meta_events=[EXIT_CODE],
        rosc_achieved=False,
    )

    assert mode == SimMode.CODE


def test_rosc_plus_exit_code_downgrades_code_to_urgent():
    manager = SimModeManager()
    manager.update(vitals=_vitals(o2_sat=80), rhythm="sinus")

    mode = manager.update(
        vitals=_vitals(o2_sat=99),
        rhythm="sinus",
        clinician_meta_events=[EXIT_CODE],
        rosc_achieved=True,
    )

    assert mode == SimMode.URGENT


def test_rosc_event_plus_exit_code_downgrades_code_to_urgent():
    manager = SimModeManager()
    manager.update(vitals=_vitals(o2_sat=80), rhythm="sinus")

    mode = manager.update(
        vitals=_vitals(o2_sat=99),
        rhythm="sinus",
        clinician_meta_events=[EXIT_CODE],
        physiology_events=[{"kind": "rosc"}],
    )

    assert mode == SimMode.URGENT


def test_explicit_code_downgrade_with_target_stable():
    manager = SimModeManager()
    manager.update(vitals=_vitals(o2_sat=80), rhythm="sinus")

    mode = manager.update(
        vitals=_vitals(),
        rhythm="sinus",
        code_downgrade_allowed=True,
        code_downgrade_target=SimMode.STABLE,
    )

    assert mode == SimMode.STABLE


def test_explicit_code_downgrade_target_code_falls_back_to_urgent():
    manager = SimModeManager()
    manager.update(vitals=_vitals(o2_sat=80), rhythm="sinus")

    mode = manager.update(
        vitals=_vitals(),
        rhythm="sinus",
        code_downgrade_allowed=True,
        code_downgrade_target=SimMode.CODE,
    )

    assert mode == SimMode.URGENT


def test_urgent_to_stable_downgrade_requires_hysteresis():
    manager = SimModeManager(downgrade_required_rounds=3)
    assert manager.update(vitals=_vitals(hr=181), rhythm="sinus") == SimMode.URGENT

    assert manager.update(vitals=_vitals(hr=80), rhythm="sinus") == SimMode.URGENT
    assert manager.update(vitals=_vitals(hr=80), rhythm="sinus") == SimMode.URGENT
    assert manager.update(vitals=_vitals(hr=80), rhythm="sinus") == SimMode.STABLE


def test_hr_hysteresis_fixture_keeps_urgent_mode_during_brief_dips():
    manager = SimModeManager(downgrade_required_rounds=3)
    modes = [
        manager.update(vitals=_vitals(hr=hr), rhythm="sinus")
        for hr in HR_FIXTURE
    ]

    assert modes == EXPECTED_HR_FIXTURE_MODES


@pytest.mark.parametrize(
    ("mode", "dt_s"),
    [
        (SimMode.STABLE, 30),
        (SimMode.URGENT, 10),
        (SimMode.CODE, 5),
    ],
)
def test_get_round_dt_s_returns_mode_dt(mode: SimMode, dt_s: int):
    assert get_round_dt_s(mode) == dt_s
