"""
Pathology rule registry and initialization signatures.

Two registries (ENGINE_DESIGN.md §3.5 + §5):
- `PATHOLOGY_INIT_SIGNATURES`: name → fn(severity: float) -> dict[str, Any]
    Pathology-conditioned hidden-state shape used during decode (step 1 of
    initialization). Severity is a scalar in [0, 1]; unspecified vars fall
    through to DEFAULT_HEALTHY.
- `PATHOLOGY_RULES`: name → fn(h: HiddenState, severity: float, dt_s: float)
    Continuous drift rule called every integrator tick.

Phase 3a ships `PATHOLOGY_INIT_SIGNATURES` populated for all 89 unique
pathology names observed in the 808-pair dataset (≥3-pair pathologies get
hand-authored signatures; 1-2-pair long-tail pathologies get a generic
"severity-scaled decompensation" shape). `PATHOLOGY_RULES` remains empty —
drift rules are Phase 4.

Design note: signatures set the **shape** (which hidden vars deviate from
baseline and in what direction). Magnitudes for the designated residual-
correction vars (`chronotropic_drive`, `SVR_index`, `CO_index`,
`ventilatory_drive`, `PaO2_effective`) get overwritten by `decode_state`'s
residual step; so init values for those are starting guesses and only
matter when residual correction fails to converge.
"""
from __future__ import annotations

from typing import Any, Callable

from .hidden_state import HiddenState


# --- Registries (populated by decorator at import time) ---
PATHOLOGY_INIT_SIGNATURES: dict[str, Callable[[float], dict[str, Any]]] = {}
PATHOLOGY_RULES: dict[str, Callable[[HiddenState, float, float], None]] = {}


def pathology_init(name: str):
    """Decorator: register an init_signature for `name`."""
    def register(fn: Callable[[float], dict[str, Any]]):
        PATHOLOGY_INIT_SIGNATURES[name] = fn
        return fn
    return register


def pathology_rule(name: str):
    """Decorator: register a drift rule for `name`."""
    def register(fn: Callable[[HiddenState, float, float], None]):
        PATHOLOGY_RULES[name] = fn
        return fn
    return register


# --- Severity scalar mapping ---
SEVERITY_SCALAR = {"mild": 0.3, "moderate": 0.6, "severe": 1.0}


def severity_to_scalar(severity: str) -> float:
    return SEVERITY_SCALAR.get(severity, 0.6)


# --- Healthy baseline (used for unspecified vars + unknown pathologies) ---
DEFAULT_HEALTHY: dict[str, Any] = {
    "rhythm":             "sinus",
    "CO_index":           1.0,
    "SVR_index":          1.0,
    "preload_index":      1.0,
    "chronotropic_drive": 0.0,
    "PaO2_effective":     95.0,
    "shunt_fraction":     0.05,
    "compliance_index":   1.0,
    "ventilatory_drive":  1.0,
    "core_temp_trend":    0.0,
    "consciousness":      1.0,
}


# ======================================================================
# Init signatures — grouped by organ system
# ======================================================================
#
# Each signature sets the pathology's physiologic "shape" — i.e. which
# hidden vars deviate from DEFAULT_HEALTHY and in what direction. Residual
# correction in `io.decode_state` overwrites chronotropic_drive / SVR_index
# / CO_index / ventilatory_drive / PaO2_effective to match the pair's
# observed vitals exactly, so the init numbers matter mostly for:
#   - `rhythm` (not residual-corrected)
#   - `shunt_fraction`, `compliance_index` (Phase 5 intervention effects)
#   - `core_temp_trend`, `consciousness` (not touched by residual either)
#
# Confidence tag comment convention (per ENGINE_DESIGN §4.1):
#   # src: author — design-level intuition, not literature-fit magnitudes.


# ----------------------------------------------------------------------
# Cardiovascular
# ----------------------------------------------------------------------

@pathology_init("pea_arrest")
def _sig_pea_arrest(sev: float) -> dict:
    # Rhythm set here is overridden by `infer_rhythm` when before.HR==0.
    return {
        "rhythm":       "PEA",
        "CO_index":     0.2 * (1 - sev),            # near-zero forward flow
        "preload_index": 0.5 * (1 - 0.5 * sev),
        "consciousness": 0.0,
    }


@pathology_init("vf_arrest")
def _sig_vf_arrest(sev: float) -> dict:
    return {
        "rhythm":       "VF",
        "CO_index":     0.0,
        "preload_index": 0.5,
        "consciousness": 0.0,
    }


@pathology_init("asystole")
def _sig_asystole(sev: float) -> dict:
    return {
        "rhythm":       "asystole",
        "CO_index":     0.0,
        "preload_index": 0.5,
        "consciousness": 0.0,
    }


@pathology_init("stemi")
def _sig_stemi(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "CO_index":           1.0 - 0.3 * sev,      # LV dysfunction
        "chronotropic_drive": 0.4 * sev,            # sympathetic response
        "SVR_index":          1.0 + 0.2 * sev,      # compensatory vasoconstriction
    }


@pathology_init("cardiogenic_shock")
def _sig_cardiogenic_shock(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "CO_index":           1.0 - 0.5 * sev,
        "chronotropic_drive": 1.0 * sev,
        "SVR_index":          1.0 + 0.4 * sev,
        "preload_index":      1.0 + 0.2 * sev,
        "consciousness":      1.0 - 0.3 * sev,
    }


@pathology_init("vt")
def _sig_vt(sev: float) -> dict:
    return {
        "rhythm":             "VT",
        "chronotropic_drive": 0.5 * sev,
        "CO_index":           1.0 - 0.2 * sev,
    }


@pathology_init("svt")
def _sig_svt(sev: float) -> dict:
    return {
        "rhythm":             "SVT",
        "chronotropic_drive": 0.3 * sev,
        "CO_index":           1.0 - 0.1 * sev,
    }


@pathology_init("bradycardia")
def _sig_bradycardia(sev: float) -> dict:
    return {
        "rhythm":             "bradycardia",
        "chronotropic_drive": -0.5 * sev,
        "CO_index":           1.0 - 0.2 * sev,
    }


@pathology_init("aortic_dissection")
def _sig_aortic_dissection(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": 0.5 * sev,
        "SVR_index":          1.0 + 0.3 * sev,      # often hypertensive
    }


@pathology_init("aortic_aneurysm_rupture")
def _sig_aortic_rupture(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "preload_index":      1.0 - 0.5 * sev,
        "chronotropic_drive": 1.0 * sev,
        "SVR_index":          1.0 + 0.3 * sev,
    }


@pathology_init("tamponade")
def _sig_tamponade(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "CO_index":           1.0 - 0.4 * sev,
        "preload_index":      1.0 + 0.3 * sev,       # back pressure
        "SVR_index":          1.0 + 0.3 * sev,
        "chronotropic_drive": 0.8 * sev,
    }


@pathology_init("pericarditis")
def _sig_pericarditis(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": 0.3 * sev,
    }


@pathology_init("chf_exacerbation")
def _sig_chf_exac(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "CO_index":           1.0 - 0.3 * sev,
        "preload_index":      1.0 + 0.3 * sev,
        "shunt_fraction":     0.05 + 0.25 * sev,
        "compliance_index":   1.0 - 0.3 * sev,
        "ventilatory_drive":  1.0 + 0.4 * sev,
    }


@pathology_init("pulmonary_edema")
def _sig_pulmonary_edema(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.15 + 0.35 * sev,
        "compliance_index":   1.0 - 0.5 * sev,
        "preload_index":      1.0 + 0.3 * sev,
        "CO_index":           1.0 - 0.2 * sev,
        "ventilatory_drive":  1.0 + 0.8 * sev,
        "chronotropic_drive": 0.5 * sev,
    }


@pathology_init("peripartum_cardiomyopathy")
def _sig_ppcm(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "CO_index":           1.0 - 0.4 * sev,
        "preload_index":      1.0 + 0.3 * sev,
        "shunt_fraction":     0.05 + 0.2 * sev,
        "chronotropic_drive": 0.6 * sev,
    }


@pathology_init("lvad_thrombosis")
def _sig_lvad_thrombosis(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "CO_index":           1.0 - 0.5 * sev,
        "preload_index":      1.0 + 0.3 * sev,
        "consciousness":      1.0 - 0.2 * sev,
    }


@pathology_init("electrical_storm")
def _sig_electrical_storm(sev: float) -> dict:
    return {
        "rhythm":             "VT",           # recurrent VT episodes
        "chronotropic_drive": 0.5 * sev,
        "CO_index":           1.0 - 0.3 * sev,
    }


@pathology_init("viral_myocarditis")
def _sig_viral_myocarditis(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "CO_index":           1.0 - 0.3 * sev,
        "chronotropic_drive": 0.4 * sev,
    }


@pathology_init("coarctation_of_aorta")
def _sig_coarctation(sev: float) -> dict:
    # Differential BP; simplified as elevated upstream pressure.
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 + 0.3 * sev,
        "CO_index":           1.0 - 0.2 * sev,
    }


@pathology_init("hypertensive_emergency")
def _sig_hypertensive_emergency(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 + 0.6 * sev,
        "chronotropic_drive": 0.3 * sev,
    }


@pathology_init("pre_excited_af")
def _sig_pre_excited_af(sev: float) -> dict:
    return {
        "rhythm":             "SVT",         # irregular tachy; approximated
        "chronotropic_drive": 0.5 * sev,
    }


# ----------------------------------------------------------------------
# Respiratory
# ----------------------------------------------------------------------

@pathology_init("anaphylaxis")
def _sig_anaphylaxis(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 - 0.4 * sev,    # distributive
        "shunt_fraction":     0.05 + 0.25 * sev,  # bronchospasm/edema
        "compliance_index":   1.0 - 0.3 * sev,
        "chronotropic_drive": 0.8 * sev,
        "ventilatory_drive":  1.0 + 0.5 * sev,
    }


@pathology_init("airway_obstruction")
def _sig_airway_obstruction(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.1 + 0.4 * sev,
        "compliance_index":   1.0 - 0.4 * sev,
        "ventilatory_drive":  1.0 + 0.6 * sev,
        "chronotropic_drive": 0.5 * sev,
    }


@pathology_init("asthma_exacerbation")
def _sig_asthma_exac(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.1 + 0.25 * sev,
        "compliance_index":   1.0 - 0.35 * sev,
        "ventilatory_drive":  1.0 + 0.6 * sev,
        "chronotropic_drive": 0.5 * sev,
    }


@pathology_init("pneumonia")
def _sig_pneumonia(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.1 + 0.3 * sev,
        "compliance_index":   1.0 - 0.2 * sev,
        "ventilatory_drive":  1.0 + 0.4 * sev,
        "chronotropic_drive": 0.3 * sev,
    }


@pathology_init("pulmonary_embolism")
def _sig_pe(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "CO_index":           1.0 - 0.3 * sev,    # RV strain
        "shunt_fraction":     0.1 + 0.3 * sev,
        "chronotropic_drive": 0.8 * sev,
        "ventilatory_drive":  1.0 + 0.6 * sev,
    }


@pathology_init("ards")
def _sig_ards(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.15 + 0.5 * sev,
        "compliance_index":   1.0 - 0.5 * sev,
        "ventilatory_drive":  1.0 + 0.6 * sev,
        "chronotropic_drive": 0.4 * sev,
    }


@pathology_init("copd_exacerbation")
def _sig_copd_exac(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.15 + 0.25 * sev,
        "compliance_index":   1.0 - 0.2 * sev,
        "ventilatory_drive":  1.0 + 0.4 * sev,
        "chronotropic_drive": 0.3 * sev,
    }


@pathology_init("tension_pneumothorax")
def _sig_tension_ptx(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "preload_index":      1.0 - 0.4 * sev,    # mediastinal compression
        "CO_index":           1.0 - 0.3 * sev,
        "shunt_fraction":     0.1 + 0.3 * sev,
        "chronotropic_drive": 0.8 * sev,
    }


@pathology_init("pneumothorax")
def _sig_pneumothorax(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.05 + 0.15 * sev,
        "ventilatory_drive":  1.0 + 0.3 * sev,
    }


@pathology_init("respiratory_failure")
def _sig_respiratory_failure(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.15 + 0.35 * sev,
        "compliance_index":   1.0 - 0.3 * sev,
        "ventilatory_drive":  1.0 + 0.5 * sev,
    }


@pathology_init("neonatal_respiratory_distress")
def _sig_neonatal_rds(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.1 + 0.3 * sev,
        "compliance_index":   1.0 - 0.4 * sev,
        "ventilatory_drive":  1.0 + 0.5 * sev,
    }


@pathology_init("bronchiolitis")
def _sig_bronchiolitis(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.1 + 0.2 * sev,
        "ventilatory_drive":  1.0 + 0.4 * sev,
    }


@pathology_init("acute_chest_syndrome")
def _sig_acs(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.1 + 0.3 * sev,
        "compliance_index":   1.0 - 0.3 * sev,
        "ventilatory_drive":  1.0 + 0.4 * sev,
        "chronotropic_drive": 0.3 * sev,
    }


@pathology_init("malignant_pleural_effusion")
def _sig_mpe(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.1 + 0.2 * sev,
        "compliance_index":   1.0 - 0.3 * sev,
        "ventilatory_drive":  1.0 + 0.3 * sev,
    }


@pathology_init("pulmonary_hemorrhage")
def _sig_pulm_hem(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.15 + 0.35 * sev,
        "preload_index":      1.0 - 0.2 * sev,
        "chronotropic_drive": 0.5 * sev,
    }


@pathology_init("airway_edema")
def _sig_airway_edema(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.1 + 0.3 * sev,
        "compliance_index":   1.0 - 0.3 * sev,
        "ventilatory_drive":  1.0 + 0.5 * sev,
    }


@pathology_init("aspiration_pneumonitis")
def _sig_aspiration(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.1 + 0.3 * sev,
        "ventilatory_drive":  1.0 + 0.4 * sev,
    }


@pathology_init("inhalation_injury")
def _sig_inhalation(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.1 + 0.3 * sev,
        "compliance_index":   1.0 - 0.3 * sev,
        "ventilatory_drive":  1.0 + 0.4 * sev,
    }


@pathology_init("drowning")
def _sig_drowning(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.15 + 0.4 * sev,
        "compliance_index":   1.0 - 0.3 * sev,
        "ventilatory_drive":  1.0 + 0.3 * sev,
        "consciousness":      1.0 - 0.4 * sev,
    }


# ----------------------------------------------------------------------
# Metabolic / Endocrine / Toxicology
# ----------------------------------------------------------------------

@pathology_init("dka")
def _sig_dka(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": 0.8 * sev,        # compensatory tachy + volume deplete
        "preload_index":      1.0 - 0.3 * sev,
        "ventilatory_drive":  1.0 + 1.0 * sev,  # Kussmaul
        "consciousness":      1.0 - 0.2 * sev,
    }


@pathology_init("hyperkalemia")
def _sig_hyperkalemia(sev: float) -> dict:
    return {
        "rhythm":             "bradycardia" if sev > 0.5 else "sinus",
        "chronotropic_drive": -0.5 * sev,
        "CO_index":           1.0 - 0.2 * sev,
    }


@pathology_init("hypoglycemia")
def _sig_hypoglycemia(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": 0.5 * sev,
        "consciousness":      1.0 - 0.5 * sev,
    }


@pathology_init("tumor_lysis")
def _sig_tumor_lysis(sev: float) -> dict:
    # Hyperkalemia-dominant electrolyte storm.
    return {
        "rhythm":             "bradycardia" if sev > 0.7 else "sinus",
        "chronotropic_drive": -0.4 * sev,
        "CO_index":           1.0 - 0.2 * sev,
    }


@pathology_init("thyroid_storm")
def _sig_thyroid_storm(sev: float) -> dict:
    # core_temp_trend moved to drift rule in Phase 4 (§5 scenario-paced).
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": 2.0 * sev,
        "SVR_index":          1.0 - 0.2 * sev,
        "ventilatory_drive":  1.0 + 0.3 * sev,
    }


@pathology_init("adrenal_crisis")
def _sig_adrenal_crisis(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 - 0.5 * sev,
        "preload_index":      1.0 - 0.3 * sev,
        "chronotropic_drive": 0.6 * sev,
        "consciousness":      1.0 - 0.3 * sev,
    }


@pathology_init("hyponatremia")
def _sig_hyponatremia(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "consciousness":      1.0 - 0.4 * sev,
    }


@pathology_init("hypothermia")
def _sig_hypothermia(sev: float) -> dict:
    # core_temp_trend moved to drift rule in Phase 4 (§5 scenario-paced).
    return {
        "rhythm":             "bradycardia" if sev > 0.5 else "sinus",
        "chronotropic_drive": -0.8 * sev,
        "CO_index":           1.0 - 0.3 * sev,
        "consciousness":      1.0 - 0.4 * sev,
    }


@pathology_init("hyperthermia")
def _sig_hyperthermia(sev: float) -> dict:
    # core_temp_trend moved to drift rule in Phase 4 (§5 scenario-paced).
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": 1.2 * sev,
        "consciousness":      1.0 - 0.3 * sev,
    }


@pathology_init("opioid_toxicity")
def _sig_opioid_toxicity(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "ventilatory_drive":  1.0 - 0.8 * sev,    # respiratory depression
        "consciousness":      1.0 - 0.7 * sev,
        "chronotropic_drive": -0.3 * sev,
    }


@pathology_init("beta_blocker_toxicity")
def _sig_bb_tox(sev: float) -> dict:
    return {
        "rhythm":             "bradycardia",
        "chronotropic_drive": -1.0 * sev,
        "CO_index":           1.0 - 0.4 * sev,
    }


@pathology_init("ccb_toxicity")
def _sig_ccb_tox(sev: float) -> dict:
    return {
        "rhythm":             "bradycardia",
        "chronotropic_drive": -0.8 * sev,
        "SVR_index":          1.0 - 0.4 * sev,
        "CO_index":           1.0 - 0.3 * sev,
    }


@pathology_init("digoxin_toxicity")
def _sig_digoxin_tox(sev: float) -> dict:
    return {
        "rhythm":             "bradycardia",
        "chronotropic_drive": -0.6 * sev,
        "CO_index":           1.0 - 0.3 * sev,
    }


@pathology_init("asa_toxicity")
def _sig_asa_tox(sev: float) -> dict:
    # core_temp_trend moved to drift rule in Phase 4 (§5 scenario-paced).
    return {
        "rhythm":             "sinus",
        "ventilatory_drive":  1.0 + 0.8 * sev,    # respiratory alkalosis
        "consciousness":      1.0 - 0.3 * sev,
        "chronotropic_drive": 0.5 * sev,
    }


@pathology_init("serotonin_syndrome")
def _sig_serotonin(sev: float) -> dict:
    # core_temp_trend moved to drift rule in Phase 4 (§5 scenario-paced).
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": 1.0 * sev,
        "consciousness":      1.0 - 0.3 * sev,
    }


@pathology_init("tca_overdose")
def _sig_tca_od(sev: float) -> dict:
    return {
        "rhythm":             "sinus",          # wide QRS, but not classified here
        "SVR_index":          1.0 - 0.3 * sev,
        "chronotropic_drive": 0.8 * sev,
        "consciousness":      1.0 - 0.5 * sev,
    }


@pathology_init("iron_overdose")
def _sig_iron_od(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "preload_index":      1.0 - 0.3 * sev,
        "SVR_index":          1.0 - 0.2 * sev,
        "chronotropic_drive": 0.7 * sev,
        "consciousness":      1.0 - 0.3 * sev,
    }


@pathology_init("toxic_alcohol")
def _sig_toxic_alcohol(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "ventilatory_drive":  1.0 + 0.5 * sev,    # anion gap acidosis
        "consciousness":      1.0 - 0.5 * sev,
    }


@pathology_init("cyanide_toxicity")
def _sig_cyanide(sev: float) -> dict:
    # O2 utilization defect — SpO2 may look high even when tissue hypoxic.
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": 0.5 * sev,
        "SVR_index":          1.0 - 0.3 * sev,
        "consciousness":      1.0 - 0.5 * sev,
        "ventilatory_drive":  1.0 + 0.5 * sev,
    }


@pathology_init("local_anesthetic_toxicity")
def _sig_last_tox(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": -0.3 * sev,
        "CO_index":           1.0 - 0.3 * sev,
        "consciousness":      1.0 - 0.5 * sev,
    }


@pathology_init("organophosphate_poisoning")
def _sig_organophosphate(sev: float) -> dict:
    return {
        "rhythm":             "bradycardia",
        "chronotropic_drive": -0.8 * sev,
        "ventilatory_drive":  1.0 - 0.3 * sev,
        "consciousness":      1.0 - 0.4 * sev,
    }


@pathology_init("mixed_overdose")
def _sig_mixed_od(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "ventilatory_drive":  1.0 - 0.4 * sev,
        "consciousness":      1.0 - 0.5 * sev,
    }


@pathology_init("sedation_respiratory_depression")
def _sig_sedation_rd(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "ventilatory_drive":  1.0 - 0.7 * sev,
        "consciousness":      1.0 - 0.6 * sev,
    }


# ----------------------------------------------------------------------
# Neurologic / Trauma
# ----------------------------------------------------------------------

@pathology_init("traumatic_brain_injury")
def _sig_tbi(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "consciousness":      1.0 - 0.6 * sev,
        "chronotropic_drive": 0.3 * sev,          # variable; sympathetic surge
        "SVR_index":          1.0 + 0.2 * sev,    # Cushing's
    }


@pathology_init("subarachnoid_hemorrhage")
def _sig_sah(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "consciousness":      1.0 - 0.5 * sev,
        "chronotropic_drive": 0.5 * sev,
        "SVR_index":          1.0 + 0.3 * sev,    # HTN with IICP
        "ventilatory_drive":  1.0 + 0.2 * sev,
    }


@pathology_init("status_epilepticus")
def _sig_status_epilepticus(sev: float) -> dict:
    # core_temp_trend moved to drift rule in Phase 4 (§5 scenario-paced).
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": 1.0 * sev,
        "consciousness":      1.0 - 0.8 * sev,
    }


@pathology_init("seizure")
def _sig_seizure(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": 0.8 * sev,
        "consciousness":      1.0 - 0.6 * sev,
    }


@pathology_init("delirium")
def _sig_delirium(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "consciousness":      1.0 - 0.4 * sev,
        "chronotropic_drive": 0.3 * sev,
    }


@pathology_init("spinal_cord_injury")
def _sig_sci(sev: float) -> dict:
    # High SCI → neurogenic shock.
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 - 0.4 * sev,
        "chronotropic_drive": -0.4 * sev,         # loss of sympathetic tone
    }


@pathology_init("neurogenic_shock")
def _sig_neurogenic(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 - 0.5 * sev,
        "chronotropic_drive": -0.3 * sev,
    }


@pathology_init("hemorrhagic_shock")
def _sig_hem_shock(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "preload_index":      1.0 - 0.5 * sev,
        "chronotropic_drive": 1.2 * sev,
        "SVR_index":          1.0 + 0.3 * sev,
        "consciousness":      1.0 - 0.3 * sev,
    }


@pathology_init("polytrauma")
def _sig_polytrauma(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "preload_index":      1.0 - 0.3 * sev,
        "chronotropic_drive": 0.8 * sev,
        "SVR_index":          1.0 + 0.2 * sev,
        "consciousness":      1.0 - 0.3 * sev,
    }


@pathology_init("burn")
def _sig_burn(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "preload_index":      1.0 - 0.3 * sev,     # fluid shifts
        "chronotropic_drive": 0.6 * sev,
    }


@pathology_init("abuse_trauma")
def _sig_abuse_trauma(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "preload_index":      1.0 - 0.3 * sev,
        "chronotropic_drive": 0.6 * sev,
    }


@pathology_init("blunt_abdominal_trauma")
def _sig_blunt_abd(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "preload_index":      1.0 - 0.4 * sev,
        "chronotropic_drive": 0.8 * sev,
    }


# ----------------------------------------------------------------------
# Sepsis / Infection
# ----------------------------------------------------------------------

@pathology_init("sepsis")
def _sig_sepsis(sev: float) -> dict:
    # core_temp_trend moved to drift rule in Phase 4 (§5 scenario-paced).
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 - 0.3 * sev,
        "chronotropic_drive": 0.8 * sev,
    }


@pathology_init("septic_shock")
def _sig_septic_shock(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 - 0.5 * sev,     # massive vasodilation
        "chronotropic_drive": 1.2 * sev,
        "CO_index":           1.0 + 0.1 * sev,     # hyperdynamic early
        "preload_index":      1.0 - 0.2 * sev,
        "consciousness":      1.0 - 0.3 * sev,
    }


@pathology_init("cholangitis")
def _sig_cholangitis(sev: float) -> dict:
    # core_temp_trend moved to drift rule in Phase 4 (§5 scenario-paced).
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 - 0.3 * sev,
        "chronotropic_drive": 0.8 * sev,
    }


@pathology_init("meningitis")
def _sig_meningitis(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "consciousness":      1.0 - 0.5 * sev,
        "chronotropic_drive": 0.5 * sev,
        "ventilatory_drive":  1.0 + 0.3 * sev,
    }


@pathology_init("mis_c")
def _sig_misc(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 - 0.3 * sev,
        "chronotropic_drive": 0.8 * sev,
        "CO_index":           1.0 - 0.2 * sev,
    }


@pathology_init("myasthenic_crisis")
def _sig_myasthenic(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "ventilatory_drive":  1.0 - 0.5 * sev,    # weak respiratory muscles
    }


# ----------------------------------------------------------------------
# OB/GYN
# ----------------------------------------------------------------------

@pathology_init("eclampsia")
def _sig_eclampsia(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 + 0.4 * sev,
        "chronotropic_drive": 0.3 * sev,
        "consciousness":      1.0 - 0.6 * sev,
    }


@pathology_init("preeclampsia")
def _sig_preeclampsia(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 + 0.3 * sev,
        "chronotropic_drive": 0.2 * sev,
    }


@pathology_init("postpartum_hemorrhage")
def _sig_pph(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "preload_index":      1.0 - 0.5 * sev,
        "chronotropic_drive": 1.0 * sev,
        "SVR_index":          1.0 + 0.2 * sev,
    }


@pathology_init("obstetric_trauma")
def _sig_ob_trauma(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "preload_index":      1.0 - 0.3 * sev,
        "chronotropic_drive": 0.7 * sev,
    }


@pathology_init("ectopic_pregnancy")
def _sig_ectopic(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "preload_index":      1.0 - 0.4 * sev,
        "chronotropic_drive": 1.0 * sev,
    }


@pathology_init("breech_delivery")
def _sig_breech(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": 0.5 * sev,          # maternal stress
    }


@pathology_init("active_labor")
def _sig_active_labor(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "chronotropic_drive": 0.4 * sev,
        "ventilatory_drive":  1.0 + 0.3 * sev,
    }


# ----------------------------------------------------------------------
# GI / Misc
# ----------------------------------------------------------------------

@pathology_init("upper_gi_bleed")
def _sig_ugi_bleed(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "preload_index":      1.0 - 0.4 * sev,
        "chronotropic_drive": 0.8 * sev,
    }


@pathology_init("pancreatitis")
def _sig_pancreatitis(sev: float) -> dict:
    return {
        "rhythm":             "sinus",
        "SVR_index":          1.0 - 0.2 * sev,
        "chronotropic_drive": 0.5 * sev,
        "preload_index":      1.0 - 0.2 * sev,
    }


# ======================================================================
# Phase 4: Pathology drift rules
# ======================================================================
#
# Each rule has signature `fn(h: HiddenState, severity: float, dt_s: float)`
# and mutates `h` in place. Called once per integrator tick (default 30s)
# for every pair with `duration_s > 0` whose pathology name is registered.
#
# Design principles (ENGINE_DESIGN.md §5, Phase 4 brief):
#
#  * Rules are pure functions of hidden state + severity + dt. No access
#    to case_id / before_dict / other pairs. Branching conditions MUST be
#    on hidden state, not scenario metadata. This is what enforces
#    cross-scenario generalization.
#
#  * Drift rates are **scenario-paced**, not ICU-physiologic. Scenarios
#    compress minutes-to-hours of real deterioration into 2-10 minute
#    pairs. Coefficients below were calibrated against the 249 pure-wait
#    pairs (src: Phase 4 calibration, 2026-04-19), not literature.
#
#  * Severity scaling is explicit per rule. For pathologies where drift
#    is strongly bimodal (mild/moderate stable → severe collapse) the
#    rule branches on `severity >= 0.9` rather than multiplying linearly.
#    For gradual deterioration we use `severity` (or `severity**2`) as a
#    scalar multiplier.
#
#  * Tipping points use hidden-state thresholds (e.g. `CO_index < 0.4 and
#    chronotropic_drive > 3`) — never case_id or pathology severity alone.
#    This lets "patient tires" transitions emerge from the same rule when
#    the integrator steps into the unstable region.
#
# Provenance tagging (§4.1 convention):
#    [calibrated]  — fit to pure-wait pair means for the pathology
#    [author]      — design-level intuition, under-determined by data
#                    (<3 pure-wait pairs for that pathology)
#    [tipping]     — non-linear branch for decompensation; threshold is
#                    author-chosen, magnitude calibrated if possible


# ---- Helper utilities for drift rules --------------------------------

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _decay_toward(x: float, target: float, rate_per_min: float,
                  dt_min: float) -> float:
    """Exponential-style approach to `target` at rate `rate_per_min`.

    Fractional step = min(1, rate * dt_min). Used for indices approaching
    a fixed point (e.g. CO_index → 0.3 under progressive heart failure).
    """
    w = min(1.0, rate_per_min * dt_min)
    return x + (target - x) * w


# ======================================================================
# Phase 4 drift rules — severity-gated + state-conditional
# ======================================================================
#
# Each rule obeys three contracts:
#
#   1. **Severity gate.** Drift fires only at `severity >= 0.9` (severe).
#      Authors write moderate/mild pairs to be stable over the observation
#      window; adding drift there pushes direction-match off on pairs that
#      the Phase 3b identity baseline was passing. Severe pathologies
#      decompensate visibly in 2-10 min, which matches the dataset.
#
#   2. **State-conditional.** Where possible, gate drift on the
#      already-decoded hidden state. For example, only drop HR further
#      when `h.chronotropic_drive` is already low; only drop BP further
#      when `h.CO_index` or `h.preload_index` is already depressed. This
#      lets drift accelerate existing deterioration without pushing
#      stable severe presentations off-course.
#
#   3. **Scenario-paced coefficients.** Rates were calibrated against
#      pure-wait pair means for the pathology (2026-04-19 calibration
#      pass). They're not literature-derived.
#
# Provenance tags in comments:
#    [calibrated]  — fit to pure-wait pair observations
#    [author]      — design intuition; data too sparse for fit
#    [tipping]     — non-linear branch for decompensation into arrest


# ---- Helper utilities for drift rules --------------------------------

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _decompensating(h) -> bool:
    """Hidden-state heuristic: is this patient already trending down?

    Used by rules that only want to accelerate visible deterioration,
    not push stable presentations off. True when HR drive is tachy/brady
    enough, CO reduced, preload dropped, or PaO2 low.
    """
    return (
        h.chronotropic_drive > 2.5  # already tachy
        or h.chronotropic_drive < -0.5  # bradycardic
        or h.CO_index < 0.8
        or h.preload_index < 0.8
        or h.PaO2_effective < 75
    )


# ======================================================================
# Cardiovascular
# ======================================================================

@pathology_rule("stemi")
def _drift_stemi(h, severity, dt_s):
    # Pure-wait obs severe (n=5): +52 HR, -49 BPs, -30 O2Sat. Phase 4
    # pinned no-op; Phase 5 light state-conditional version (fires only if
    # CO or preload already depressed) improves HR/BP MAE on pure-wait
    # without regressing strict count. [state-conditional]
    if severity < 0.9:
        return
    if h.CO_index >= 0.9 and h.preload_index >= 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive += 0.15 * dt_min
    h.CO_index = _clip(h.CO_index * (1 - 0.015 * dt_min), 0.1, 4.0)


@pathology_rule("aortic_dissection")
def _drift_aortic_dissection(h, severity, dt_s):
    # Pure-wait obs severe (n=4): HR/BP drift mixed. Phase 4 pinned no-op;
    # Phase 5 light O2-only drift (no BP/preload) is neutral on strict and
    # provides tiny O2 MAE improvement. [state-conditional, minimal]
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 0.5 * dt_min, 0, 600)


@pathology_rule("hemorrhagic_shock")
def _drift_hemorrhagic_shock(h, severity, dt_s):
    # Pure-wait obs: mild — short observation window, already in shock.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.preload_index = _clip(h.preload_index - 0.02 * dt_min, 0, 1.8)


@pathology_rule("upper_gi_bleed")
def _drift_ugi_bleed(h, severity, dt_s):
    # Pure-wait obs (n=1 severe): -85 BPs, -96 O2Sat — exsanguination.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.preload_index = _clip(h.preload_index - 0.04 * dt_min, 0, 1.8)
    h.PaO2_effective = _clip(h.PaO2_effective - 3.0 * dt_min, 0, 600)


@pathology_rule("ectopic_pregnancy")
def _drift_ectopic(h, severity, dt_s):
    # Pure-wait obs severe (n=3): -22 HR, -25 O2Sat. Late hemorrhagic shock.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.preload_index = _clip(h.preload_index - 0.04 * dt_min, 0, 1.8)
    h.PaO2_effective = _clip(h.PaO2_effective - 2.0 * dt_min, 0, 600)
    # [tipping] hemorrhagic arrest
    if h.preload_index < 0.2 and h.CO_index < 0.3 and h.rhythm == "sinus":
        h.rhythm = "PEA"
        h.CO_index = 0.0


@pathology_rule("subarachnoid_hemorrhage")
def _drift_sah(h, severity, dt_s):
    # Pure-wait obs severe (n=4): -7 HR, -32 BPs, -12 BPd. IICP collapse.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.CO_index = _clip(h.CO_index * (1 - 0.03 * dt_min), 0.1, 4.0)


@pathology_rule("traumatic_brain_injury")
def _drift_traumatic_brain_injury(h, severity, dt_s):
    # Pure-wait obs severe (n=3): mild Cushing.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.SVR_index = _clip(h.SVR_index * (1 + 0.02 * dt_min), 0.2, 4.0)
    h.PaO2_effective = _clip(h.PaO2_effective - 1.5 * dt_min, 0, 600)


@pathology_rule("adrenal_crisis")
def _drift_adrenal_crisis(h, severity, dt_s):
    # Pure-wait obs severe (n=6): +31 HR, -20 BPs, -35 O2Sat — but
    # individual pair drifts vary widely. Phase 4 pinned no-op (uniform
    # drift net-negative). Phase 5 un-pinned with state-conditional
    # gating: drift fires only when patient is already decompensated
    # (CO or SVR depressed), so author-stable severe pairs don't move.
    # Net +1 strict pure-wait, neutral action. [tipping none]
    if severity < 0.9:
        return
    if h.CO_index >= 0.9 and h.SVR_index >= 0.9:
        return
    dt_min = dt_s / 60.0
    h.SVR_index = _clip(h.SVR_index * (1 - 0.015 * dt_min), 0.2, 4.0)


@pathology_rule("tamponade")
def _drift_tamponade(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.CO_index = _clip(h.CO_index * (1 - 0.03 * dt_min), 0.1, 4.0)
    h.PaO2_effective = _clip(h.PaO2_effective - 2.0 * dt_min, 0, 600)


@pathology_rule("tension_pneumothorax")
def _drift_tension_pneumothorax(h, severity, dt_s):
    # Pure-wait obs severe (n=4): -25 HR, -30 BPs, -25 O2Sat.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.preload_index = _clip(h.preload_index - 0.04 * dt_min, 0, 1.8)
    h.CO_index = _clip(h.CO_index * (1 - 0.03 * dt_min), 0.1, 4.0)
    h.PaO2_effective = _clip(h.PaO2_effective - 2.0 * dt_min, 0, 600)
    if h.preload_index < 0.25 and h.CO_index < 0.3 and h.rhythm == "sinus":
        h.rhythm = "PEA"
        h.CO_index = 0.0


@pathology_rule("pulmonary_embolism")
def _drift_pulmonary_embolism(h, severity, dt_s):
    # Pure-wait obs severe (n=3): -62 HR, -75 BPs, -63 O2Sat. Massive PE.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.CO_index = _clip(h.CO_index * (1 - 0.06 * dt_min), 0.1, 4.0)
    h.PaO2_effective = _clip(h.PaO2_effective - 4.0 * dt_min, 0, 600)
    h.chronotropic_drive -= 0.3 * dt_min
    if h.CO_index < 0.3 and h.rhythm == "sinus":
        h.rhythm = "PEA"
        h.CO_index = 0.0


@pathology_rule("vt")
def _drift_vt(h, severity, dt_s):
    # Pure-wait obs: unstable VT deteriorates.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive += 0.15 * dt_min
    h.CO_index = _clip(h.CO_index * (1 - 0.02 * dt_min), 0.1, 4.0)


# ======================================================================
# Respiratory
# ======================================================================

@pathology_rule("airway_obstruction")
def _drift_airway_obstruction(h, severity, dt_s):
    # Pure-wait obs severe (n=5): -36 HR, -30 O2Sat (hypoxic brady arrest)
    # BUT the specific pairs vary hugely in starting vitals (some stable,
    # some already collapsing). Per-rule impact test showed any uniform
    # drift was net-negative; leave pinned but no-op, to be revived with
    # Phase 5 tipping once intervention composition works.
    return


@pathology_rule("asthma_exacerbation")
def _drift_asthma_exacerbation(h, severity, dt_s):
    # Pure-wait obs severe (n=8): mild decline. Phase 4 pinned no-op;
    # Phase 5 re-evaluation with PaO2 + ventilatory-drive drift recovers
    # +1 pure-wait pair, no action-pair regression. [calibrated]
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 1.5 * dt_min, 0, 600)
    h.ventilatory_drive = _clip(h.ventilatory_drive - 0.02 * dt_min, 0, 5.0)


@pathology_rule("copd_exacerbation")
def _drift_copd_exacerbation(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 1.0 * dt_min, 0, 600)


@pathology_rule("pneumonia")
def _drift_pneumonia(h, severity, dt_s):
    # Pure-wait obs: mild O2 decline.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 2.0 * dt_min, 0, 600)


@pathology_rule("respiratory_failure")
def _drift_respiratory_failure(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 1.5 * dt_min, 0, 600)


@pathology_rule("aspiration_pneumonitis")
def _drift_aspiration(h, severity, dt_s):
    # n=1 severe: HR 120→0. Hypoxic arrest.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 4.0 * dt_min, 0, 600)
    if h.PaO2_effective < 30 and h.rhythm == "sinus":
        h.rhythm = "PEA"
        h.CO_index = 0.0


# ======================================================================
# Infection / sepsis
# ======================================================================

@pathology_rule("septic_shock")
def _drift_septic_shock(h, severity, dt_s):
    # Pure-wait obs severe: -3 BPs, -14 O2Sat. Phase 4 pinned no-op; Phase 5
    # re-evaluation unpinned with state-conditional SVR drift (fires only
    # when SVR already depressed) — neutral strict but BP MAE -0.8/-0.8.
    # [state-conditional]
    if severity < 0.9:
        return
    if h.SVR_index >= 0.85:
        return
    dt_min = dt_s / 60.0
    h.SVR_index = _clip(h.SVR_index * (1 - 0.02 * dt_min), 0.2, 4.0)


@pathology_rule("anaphylaxis")
def _drift_anaphylaxis(h, severity, dt_s):
    # Pure-wait obs severe (n=7): -7 HR, -12 RR, -5 O2Sat.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 1.0 * dt_min, 0, 600)
    h.ventilatory_drive = _clip(
        h.ventilatory_drive - 0.1 * dt_min, 0, 5.0)


# ======================================================================
# Metabolic / Endocrine / Toxicology
# ======================================================================

@pathology_rule("hyperkalemia")
def _drift_hyperkalemia(h, severity, dt_s):
    # Pure-wait obs severe (n=4): -37 HR, -58 BPs, -73 O2Sat. Brady→arrest.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive -= 0.3 * dt_min
    h.CO_index = _clip(h.CO_index * (1 - 0.03 * dt_min), 0.1, 4.0)
    h.PaO2_effective = _clip(h.PaO2_effective - 3.0 * dt_min, 0, 600)
    # [tipping] bradycardic arrest — only when patient decoded as bradycardic
    if h.rhythm == "bradycardia" and h.chronotropic_drive < -1.5:
        h.rhythm = "asystole"
        h.CO_index = 0.0


@pathology_rule("bradycardia")
def _drift_bradycardia(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive -= 0.15 * dt_min
    h.CO_index = _clip(h.CO_index * (1 - 0.03 * dt_min), 0.1, 4.0)
    # [tipping] decompensating brady — only if already brady and very depressed
    if h.rhythm == "bradycardia" and h.chronotropic_drive < -2.0 \
            and h.CO_index < 0.4:
        h.rhythm = "asystole"
        h.CO_index = 0.0


@pathology_rule("beta_blocker_toxicity")
def _drift_beta_blocker_tox(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive -= 0.1 * dt_min
    h.CO_index = _clip(h.CO_index * (1 - 0.015 * dt_min), 0.1, 4.0)


@pathology_rule("ccb_toxicity")
def _drift_ccb_toxicity(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive -= 0.1 * dt_min
    h.SVR_index = _clip(h.SVR_index * (1 - 0.02 * dt_min), 0.2, 4.0)


@pathology_rule("digoxin_toxicity")
def _drift_digoxin_toxicity(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.CO_index = _clip(h.CO_index * (1 - 0.02 * dt_min), 0.1, 4.0)


@pathology_rule("tca_overdose")
def _drift_tca_overdose(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.SVR_index = _clip(h.SVR_index * (1 - 0.03 * dt_min), 0.2, 4.0)
    h.PaO2_effective = _clip(h.PaO2_effective - 2.0 * dt_min, 0, 600)


@pathology_rule("cyanide_toxicity")
def _drift_cyanide(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive -= 0.2 * dt_min
    h.PaO2_effective = _clip(h.PaO2_effective - 3.0 * dt_min, 0, 600)


@pathology_rule("tumor_lysis")
def _drift_tumor_lysis(h, severity, dt_s):
    # Hyperkalemia-dominant presentation with arrest potential.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive -= 0.2 * dt_min
    h.PaO2_effective = _clip(h.PaO2_effective - 2.0 * dt_min, 0, 600)
    if h.rhythm == "bradycardia" and h.chronotropic_drive < -2.0 \
            and h.CO_index < 0.4:
        h.rhythm = "asystole"
        h.CO_index = 0.0


@pathology_rule("asa_toxicity")
def _drift_asa_toxicity(h, severity, dt_s):
    # Pure-wait obs: stable across vitals. T trend moved here.
    h.core_temp_trend = 0.0
    if severity < 0.9:
        return


@pathology_rule("serotonin_syndrome")
def _drift_serotonin(h, severity, dt_s):
    # n=2: +37 HR, -62 BPs, -50 O2Sat.
    h.core_temp_trend = 0.015 * severity  # hyperthermia runs even moderate
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive += 0.2 * dt_min
    h.SVR_index = _clip(h.SVR_index * (1 - 0.03 * dt_min), 0.2, 4.0)
    h.PaO2_effective = _clip(h.PaO2_effective - 2.0 * dt_min, 0, 600)


@pathology_rule("thyroid_storm")
def _drift_thyroid_storm(h, severity, dt_s):
    # n=2: small drift. Hyperthermia runs for all severity levels.
    h.core_temp_trend = 0.02 * severity
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 1.0 * dt_min, 0, 600)


@pathology_rule("hyperthermia")
def _drift_hyperthermia(h, severity, dt_s):
    # core_temp_trend runs for moderate too (single scenario case).
    h.core_temp_trend = 0.015 * severity


@pathology_rule("hypothermia")
def _drift_hypothermia(h, severity, dt_s):
    # Pure-wait obs severe (n=5): +0.4 °C from passive warming.
    # Trend runs at all severity levels.
    h.core_temp_trend = 0.04 * severity
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive += 0.05 * dt_min


@pathology_rule("status_epilepticus")
def _drift_status_epilepticus(h, severity, dt_s):
    # T stable in pure-wait obs.
    h.core_temp_trend = 0.0
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 1.5 * dt_min, 0, 600)


@pathology_rule("sepsis")
def _drift_sepsis(h, severity, dt_s):
    # T stable.
    h.core_temp_trend = 0.0
    if severity < 0.9:
        return


@pathology_rule("cholangitis")
def _drift_cholangitis(h, severity, dt_s):
    h.core_temp_trend = 0.0
    if severity < 0.9:
        return


@pathology_rule("dka")
def _drift_dka(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 3.0 * dt_min, 0, 600)


@pathology_rule("opioid_toxicity")
def _drift_opioid_toxicity(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 0.8 * dt_min, 0, 600)
    h.ventilatory_drive = _clip(
        h.ventilatory_drive - 0.03 * dt_min, 0, 5.0)


@pathology_rule("organophosphate_poisoning")
def _drift_organophosphate(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive -= 0.04 * dt_min
    h.ventilatory_drive = _clip(
        h.ventilatory_drive - 0.02 * dt_min, 0, 5.0)


@pathology_rule("iron_overdose")
def _drift_iron(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.preload_index = _clip(h.preload_index - 0.02 * dt_min, 0, 1.8)
    h.PaO2_effective = _clip(h.PaO2_effective - 1.5 * dt_min, 0, 600)


@pathology_rule("mixed_overdose")
def _drift_mixed_overdose(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 0.8 * dt_min, 0, 600)


@pathology_rule("local_anesthetic_toxicity")
def _drift_last_tox(h, severity, dt_s):
    # n=2: +71 HR, -78 BPs for severe.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive += 0.2 * dt_min
    h.SVR_index = _clip(h.SVR_index * (1 - 0.04 * dt_min), 0.2, 4.0)


# ======================================================================
# Other / miscellaneous
# ======================================================================

@pathology_rule("hyponatremia")
def _drift_hyponatremia(h, severity, dt_s):
    # Pure-wait obs: +28 HR, +38 BPs in severe.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive += 0.3 * dt_min
    h.SVR_index = _clip(h.SVR_index * (1 + 0.02 * dt_min), 0.2, 4.0)


@pathology_rule("burn")
def _drift_burn(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.preload_index = _clip(h.preload_index - 0.04 * dt_min, 0, 1.8)


@pathology_rule("aortic_aneurysm_rupture")
def _drift_aortic_rupture(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.preload_index = _clip(h.preload_index - 0.06 * dt_min, 0, 1.8)


@pathology_rule("cardiogenic_shock")
def _drift_cardiogenic_shock(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.CO_index = _clip(h.CO_index * (1 - 0.02 * dt_min), 0.1, 4.0)


@pathology_rule("pulmonary_edema")
def _drift_pulmonary_edema(h, severity, dt_s):
    # Template from ENGINE_DESIGN §5.
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.shunt_fraction = _clip(h.shunt_fraction + 0.01 * dt_min, 0, 0.8)
    h.compliance_index = _clip(h.compliance_index - 0.015 * dt_min, 0.2, 1.2)
    h.PaO2_effective = _clip(h.PaO2_effective - 1.0 * dt_min, 0, 600)
    # [tipping] compensation fails
    if h.chronotropic_drive > 2.5 and h.compliance_index < 0.5:
        h.ventilatory_drive = _clip(
            h.ventilatory_drive - 0.15 * dt_min, 0, 5.0)


@pathology_rule("chf_exacerbation")
def _drift_chf_exacerbation(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 0.6 * dt_min, 0, 600)


@pathology_rule("viral_myocarditis")
def _drift_viral_myocarditis(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.CO_index = _clip(h.CO_index * (1 - 0.02 * dt_min), 0.1, 4.0)


@pathology_rule("electrical_storm")
def _drift_electrical_storm(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 2.0 * dt_min, 0, 600)


@pathology_rule("drowning")
def _drift_drowning(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive -= 0.2 * dt_min


@pathology_rule("myasthenic_crisis")
def _drift_myasthenic_crisis(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.ventilatory_drive = _clip(
        h.ventilatory_drive - 0.04 * dt_min, 0, 5.0)


@pathology_rule("vf_arrest")
def _drift_vf_arrest(h, severity, dt_s):
    # n=2: post-ROSC stabilization. No drift.
    return


@pathology_rule("pea_arrest")
def _drift_pea_arrest(h, severity, dt_s):
    # n=7: paradoxical recovery (author bundles ROSC into pea_arrest label).
    # No drift — cannot predict scenario-driven ROSC.
    return


@pathology_rule("polytrauma")
def _drift_polytrauma(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.preload_index = _clip(h.preload_index - 0.02 * dt_min, 0, 1.8)


@pathology_rule("hypoglycemia")
def _drift_hypoglycemia(h, severity, dt_s):
    if severity < 0.9:
        return


@pathology_rule("hypertensive_emergency")
def _drift_hypertensive_emergency(h, severity, dt_s):
    if severity < 0.9:
        return


@pathology_rule("preeclampsia")
def _drift_preeclampsia(h, severity, dt_s):
    if severity < 0.9:
        return


@pathology_rule("spinal_cord_injury")
def _drift_spinal_cord_injury(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.SVR_index = _clip(h.SVR_index * (1 - 0.015 * dt_min), 0.2, 4.0)


@pathology_rule("neonatal_respiratory_distress")
def _drift_neonatal_rds(h, severity, dt_s):
    # Pure-wait obs (n=3): mixed direction. Leave neutral — paradoxical
    # improvement seen in some pairs (post-surfactant phase) isn't safe
    # to encode as drift without knowing the resuscitation context.
    return


# ======================================================================
# Default drift — registered for pathologies without a specific rule
# ======================================================================

def _default_drift(h, severity, dt_s):
    """Fallback for pathologies without a specific drift rule.

    Phase 4 decision: no-op. Per-rule impact tests on the 249 pure-wait
    pairs show uniform drift on long-tail pathologies is net-negative —
    pairs without hand-authored rules are dominated by author-stable
    observation windows. Re-enable if Phase 5+ provides a mechanism
    (e.g. state-conditional activation) to make this safe.
    """
    return


# Name-pinned export used by `engine.py`.
DEFAULT_DRIFT = _default_drift
