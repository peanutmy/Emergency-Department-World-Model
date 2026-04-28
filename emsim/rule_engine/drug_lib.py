"""
Drug-class pharmacodynamic templates and runtime helpers (Phase 6).

Strategy (ENGINE_DESIGN.md §5): the 54 drugs in the corpus are grouped into
~12 pharmacologic classes. Every class has a shared PD template (a list of
``{target, magnitude, half_life_s}`` effects and a common ``onset_s``).
Per-drug standard doses and occasional per-drug overrides fill the gap.

Runtime contract (L1 stateless step):
  - ``start_drug(h, action)`` schedules one ``DrugEffect`` per class-template
    entry (plus any overrides) onto ``h.active_drug_effects`` when the engine
    fires a drug action at t=0. Magnitude scales linearly with
    ``action.dose / standard_dose`` (clipped to [0.1, 3.0]).
  - ``tick(h, dt_s, t_sim_s)`` advances each effect's PK clock by ``dt_s`` and
    applies the delta in contribution (``contribution_at`` on the age) to the
    target hidden var.

Within a single ``step()`` call the engine decodes a fresh HiddenState from
``before`` (no drug memory from previous pairs), fires all drug actions at
t=0, and ticks PD over ``duration_s``. Cross-pair drug memory is deferred to
L2 Session (ENGINE_DESIGN §6.5) — it lives on engine-internal HiddenState,
not in the corpus.

Magnitudes are author-initial starting points per §4.1; Phase 6 calibration
fits them against single-drug pair subsets. All class templates tag the
confidence level and the clinical reference (ACLS / anesthesia text).
"""
from __future__ import annotations

import warnings
from typing import Any

from .hidden_state import DrugEffect, HiddenState
from .mapping import clip


# =====================================================================
# Target-variable clip ranges
# =====================================================================
# Mirrors the residual-widened bounds used by io._residual_correct so drug
# PD can't leave hidden state in a range the decoder refuses to reconstruct.
# `chronotropic_drive` range widened on the negative end so sedation /
# bradycardic pushes can still compose with low-baseline patients.

_VAR_CLIP: dict[str, tuple[float, float]] = {
    "CO_index":           (0.05, 4.0),
    "SVR_index":          (0.2,  4.0),
    "preload_index":      (0.0,  1.8),
    "chronotropic_drive": (-4.0, 10.0),
    "PaO2_effective":     (0.0,  600.0),
    "shunt_fraction":     (0.0,  0.8),
    "compliance_index":   (0.2,  1.2),
    "ventilatory_drive":  (0.0,  5.0),
    "core_temp_trend":    (-0.2, 0.2),
    "consciousness":      (0.0,  1.0),
}


def _clip_var(var: str, value: float) -> float:
    lo, hi = _VAR_CLIP.get(var, (-1e9, 1e9))
    return clip(value, lo, hi)


# =====================================================================
# Class templates: {class_name: {"onset_s": int, "effects": [{target, magnitude, half_life_s}, ...]}}
# =====================================================================
# Magnitudes are author-initial; calibrated against drug-only single-action
# pair subsets in Phase 6. Half-lives are scenario-paced (per ENGINE_DESIGN
# §5): author-scale durations that match 60-600s observation windows, not
# ICU-published pharmacologic half-lives (which would barely move over a
# scenario pair). `half_life_s` represents "time until contribution halves"
# *within the scenario's narrative cadence*.

DRUG_CLASSES: dict[str, dict[str, Any]] = {
    # -----------------------------------------------------------------
    # Tier 1 — cover the top-10 drugs (60% of 214 uses)
    # -----------------------------------------------------------------

    # Mixed alpha/beta pressor: epi (ACLS push, anaphylaxis IM).
    # 30+ of 35 corpus epi pairs are arrest (output always 0 via arrest
    # rhythm branch). The few non-arrest pairs (anaphylaxis IM) show modest
    # HR/BP rises. Keep magnitudes small and narrowly indicated.
    "pressor_alpha_beta": {
        "onset_s": 60,
        "effects": [
            {"target": "chronotropic_drive", "magnitude": +0.25, "half_life_s": 300},
            {"target": "SVR_index",          "magnitude": +0.08, "half_life_s": 300},
        ],
    },

    # Pure alpha pressor: norepi, phenylephrine, vasopressin.
    # Author BP deltas vary widely (0-30). Small magnitude keeps us within
    # BP deadzone on stable pairs while still nudging on severe shock.
    "pressor_alpha_pure": {
        "onset_s": 60,
        "effects": [
            {"target": "SVR_index",          "magnitude": +0.06, "half_life_s": 1200},
        ],
    },

    # Beta-1 inotrope: dobutamine, milrinone.
    "inotrope_beta": {
        "onset_s": 180,
        "effects": [
            {"target": "CO_index",           "magnitude": +0.15, "half_life_s": 1800},
            {"target": "chronotropic_drive", "magnitude": +0.25, "half_life_s": 1800},
        ],
    },

    # GABAergic sedative: propofol, midazolam, etomidate. `consciousness` has
    # no direct vital path (RR lost its consciousness factor in Phase 3a),
    # so dropping it is a free observation. HR/BP via SVR only — drops RR
    # directly caused persistent author-stable regressions, so omitted.
    "sedative": {
        "onset_s": 120,
        "effects": [
            {"target": "consciousness",      "magnitude": -0.4,  "half_life_s": 1800},
            {"target": "SVR_index",          "magnitude": -0.18, "half_life_s": 1800},
        ],
    },

    # Neuromuscular blocker: rocuronium, succinylcholine, vecuronium.
    # Corpus drug-only paralytic pairs don't co-register the ventilator,
    # so authors often show RR unchanged despite physiology. Keep tiny
    # magnitude — effect is meaningful only when paired with intubate in
    # mixed pairs (which decodes into carried drugs on subsequent pairs).
    "paralytic": {
        "onset_s": 60,
        "effects": [
            {"target": "ventilatory_drive",  "magnitude": -0.2, "half_life_s": 1800},
        ],
    },

    # Class III antiarrhythmic: amiodarone, sotalol.
    # Author-stable dominant: 5/7 baseline amiodarone pairs pass via pass-
    # through. Keep magnitude tiny so deadzone stays respected — this drug
    # in the corpus is mostly used as an arrest-pair marker (monitor still
    # 0 HR/BP during arrest regardless of PD).
    "antiarrhythmic_3": {
        "onset_s": 240,
        "effects": [
            {"target": "chronotropic_drive", "magnitude": -0.1, "half_life_s": 3600},
        ],
    },

    # Vagolytic: atropine, glycopyrrolate. Corpus atropine is ~90% author-
    # stable (drug as brady marker) and 10% huge response (HR 35→110 on
    # procedural sedation) — no middle ground. Tiny magnitude keeps us
    # within deadzone on stable pairs; the responder is unrecoverable
    # without a specific override.
    "vagolytic": {
        "onset_s": 60,
        "effects": [
            {"target": "chronotropic_drive", "magnitude": +0.1, "half_life_s": 1800},
        ],
    },

    # Electrolyte correction: Ca, Mg, NaHCO3, hypertonic saline.
    # Most electrolyte drug-only pairs are author-stable (sent to mark the
    # dose without a visible vital change). Keep base PD near zero; specific
    # reversal drugs override for their clinical scenario.
    "electrolyte": {
        "onset_s": 180,
        "effects": [],
    },

    # -----------------------------------------------------------------
    # Tier 2 — less frequent but clinically important
    # -----------------------------------------------------------------

    # Opioid: fentanyl, morphine, hydromorphone.
    "opioid": {
        "onset_s": 240,
        "effects": [
            {"target": "consciousness",      "magnitude": -0.15, "half_life_s": 2400},
            {"target": "ventilatory_drive",  "magnitude": -0.15, "half_life_s": 2400},
            {"target": "chronotropic_drive", "magnitude": -0.1,  "half_life_s": 2400},
        ],
    },

    # Beta-2 bronchodilator: salbutamol, ipratropium (nebulized combo).
    "bronchodilator": {
        "onset_s": 180,
        "effects": [
            {"target": "shunt_fraction",     "magnitude": -0.04, "half_life_s": 2400},
            {"target": "chronotropic_drive", "magnitude": +0.15, "half_life_s": 2400},
        ],
    },

    # Anticonvulsant: phenytoin, levetiracetam. Main clinical effect is
    # stopping seizures (handled at tipping/pathology level, not PD).
    "anticonvulsant": {
        "onset_s": 300,
        "effects": [],
    },

    # Reversal agents: naloxone, flumazenil, digoxin_immune_fab, hydroxocobalamin.
    # Author naloxone pairs often show RR unchanged despite bolus dosing.
    # `consciousness` (no vital path) is safe to touch; ventilatory_drive
    # must stay small to respect the dead-zone.
    "reversal": {
        "onset_s": 120,
        "effects": [
            {"target": "consciousness",      "magnitude": +0.3,  "half_life_s": 1200},
            {"target": "ventilatory_drive",  "magnitude": +0.06, "half_life_s": 1200},
        ],
    },

    # Rate-controlling beta-blocker: esmolol, labetalol, metoprolol.
    "beta_blocker": {
        "onset_s": 120,
        "effects": [
            {"target": "chronotropic_drive", "magnitude": -0.4,  "half_life_s": 1200},
            {"target": "SVR_index",          "magnitude": -0.05, "half_life_s": 1200},
        ],
    },

    # Vasodilator (nitro SL/drip, hydralazine).
    "vasodilator": {
        "onset_s": 120,
        "effects": [
            {"target": "SVR_index",          "magnitude": -0.12, "half_life_s": 900},
            {"target": "preload_index",      "magnitude": -0.03, "half_life_s": 900},
        ],
    },

    # Diuretic: furosemide. Preload drop over minutes (slow onset).
    "diuretic": {
        "onset_s": 900,
        "effects": [
            {"target": "preload_index",      "magnitude": -0.08, "half_life_s": 3600},
        ],
    },

    # Neutral (monitored-only drugs: ASA, heparin, antibiotics, thrombolytics).
    # These have real mechanisms (antiplatelet, anticoagulant, fibrinolytic)
    # but no immediate hidden-state vital signature over scenario timescales.
    # Keep an empty template so the name maps cleanly to a class.
    "neutral": {
        "onset_s": 60,
        "effects": [],
    },
}


# =====================================================================
# Drug name → class
# =====================================================================
# 54 drugs total. Route / dose discrepancies are resolved by DRUG_OVERRIDES.

DRUG_TO_CLASS: dict[str, str] = {
    # Pressors / inotropes
    "epinephrine":            "pressor_alpha_beta",
    "norepinephrine":         "pressor_alpha_pure",
    "phenylephrine":          "pressor_alpha_pure",
    "dobutamine":             "inotrope_beta",

    # Vagolytic / antiarrhythmic
    "atropine":               "vagolytic",
    "amiodarone":             "antiarrhythmic_3",
    "procainamide":           "antiarrhythmic_3",
    "adenosine":              "antiarrhythmic_3",    # overridden (transient)

    # Sedatives / analgesics / paralytics
    "propofol":               "sedative",
    "midazolam":              "sedative",
    "ketamine":               "sedative",            # overridden (sympathomimetic)
    "lorazepam":              "sedative",
    "fentanyl":               "opioid",
    "hydromorphone":          "opioid",
    "rocuronium":             "paralytic",
    "succinylcholine":        "paralytic",

    # Beta-blockers (rate control)
    "esmolol":                "beta_blocker",
    "labetalol":              "beta_blocker",

    # Vasodilators
    "nitroglycerin":          "vasodilator",
    "hydralazine":            "vasodilator",

    # Diuretic
    "furosemide":             "diuretic",

    # Electrolyte / metabolic
    "calcium_gluconate":      "electrolyte",
    "calcium_chloride":       "electrolyte",
    "magnesium":              "electrolyte",
    "sodium_bicarbonate":     "electrolyte",
    "hypertonic_saline_3pct": "electrolyte",
    "dextrose_50":            "electrolyte",          # overridden (hypoglycemia reversal)
    "dextrose_10":            "electrolyte",          # overridden
    "insulin_regular":        "electrolyte",          # overridden (hyperK / DKA)
    "thiamine":               "neutral",
    "mannitol":               "diuretic",

    # Bronchodilator
    "salbutamol":             "bronchodilator",

    # Anticonvulsants
    "phenytoin":              "anticonvulsant",
    "levetiracetam":          "anticonvulsant",

    # Reversal / antidotes
    "naloxone":               "reversal",
    "digoxin_immune_fab":     "reversal",
    "hydroxocobalamin":       "reversal",
    "glucagon":               "reversal",            # overridden (beta-blocker tox reversal)
    "pralidoxime":            "reversal",            # overridden (organophosphate)
    "fomepizole":             "neutral",             # antidote for toxic alcohols, slow
    "deferoxamine":           "neutral",             # iron chelator, slow
    "lipid_emulsion_20":      "reversal",            # overridden (LAST rescue)
    "flumazenil":             "reversal",

    # Steroids
    "hydrocortisone":         "neutral",             # overridden for adrenal_crisis context
    "methylprednisolone":     "neutral",

    # Coagulation / fibrinolytic / antiplatelet
    "heparin":                "neutral",
    "TXA":                    "neutral",
    "alteplase":              "neutral",
    "tenecteplase":           "neutral",
    "ASA":                    "neutral",
    "rVIII":                  "neutral",

    # Antibiotics
    "piperacillin_tazobactam":"neutral",

    # OB-GYN
    "oxytocin":               "neutral",             # uterotonic, no vital signature
    "carboprost":             "neutral",

    # Pediatric ductal patency
    "prostaglandin_E1":       "neutral",
}


# =====================================================================
# Per-drug standard dose
# =====================================================================
# (dose, unit) is the canonical ED "first dose" — used as denominator for
# linear dose scaling. Units listed match the corpus extraction; scaling
# is only applied when action.unit matches.

DRUG_STANDARD_DOSE: dict[str, tuple[float, str]] = {
    "epinephrine":            (1.0,   "mg"),
    "norepinephrine":         (0.1,   "mcg/kg/min"),
    "phenylephrine":          (100,   "mcg"),
    "dobutamine":             (5.0,   "mcg/kg/min"),
    "atropine":               (0.5,   "mg"),
    "amiodarone":             (300,   "mg"),
    "procainamide":           (50,    "mg/min"),
    "adenosine":              (6,     "mg"),
    "propofol":               (150,   "mg"),
    "midazolam":              (2.0,   "mg"),
    "ketamine":               (100,   "mg"),
    "lorazepam":              (2.0,   "mg"),
    "fentanyl":               (100,   "mcg"),
    "hydromorphone":          (1.0,   "mg"),
    "rocuronium":             (80,    "mg"),
    "succinylcholine":        (120,   "mg"),
    "esmolol":                (500,   "mcg/kg"),
    "labetalol":              (20,    "mg"),
    "nitroglycerin":          (0.4,   "mg"),
    "hydralazine":            (10,    "mg"),
    "furosemide":             (40,    "mg"),
    "calcium_gluconate":      (1,     "g"),
    "calcium_chloride":       (1,     "g"),
    "magnesium":              (2,     "g"),
    "sodium_bicarbonate":     (50,    "mEq"),
    "hypertonic_saline_3pct": (250,   "ml"),
    "dextrose_50":            (25,    "g"),
    "dextrose_10":            (2.5,   "g"),
    "insulin_regular":        (10,    "U"),
    "mannitol":               (50,    "g"),
    "salbutamol":             (5.0,   "mg"),
    "phenytoin":              (1000,  "mg"),
    "levetiracetam":          (1000,  "mg"),
    "naloxone":               (0.4,   "mg"),
    "digoxin_immune_fab":     (5,     "vials"),
    "hydroxocobalamin":       (5,     "g"),
    "glucagon":               (3.0,   "mg"),
    "pralidoxime":            (2000,  "mg"),
    "flumazenil":             (0.2,   "mg"),
    "lipid_emulsion_20":      (100,   "mL"),
    "hydrocortisone":         (100,   "mg"),
    "methylprednisolone":     (125,   "mg"),
    # neutral-class drugs still benefit from dose normalization for audit.
    "heparin":                (5000,  "U"),
    "TXA":                    (1,     "g"),
    "alteplase":              (90,    "mg"),
    "tenecteplase":           (50,    "mg"),
    "ASA":                    (160,   "mg"),
    "oxytocin":               (10,    "U"),
    "carboprost":             (0.25,  "mg"),
    "prostaglandin_E1":       (0.1,   "mcg/kg/min"),
    "thiamine":               (100,   "mg"),
    "fomepizole":             (1000,  "mg"),
    "deferoxamine":           (1000,  "mg/hr"),
    "piperacillin_tazobactam":(4.5,   "g"),
    "rVIII":                  (2500,  "U"),
}


# =====================================================================
# Per-drug overrides
# =====================================================================
# Entry keys:
#   - "extra_effects": appended to the class template's effects list
#   - "replace_effects": fully replaces the class template's effects list
#   - "onset_s": override class default for this drug only
#   - "unit": override canonical unit (if class-based scaling would mis-scale)
#
# Keep this short — the class template should handle the common case.

DRUG_OVERRIDES: dict[str, dict[str, Any]] = {
    # Dissociative anesthetic: sedates but sympathomimetic (BP/HR up).
    # Keep the class sedative effects (consciousness/ventilatory_drive drop)
    # but offset the BP/HR drops with sympathomimetic extras.
    "ketamine": {
        "extra_effects": [
            {"target": "chronotropic_drive", "magnitude": +0.35, "half_life_s": 1200},
            {"target": "SVR_index",          "magnitude": +0.12, "half_life_s": 1200},
        ],
    },

    # Adenosine: transient AV block (~6 s) induces brief pause then sinus
    # conversion in SVT. Short-lived chrono negative.
    "adenosine": {
        "replace_effects": [
            {"target": "chronotropic_drive", "magnitude": -1.5, "half_life_s": 20},
        ],
        "onset_s": 5,
    },

    # Glucagon: rescue therapy in beta-blocker OD → chrono up via cAMP bypass.
    "glucagon": {
        "replace_effects": [
            {"target": "chronotropic_drive", "magnitude": +0.4, "half_life_s": 1800},
            {"target": "CO_index",           "magnitude": +0.1, "half_life_s": 1800},
        ],
    },

    # Digoxin_immune_fab: reverses digoxin toxicity (brady/AV block).
    "digoxin_immune_fab": {
        "replace_effects": [
            {"target": "chronotropic_drive", "magnitude": +0.3, "half_life_s": 1800},
            {"target": "CO_index",           "magnitude": +0.1, "half_life_s": 1800},
        ],
    },

    # Hydroxocobalamin: cyanide antidote → restores ox-phos → BP/HR normalize.
    "hydroxocobalamin": {
        "replace_effects": [
            {"target": "CO_index",           "magnitude": +0.15, "half_life_s": 1800},
            {"target": "SVR_index",          "magnitude": +0.1,  "half_life_s": 1800},
        ],
    },

    # Lipid emulsion for LAST: restores cardiac function via lipid sink.
    "lipid_emulsion_20": {
        "replace_effects": [
            {"target": "CO_index",           "magnitude": +0.15, "half_life_s": 1800},
            {"target": "chronotropic_drive", "magnitude": +0.25, "half_life_s": 1800},
        ],
    },

    # Flumazenil: benzo reversal specifically.
    "flumazenil": {
        "replace_effects": [
            {"target": "consciousness",      "magnitude": +0.3, "half_life_s": 900},
            {"target": "ventilatory_drive",  "magnitude": +0.2, "half_life_s": 900},
        ],
    },

    # Hydrocortisone in adrenal crisis → slow SVR restoration.
    "hydrocortisone": {
        "replace_effects": [
            {"target": "SVR_index",          "magnitude": +0.08, "half_life_s": 3600},
        ],
        "onset_s": 900,
    },

    # Insulin (hyperK / DKA): drives K intracellular → brady correction.
    "insulin_regular": {
        "replace_effects": [
            {"target": "chronotropic_drive", "magnitude": +0.1, "half_life_s": 1800},
        ],
    },

    # Dextrose: reverses hypoglycemia → consciousness restoration.
    "dextrose_50": {
        "replace_effects": [
            {"target": "consciousness",      "magnitude": +0.3, "half_life_s": 1800},
        ],
    },
    "dextrose_10": {
        "replace_effects": [
            {"target": "consciousness",      "magnitude": +0.15, "half_life_s": 1800},
        ],
    },

    # NaHCO3, calcium: baseline (pass-through) was 71% / 22% — author-
    # stable-dominated. Any chrono bump regresses the stable majority.
    # Leave as neutral electrolyte class (empty effects).

    # Prostaglandin E1: keeps ductus open in neonate — improves mixing/O2.
    "prostaglandin_E1": {
        "replace_effects": [
            {"target": "shunt_fraction",     "magnitude": -0.08, "half_life_s": 3600},
            {"target": "PaO2_effective",     "magnitude": +8,    "half_life_s": 3600},
        ],
    },

    # Oxytocin in PPH: uterine contraction stops hemorrhage.
    "oxytocin": {
        "replace_effects": [
            {"target": "preload_index",      "magnitude": +0.05, "half_life_s": 1800},
        ],
    },

    # Mannitol: osmotic diuretic for elevated ICP.
    "mannitol": {
        "replace_effects": [
            {"target": "preload_index",      "magnitude": +0.05, "half_life_s": 1800},
        ],
    },

    # Pralidoxime (organophosphate): reactivates ACh-E → HR normalizes.
    "pralidoxime": {
        "replace_effects": [
            {"target": "chronotropic_drive", "magnitude": +0.2, "half_life_s": 1800},
        ],
    },
}


# =====================================================================
# Start / tick
# =====================================================================

_warned_drugs: set[str] = set()


def _resolve_effects(name: str) -> tuple[int, list[dict[str, Any]]]:
    """Return (onset_s, effects_list) for a drug, applying class + override."""
    cls = DRUG_TO_CLASS.get(name)
    if cls is None:
        if name not in _warned_drugs:
            _warned_drugs.add(name)
            warnings.warn(
                f"drug_lib: drug {name!r} has no class mapping — registering as neutral",
                RuntimeWarning,
                stacklevel=3,
            )
        return 60, []
    template = DRUG_CLASSES.get(cls, {"onset_s": 60, "effects": []})
    onset_s = int(template.get("onset_s", 60))
    effects = list(template.get("effects", []))

    override = DRUG_OVERRIDES.get(name, {})
    if override:
        if "onset_s" in override:
            onset_s = int(override["onset_s"])
        # replace_effects fully replaces the class template's effects
        if override.get("replace_effects"):
            effects = list(override["replace_effects"])
        # extra_effects append to the current list (after replace, if any)
        if override.get("extra_effects"):
            effects = effects + list(override["extra_effects"])
    return onset_s, effects


def _dose_ratio(action: dict[str, Any]) -> float:
    """Linear dose scaling ratio, clipped to [0.1, 3.0]. Returns 1.0 when
    the action has no dose or the unit mismatches the standard."""
    name = action.get("name", "")
    dose = action.get("dose")
    unit = action.get("unit", "")
    standard = DRUG_STANDARD_DOSE.get(name)
    if not standard or dose is None:
        return 1.0
    std_dose, std_unit = standard
    if unit and std_unit and unit != std_unit:
        # Unit mismatch — keep unscaled; avoids accidentally 1000× scaling
        # mcg-vs-mg mistakes. Logs once per drug+unit pair.
        key = f"{name}:{unit}vs{std_unit}"
        if key not in _warned_drugs:
            _warned_drugs.add(key)
            warnings.warn(
                f"drug_lib: {name} dose unit {unit!r} doesn't match standard "
                f"{std_unit!r} — skipping dose scaling",
                RuntimeWarning,
                stacklevel=3,
            )
        return 1.0
    try:
        d = float(dose)
    except (TypeError, ValueError):
        return 1.0
    if std_dose <= 0:
        return 1.0
    return clip(d / std_dose, 0.1, 3.0)


# =====================================================================
# Clinical-indication gating
# =====================================================================
# Drug PD in the corpus only reliably matches authored vitals when the
# patient is in the physiologic state the drug is clinically indicated
# for. Pressors given to a normotensive patient show as author-stable;
# atropine given to a non-bradycardic patient shows no HR bump. Firing
# the full PD on all pairs regresses author-stable pairs out of the
# direction dead-zone. Gating per `(target_var, sign)` keeps the effect
# silent when the indication isn't present.

def _gate_scale(target_var: str, magnitude: float, h: HiddenState) -> float:
    """Return multiplier in [0, 1] for how *indicated* this effect is.

    Binary-ish gating with a short transition band. Drugs fire at full
    magnitude only when the target var is in the "indication zone"
    (clinically abnormal in the direction the drug corrects); fire at
    small scale in a narrow transition band; silent otherwise. Prevents
    author-stable pairs (drug as narrative marker) from crossing the
    direction dead-zone.
    """
    if magnitude == 0.0:
        return 0.0

    def _ramp(val: float, full_at: float, none_at: float) -> float:
        if full_at < none_at:
            if val <= full_at: return 1.0
            if val >= none_at: return 0.0
            return (none_at - val) / (none_at - full_at)
        if val >= full_at: return 1.0
        if val <= none_at: return 0.0
        return (val - none_at) / (full_at - none_at)

    # --- chronotropic_drive ---
    # Stricter thresholds: fire only on clearly abnormal patients. Authors
    # frequently give pressors/vagolytics with modest HR and expect no
    # noticeable delta — keeping the ramp narrow preserves those.
    if target_var == "chronotropic_drive":
        if magnitude > 0:          # pressors, vagolytics, reversals
            return _ramp(h.chronotropic_drive, -1.0, -0.2)
        return _ramp(h.chronotropic_drive, 1.5, 0.5)       # beta-blockers, adenosine

    # --- SVR_index ---
    if target_var == "SVR_index":
        if magnitude > 0:           # pressors
            return _ramp(h.SVR_index, 0.8, 1.0)
        return _ramp(h.SVR_index, 1.3, 1.05)               # vasodilators / sedative BP drop

    # --- CO_index ---
    if target_var == "CO_index":
        if magnitude > 0:            # inotropes, epi
            return _ramp(h.CO_index, 0.7, 1.0)
        return _ramp(h.CO_index, 1.2, 0.95)

    # --- preload_index ---
    if target_var == "preload_index":
        if magnitude > 0:
            return _ramp(h.preload_index, 0.7, 1.0)
        return _ramp(h.preload_index, 1.3, 0.9)             # diuretics

    # --- ventilatory_drive ---
    if target_var == "ventilatory_drive":
        if magnitude > 0:            # reversals (naloxone, flumazenil)
            return _ramp(h.ventilatory_drive, 0.3, 0.7)
        return _ramp(h.ventilatory_drive, 1.0, 0.5)        # sedatives, opioids, paralytics

    # --- consciousness (no direct vital path — free) ---
    if target_var == "consciousness":
        if magnitude > 0:
            return _ramp(h.consciousness, 0.3, 0.75)
        return _ramp(h.consciousness, 0.85, 0.4)

    # --- PaO2_effective, shunt_fraction ---
    if target_var == "PaO2_effective":
        if magnitude > 0:
            return _ramp(h.PaO2_effective, 60.0, 90.0)
        return _ramp(h.PaO2_effective, 200.0, 70.0)
    if target_var == "shunt_fraction":
        if magnitude < 0:
            return _ramp(h.shunt_fraction, 0.3, 0.1)
        return _ramp(h.shunt_fraction, 0.05, 0.3)

    return 1.0


def _enqueue_effects(h: HiddenState,
                     name: str,
                     t_start_s: float,
                     dose_ratio: float) -> None:
    """Schedule one DrugEffect per class-template entry onto h.active_drug_effects."""
    onset_s, effects = _resolve_effects(name)
    if not effects:
        # Still enqueue a single book-keeping entry so round-trip over
        # h.active_drug_effects doesn't drop the drug silently.
        h.active_drug_effects.append(
            DrugEffect(
                target_var="",
                magnitude=0.0,
                t_start_s=t_start_s,
                half_life_s=0.0,
                onset_s=onset_s,
                drug_name=name,
                _prev_contrib=0.0,
            )
        )
        return
    for eff in effects:
        target = eff["target"]
        base_mag = eff["magnitude"] * dose_ratio
        # Graded clinical-indication gate: scale magnitude by how indicated
        # this effect is given current hidden state. Prevents author-stable
        # pairs (drug given as a "marker" without authored vital change)
        # from crossing the direction dead-zone.
        gate = _gate_scale(target, base_mag, h)
        if gate <= 0.0:
            continue
        mag = base_mag * gate
        effect = DrugEffect(
            target_var=target,
            magnitude=mag,
            t_start_s=t_start_s,
            half_life_s=float(eff.get("half_life_s", 600.0)),
            onset_s=onset_s,
            drug_name=name,
        )
        h.active_drug_effects.append(effect)


def start_drug(h: HiddenState,
               action: dict[str, Any],
               t_sim_s: float = 0.0) -> None:
    """Schedule PD effects for a newly-administered drug (age = 0 at t_sim_s).

    Called from the engine's t=0 action loop. Multi-effect class templates
    expand into multiple ``DrugEffect`` entries, each tracking its own target
    var and half-life. Magnitudes scale linearly with
    ``action.dose / DRUG_STANDARD_DOSE[name]`` (clipped [0.1, 3.0]).
    """
    name = action.get("name", "")
    if not name:
        return
    ratio = _dose_ratio(action)
    _enqueue_effects(h, name, t_start_s=float(t_sim_s), dose_ratio=ratio)


import os as _os
_DRUG_PD_DISABLED = _os.environ.get("EMSIM_DISABLE_DRUG_PD", "0") == "1"


def tick(h: HiddenState, dt_s: float, t_sim_s: float = 0.0) -> None:
    """Advance PD integration by ``dt_s`` seconds. Apply delta contributions
    onto each effect's target hidden var.

    Integrator guarantee: hidden state after the tick equals
    ``hidden_before_tick + Σ (contribution_at(new_age) - _prev_contrib)``
    for every effect, clipped to the var's physiologic range. Effects past
    5 half-lives contribute 0 but remain in the list (design freeze: drugs
    are never removed within a pair so L2 Session can later persist them
    across pair boundaries).

    Set ``EMSIM_DISABLE_DRUG_PD=1`` to bypass all PD (keeps drug plumbing
    alive but does not mutate hidden state); useful for A/B isolating drug
    PD effects from the drift-composition gate.
    """
    if _DRUG_PD_DISABLED:
        return
    if not h.active_drug_effects or dt_s <= 0:
        return
    new_t = t_sim_s + dt_s
    for effect in h.active_drug_effects:
        if not effect.target_var or effect.magnitude == 0.0:
            continue
        new_age = new_t - effect.t_start_s
        new_contrib = effect.contribution_at(new_age)
        delta = new_contrib - effect._prev_contrib
        if delta == 0.0:
            continue
        current = getattr(h, effect.target_var, 0.0)
        setattr(h, effect.target_var,
                _clip_var(effect.target_var, current + delta))
        effect._prev_contrib = new_contrib
