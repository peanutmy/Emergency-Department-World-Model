# M2a Handoff

## Files Created or Modified

- `ed_world_model/actions/registry.py`
  - Added `ActionFamily` string constants for the six clinician-facing action
    families.
  - Added `KindHint` string constants for the 33 canonical engine-facing
    `kind_hint` values.
  - Added frozen `ActionDefinition` dataclass with:
    - `kind_hint`
    - `family`
    - `params_template`
    - `param_options`
    - `param_guidance`
    - `clinician_selectable`
    - optional `description`
  - Added canonical params templates for:
    - `oxygen_support`
    - `airway_management`
    - `fluid_bolus`
    - `blood_transfusion`
    - medication-like actions
    - procedure-like actions
    - `no_action`
  - Added `ActionRegistry` lookup helpers:
    - `list_families`
    - `list_kind_hints`
    - `list_kind_hints_for_family`
    - `get_definition`
    - `get_params_template`
    - `get_param_options`
    - `get_param_guidance`
    - `is_known_kind_hint`
    - `is_known_family`
  - Registry methods only expose taxonomy data, params templates,
    prompt-guidance options, and open-field guidance. They do not validate
    proposals, mutate state, generate actions, or run engines.
- `tests/test_action_registry.py`
  - Added 23 focused tests covering the M2a registry boundary, kind-hint count,
    family coverage, params templates, prompt-guidance options, open-field
    guidance, unknown lookup errors, `raw_text` absence, and no imports from
    state or transition engines.
- `docs/milestones/M2a_handoff.md`
  - Added this handoff.

## Files Intentionally Not Touched

- `docs/v1_3_1.md`
- `ed_world_model/state/state_manager.py`
- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `pdf/`
- `out/`
- existing engine code
- existing test files

No ActionValidator, Orchestrator, ObservationBuilder, adapters, agents, or turn
loop code was implemented.

## Tests Run and Results

Latest M2a patch focused run:

```bash
python -m pytest tests/test_action_registry.py
```

Result: passed.

```text
23 passed in 0.03s
```

Initial M2a adjacent state run before the param-options patch:

```bash
python -m pytest tests/test_global_state.py tests/test_state_manager.py tests/test_action_registry.py
```

Result: passed.

```text
56 passed in 0.10s
```

Broad pytest was not run because M2a only changes ActionRegistry data and
focused registry tests passed.

## Deviations From `docs/v1_3_1.md`

- No behavioral deviations were intentionally introduced.
- The design doc specifies hierarchical action selection and engine-facing
  `kind_hint` values, but it does not list the full params-template assignment
  for each `kind_hint`. M2a follows the milestone-provided grouping:
  specialized templates for oxygen, airway management, fluids, blood
  transfusion, and `no_action`; medication-like templates for medication class
  actions; and procedure-like templates for procedure/device/energy actions.
- M2a patch adds `param_options` only for menu-like discrete fields. Numeric or
  context-dependent fields use `param_guidance` instead of fixed option lists.
- `airway_management.params_template` now contains `stage`,
  `rsi_medication_used`, `bvm_before`, and `intubated`; the earlier
  `procedure` field was removed by M2a patch decision.
- `no_action` remains in the registry as an engine-facing/system-generated
  action and is marked `clinician_selectable = False`.
- Canonical ActionRegistry family names are `time_progression`,
  `respiratory_support`, `circulation_hemodynamics`, `cardiac_rhythm`,
  `medication_toxicology_metabolic`, and `procedures`. These family names are
  intentionally broader than `kind_hint` values. The example in
  `docs/v1_3_1.md` that pairs `family: "oxygen_support"` with
  `kind_hint: "oxygen_support"` should be treated as illustrative, not
  canonical. Downstream clinician proposals should use
  `family: "respiratory_support"` with `kind_hint: "oxygen_support"`.
  `no_action` remains engine-facing/system-generated and is not
  clinician-selectable.
- `ActionRegistry.get_definition(...)` raises `ValueError` for an unknown
  `kind_hint` with a clear message. This is lookup error reporting only, not
  proposal validation.

## Known Limitations

- `ActionRegistry` is data + lookup only.
- `param_options` are prompt guidance only and are not validation constraints in
  v1.3.1.
- It does not validate clinician proposals, role permissions, one-action-per
  turn, diagnostic test availability, or params compatibility beyond canonical
  lookup presence.
- It does not create runtime action proposals or engine-facing action payloads.
- It does not generate or store `raw_text`.
- It does not mutate `GlobalState`.
- It does not import or call transition engines.

## Exact Next Step

M2b ActionValidator.
