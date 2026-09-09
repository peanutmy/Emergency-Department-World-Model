# M14b Generate Utterances Physiology Mode Handoff

## Scope

This patch changes `generate_utterances.py` so real-agent utterance generation no
longer hardcodes fake physiology.

No scenario JSON, transition engine, adapter, or core turn-loop behavior was
changed.

## Changes

- `generate_utterances.py`
  - Added `--physiology-mode`.
  - Default mode is now `hybrid` for this script.
  - Accepted command-line aliases:
    - `true`, `real`, `yes`, `1` -> `hybrid`
    - `false`, `no`, `0` -> `fake`
  - Both the normal real-agent path and the `--emotion` path pass the selected
    physiology mode through instead of hardcoding `fake`.
  - The `--emotion --output` JSON now records:
    - top-level `physiology_mode`
    - per-turn `active_agents`
    - per-turn `physiology_action_kind_hint`
    - per-turn `physiology_action`
    - per-turn `state_after`
    - per-turn `events`

## Behavior Notes

- The canonical internal mode remains `hybrid`, not boolean `true`.
- `--physiology-mode true` is accepted for convenience and maps to `hybrid`.
- `hybrid` uses the existing `HybridPhysiologyAdapter` through
  `build_demo_physiology_adapter(...)`.
- The separate case-specific BiPAP availability issue remains unresolved by
  this patch. This change only ensures the script can use the real physiology
  adapter instead of fake physiology.

## Verification

```bash
python -m py_compile generate_utterances.py
```

Result: passed.

```bash
python generate_utterances.py --help
```

Result: passed; help shows `--physiology-mode`.

```bash
python - <<'PY'
import generate_utterances as g
for value in ['hybrid', 'true', 'real', '1', 'yes', 'fake', 'false', '0', 'no']:
    print(value, '->', g._normalize_physiology_mode(value))
PY
```

Result:

```text
hybrid -> hybrid
true -> hybrid
real -> hybrid
1 -> hybrid
yes -> hybrid
fake -> fake
false -> fake
0 -> fake
no -> fake
```
