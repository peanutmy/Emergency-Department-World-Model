# M14v Provider Verbal Style Policy Handoff

## Summary

Updated clinician and nurse verbal prompting to use a style-dependent verbal
policy instead of clinician/nurse traits or default reassurance.

## Change

Changed:

- `ed_world_model/agents/prompting.py`
- `ed_world_model/agents/clinician.py`
- `ed_world_model/agents/nurse.py`

The new shared policy says clinician/nurse verbal style may be warm, neutral,
blunt, terse, rushed, emotionally cold, or mildly impatient when plausible.
Reassurance is optional and should not be the default.

Clinical competence remains required: prompts still forbid inventing facts,
orders, diagnoses, diagnostic results, or unsupported rationale. The policy also
forbids abusive, mocking, or cartoonishly cruel speech.

Clinician/nurse prompts now treat this as generation guidance, not as
AgentProfile traits. If a clinician/nurse profile is present, prompt rendering
omits `traits` so provider style is not driven by clinician/nurse trait fields.
Patient/relative profile traits remain visible in the generic prompt path.

## Verification

Ran:

```bash
python -m py_compile ed_world_model/agents/prompting.py ed_world_model/agents/clinician.py ed_world_model/agents/nurse.py
```

Rendered prompt smoke checks confirmed:

- clinician Stage 2 prompt includes `Clinical verbal style policy`
- nurse prompt includes `Clinical verbal style policy`
- clinician/nurse rendered prompts do not include `Profile traits`
- clinician/nurse rendered prompts do not leak profile trait keys/values
- patient rendered prompts still include profile traits

Ran:

```bash
python -m pytest tests/test_clinician_agent.py tests/test_nurse_agent.py tests/test_agent_prompting.py
```

Result: 99 passed, 4 failed. The failures are old prompt-text assertions that
still expect clinician/nurse prompts to expose `Profile traits` or instruct
nurse output to reflect `AgentProfile traits`. Those expectations conflict with
the new no-provider-traits verbal policy and were not edited in this session.
