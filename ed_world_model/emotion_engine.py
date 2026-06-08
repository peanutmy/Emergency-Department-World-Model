"""Emotion State Transition Engine.

Two-stage, LLM-driven emotion engine for the ED world model. Converted from
``emotion_engine.ipynb``.

## Design

- **4 emotion dimensions**: ``fear``, ``sadness``, ``overwhelm``, ``shame``.
- **4 ordinal levels**: ``not at all`` < ``low`` < ``high`` < ``extreme``.
  ``not at all`` means the emotion is *genuinely absent* / not engaged at all - it
  is NOT the low end of an intensity scale. Any actual presence of an emotion is
  at least ``low``; ``not at all`` is reserved for "not present".
- **All emotions are tracked**: each agent carries a full four-dimension state
  vector. Any emotion can move independently at each tick.
- **Driver**: only an utterance from *another* agent can move emotions.
  Self-utterance cannot change one's own emotion.

Two LLM calls drive the engine:

1. ``init_emotion(scenario, initial_vitals, agent_traits)`` -> initial
   four-emotion state vector.
2. ``step_emotion(trajectory, utterance, speaker_role)`` -> new four-emotion
   state vector.

## Configuration

    export ANTHROPIC_API_KEY="<your key>"
    export ANTHROPIC_BASE_URL="http://148.113.224.153:3000"   # optional relay

"""
from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Point the Anthropic SDK at the relay the notebook used, unless the caller has
# already chosen a base URL. The API key must come from the environment.
os.environ.setdefault("ANTHROPIC_BASE_URL", "http://148.113.224.153:3000")

# Model the engine calls. Override per-run with the EMOTION_ENGINE_MODEL env var
# (e.g. a model name your relay serves) without editing code.
# NOTE: claude-sonnet-4-20250514 is deprecated (end-of-life 2026-06-15).
DEFAULT_MODEL = os.environ.get("EMOTION_ENGINE_MODEL", "claude-sonnet-4-20250514")


# ---------------------------------------------------------------------------
# 1. Type aliases and constants
# ---------------------------------------------------------------------------

EmotionName = Literal[
    "fear", "sadness", "overwhelm", "shame"
]
EmotionLevel = Literal["not at all", "low", "high", "extreme"]

EMOTION_NAMES: tuple[EmotionName, ...] = (
    "fear", "sadness", "overwhelm", "shame",
)
EMOTION_LEVELS: tuple[EmotionLevel, ...] = ("not at all", "low", "high", "extreme")


# ---------------------------------------------------------------------------
# 2. State models
# ---------------------------------------------------------------------------

class EmotionState(BaseModel):
    """Snapshot of all four emotions for one agent at one tick."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    fear: EmotionLevel = "not at all"
    sadness: EmotionLevel = "not at all"
    overwhelm: EmotionLevel = "not at all"
    shame: EmotionLevel = "not at all"

    def level_of(self, name: EmotionName) -> EmotionLevel:
        return getattr(self, name)

    def summary(self) -> dict[str, str]:
        return {n: self.level_of(n) for n in EMOTION_NAMES}


class EmotionTrajectory(BaseModel):
    """Per-agent emotion state across a trajectory.

    All four emotions are tracked and can change independently at each tick.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    agent_role: str
    state: EmotionState
    history: list[EmotionState] = Field(default_factory=list)
    init_rationale: str | None = None
    last_evidence: str | None = None
    last_reasoning: str | None = None


# ---------------------------------------------------------------------------
# 3. LLM output schemas
# ---------------------------------------------------------------------------

class InitOutput(BaseModel):
    """Schema returned by the init LLM call."""

    model_config = ConfigDict(extra="forbid")

    initial_state: EmotionState
    rationale: str


class StepOutput(BaseModel):
    """Schema returned by the step LLM call.

    Returns the full new emotional state (all four dimensions) plus
    evidence and reasoning.
    """

    model_config = ConfigDict(extra="forbid")

    new_state: EmotionState
    evidence: str
    reasoning: str


# ---------------------------------------------------------------------------
# 4. Prompts
# ---------------------------------------------------------------------------

INIT_PROMPT_TEMPLATE = """\
You are modeling the emotional state of an agent in an Emergency Department \
simulation. Decide the initial intensity of ALL four emotions given the \
scenario and vitals below.

Emotion dimensions (each takes one of: not at all, low, high, extreme):
  - fear: Fear is a response characterized by worries for one's health and safety, triggered by awareness of vulnerability to a threat such as disease, treatment delays, or possible contagion. It can be buffered by empathic, clear doctor-patient communication; when un-buffered it can progress toward hopelessness.
  - sadness: Sadness is an emotional expression of grief, unhappiness, loss, hopelessness, helplessness, or sorrow. It can also be experienced as loneliness, discouragement, rejection, or dissatisfaction with oneself. It is a normal response to distressing life events such as illness, but can also be a harbinger of conditions such as major depression, adjustment disorder, grief, or demoralization.
  - overwhelm: Overwhelm is the phenomenon where decision-making capacity is exceeded by the sheer complexity or volume of information at hand, such that the patient cannot attain the understanding needed for informed consent. It has two distinct subtypes: emotional overwhelm (distress from the illness experience interferes with processing) and informational overload (the information itself is inherently too complex).
  - shame: Shame is a negative, self-conscious emotion that occurs when a person feels themselves to have been seen and judged to be flawed in some crucial way - with some aspect of their identity or selfhood perceived to be inadequate, damaged, inappropriate, or immoral. It is distinct from stigma (which is a label or category) and includes variants such as embarrassment, humiliation, mortification, feelings of defectiveness, and low self-worth.

"not at all" means the emotion is genuinely ABSENT - the scenario does not \
activate it at all. "not at all" is NOT the low end of an intensity scale and \
must NOT be used to mean "a little bit": if the scenario activates an emotion \
to ANY degree, its minimum level is "low". So each emotion is either absent \
("not at all") or present at "low" / "high" / "extreme".

For each emotion, pick the level that matches how strongly the scenario \
activates it BEFORE any dialogue starts. Multiple emotions can be active \
simultaneously; set any emotion that is not activated at all to "not at all".

Return JSON only:
{{
  "initial_state": {{
    "fear": one of {emotion_levels},
    "sadness": one of {emotion_levels},
    "overwhelm": one of {emotion_levels},
    "shame": one of {emotion_levels}
  }},
  "rationale": "two or three sentences explaining the initial emotion profile"
}}

Inputs
------
Agent role: {agent_role}

Scenario:
{scenario}

Patient initial vitals:
{vitals}

Agent traits (level + definition):
{traits}
"""

STEP_PROMPT_TEMPLATE = """\
You are tracking one agent's full emotional state in an Emergency Department \
simulation. All four emotions are tracked and any of them can change.

Emotion dimensions: fear, sadness, overwhelm, shame.
Levels (ordinal): not at all < low < high < extreme.

Context for this tick:
  - Listener (the agent whose state you update): {listener_role}
  - Speaker (who said the utterance below):       {speaker_role}
  - Listener's CURRENT emotional state:
    fear={fear}, sadness={sadness}, overwhelm={overwhelm}, shame={shame}

Who the listener is and how they are doing clinically (this constrains how \
they can react - read it carefully):
{clinical_context}

Recent dialogue leading up to this utterance (oldest first; for context only - \
you are scoring ONLY the final "Utterance" below):
{dialogue}

Given the current state, the clinical context, and the utterance, decide the \
NEW level for each emotion. Apply these rules:
  - LEVEL SEMANTICS. "not at all" means the emotion is genuinely ABSENT / not \
    present in the listener's state. It is NOT the low end of an intensity \
    scale and must NOT be used to mean "a little bit": any actual presence of \
    an emotion is at least "low". So an emotion is either absent ("not at all") \
    or present at one of "low" / "high" / "extreme". \
    An emotion the current utterance does not address KEEPS its current level (see DEFAULT TO NO \
    CHANGE) - do NOT drop it to "not at all" just because this utterance does not \
    mention it. Only return an emotion to "not at all" when its trigger is fully \
    resolved or gone, so it is no longer present at all.

  - DEFAULT TO NO CHANGE. An emotion stays at its current level unless the \
    utterance gives a concrete reason to move it. When in doubt, leave it \
    unchanged. Emotions are sticky: a distressing state persists until \
    something actually changes it.

  - MOVE AT MOST ONE LEVEL per utterance (e.g. extreme->high, low->not at all). \
    A single sentence almost never justifies a two-level swing such as \
    extreme->low or not at all->high. Jumping two levels is allowed ONLY when \
    the utterance is itself a sudden, severe shock (e.g. being told of \
    imminent death or a catastrophic finding). Routine care ("we're giving \
    you oxygen", "starting some fluids") does NOT qualify.


  - ACKNOWLEDGMENT IS NOT RELIEF. Merely naming, validating, or empathizing \
    with an emotion ("I can see you're overwhelmed", "that sounds hard") does \
    NOT lower it. De-escalation requires the underlying trigger to be actually \
    addressed - a concrete reassurance, a plan or next step, symptom relief, \
    or a resolved uncertainty. Empathy with no resolution holds the level \
    steady (at most a marginal softening); on its own it never produces a \
    level drop.

  - QUESTIONS ARE NOT SOOTHING by default. Asking the patient to focus on, \
    rate, or recount distressing content ("what symptom is bothering you \
    most right now?") typically sustains overwhelm/fear and may mildly raise \
    it; it does not lower it.

  - Who the speaker is - the same words from a clinician, a family member, \
    or a stranger land differently.
    
  - "extreme" is a TERMINAL crisis level: reaching it ENDS the encounter. \
    Reserve it for a genuinely catastrophic emotional state (total panic, \
    collapse, complete breakdown), not merely "very distressed". It requires a \
    strong, explicit cue; never assign it from "not at all" or "low" without one. \
    The normal working range is not at all / low / high - keep emotions there \
    unless the situation is truly catastrophic.
  - Most utterances only affect one or two emotions; leave the rest \
    unchanged.

Return JSON only:
{{
  "new_state": {{
    "fear": one of ["not at all", "low", "high", "extreme"],
    "sadness": one of ["not at all", "low", "high", "extreme"],
    "overwhelm": one of ["not at all", "low", "high", "extreme"],
    "shame": one of ["not at all", "low", "high", "extreme"]
  }},
  "evidence": "short phrase from the utterance driving the change",
  "reasoning": "for EACH emotion that changed, name it, give old->new, and say \
why; then briefly note why the others stayed put"
}}

Utterance (score this one):
{utterance}
"""


# ---------------------------------------------------------------------------
# 5. LLM seam
# ---------------------------------------------------------------------------

def _call_llm_json(prompt: str, *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Call the LLM and return parsed JSON."""
    from anthropic import Anthropic  # type: ignore

    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return json.loads(_extract_json(text))


def _extract_json(text: str) -> str:
    """Pull the first {...} block out of a model response, tolerant of fences."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in LLM output:\n{text}")
    return text[start : end + 1]


def _format_traits(traits: dict[str, Any]) -> str:
    """Render trait dict as a readable block for prompts.

    Accepts either {trait_name: level_str} or
    {trait_name: {"level": ..., "definition": ...}}.
    """
    if not traits:
        return "(none)"
    lines: list[str] = []
    for name, value in traits.items():
        if isinstance(value, dict):
            level = value.get("level")
            definition = value.get("definition")
            lines.append(f"- {name} = {level}: {definition}")
        else:
            lines.append(f"- {name} = {value}")
    return "\n".join(lines)


def _format_vitals(vitals: dict[str, Any]) -> str:
    if not vitals:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in vitals.items() if v is not None)


# ---------------------------------------------------------------------------
# 6. init_emotion - LLM call #1
# ---------------------------------------------------------------------------

def init_emotion(
    *,
    agent_role: str,
    scenario: str,
    initial_vitals: dict[str, Any],
    agent_traits: dict[str, Any],
    llm_call=_call_llm_json,
) -> EmotionTrajectory:
    """First LLM call: set the initial level of all six emotions."""

    prompt = INIT_PROMPT_TEMPLATE.format(
        emotion_levels=list(EMOTION_LEVELS),
        agent_role=agent_role,
        scenario=scenario.strip(),
        vitals=_format_vitals(initial_vitals),
        traits=_format_traits(agent_traits),
    )
    raw = llm_call(prompt)
    parsed = InitOutput.model_validate(raw)

    return EmotionTrajectory(
        agent_role=agent_role,
        state=parsed.initial_state,
        history=[parsed.initial_state],
        init_rationale=parsed.rationale,
    )


# ---------------------------------------------------------------------------
# 7. step_emotion - LLM call #2
# ---------------------------------------------------------------------------

def step_emotion(
    trajectory: EmotionTrajectory,
    *,
    utterance: str,
    speaker_role: str,
    clinical_context: str = "",
    dialogue: str = "",
    llm_call=_call_llm_json,
) -> EmotionTrajectory:
    """Second LLM call: update all four emotions from the utterance.

    ``clinical_context`` describes who the listener is and how they are doing
    clinically (e.g. an obtunded child who cannot follow the conversation); it
    constrains how strongly an utterance can move emotion. ``dialogue`` is a
    short digest of the recent conversation so the utterance is scored in
    context rather than in a vacuum. Both are optional and default to a
    "(not provided)" placeholder.
    """
    if speaker_role == trajectory.agent_role:
        raise ValueError(
            f"Self-utterance cannot change emotion (speaker={speaker_role}, "
            f"agent={trajectory.agent_role})"
        )

    st = trajectory.state
    prompt = STEP_PROMPT_TEMPLATE.format(
        listener_role=trajectory.agent_role,
        speaker_role=speaker_role,
        fear=st.fear,
        sadness=st.sadness,
        overwhelm=st.overwhelm,
        shame=st.shame,
        clinical_context=clinical_context.strip() or "(not provided)",
        dialogue=dialogue.strip() or "(no prior dialogue this encounter)",
        utterance=utterance.strip(),
    )
    raw = llm_call(prompt)
    parsed = StepOutput.model_validate(raw)

    return EmotionTrajectory(
        agent_role=trajectory.agent_role,
        state=parsed.new_state,
        history=[*trajectory.history, parsed.new_state],
        init_rationale=trajectory.init_rationale,
        last_evidence=parsed.evidence,
        last_reasoning=parsed.reasoning,
    )


__all__ = [
    "DEFAULT_MODEL",
    "EMOTION_LEVELS",
    "EMOTION_NAMES",
    "EmotionLevel",
    "EmotionName",
    "EmotionState",
    "EmotionTrajectory",
    "InitOutput",
    "StepOutput",
    "init_emotion",
    "step_emotion",
]
