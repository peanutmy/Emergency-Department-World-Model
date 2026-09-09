"""Emotion State Transition Engine.

Single-step, LLM-driven emotion engine for the ED world model. Converted from
``emotion_engine.ipynb``.

## Design

- **4 emotion dimensions**: ``fear``, ``sadness``, ``overwhelm``, ``shame``.
- **3 ordinal levels**: ``absent`` < ``present`` < ``extreme``.
  ``absent`` means the emotion is genuinely not present. ``present`` means the
  emotion is active but not crisis-level. ``extreme`` is a terminal, absorbing
  crisis-level emotional state.
- **All emotions are tracked**: each agent carries a full four-dimension state
  vector. Any emotion can move independently at each tick.
- **Driver**: An agent's emotional state transitions only in response to an utterance it receives from another agent; the agent must be the recipient for any transition to occur.
  Self-utterance cannot change one's own emotion.

The opening state is deterministic; LLM calls only drive dialogue updates:

1. ``init_emotion`` -> all four emotions start at ``absent``.
2. ``step_emotion(trajectory, utterance, speaker_role)`` -> LLM-scored new four-emotion
   state vector.

## Configuration

Credentials and endpoints come from the shell; nothing is hardcoded here:

    export ANTHROPIC_API_KEY="<your key>"
    export ANTHROPIC_BASE_URL="<relay url>"        # optional
    export EMOTION_ENGINE_MODEL="claude-opus-4-8"  # optional model override

"""
from __future__ import annotations

import json
import os
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

# Model the engine calls. Override per-run with the EMOTION_ENGINE_MODEL env var
# (e.g. a model name your relay serves) without editing code. claude-opus-4-8
# scored best on the 28-utterance human-gold eval (see eval_data/).
DEFAULT_MODEL = os.environ.get("EMOTION_ENGINE_MODEL", "claude-opus-4-8")
DEFAULT_PROMPT_STRATEGY = os.environ.get(
    "EMOTION_ENGINE_PROMPT_STRATEGY", "zero_shot"
)


# ---------------------------------------------------------------------------
# 1. Type aliases and constants
# ---------------------------------------------------------------------------

EmotionName = Literal[
    "fear", "sadness", "overwhelm", "shame"
]
EmotionLevel = Literal["absent", "present", "extreme"]
PromptStrategy = Literal["zero_shot", "few_shot", "react"]

EMOTION_NAMES: tuple[EmotionName, ...] = (
    "fear", "sadness", "overwhelm", "shame",
)
EMOTION_LEVELS: tuple[EmotionLevel, ...] = ("absent", "present", "extreme")
PROMPT_STRATEGIES: tuple[PromptStrategy, ...] = ("zero_shot", "few_shot", "react")


# ---------------------------------------------------------------------------
# 2. State models
# ---------------------------------------------------------------------------

class EmotionState(BaseModel):
    """Snapshot of all four emotions for one agent at one tick."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    fear: EmotionLevel = "absent"
    sadness: EmotionLevel = "absent"
    overwhelm: EmotionLevel = "absent"
    shame: EmotionLevel = "absent"

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


class StepOutput(BaseModel):
    """Schema returned by the step LLM call.

    Returns the full new emotional state (all four dimensions) plus
    evidence and reasoning.
    """

    model_config = ConfigDict(extra="forbid")

    new_state: EmotionState
    evidence: str
    reasoning: str


def _stringify_llm_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _coerce_step_output(raw: Any) -> StepOutput:
    """Normalize common LLM JSON shape mistakes before strict validation.

    Claude sometimes returns the requested four emotion fields plus an extra
    ``reasoning`` sentence inside ``new_state``. ``EmotionState`` correctly
    forbids that extra key, but failing the whole turn silently freezes emotion.
    Keep strict validation for the four levels while tolerating misplaced
    reasoning/evidence metadata.
    """

    if not isinstance(raw, dict):
        return StepOutput.model_validate(raw)

    raw_state = raw.get("new_state")
    if not isinstance(raw_state, dict):
        raw_state = raw

    state = {
        emotion: raw_state.get(emotion)
        for emotion in EMOTION_NAMES
        if raw_state.get(emotion) is not None
    }
    evidence = raw.get("evidence", raw_state.get("evidence", ""))
    reasoning = raw.get(
        "reasoning",
        raw_state.get("reasoning", raw.get("rationale", raw_state.get("rationale", ""))),
    )

    return StepOutput.model_validate(
        {
            "new_state": state,
            "evidence": _stringify_llm_field(evidence),
            "reasoning": _stringify_llm_field(reasoning),
        }
    )


# ---------------------------------------------------------------------------
# 4. Prompts
# ---------------------------------------------------------------------------

# Shared, verbatim emotion definitions. Keep these in sync across the step
# and evaluation-item augmentation prompts.
EMOTION_DEFINITIONS_TEXT = """\
- fear: Fear is a response characterized by worry, alarm, or panic about one's \
health, safety, survival, or the safety of someone closely tied to the patient. \
It is triggered by awareness of vulnerability to a threat such as worsening \
illness, dangerous symptoms or test results, treatment delays, possible \
contagion, invasive procedures, loss of control, or risk of death. Clear, \
empathic communication and a concrete plan can buffer fear, but serious new or \
worsening threats can still raise it.

- sadness: Sadness is an emotional response to loss, grief, helplessness, \
hopelessness, or sorrow. It can arise from loss already incurred, such as death, \
permanent damage, loss of function, loss of independence, or a poor prognosis. \
It can also rise from imminent or deeply threatened loss when the utterance \
makes a personally devastating outcome feel close or likely, such as losing a \
baby, family member, survival, bodily function, dignity, or future life. \
Sadness is distinct from fear: fear is primarily about danger and threat, while \
sadness is about actual or anticipated loss and helplessness.

- overwhelm: Overwhelm is acute emotional or cognitive overload, where the \
listener's capacity to process what is happening or what is being asked of them \
is exceeded. It can be caused by too much information, multiple technical \
findings, complex explanations, simultaneous interventions, urgent decisions, \
sensory/procedural intensity, or distress from the illness experience \
interfering with processing. Overwhelm includes both informational overload and \
emotional overload.
- shame: Shame is a negative self-conscious emotion that occurs when the \
    listener feels exposed, blamed, judged, humiliated, or seen as flawed, \
    irresponsible, inadequate, immoral, disgusting, or personally at fault. It can \
    be triggered by blame for non-adherence, substance use, self-harm, stigmatized \
    behavior or illness, embarrassing bodily exposure, loss of dignity, or language \
    that makes the listener feel defective rather than cared for. Neutral, \
nonjudgmental medical disclosure does not by itself raise shame; shame rises \
when the utterance carries judgment, exposure, blame, humiliation, or loss of \
dignity.
"""

# Shared, verbatim emotion/level/transition rules. This block is the SINGLE
# SOURCE OF TRUTH for emotion semantics: the step prompt below embeds it,
# and the augmentation dataset builder imports it so generated eval items
# are calibrated to exactly the same definitions the engine applies.
STEP_EMOTION_RULES = """\
  - LEVEL SEMANTICS. Levels are ordinal: absent < present < extreme. \
    "absent" means the emotion is genuinely not present in the listener's \
    state. "present" means the emotion is active to any non-crisis degree, \
    from mild/background to strongly present. "extreme" is a terminal, \
    absorbing crisis-level state. If an emotion is currently extreme, output \
    extreme for that same emotion; the engine should not be called \
    after termination, but if it is, extreme cannot de-escalate. Only return a \
    non-extreme emotion to "absent" when its trigger is fully resolved or \
    gone, so it is no longer present at all.

  - An emotion stays at its current level unless the \
    utterance gives a concrete reason to move it. An emotion the utterance \
    does not address KEEPS its current level - do NOT drop it to "absent" \
    just because this utterance does not mention it. When in doubt, leave it \
    unchanged. Emotions are sticky by default: a distressing state persists \
    until something actually changes it. However, if the utterance explicitly \
    resolves the specific trigger for a non-extreme emotion, that emotion may \
    drop to absent even if the overall medical situation remains serious.

  - CLINICAL THREAT ESCALATION. First judge the VALENCE of the disclosure - is \
    it bad/worsening news or good/improving news? Direct disclosure to the \
    listener of a NEWLY discovered serious abnormality, a worsening condition, or \
    a need to escalate care (e.g. a dangerous lab/vital, a new serious \
    diagnosis, organ dysfunction, infection/sepsis, shock, or the need for a \
    major intervention such as intubation, ICU care, or surgery) is bad news \
    and a concrete reason to raise fear from absent to present, unless fear is \
    already present/extreme. This is a STRONG rule: a calm tone, the listener's trust in the \
    provider, and a clear treatment plan may buffer the SIZE of the rise but \
    they DO NOT cancel it - when serious bad news is delivered, fear must be at \
    least present, but it should not become extreme unless the separate extreme \
    threshold is met. A serious clinical disclosure is enough for \
    absent->present fear when the patient is symptomatic or unstable, whether \
    the listener is the patient or a relative. If the utterance \
    also piles on several technical findings, urgent steps, or a new diagnosis \
    the listener must process at once, raise overwhelm from absent to present \
    too. Judge by meaning, not keywords: the SAME clinical word can be good or \
    bad news. "Your oxygen is dangerously low" raises fear, but "your oxygen \
    has recovered to normal", "we will NOT need to intubate", or "your lactate \
    is coming down" are reassuring resolutions that do NOT raise fear and may \
    lower it. Improvement, stabilization, or a resolved threat is good news.

  - APPLY ALL FOUR DEFINITIONS INDEPENDENTLY. Each utterance can raise or \
    lower any of the four emotions; judge all four every tick using the \
    emotion definitions above. Relief, resolution, restored dignity, or removal \
    of processing burden can hold an emotion steady or lower it; support without \
    concrete resolution usually buffers escalation but should not make an emotion \
    absent. Never raise an emotion merely because a clinical term appears. As practical \
    cues: sadness tracks actual or anticipated devastating loss; shame tracks \
    blame, exposure, humiliation, or loss of dignity; overwhelm tracks the \
    amount, urgency, complexity, and burden of what the listener must process.

  - MOVE AT MOST ONE LEVEL per utterance (e.g. absent->present, \
    present->extreme, present->absent). The two-level jump absent->extreme is \
    allowed ONLY when the utterance itself meets the threshold in the \
    "extreme" rule below; it is RARE, reserved for extremely severe \
    utterances, not for ordinary bad news - when in doubt, move at most one \
    level. Routine care ("we're giving you oxygen", "starting some fluids") \
    never qualifies.

  - ACKNOWLEDGMENT IS NOT RELIEF. Merely naming, validating, or empathizing \
    with an emotion ("I can see you're overwhelmed", "that sounds hard") does \
    NOT lower it. De-escalation requires the underlying trigger to be actually \
    addressed - a concrete reassurance, a concrete plan or next step, symptom \
    relief, or a resolved uncertainty. Empathy with no resolution holds the level \
    steady (at most a marginal softening); on its own it never produces a \
    level drop.

  - QUESTIONS ARE NOT SOOTHING by default. Asking the listener to focus on, \
    rate, or recount distressing content ("what symptom is bothering you \
    most right now?") typically sustains overwhelm/fear and may mildly raise \
    it; it does not lower it.

  - Who the speaker is - the same words from a clinician, a family member, \
    or a stranger land differently.

  - "extreme" is a TERMINAL crisis level: reaching it ENDS the encounter. \
    Because extreme ends the simulation, do not use it for strong but \
    cooperative distress. Extreme describes the LISTENER'S BREAKDOWN, not the \
    severity of the news. Assign extreme only when the utterance would leave \
    the listener unable to keep participating in the encounter: panic attack, \
    emotional collapse, complete shutdown, unbearable humiliation, or \
    catastrophic grief. A concrete protective plan usually buffers extreme, \
    but it does not automatically block extreme when the utterance conveys \
    imminent death, catastrophic loss, total loss of control, or devastating \
    humiliation/blame in a way that would plausibly make breakdown likely. \
    Explicitly life-threatening disclosures can remain present when the \
    patient can still listen, answer, and cooperate; they become extreme only \
    when the utterance makes breakdown likely. Never assign extreme from \
    absent unless this utterance itself contains such a terminal cue. Extreme \
    never de-escalates: once an emotion reaches extreme the encounter is over, \
    so never move an emotion down from extreme.

"""

FEW_SHOT_PROMPT_GUIDANCE_TEXT = """\
Calibration examples for applying the rules above. These are examples only;
do not copy their wording, and do not infer facts from them into the live case.

Example 1 input:
- Listener: patient
- Speaker: nurse
- Current state: fear=absent, sadness=absent, overwhelm=absent, shame=absent
- Clinical context: Adult patient with shortness of breath, awake and able to answer.
- Utterance: "I am putting this blood pressure cuff on your arm. It will squeeze for a moment."
Example 1 output:
{
  "new_state": {
    "fear": "absent",
    "sadness": "absent",
    "overwhelm": "absent",
    "shame": "absent"
  },
  "evidence": "no concrete new or resolving trigger",
  "reasoning": "No emotion changes. Routine bedside care introduces no threat, loss, processing burden, or dignity/blame cue."
}

Example 2 input:
- Listener: patient
- Speaker: clinician
- Current state: fear=absent, sadness=absent, overwhelm=absent, shame=absent
- Clinical context: Adult patient is awake, symptomatic, and unstable.
- Utterance: "Your oxygen level is dangerously low, and we need to move quickly to put you on a tight breathing mask."
Example 2 output:
{
  "new_state": {
    "fear": "present",
    "sadness": "absent",
    "overwhelm": "present",
    "shame": "absent"
  },
  "evidence": "oxygen level is dangerously low",
  "reasoning": "Fear changes absent->present because the clinician directly discloses a serious current threat. Overwhelm changes absent->present because the listener must process urgency and an immediate intervention. Sadness and shame stay absent."
}

Example 3 input:
- Listener: patient
- Speaker: clinician
- Current state: fear=present, sadness=absent, overwhelm=present, shame=absent
- Clinical context: Adult patient is awake after treatment for respiratory distress.
- Utterance: "Your oxygen level is back in the safe range now, and you are breathing comfortably on the smaller oxygen tube."
Example 3 output:
{
  "new_state": {
    "fear": "absent",
    "sadness": "absent",
    "overwhelm": "absent",
    "shame": "absent"
  },
  "evidence": "oxygen level is back in the safe range",
  "reasoning": "Fear changes present->absent because the specific oxygen threat is resolved. Overwhelm changes present->absent because the urgent processing burden is removed. Sadness and shame stay absent."
}

Example 4 input:
- Listener: patient
- Speaker: clinician
- Current state: fear=absent, sadness=absent, overwhelm=absent, shame=absent
- Clinical context: Adult patient is awake and worried about being judged.
- Utterance: "You skipped your medicines again, and that is why you are in this mess."
Example 4 output:
{
  "new_state": {
    "fear": "absent",
    "sadness": "absent",
    "overwhelm": "absent",
    "shame": "present"
  },
  "evidence": "You skipped your medicines again",
  "reasoning": "Shame changes absent->present because the utterance assigns personal blame. Fear, sadness, and overwhelm stay absent because no threat, loss, or processing burden is introduced."
}
"""

REACT_PROMPT_GUIDANCE_TEXT = """\
Use a ReAct-style internal review, but do not output the review steps and do
not call tools. This is a single-call judge task.

Internal process:
1. Observe: identify the listener, speaker, current levels, and exact utterance
   phrases that could be triggers or resolutions.
2. Reason: apply the four emotion definitions independently, using the current
   state and clinical context. Distinguish threat, loss, processing burden, and
   judgment/dignity cues.
3. Act: choose the next valid level for each emotion under the transition
   rules.
4. Check: verify stickiness, one-level movement, rare absent->extreme threshold,
   and extreme absorbing behavior before returning JSON.

Return only the final JSON object requested below.
"""

STEP_PROMPT_TEMPLATE = (
    """\
You are tracking one agent's full emotional state in an Emergency Department \
simulation.

Emotion definitions:
"""
    + EMOTION_DEFINITIONS_TEXT
    + """

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
"""
    + STEP_EMOTION_RULES
    + """

{strategy_guidance}

Return JSON only:
{{
  "new_state": {{
    "fear": one of ["absent", "present", "extreme"],
    "sadness": one of ["absent", "present", "extreme"],
    "overwhelm": one of ["absent", "present", "extreme"],
    "shame": one of ["absent", "present", "extreme"]
  }},
  "evidence": "short phrase from the utterance driving the change, or 'no concrete new or resolving trigger' for no-change",
  "reasoning": "for EACH emotion that changed, name it, give old->new, and say \
why; then briefly note why the others stayed put"
}}

Utterance (score this one):
{utterance}
"""
)


def _normalize_prompt_strategy(value: str | None = None) -> PromptStrategy:
    raw = DEFAULT_PROMPT_STRATEGY if value is None else value
    strategy = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "zero": "zero_shot",
        "zeroshot": "zero_shot",
        "baseline": "zero_shot",
        "few": "few_shot",
        "fewshot": "few_shot",
        "examples": "few_shot",
        "react_style": "react",
        "structured_react": "react",
    }
    strategy = aliases.get(strategy, strategy)
    if strategy not in PROMPT_STRATEGIES:
        allowed = ", ".join(PROMPT_STRATEGIES)
        raise ValueError(
            f"Unknown emotion prompt strategy {raw!r}; expected one of {allowed}."
        )
    return cast(PromptStrategy, strategy)


def _prompt_strategy_guidance(prompt_strategy: str | None = None) -> str:
    strategy = _normalize_prompt_strategy(prompt_strategy)
    if strategy == "zero_shot":
        return ""
    if strategy == "few_shot":
        return FEW_SHOT_PROMPT_GUIDANCE_TEXT
    return REACT_PROMPT_GUIDANCE_TEXT


def build_step_prompt(
    *,
    listener_role: str,
    speaker_role: str,
    state: EmotionState,
    utterance: str,
    clinical_context: str = "",
    dialogue: str = "",
    prompt_strategy: str | None = None,
) -> str:
    """Build the single-utterance emotion judge prompt."""

    return STEP_PROMPT_TEMPLATE.format(
        listener_role=listener_role,
        speaker_role=speaker_role,
        fear=state.fear,
        sadness=state.sadness,
        overwhelm=state.overwhelm,
        shame=state.shame,
        clinical_context=clinical_context.strip() or "(not provided)",
        dialogue=dialogue.strip() or "(no prior dialogue this encounter)",
        strategy_guidance=_prompt_strategy_guidance(prompt_strategy),
        utterance=utterance.strip(),
    )


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
        temperature=0.0,
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


# ---------------------------------------------------------------------------
# 6. init_emotion - deterministic baseline
# ---------------------------------------------------------------------------

def init_emotion(
    *,
    agent_role: str,
    scenario: str,
    initial_vitals: dict[str, Any],
    agent_traits: dict[str, Any],
    llm_call=_call_llm_json,
) -> EmotionTrajectory:
    """Set the opening emotion state to all ``absent`` without an LLM call.

    Scenario, vitals, traits, and ``llm_call`` are accepted for backwards
    compatibility with older callers, but opening emotion is intentionally a
    neutral baseline. Conversation utterances are the only driver of emotion
    movement.
    """

    initial_state = EmotionState()

    return EmotionTrajectory(
        agent_role=agent_role,
        state=initial_state,
        history=[initial_state],
        init_rationale="Initial emotion state is fixed to all absent.",
    )


# ---------------------------------------------------------------------------
# 7. step_emotion - LLM update
# ---------------------------------------------------------------------------

def step_emotion(
    trajectory: EmotionTrajectory,
    *,
    utterance: str,
    speaker_role: str,
    clinical_context: str = "",
    dialogue: str = "",
    prompt_strategy: str | None = None,
    llm_call=_call_llm_json,
) -> EmotionTrajectory:
    """LLM call: update all four emotions from the utterance.

    ``clinical_context`` describes who the listener is and how they are doing
    clinically (e.g. an obtunded child who cannot follow the conversation); it
    constrains how strongly an utterance can move emotion. ``dialogue`` is a
    short digest of the recent conversation so the utterance is scored in
    context rather than in a vacuum. Both are optional; when empty they fall
    back to explicit placeholders ("(not provided)" / "(no prior dialogue this
    encounter)").
    """
    if speaker_role == trajectory.agent_role:
        raise ValueError(
            f"Self-utterance cannot change emotion (speaker={speaker_role}, "
            f"agent={trajectory.agent_role})"
        )

    st = trajectory.state
    prompt = build_step_prompt(
        listener_role=trajectory.agent_role,
        speaker_role=speaker_role,
        state=st,
        clinical_context=clinical_context,
        dialogue=dialogue,
        prompt_strategy=prompt_strategy,
        utterance=utterance.strip(),
    )
    raw = llm_call(prompt)
    parsed = _coerce_step_output(raw)
    new_values = {name: parsed.new_state.level_of(name) for name in EMOTION_NAMES}
    clamped_extremes: list[EmotionName] = []
    for name in EMOTION_NAMES:
        if st.level_of(name) == "extreme" and new_values[name] != "extreme":
            new_values[name] = "extreme"
            clamped_extremes.append(name)

    new_state = EmotionState(**new_values)
    reasoning = parsed.reasoning
    if clamped_extremes:
        note = (
            "Terminal rule kept "
            + ", ".join(clamped_extremes)
            + " at extreme."
        )
        reasoning = f"{reasoning.rstrip()} {note}" if reasoning else note

    return EmotionTrajectory(
        agent_role=trajectory.agent_role,
        state=new_state,
        history=[*trajectory.history, new_state],
        init_rationale=trajectory.init_rationale,
        last_evidence=parsed.evidence,
        last_reasoning=reasoning,
    )


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT_STRATEGY",
    "EMOTION_DEFINITIONS_TEXT",
    "EMOTION_LEVELS",
    "EMOTION_NAMES",
    "FEW_SHOT_PROMPT_GUIDANCE_TEXT",
    "PROMPT_STRATEGIES",
    "PromptStrategy",
    "REACT_PROMPT_GUIDANCE_TEXT",
    "EmotionLevel",
    "EmotionName",
    "EmotionState",
    "EmotionTrajectory",
    "StepOutput",
    "build_step_prompt",
    "init_emotion",
    "step_emotion",
]
