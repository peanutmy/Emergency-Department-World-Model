# EMSim-Based Multi-Agent ED Simulation System Plan

## 1. Overall Goal

The current system has only one core component:

> **EMSim physiology engine**

The goal is to build a multi-agent emergency department simulation layer on top of EMSim.

The responsibility split should be:

```text
EMSim = patient body / physiology truth
Orchestrator = ED world controller
Workflow Engine = orders, nurse tasks, lab delays
Memory System = who knows what
LLM Agents = role behavior and language generation
```

The final system should look like:

```text
┌────────────────────────────────────────────┐
│              Multi-Agent ED System          │
│                                            │
│  Clinician Agent / Human Clinician          │
│  Nurse Agent                                │
│  Patient Agent                              │
│  Relative Agent                             │
│                                            │
│  Orchestrator                               │
│  Workflow Engine                            │
│  Memory System                              │
│  Observation Gateway                        │
│  Evaluation Logger                          │
└──────────────────────┬─────────────────────┘
                       ↓
┌────────────────────────────────────────────┐
│              EMSim EngineSession            │
│                                            │
│  HiddenState                                │
│  Pathology drift                            │
│  Drug PD                                    │
│  Intervention rules                         │
│  Hidden state → observable vitals mapping   │
└────────────────────────────────────────────┘
```

The most important architectural principle:

> **EMSim decides how the patient’s body changes.  
> Multi-agent system decides how people interact, communicate, order, execute, observe, and remember.**

---

## 2. Phase 1 — Wrap EMSim as a Live EngineSession

### 2.1 Motivation

The current EMSim engine is mainly a transition engine:

```text
before + actions + duration_s → after
```

But a live ED simulation requires a persistent patient state:

```text
patient starts at t=0
doctor orders something
nurse later executes
drug starts working
disease keeps progressing
labs come back
patient reacts
```

Therefore, the first step is to wrap EMSim into an `EngineSession`.

---

### 2.2 EngineSession Responsibilities

```python
class EngineSession:
    def __init__(self, initial_state: dict, scenario_seed: int):
        self.now_s = 0
        self.hidden_state = ...
        self.intervention_flags = ...
        self.active_drug_effects = ...
        self.rng = ...

    def observe_vitals(self) -> dict:
        # Return observable vitals only.
        ...

    def apply_emsim_action(self, action: dict) -> None:
        # Apply EMSim-compatible intervention or drug action.
        ...

    def advance(self, dt_s: int) -> dict:
        # Advance physiology by dt_s and return new observable state.
        ...
```

`EngineSession` should be the only live source of physiological truth.

Do **not** duplicate hidden physiology inside agent memory.

---

### 2.3 EMSim Action Format

The existing EMSim action space can be reused directly.

Intervention example:

```json
{
  "type": "intervention",
  "name": "apply_NRB"
}
```

Drug example:

```json
{
  "type": "drug",
  "name": "epinephrine",
  "dose": 1,
  "unit": "mg",
  "route": "IV"
}
```

---

### 2.4 Phase 1 Deliverables

```text
✅ EngineSession
✅ observe_vitals()
✅ apply_emsim_action()
✅ advance(dt_s)
✅ Simple test: apply_NRB / give_fluids / drug changes vitals
```

---

## 3. Phase 2 — Build World State and Workflow Engine

EMSim should not manage hospital workflow.

The following should be handled outside EMSim:

```text
What orders the doctor placed
Whether the nurse is busy
When labs return
Whether a test result is known
Whether the family has been updated
```

These belong to the **ED workflow layer**.

---

### 3.1 WorldState

```python
@dataclass
class WorldState:
    now_s: int
    room_phase: str
    orders: dict[str, ClinicalOrder]
    nurse_tasks: dict[str, NurseTask]
    pending_events: list[PendingEvent]
    conversation_log: list[ConversationTurn]
    active_agents: list[Role]
```

---

### 3.2 OrderManager

A clinician physical action should first become an order, not an immediate EMSim action.

```python
@dataclass
class ClinicalOrder:
    order_id: str
    ordered_by: Role
    order_type: Literal["drug", "intervention", "lab", "imaging", "procedure"]
    payload: dict
    priority: Literal["stat", "urgent", "routine"]
    status: Literal[
        "ordered",
        "validated",
        "pending_execution",
        "in_progress",
        "completed",
        "failed",
        "cancelled"
    ]
    ordered_at_s: int
    started_at_s: int | None = None
    completed_at_s: int | None = None
```

Example:

```json
{
  "order_type": "drug",
  "payload": {
    "name": "procainamide",
    "dose": 1000,
    "unit": "mg",
    "route": "IV"
  },
  "priority": "stat",
  "status": "ordered"
}
```

---

### 3.3 NurseTaskQueue

Whether the nurse is busy should be determined by a task queue.

```python
@dataclass
class NurseTask:
    task_id: str
    source_order_id: str | None
    task_type: Literal[
        "administer_drug",
        "apply_intervention",
        "draw_lab",
        "measure_vitals",
        "assist_procedure",
        "report_result"
    ]
    priority: Literal["stat", "urgent", "routine"]
    duration_s: int
    remaining_s: int
    status: Literal["queued", "active", "done", "failed"]
```

For MVP:

```python
MAX_ACTIVE_TASKS_PER_NURSE = 1
```

If more execution capacity is needed, model multiple nurses instead of letting one nurse do many tasks at once:

```python
nurse_pool.capacity = 2
```

---

### 3.4 LabScheduler

Lab results should be modeled as pending events.

```python
@dataclass
class LabOrder:
    order_id: str
    test_name: str
    status: Literal["ordered", "collected", "processing", "resulted"]
    ordered_at_s: int
    collected_at_s: int | None
    resulted_at_s: int | None
    turnaround_s: int
    result_payload: dict | None
```

MVP fixed turnaround table:

```python
LAB_TURNAROUND_S = {
    "glucose": 30,
    "ecg": 60,
    "abg": 180,
    "vbg": 180,
    "lactate": 360,
    "cbc": 600,
    "bmp": 600,
    "troponin": 900,
    "cxr": 600,
    "ct_head": 1800,
}
```

---

### 3.5 Phase 2 Deliverables

```text
✅ WorldState
✅ OrderManager
✅ NurseTaskQueue
✅ LabScheduler
✅ PendingEventQueue
✅ No-LLM demo:
   doctor order → nurse task → EMSim action → lab result returns
```

---

## 4. Phase 3 — Build the Memory System

The memory system should not be “one infinite chat history per agent.”

Medical simulation needs permission-controlled memory.

---

### 4.1 Memory System Structure

```text
MemorySystem
├── GroundTruthMemory
├── WorldStateMemory
├── DiscoveredClinicalMemory
├── ConversationMemory
├── AgentPrivateMemory
└── EventMemory
```

---

### 4.2 GroundTruthMemory

This stores true patient facts.

```python
@dataclass
class GroundTruthMemory:
    diagnosis: str | None
    pathology: str
    allergies: list[str]
    medications: list[str]
    past_medical_history: list[str]
    symptom_truth: dict
    family_known_info: dict
```

Access permissions:

```text
Orchestrator ✅
ObservationGateway ✅
Evaluator ✅
Patient/Relative partial ✅
Clinician ❌
Nurse ❌
```

---

### 4.3 DiscoveredClinicalMemory

This stores what the clinical team currently knows.

```python
@dataclass
class DiscoveredClinicalMemory:
    allergies_known: bool = False
    allergies: list[str] = field(default_factory=list)

    medications_known: bool = False
    medications: list[str] = field(default_factory=list)

    pmh_known: bool = False
    past_medical_history: list[str] = field(default_factory=list)

    symptom_history: dict = field(default_factory=dict)
    vitals_history: list[dict] = field(default_factory=list)
    exam_findings: dict = field(default_factory=dict)
    test_results: dict = field(default_factory=dict)
```

This is the central mechanism for information asymmetry.

Example:

```text
True allergy = sulfa
Clinician has not asked about allergy
→ DiscoveredClinicalMemory.allergies_known = False
→ Clinician observation does not include sulfa allergy
→ If clinician orders sulfa, EMSim/adverse logic can still react based on ground truth
```

---

### 4.4 ConversationMemory

```python
@dataclass
class ConversationMemory:
    raw_turns: list[ConversationTurn]
    recent_window: list[ConversationTurn]
    running_summary: str
```

Only include in prompts:

```text
recent_window + running_summary
```

Do not put the entire raw transcript into the prompt.

---

### 4.5 AgentPrivateMemory

Patient private memory:

```python
@dataclass
class PatientPrivateMemory:
    personality: str
    cooperation_baseline: float
    health_literacy: str
    symptom_knowledge: dict
    sensitive_topics: list[str]
```

Relative private memory:

```python
@dataclass
class RelativePrivateMemory:
    relationship: str
    knows_allergies: bool
    knows_medications: bool
    knows_recent_events: bool
    personality: str
```

---

### 4.6 EventMemory

Used for replay, debugging, and evaluation.

```python
@dataclass
class EventMemory:
    action_log: list[ActionRecord]
    emsim_calls: list[dict]
    rejected_actions: list[dict]
    physiology_snapshots: list[dict]
    workflow_events: list[dict]
```

---

### 4.7 Phase 3 Deliverables

```text
✅ GroundTruthMemory
✅ DiscoveredClinicalMemory
✅ ConversationMemory
✅ EventMemory
✅ Permission test:
   clinician cannot see hidden truth or allergy until asked/discovered
```

---

## 5. Phase 4 — Build the Observation Gateway

The Observation Gateway converts the same world state into different role-specific observations.

---

### 5.1 Clinician Observation

```python
def build_clinician_observation():
    return {
        "time": world.now_s,
        "known_vitals": discovered.vitals_history[-3:],
        "known_history": discovered.symptom_history,
        "known_allergies": discovered.allergies if discovered.allergies_known else "unknown",
        "known_meds": discovered.medications if discovered.medications_known else "unknown",
        "test_results": discovered.test_results,
        "recent_conversation": conversation.recent_window,
        "pending_clarifications": ...
    }
```

Clinician should not see:

```text
hidden state
pathology label
true diagnosis
undiscovered allergy
drug PD internals
```

---

### 5.2 Nurse Observation

```python
def build_nurse_observation():
    return {
        "current_vitals": emsim.observe_vitals(),
        "pending_orders": order_manager.pending_orders(),
        "current_task": nurse_state.current_task,
        "queue": nurse_state.queue,
        "recent_conversation": conversation.recent_window,
    }
```

Nurse acts as sensor and executor.

---

### 5.3 Patient Observation

Patient should not see numeric vitals unless staff told them.

```python
def build_patient_observation():
    vitals = emsim.observe_vitals()
    symptoms = symptom_renderer(vitals, emsim_public_state)

    return {
        "symptoms": symptoms,
        "pain_distress": patient_emotion.pain_distress,
        "dyspnea": symptoms.dyspnea,
        "dizziness": symptoms.dizziness,
        "consciousness_capacity": symptoms.communication_capacity,
        "recent_conversation": conversation.recent_window,
        "asked_question": response_opportunity_for_patient,
    }
```

---

### 5.4 Relative Observation

```python
def build_relative_observation():
    return {
        "visible_patient_state": visible_signs,
        "known_family_info": relative_private_memory,
        "emotion_state": relative_emotion,
        "recent_conversation": conversation.recent_window,
        "asked_question": response_opportunity_for_relative,
    }
```

---

### 5.5 Phase 4 Deliverables

```text
✅ Role-specific observations
✅ Symptom renderer
✅ Patient does not directly see numeric vitals
✅ Clinician does not see hidden state
✅ Nurse can report current vitals
```

---

## 6. Phase 5 — Build the Action System

Agents should not output unconstrained free-form actions.

Each turn should produce structured actions.

---

### 6.1 AgentTurnOutput

```python
@dataclass
class AgentTurnOutput:
    verbal_action: VerbalAction | None
    information_action: InformationAction | None
    physical_action: PhysicalAction | None
```

---

### 6.2 VerbalAction

```python
@dataclass
class VerbalAction:
    speaker: Role
    target: Role | None
    content: str
    speech_act: Literal[
        "question",
        "answer",
        "order",
        "report",
        "reassure",
        "explain",
        "refuse",
        "emotional_reaction"
    ]
    expects_response: bool = False
```

---

### 6.3 InformationAction

```python
@dataclass
class InformationAction:
    actor: Role
    info_type: Literal[
        "measure_vitals",
        "ask_history",
        "perform_exam",
        "review_monitor",
        "review_test_result"
    ]
    target: Role | None
    slots: list[str]
```

Information actions update:

```text
DiscoveredClinicalMemory
```

They should not directly mutate EMSim hidden state.

---

### 6.4 PhysicalAction

There are two major types.

#### Clinician Physical Action = Order Bundle

```python
@dataclass
class PhysicalOrderBundle:
    actor: Literal["clinician"]
    orders: list[ClinicalOrder]
```

Clinicians can place multiple orders in one turn, but these are only orders, not immediate world mutations.

Suggested limits:

```python
MAX_ORDERS_PER_TURN_NORMAL = 5
MAX_ORDERS_PER_TURN_RESUS = 7
```

#### Nurse Physical Action = Execution

```python
@dataclass
class PhysicalExecution:
    actor: Literal["nurse"]
    task_id: str
    emsim_action: dict | None
```

Only nurse execution or system protocols should call EMSim.

---

### 6.5 Patient / Relative Physical Action = Scene Behavior

```python
@dataclass
class SceneBehavior:
    actor: Role
    behavior_type: Literal[
        "grimace",
        "cough",
        "vomit",
        "pull_mask_off",
        "become_unresponsive",
        "show_medication_list",
        "interrupt",
        "leave_room"
    ]
    intensity: float
```

Patient/relative physical actions should not directly mutate hidden physiology.

If a patient pulls off an oxygen mask, the orchestrator may convert that into a workflow/intervention change according to explicit rules.

---

### 6.6 Per-Round Action Limits

Recommended rule:

```text
Each agent per round:
  max 1 verbal action
  max 1 non-verbal action

Clinician:
  max 1 verbal action
  max 1 order bundle, up to 5 orders

Nurse:
  max 1 verbal action
  max 1 physical task execution

Patient:
  max 1 verbal action
  max 1 scene behavior

Relative:
  max 1 verbal action
  max 1 scene behavior
```

Core summary:

> **Each agent can “say one thing and do one thing” per round.  
> The clinician’s “one thing” can be an order bundle.  
> The nurse’s “one thing” is one active task execution.**

---

### 6.7 Phase 5 Deliverables

```text
✅ VerbalAction schema
✅ InformationAction schema
✅ PhysicalOrderBundle schema
✅ PhysicalExecution schema
✅ SceneBehavior schema
✅ Action validator
✅ Order → NurseTask → EMSimAction mapping
```

---

## 7. Phase 6 — Dialogue Obligation / Response Opportunity

A question should not force an answer.

Instead, it should create a **ResponseOpportunity**.

---

### 7.1 ResponseOpportunity

```python
@dataclass
class ResponseOpportunity:
    id: str
    asked_by: Role
    addressed_to: Role
    question_text: str
    question_type: Literal[
        "symptom",
        "medical_history",
        "allergy",
        "medication",
        "vitals_request",
        "clarification",
        "family_question",
        "sensitive_topic"
    ]
    requiredness: Literal["required", "expected", "optional"]
    sensitivity: Literal["low", "medium", "high"]
    urgency: Literal["stat", "urgent", "routine"]
    status: Literal[
        "open",
        "answered",
        "partially_answered",
        "refused",
        "evaded",
        "ignored",
        "unable_to_answer",
        "expired"
    ] = "open"
```

---

### 7.2 ResponseMode

```python
class ResponseMode(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    PARTIAL_ANSWER = "partial_answer"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    EVADE = "evade"
    EMOTIONAL_REACTION = "emotional_reaction"
    SILENCE = "silence"
    UNABLE = "unable"
```

---

### 7.3 Response Logic

```python
def respond_to_opportunity(agent, opportunity):
    capacity = estimate_capacity(agent)
    knowledge = check_if_agent_knows_answer(agent, opportunity)
    emotion = agent.emotion_state

    mode = choose_response_mode(capacity, knowledge, emotion, opportunity)

    response = llm_generate_response(
        role=agent.role,
        opportunity=opportunity,
        response_mode=mode,
        memory=agent_private_memory,
        emotion=emotion,
        observation=current_observation
    )

    update_opportunity_status(opportunity, mode)
    update_emotion(agent, opportunity, response)
    update_discovered_memory_if_answered(response)
```

---

### 7.4 Important Principle

A question creates interaction pressure, not a guaranteed answer.

Examples:

```text
Clinician: "Do you use cocaine or any stimulants?"

Possible patient responses:
- Direct answer: "Yes, I used cocaine earlier today."
- Refusal: "I don't want to talk about that."
- Evasion: "Why does that matter?"
- Unable: patient is too dyspneic or altered to respond.
```

Only direct or partial answers should update `DiscoveredClinicalMemory`.

---

### 7.5 Phase 6 Deliverables

```text
✅ Question detector
✅ ResponseOpportunityQueue
✅ Patient/Relative can answer, partially answer, evade, refuse, or stay silent
✅ Answer updates DiscoveredClinicalMemory
✅ Non-answer updates emotion/trust/cooperation
```

---

## 8. Phase 7 — Emotion / Affective State

Emotion should be part of patient and relative private state.

EMSim controls physiology. Emotion controls expression, cooperation, trust, and response style.

---

### 8.1 EmotionState

```python
@dataclass
class EmotionState:
    anxiety: float = 0.5
    fear: float = 0.5
    anger: float = 0.0
    trust: float = 0.5
    cooperation: float = 0.8
    confusion: float = 0.0
    pain_distress: float = 0.0
```

---

### 8.2 Physiology → Emotion

```python
def update_patient_emotion_from_physiology(emotion, obs):
    if obs["O2Sat"] < 90:
        emotion.fear += 0.1
        emotion.cooperation -= 0.05

    if obs["BP_sys"] < 90:
        emotion.confusion += 0.1

    if obs["HR"] > 140:
        emotion.anxiety += 0.05
```

EMSim can affect emotion through observable condition.

Emotion should not directly mutate EMSim.

---

### 8.3 Interaction → Emotion

```python
def update_emotion_from_interaction(emotion, event):
    if event.type == "clear_explanation":
        emotion.trust += 0.1
        emotion.anxiety -= 0.05

    if event.type == "ignored_question":
        emotion.trust -= 0.1
        emotion.anxiety += 0.1

    if event.type == "sensitive_question_without_explanation":
        emotion.trust -= 0.05
        emotion.cooperation -= 0.05
```

---

### 8.4 Phase 7 Deliverables

```text
✅ Patient EmotionState
✅ Relative EmotionState
✅ Emotion affects response mode
✅ Physiology affects patient emotion
✅ Clinician communication affects trust/cooperation
```

---

## 9. Phase 8 — Agent Layer

Only after the world, workflow, memory, observation, action, and emotion layers are stable should LLM agents be added.

Recommended order:

```text
1. Human clinician + AI nurse/patient/relative
2. AI clinician later
```

---

### 9.1 BaseAgent Interface

```python
class BaseAgent:
    role: Role

    def act(self, observation: dict) -> AgentTurnOutput:
        ...
```

---

### 9.2 Clinician Agent

For MVP, use a human clinician first.

Later, implement:

```python
class ClinicianAgent(BaseAgent):
    def act(self, observation):
        # Generate:
        # - verbal action
        # - information request
        # - order bundle
        ...
```

Clinician prompt principles:

```text
You only know discovered clinical information.
You cannot access hidden diagnosis or hidden physiology.
When ordering medications, specify drug, dose, route, and priority.
You may ask questions, request vitals, perform exams, or place orders.
```

---

### 9.3 Nurse Agent

Nurse responsibilities:

```text
report
execute
clarify
warn
```

Prompt principles:

```text
You are an ED nurse.
You report observable vitals and clinical observations.
You execute validated orders assigned to you.
If an order is incomplete, ask for clarification.
If the patient deteriorates, interrupt with concise urgent updates.
You cannot reveal hidden diagnosis or physiology.
```

---

### 9.4 Patient Agent

Prompt principles:

```text
You are the patient.
You only know your symptoms, feelings, and memories.
You do not know exact vital signs unless staff told you.
You do not know hidden diagnosis.
Your response depends on pain, dyspnea, consciousness, anxiety, trust, and cooperation.
You may answer, partially answer, refuse, evade, remain silent, or be unable to answer.
```

---

### 9.5 Relative Agent

Prompt principles:

```text
You are the patient's relative.
You know only the family history and events specified in your private memory.
You may provide collateral history, ask questions, interrupt, or react emotionally.
You do not know hidden diagnosis or exact physiology.
```

---

### 9.6 Phase 8 Deliverables

```text
✅ BaseAgent
✅ NurseAgent
✅ PatientAgent
✅ RelativeAgent
✅ HumanClinician mode
✅ Later: AI ClinicianAgent
```

---

## 10. Phase 9 — Main Orchestrator Loop

The orchestrator is the heart of the system.

---

### 10.1 Recommended Round Flow

```text
1. Receive clinician input
2. Parse clinician verbal / information / order actions
3. Add ResponseOpportunities if questions exist
4. Validate and record orders
5. Convert valid orders to workflow tasks
6. Resolve nurse task queue
7. Execute completed physical tasks into EMSim
8. Advance EMSim physiology
9. Fire pending lab/imaging/events
10. Build observations for patient/relative/nurse
11. Agents respond/react
12. Update conversation memory
13. Update emotion states
14. Log everything
15. Advance simulation time
```

---

### 10.2 Pseudocode

```python
def run_round(clinician_input: str):
    # 1. Parse clinician input
    clinician_turn = clinician_parser.parse(clinician_input)

    # 2. Save verbal action
    if clinician_turn.verbal_action:
        conversation.add(clinician_turn.verbal_action)
        response_queue.add_from_verbal_action(clinician_turn.verbal_action)

    # 3. Process information actions
    for info_action in clinician_turn.information_actions:
        process_information_action(info_action)

    # 4. Process order bundle
    for order in clinician_turn.order_bundle:
        validation = order_validator.validate(order)
        if validation.ok:
            order_manager.add(order)
            workflow_engine.create_tasks_for_order(order)
        else:
            event_memory.rejected_actions.append(validation)

    # 5. Update nurse task queue
    completed_tasks = nurse_task_queue.advance(dt_s=ROUND_DT_S)

    # 6. Execute completed nurse tasks
    for task in completed_tasks:
        if task.emsim_action:
            emsim.apply_emsim_action(task.emsim_action)

    # 7. Advance physiology
    new_state = emsim.advance(ROUND_DT_S)

    # 8. Fire pending events
    fired_events = event_queue.fire_due_events(world.now_s)
    for event in fired_events:
        process_event(event)

    # 9. Build observations
    nurse_obs = obs_gateway.for_nurse()
    patient_obs = obs_gateway.for_patient()
    relative_obs = obs_gateway.for_relative()

    # 10. Nurse response
    nurse_turn = nurse_agent.act(nurse_obs)
    process_agent_turn(nurse_turn)

    # 11. Patient response / reaction
    patient_turn = patient_agent.act(patient_obs)
    process_agent_turn(patient_turn)

    # 12. Relative response / reaction
    relative_turn = relative_agent.act(relative_obs)
    process_agent_turn(relative_turn)

    # 13. Update affective states
    affect_engine.update_all()

    # 14. Log
    logger.log_round(...)

    # 15. Advance world time
    world.now_s += ROUND_DT_S
```

---

### 10.3 Phase 9 Deliverables

```text
✅ run_round()
✅ deterministic world time
✅ order → task → EMSim
✅ question → response opportunity → answer/refusal/emotion
✅ full round log
```

---

## 11. Phase 10 — Scenario Format

Do not treat transition pairs as full scripts.

Transition pairs are useful for physiology calibration and benchmarks, but interactive simulation needs a scenario format.

---

### 11.1 ScenarioSpec Example

```yaml
scenario:
  case_id: wpw_wide_complex_tachycardia
  title: Wide Complex Tachycardia with WPW
  category: Cardiology
  mode: interactive_training

patient:
  age: 35
  sex: male
  chief_complaint: palpitations and dizziness
  private_memory:
    symptom_onset: 45 minutes ago
    pain: none
    dyspnea: mild
    knows_meds: partial
  personality:
    anxiety: 0.7
    cooperation: 0.8
    health_literacy: low

relative:
  present: true
  relationship: wife
  private_memory:
    knows_medications: true
    knows_allergies: true
  personality:
    anxiety: 0.8
    interruptiveness: 0.5

ground_truth:
  allergies: []
  medications: []
  past_medical_history: []
  pathology: svt
  severity: moderate
  hidden_notes:
    diagnosis: pre-excited atrial fibrillation / WPW concern

initial_state:
  vitals:
    HR: 180
    BP_sys: 125
    BP_dia: 85
    RR: 22
    O2Sat: 97
    T: 37.1
  interventions:
    airway: false
    O2_device: null
    FiO2: 0.21
    intubated: false
    CPR_active: false
  mechanism:
    pathology:
      name: svt
      severity: moderate
    rhythm: VT

tests:
  ECG:
    result: "wide complex regular tachycardia with delta wave history concerning for WPW"
    turnaround_s: 60
  troponin:
    value: 0.02
    unit: ng/mL
    turnaround_s: 900

evaluation:
  critical_actions:
    - avoid_av_nodal_blocker
    - consider_procainamide_or_cardioversion
  unsafe_actions:
    - adenosine
    - diltiazem
```

---

### 11.2 Phase 10 Deliverables

```text
✅ Scenario YAML schema
✅ Loader
✅ Validator
✅ One WPW case
✅ One sepsis case
✅ One anaphylaxis / respiratory case
```

---

## 12. Phase 11 — Evaluation Logger

The system should be replayable and evaluable.

---

### 12.1 RoundLog

```python
@dataclass
class RoundLog:
    round_id: int
    time_s: int

    clinician_input: str
    parsed_actions: dict

    orders_created: list[dict]
    tasks_created: list[dict]
    tasks_completed: list[dict]

    emsim_actions: list[dict]
    vitals_before: dict
    vitals_after: dict

    response_opportunities: list[dict]
    agent_turns: list[dict]

    discovered_memory_delta: dict
    emotion_state_before_after: dict

    rejected_actions: list[dict]
```

---

### 12.2 Evaluation Metrics

#### Physiology realism

```text
vital MAE
direction match
tolerance hit rate
```

#### Workflow realism

```text
time from order to execution
time from lab order to result
nurse task queue delay
missed reassessments
```

#### Clinical decision quality

```text
time to critical action
unsafe orders
missing key history
failure to reassess
inappropriate disposition
```

#### Communication quality

```text
answered patient/family concerns
clear explanation
trust improved or worsened
patient cooperation
relative escalation
```

---

### 12.3 Phase 11 Deliverables

```text
✅ JSON run log
✅ Replayable trajectory
✅ Basic scoring
✅ Unsafe action detection
✅ Clinician action timeline
```

---

## 13. Phase 12 — From Human Clinician to AI Clinician

Do not start with a fully autonomous 4-agent system.

Start with:

```text
Human clinician + AI nurse + AI patient + AI relative + EMSim
```

Then later:

```text
AI clinician + AI nurse + AI patient + AI relative + EMSim
```

---

### 13.1 Why Human Clinician First?

If everything is AI from the beginning, debugging becomes difficult.

You will not know whether a bad run is caused by:

```text
AI clinician reasoning failure
Nurse workflow failure
Patient response failure
EMSim physiology issue
Parser/order extraction error
Orchestrator routing bug
```

---

### 13.2 AI Clinician Output Must Be Structured

Example:

```json
{
  "verbal_action": {
    "target": "nurse",
    "content": "Please place him on a non-rebreather and get an ECG."
  },
  "information_actions": [
    {
      "type": "request_vitals"
    }
  ],
  "order_bundle": [
    {
      "order_type": "intervention",
      "payload": {"name": "apply_NRB"},
      "priority": "stat"
    },
    {
      "order_type": "lab",
      "payload": {"name": "troponin"},
      "priority": "urgent"
    }
  ]
}
```

Do not let the AI clinician output only free text.

---

## 14. Recommended Code Directory

```text
ed_multiagent/
├── main.py
├── config.py
│
├── emsim_adapter/
│   ├── engine_session.py
│   └── action_mapper.py
│
├── world/
│   ├── state.py
│   ├── orders.py
│   ├── nurse_tasks.py
│   ├── events.py
│   └── workflow_engine.py
│
├── memory/
│   ├── ground_truth.py
│   ├── discovered.py
│   ├── conversation.py
│   ├── private_memory.py
│   └── event_memory.py
│
├── observation/
│   ├── gateway.py
│   ├── symptom_renderer.py
│   └── role_views.py
│
├── actions/
│   ├── schema.py
│   ├── parser.py
│   ├── validator.py
│   └── response_opportunity.py
│
├── agents/
│   ├── base.py
│   ├── clinician.py
│   ├── nurse.py
│   ├── patient.py
│   └── relative.py
│
├── affect/
│   ├── emotion_state.py
│   └── affect_engine.py
│
├── orchestrator/
│   ├── simulation.py
│   └── round_loop.py
│
├── scenario/
│   ├── schema.py
│   ├── loader.py
│   └── scenarios/
│
└── evaluation/
    ├── logger.py
    ├── scorer.py
    └── replay.py
```

---

## 15. Implementation Milestones

### Milestone 1 — No LLM, Run the World

Goal:

```text
structured clinician order
→ order manager
→ nurse task queue
→ EMSim
→ vitals change
→ logger
```

Test case:

```text
initial patient with hypoxia
doctor orders apply_NRB
nurse executes after 30s
EMSim updates O2Sat
log records everything
```

---

### Milestone 2 — Add Lab / Workflow Delay

Goal:

```text
doctor orders lactate
nurse draws lab
event scheduled
result returns after 6 min
discovered memory updated
nurse reports result
```

---

### Milestone 3 — Add Memory and Role-Specific Observation

Goal:

```text
clinician does not know allergy
relative knows allergy
clinician asks relative
discovered memory updates
```

---

### Milestone 4 — Add Patient/Relative Response Opportunity

Goal:

```text
doctor asks patient question
patient can answer / partially answer / evade / be unable
emotion state changes
discovered memory updates only if answered
```

---

### Milestone 5 — Add Nurse LLM

Goal:

```text
nurse can report vitals
nurse can ask clarification
nurse can say currently busy
actual execution still comes from task queue
```

---

### Milestone 6 — Add Patient / Relative LLM

Goal:

```text
patient speaks based on symptom renderer + emotion + memory
relative speaks based on private memory + emotion
```

---

### Milestone 7 — Human Clinician Interactive Demo

Goal:

```text
human user acts as doctor
system has nurse/patient/relative
EMSim controls patient physiology
```

This should be the most important first demo.

---

### Milestone 8 — AI Clinician Evaluation

Goal:

```text
AI clinician plays case
logger scores:
  unsafe actions
  time to treatment
  missing history
  outcome
```

---

## 16. MVP Minimal Feature Set

### Roles

```text
human clinician
AI nurse
AI patient
AI relative
```

### Actions

```text
clinician:
  ask question
  request vitals
  order intervention/drug/lab

nurse:
  report vitals
  execute task
  ask clarification

patient:
  answer / evade / unable
  symptom expression

relative:
  provide history
  ask questions
  emotional reaction
```

### System Components

```text
EngineSession
OrderManager
NurseTaskQueue
LabScheduler
DiscoveredClinicalMemory
ConversationMemory
ResponseOpportunity
EmotionState
Logger
```

### Defer Until Later

```text
imaging complex workflow
multiple nurses
shift change
hospital bed availability
autonomous clinician
full medication safety database
long-term memory retrieval
```

---

## 17. Recommended Round Duration

Use a fixed round duration for MVP:

```python
ROUND_DT_S = 30
```

Suggested task durations:

```text
ask/answer: 15–30s
measure vitals: 30s
apply_NRB: 30s
IV push drug: 30–60s
IV infusion setup: 120s
draw labs: 90s
ECG: 60s
intubation: 180s
```

---

## 18. Important Pitfalls to Avoid

### Pitfall 1 — Letting LLM decide vitals

Do not do this.

Vitals must come from EMSim.

---

### Pitfall 2 — Letting the patient know hidden state

Do not do this.

Patient only knows symptoms and feelings, not BP/O2Sat numbers unless told.

---

### Pitfall 3 — Making drugs work immediately after order

Do not do this.

Correct flow:

```text
doctor order
→ nurse task delay
→ administered
→ EMSim drug onset
→ physiology changes
```

---

### Pitfall 4 — Using conversation memory as clinical memory

Do not do this.

Critical clinical facts should be stored structurally:

```text
allergies
medications
symptom onset
test results
vitals history
```

Do not rely on LLM rereading chat history.

---

### Pitfall 5 — Letting four agents freely act at the same time

Do not do this.

The main loop should be:

```text
clinician turn + environment reaction
```

not:

```text
four autonomous agents freely talking and acting at once
```

---

## 19. Final System Definition

The target system can be summarized as:

> **A deterministic EMSim-backed ED world simulator with role-specific LLM agents, workflow-aware clinical orders, controlled information asymmetry, affective patient/family behavior, and replayable evaluation logs.**

In Chinese:

> **一个以 EMSim 为身体引擎的急诊世界模拟器；LLM agents 只负责角色行为和语言；医嘱、护士任务、lab 延迟、信息发现、情绪变化和评估日志都由 orchestrator 管控。**

The shortest implementation path is:

```text
EngineSession
→ Workflow Engine
→ Memory System
→ Observation Gateway
→ Action System
→ ResponseOpportunity + Emotion
→ Nurse/Patient/Relative LLM
→ Human clinician demo
→ AI clinician evaluation
```

This makes the system a real controllable, reproducible, and evaluable ED simulation environment rather than a simple LLM chat room.


---

## 20. V2 Revisions Integrated from Design Review

This section updates the original plan with additional design requirements that are important for resuscitation scenarios, clinical safety evaluation, patient realism, and implementation stability.

These revisions should be treated as part of the core architecture, not optional polish.

---

## 20.1 Dynamic Round Granularity

The original MVP suggested:

```python
ROUND_DT_S = 30
```

This is acceptable for stable or routine ED interactions, but it is too coarse for resuscitation or code blue scenarios.

For example, in VF arrest, a 30-second delay between a clinician ordering defibrillation and the simulated execution of defibrillation would be clinically unrealistic and would corrupt time-to-defib evaluation.

Therefore, the simulation should use mode-dependent time steps.

```python
ROUND_DT_S_STABLE = 30
ROUND_DT_S_URGENT = 10
ROUND_DT_S_CODE = 5
```

### Sim Mode

```python
class SimMode(str, Enum):
    STABLE = "stable"
    URGENT = "urgent"
    CODE = "code"
```

### Mode Switching Logic

```python
def determine_sim_mode(vitals: dict, rhythm: str | None, clinician_declared_code: bool = False) -> SimMode:
    if clinician_declared_code:
        return SimMode.CODE

    if rhythm in {"VF", "asystole", "PEA"}:
        return SimMode.CODE

    if vitals["O2Sat"] < 85:
        return SimMode.CODE

    if vitals["HR"] > 180 or vitals["HR"] < 40:
        return SimMode.URGENT

    if vitals["BP_sys"] < 80:
        return SimMode.URGENT

    return SimMode.STABLE
```

### Time Step Selection

```python
def get_round_dt_s(mode: SimMode) -> int:
    if mode == SimMode.CODE:
        return ROUND_DT_S_CODE
    if mode == SimMode.URGENT:
        return ROUND_DT_S_URGENT
    return ROUND_DT_S_STABLE
```

### Design Principle

```text
Stable patient:
  30-second rounds are acceptable.

Urgent patient:
  10-second rounds preserve tighter clinical timing.

Code / arrest:
  5-second rounds are required for time-critical interventions.
```

Dynamic round granularity is required for meaningful evaluation of:

```text
time-to-defib
time-to-CPR
time-to-airway
time-to-pressor
time-to-reassessment
```

---

## 20.2 Subjective State Renderer

The original plan mentioned `symptom_renderer`, but this component is central and should be explicitly designed.

Patient speech and behavior should not be generated directly from vitals alone. Some clinically important subjective states come from EMSim hidden variables.

For example:

```text
consciousness → ability to answer questions
chronotropic_drive / rhythm → palpitations
shunt_fraction / PaO2_effective / ventilatory_drive → dyspnea severity
preload_index / CO_index → dizziness or presyncope
```

Therefore, add a dedicated subjective rendering layer.

Recommended location:

```text
emsim_adapter/subjective.py
```

or, if integrated directly into EMSim:

```text
rule_engine/subjective.py
```

---

### SubjectiveState Schema

```python
@dataclass
class SubjectiveState:
    dyspnea_severity: float
    confusion_level: float
    palpitation: bool
    dizziness: float
    pain_distress: float
    speech_capacity: Literal[
        "normal",
        "short_phrases",
        "single_words",
        "unable"
    ]
    visible_distress: float
```

---

### subjective_from_hidden()

```python
def subjective_from_hidden(h: HiddenState, vitals: dict) -> SubjectiveState:
    return SubjectiveState(
        dyspnea_severity=_dyspnea(
            h.PaO2_effective,
            h.shunt_fraction,
            h.ventilatory_drive,
            vitals["O2Sat"]
        ),
        confusion_level=1.0 - h.consciousness,
        palpitation=h.rhythm in {"SVT", "VT"} or h.chronotropic_drive > 1.5,
        dizziness=_dizziness(
            h.CO_index,
            h.preload_index,
            vitals["BP_sys"]
        ),
        pain_distress=0.0,
        speech_capacity=_speech_capacity(
            h.consciousness,
            h.PaO2_effective,
            h.ventilatory_drive
        ),
        visible_distress=_visible_distress(...)
    )
```

---

### Speech Capacity as a Physical Constraint

The patient agent should not freely decide whether it can answer.

If physiology implies inability to answer, the response mode should be forced.

```python
if subjective_state.speech_capacity == "unable":
    response_mode = ResponseMode.UNABLE
```

Examples:

```text
consciousness < 0.5:
  patient cannot provide coherent history

severe dyspnea:
  patient can only speak in short phrases

single-word speech:
  patient can answer yes/no or very brief questions

normal speech:
  patient can answer history questions normally
```

---

### Revised Observation Flow

```text
EMSim HiddenState
        ↓
subjective_from_hidden()
        ↓
SubjectiveState
        ↓
ObservationGateway
        ↓
PatientAgent prompt
```

The patient agent should receive subjective state, not raw hidden variables.

---

## 20.3 OrderValidator Boundary

The OrderValidator is critical, but its job must be carefully defined.

### Core Principle

```text
Validator protects system integrity.
Evaluator judges clinical safety.
EMSim simulates consequences.
```

The validator should check whether an order is structurally executable.

It should **not** block clinically unsafe decisions unless they are impossible to execute.

This is essential for medical education and AI clinician evaluation. If the simulator blocks unsafe orders, it cannot teach or evaluate the consequences of unsafe decisions.

---

### Validation Result Types

```python
class ValidationResultStatus(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    REJECT = "reject"
```

---

### Recommended Validator Behavior

| Situation | Validator Behavior | EMSim Execution |
|---|---|---|
| Drug name does not exist | Reject | No |
| Route is invalid | Reject | No |
| Required dose is missing | Reject or ask clarification | No |
| Dose is unusually high | Warn | Yes, if clinician confirms or override allowed |
| Known allergy in DiscoveredClinicalMemory | Warn | Yes, if clinician proceeds |
| Allergy exists in GroundTruth but is undiscovered | No warning | Yes; EMSim/adverse logic may trigger reaction |
| Clinical contraindication, such as AV nodal blocker in WPW | Do not reject | Yes; simulator should model harm |
| Guideline deviation | Do not reject | Yes; evaluator records unsafe or suboptimal action |

---

### Why Unsafe Actions Must Be Allowed

Some scenarios depend on harmful actions producing realistic consequences.

Example:

```text
WPW / pre-excited atrial fibrillation
Clinician gives adenosine or another AV nodal blocker
→ simulator should allow it
→ physiology may deteriorate into polymorphic VT
→ evaluator marks unsafe action
```

If the validator rejects the unsafe medication, the teaching scenario fails.

---

### Practical Validator Design

```python
@dataclass
class ValidationResult:
    status: ValidationResultStatus
    messages: list[str]
    requires_clarification: bool = False
    requires_override_confirmation: bool = False
```

Examples:

```text
Reject:
  "Unknown drug name: procainamid"

Warn:
  "Patient has a documented sulfa allergy in discovered history."

Allow:
  structurally valid order with no known warning
```

---

## 20.4 Termination and Disposition

The original plan mentioned evaluation of inappropriate disposition but did not define disposition as an action or termination condition.

ED simulation requires explicit endpoints.

---

### Add Order Types

Extend `ClinicalOrder.order_type`:

```python
OrderType = Literal[
    "drug",
    "intervention",
    "lab",
    "imaging",
    "procedure",
    "consult",
    "disposition"
]
```

---

### DispositionOrder

```python
@dataclass
class DispositionOrder:
    disposition: Literal[
        "discharge",
        "admit_floor",
        "admit_telemetry",
        "admit_icu",
        "transfer_or",
        "transfer_cath_lab",
        "death"
    ]
    rationale: str
```

---

### Termination Conditions

```python
@dataclass
class Termination:
    reason: Literal[
        "disposition_set",
        "death_no_ROSC",
        "max_time",
        "stabilized",
        "scenario_complete"
    ]
    time_s: int
    final_state: dict
```

```python
def check_termination(world, emsim, scenario):
    if world.disposition is not None:
        return Termination(
            reason="disposition_set",
            time_s=world.now_s,
            final_state=emsim.observe_vitals()
        )

    if world.no_ROSC_duration_s >= 300:
        return Termination(
            reason="death_no_ROSC",
            time_s=world.now_s,
            final_state=emsim.observe_vitals()
        )

    if world.now_s >= scenario.max_time_s:
        return Termination(
            reason="max_time",
            time_s=world.now_s,
            final_state=emsim.observe_vitals()
        )

    if scenario.required_critical_actions_done and patient_stabilized(emsim):
        return Termination(
            reason="stabilized",
            time_s=world.now_s,
            final_state=emsim.observe_vitals()
        )

    return None
```

---

### Disposition as Evaluation Target

Disposition quality should be evaluated.

Examples:

```text
Discharging unstable chest pain patient:
  unsafe disposition

Admitting septic shock to floor instead of ICU:
  inappropriate disposition

Transferring STEMI patient to cath lab:
  appropriate disposition

Continuing resuscitation after prolonged no-ROSC:
  depends on scenario/rubric
```

---

## 20.5 Conversation Memory: Avoid LLM Summaries for Clinical Facts

The original plan included:

```text
recent_window + running_summary
```

This is risky in medical settings.

LLM-generated summaries can corrupt clinically important facts.

Example:

```text
Original:
  "Chest pain started 45 minutes ago."

Bad summary:
  "Chest discomfort earlier."
```

This loses critical timing information.

---

### Revised Memory Principle

```text
Clinical facts:
  structured, slot-based, never compressed by LLM summary

Narrative / emotional context:
  may be summarized

Raw transcript:
  retained for replay and audit

Recent window:
  used for local dialogue continuity
```

---

### Prompt Context Construction

```python
PromptContext = {
    "structured_clinical_facts": build_from_discovered_memory(),
    "recent_dialogue": conversation.recent_window,
    "narrative_summary": conversation.emotional_or_social_summary,
}
```

---

### Buffer Management

When the conversation buffer becomes long:

```text
If a turn has already updated DiscoveredClinicalMemory:
  it can be dropped from recent_window because the clinical fact has been structurally stored.

If a turn contains emotional or social context:
  it can be summarized into narrative_summary.

If a turn contains unresolved clinical information:
  keep it or extract structured slots before dropping.
```

---

### Hard Rule

```text
Do not use LLM summaries as the source of truth for clinical facts.
```

Clinical facts should be generated from:

```text
DiscoveredClinicalMemory
structured test results
vitals history
orders
events
```

---

## 20.6 Two-Stage Emotion Update

The original round flow updated emotion at the end of the round.

That creates a one-round delay.

Example problem:

```text
t = 0:
  Clinician asks a rude or insensitive question.

Patient response in same round:
  uses old cooperation score.

t = 1:
  emotion updates, but too late.
```

---

### Revised Affect Update Design

Emotion should update in two stages:

```text
1. Physiology → emotion
   before building patient observation

2. Interaction → emotion
   immediately after relevant verbal or behavioral events
```

---

### Revised Round Flow with Emotion Timing

```text
1. Receive clinician input
2. Parse clinician action
3. Apply immediate interaction-based emotion update
   - rude tone
   - reassurance
   - sensitive question
   - ignored concern
4. Process orders and information actions
5. Advance workflow and EMSim
6. Apply physiology-based patient emotion update
7. Build observations
8. Nurse acts
9. Nurse action may update patient/relative emotion
10. Relative acts
11. Relative emotional reaction may update patient emotion
12. Patient acts using updated emotion
13. Apply final affect decay/clamp
14. Log
15. Advance world time
```

---

### Emotion Decay / Clamp

At the end of each round:

```python
def finalize_emotion(emotion: EmotionState):
    emotion.anxiety = clamp(decay_toward_baseline(emotion.anxiety), 0.0, 1.0)
    emotion.fear = clamp(decay_toward_baseline(emotion.fear), 0.0, 1.0)
    emotion.anger = clamp(decay_toward_baseline(emotion.anger), 0.0, 1.0)
    emotion.trust = clamp(emotion.trust, 0.0, 1.0)
    emotion.cooperation = clamp(emotion.cooperation, 0.0, 1.0)
```

---

## 20.7 LLM Cost and Latency Planning

Even in MVP, LLM cost and wall-time should be considered.

A naive setup:

```text
20 rounds × 4 agents = 80 LLM calls per case
```

If each call takes 2–5 seconds and agents run serially, one case can take several minutes.

This is bad for interactive demos.

---

### Mitigation 1: Static Prompt Caching

Separate prompts into static and dynamic parts.

```text
Static:
  system prompt
  role instruction
  scenario persona
  allowed action schema

Dynamic:
  current observation
  recent dialogue
  response opportunity
  task queue state
```

This enables prompt caching or efficient prompt reuse.

---

### Mitigation 2: Agent Parallelism

When agents do not depend on each other’s latest response, run them concurrently.

```python
results = await asyncio.gather(
    nurse_agent.act(nurse_obs),
    patient_agent.act(patient_obs),
    relative_agent.act(relative_obs),
)
```

For higher fidelity, use ordered generation:

```text
nurse → relative → patient
```

For lower latency, use parallel generation:

```text
nurse / patient / relative in parallel
```

Recommended modes:

```text
fast_demo:
  parallel non-clinician agents

high_fidelity:
  ordered social/emotional reaction chain
```

---

## 20.8 ECG and Imaging Representation

MVP can use text-only diagnostic results.

Example:

```yaml
ECG:
  result: "wide complex regular tachycardia concerning for WPW"
  turnaround_s: 60
```

However, cardiology education often requires waveform interpretation.

The system should explicitly record this as an MVP simplification.

---

### Future DiagnosticResult Schema

```python
@dataclass
class DiagnosticResult:
    modality: Literal["text", "image", "waveform", "pdf"]
    content_text: str | None
    asset_path: str | None
    structured_findings: dict
```

---

### Design Note

```text
MVP:
  text-only ECG / imaging results

Future:
  multimodal ECG, imaging, waveform, and document observations
```

ObservationGateway should be designed so that future multimodal diagnostic results can be added without rewriting the whole system.

---

## 20.9 Refined Milestone 1

The original Milestone 1 was:

```text
No LLM, run the world.
```

This should be split into smaller milestones.

---

### M1a — No Orders, Just Physiology

Goal:

```text
Load ScenarioSpec
Construct EngineSession
Run 30 rounds of advance(dt_s)
Record vitals drift
```

Validates:

```text
EngineSession can persist state
EMSim can advance over time
Vitals can be logged continuously
```

---

### M1b — Hand-Typed Orders → EMSim

Goal:

```text
Hand-type JSON order
OrderManager receives it
NurseTaskQueue delays execution
Order converts to EMSim action
EMSim applies action
Vitals change
```

Suggested WPW test:

```text
t = 0:
  procainamide 1000 mg IV

t ≈ 20 min:
  BP should trend toward the authored transition:
  125/85 → 88/60
  HR remains around 180
```

This validates:

```text
Order → Task → Execution → EMSim plumbing
```

---

### M1c — No LLM, Full Role Loop

Goal:

```text
Add DiscoveredClinicalMemory
Add ResponseOpportunity
Use hardcoded scripts for nurse/patient/relative
```

Test:

```text
Clinician asks allergy
ResponseOpportunity is created
Hardcoded patient/relative answers
DiscoveredClinicalMemory updates
Next clinician observation includes allergy
```

This validates:

```text
information asymmetry
response opportunity state machine
discovered memory update
role-specific observation
```

Only after M1a–M1c are stable should LLM agents be added.

---

## 21. Revised Phase List

After integrating the V2 review, the recommended phase structure becomes:

```text
Phase 1: EMSim EngineSession
Phase 2: Workflow Engine
Phase 3: Memory System
Phase 4: Observation Gateway
Phase 4.5: Subjective State Renderer
Phase 5: Action System
Phase 5.5: OrderValidator / Safety Evaluation Boundary
Phase 6: ResponseOpportunity
Phase 7: Emotion/Affect Engine with Two-Stage Updates
Phase 8: Termination / Disposition
Phase 9: Agent Layer
Phase 10: Orchestrator Loop with Dynamic Time Step
Phase 11: Scenario Format
Phase 12: Evaluation Logger
Phase 13: Performance Optimization
```

---

## 22. Highest-Priority V2 Changes

The most important changes to implement are:

```text
1. Dynamic round time step
   stable / urgent / code

2. Subjective renderer from hidden state
   hidden physiology → patient symptoms and speech capacity

3. Validator boundary
   reject schema errors
   warn clinical risks
   allow unsafe executable actions for teaching

4. Termination and disposition
   simulation must have explicit endpoints

5. Structured clinical memory
   no LLM summaries for critical clinical facts

6. Two-stage affect update
   physiology and interaction should affect emotion at the right time

7. Refined M1a/M1b/M1c implementation path
   validate plumbing before adding LLM agents
```

These changes make the system more clinically credible, more evaluable, and less likely to produce unrealistic resuscitation behavior.


---

## 23. V3 Revisions: SimMode Engineering, CODE Mode Cost Control, and Prompt Contract

This section integrates additional implementation details introduced by dynamic simulation modes.

The major lesson is:

> **Dynamic time step is not just changing `dt_s`.  
> It introduces a mode state machine, task-duration semantics, and agent activation policy.**

These details should be decided before implementing LLM agents.

---

## 23.1 Stateful SimModeManager

The previous V2 design used `determine_sim_mode()` as a pure function.

That is insufficient.

A stateless mode function can cause mode oscillation near threshold boundaries.

Example:

```text
HR = 181 → URGENT
HR = 178 → STABLE
HR = 182 → URGENT
```

This causes:

```text
unstable dt_s switching
non-reproducible time-to-action metrics
wasted round budget
confusing logs
```

Therefore, SimMode should be managed by a stateful manager.

---

### 23.1.1 SimMode Priority

```python
class SimMode(str, Enum):
    STABLE = "stable"
    URGENT = "urgent"
    CODE = "code"


SIM_MODE_PRIORITY = {
    SimMode.STABLE: 0,
    SimMode.URGENT: 1,
    SimMode.CODE: 2,
}
```

---

### 23.1.2 Target Mode Detection

`determine_target_mode()` may still be a pure function.

```python
def determine_target_mode(
    vitals: dict,
    rhythm: str | None,
    clinician_event: str | None = None,
) -> SimMode:
    if clinician_event == "declare_code":
        return SimMode.CODE

    if rhythm in {"VF", "asystole", "PEA"}:
        return SimMode.CODE

    if vitals["O2Sat"] < 85:
        return SimMode.CODE

    if vitals["HR"] > 180 or vitals["HR"] < 40:
        return SimMode.URGENT

    if vitals["BP_sys"] < 80:
        return SimMode.URGENT

    return SimMode.STABLE
```

---

### 23.1.3 Stateful Mode Update with Hysteresis

Mode upgrades should happen immediately.

Mode downgrades should require either:

```text
1. hysteresis: several consecutive rounds satisfy the lower mode, or
2. explicit clinician declaration.
```

Recommended hybrid design:

```text
STABLE → URGENT:
  automatic and immediate

URGENT → CODE:
  automatic and immediate

URGENT → STABLE:
  requires 3 consecutive stable checks

CODE → URGENT/STABLE:
  requires explicit exit-code event, ROSC/stabilization criteria, or scenario-specific rule
```

Implementation:

```python
class SimModeManager:
    def __init__(self, downgrade_required_rounds: int = 3):
        self.current_mode = SimMode.STABLE
        self.downgrade_counter = 0
        self.downgrade_required_rounds = downgrade_required_rounds

    def update(
        self,
        vitals: dict,
        rhythm: str | None,
        clinician_event: str | None = None,
        rosc_achieved: bool = False,
    ) -> SimMode:
        if clinician_event == "exit_code" and rosc_achieved:
            self.current_mode = SimMode.URGENT
            self.downgrade_counter = 0
            return self.current_mode

        target_mode = determine_target_mode(vitals, rhythm, clinician_event)

        current_priority = SIM_MODE_PRIORITY[self.current_mode]
        target_priority = SIM_MODE_PRIORITY[target_mode]

        # Upgrade immediately.
        if target_priority > current_priority:
            self.current_mode = target_mode
            self.downgrade_counter = 0
            return self.current_mode

        # Same mode: reset downgrade counter.
        if target_mode == self.current_mode:
            self.downgrade_counter = 0
            return self.current_mode

        # Downgrade from CODE should usually require explicit exit.
        if self.current_mode == SimMode.CODE:
            return self.current_mode

        # Downgrade from URGENT to STABLE requires hysteresis.
        if target_priority < current_priority:
            self.downgrade_counter += 1
            if self.downgrade_counter >= self.downgrade_required_rounds:
                self.current_mode = target_mode
                self.downgrade_counter = 0

        return self.current_mode
```

---

### 23.1.4 Time Step Selection

```python
def get_round_dt_s(mode: SimMode) -> int:
    if mode == SimMode.CODE:
        return 5
    if mode == SimMode.URGENT:
        return 10
    return 30
```

---

### 23.1.5 Design Principle

```text
Mode upgrades:
  immediate, because deterioration matters.

Mode downgrades:
  delayed or explicit, because clinical teams do not instantly relax after one good vital sign.
```

---

## 23.2 Task Duration Across Mode Changes

Dynamic SimMode changes must not corrupt task duration.

`NurseTask.remaining_s` must be measured in **simulated seconds**, not number of rounds.

---

### 23.2.1 Required Docstring

```python
@dataclass
class NurseTask:
    """
    duration_s and remaining_s are measured in simulated seconds,
    not number of simulation rounds.

    Every round must decrement remaining_s by the actual dt_s used
    for that round.

    This keeps task duration stable across SimMode changes.
    """
    task_id: str
    duration_s: int
    remaining_s: int
```

---

### 23.2.2 Correct Update Semantics

```python
def advance_task(task: NurseTask, dt_s: int) -> tuple[NurseTask, bool]:
    task.remaining_s -= dt_s

    if task.remaining_s <= 0:
        task.remaining_s = 0
        task.status = "done"
        return task, True

    return task, False
```

This is sufficient for MVP.

---

### 23.2.3 More Precise Future Version: Sub-Round Completion

In a more precise version, if a task has 3 seconds remaining and the current round is 10 seconds, the task should complete after 3 seconds, and its EMSim action should begin from that sub-timepoint.

Future design:

```python
def advance_task_precise(task: NurseTask, dt_s: int):
    if task.remaining_s <= dt_s:
        completion_offset_s = task.remaining_s
        task.remaining_s = 0
        task.status = "done"
        return task, completion_offset_s

    task.remaining_s -= dt_s
    return task, None
```

MVP can ignore sub-round completion, but the semantic distinction should be documented early.

---

## 23.3 CODE Mode Agent Activation Policy

In CODE mode, `dt_s = 5`, which means 12 rounds per simulated minute.

Calling all agents every 5 seconds is unnecessary and expensive.

Example:

```text
5-minute code = 60 rounds

If nurse + patient + relative all call LLM every round:
  60 × 3 = 180 LLM calls
```

This is wasteful, especially when the patient is unresponsive and the relative is repeating distress.

Therefore, CODE mode needs an agent activation policy.

---

### 23.3.1 Patient Agent in CODE Mode

If the patient is physiologically unable to speak, skip the LLM call.

```python
def should_call_patient_agent(
    mode: SimMode,
    subjective_state: SubjectiveState,
    response_opportunity: ResponseOpportunity | None,
) -> bool:
    if response_opportunity is not None:
        return True

    if mode == SimMode.CODE and subjective_state.speech_capacity == "unable":
        return False

    return True
```

Default deterministic patient output:

```json
{
  "verbal_action": null,
  "physical_behavior": {
    "behavior_type": "unresponsive",
    "intensity": 1.0
  }
}
```

---

### 23.3.2 Relative Agent in CODE Mode

Relative agent should not act every 5 seconds.

Recommended policy:

```text
In CODE mode:
  relative acts every 30 seconds,
  or immediately when a major event occurs.
```

Major events include:

```text
defibrillation delivered
ROSC achieved
patient deteriorates
clinician explains plan
clinician ignores family for too long
termination discussion begins
```

Implementation:

```python
def should_call_relative_agent(
    mode: SimMode,
    now_s: int,
    last_relative_act_s: int,
    major_event: bool,
) -> bool:
    if major_event:
        return True

    if mode == SimMode.CODE:
        return now_s - last_relative_act_s >= 30

    return True
```

Between calls, use cached relative state:

```text
relative remains distressed
relative crying silently
relative pacing anxiously
relative held back by staff
```

---

### 23.3.3 Nurse Agent in CODE Mode

The nurse cannot be fully skipped, because nurse tasks execute CPR, defibrillation, medications, and airway support.

However, nurse **task execution** should be deterministic and run every round, while nurse **LLM verbalization** should only happen when needed.

```text
NurseTaskQueue:
  always advances every round

Nurse LLM:
  called only if a verbal report, clarification, or warning is needed
```

```python
def should_call_nurse_agent(mode: SimMode, nurse_state, event) -> bool:
    if event.requires_nurse_report:
        return True

    if nurse_state.needs_clarification:
        return True

    if event.major_clinical_change:
        return True

    return False
```

Template outputs may be sufficient in many CODE rounds:

```text
"CPR ongoing."
"Shock delivered."
"Preparing epinephrine."
"Pulse check in progress."
"No pulse."
"We have ROSC."
```

---

### 23.3.4 Agent Activation Modes

Support at least two runtime modes:

```text
fast_demo:
  skip nonessential LLM calls
  parallelize independent agents
  use templates in CODE mode

high_fidelity:
  allow more frequent relative/patient reactions
  preserve social reaction ordering
```

---

## 23.4 Static / Dynamic Prompt Contract

Prompt caching is not merely an optimization.

It requires architectural separation between static and dynamic prompt parts.

This contract should be defined before implementing the first LLM agent.

---

### 23.4.1 BaseAgent Prompt Interface

```python
class BaseAgent:
    role: Role

    def build_static_prompt(self) -> str:
        """
        Static content:
        - role instructions
        - safety boundaries
        - action schema
        - scenario persona
        - invariant constraints

        This part can be cached.
        """
        ...

    def build_dynamic_prompt(self, observation: dict) -> str:
        """
        Dynamic content:
        - current observation
        - recent dialogue
        - response opportunity
        - task queue state
        - emotion state

        This part changes every round.
        """
        ...

    async def act(self, observation: dict) -> AgentTurnOutput:
        static_prompt = self.build_static_prompt()
        dynamic_prompt = self.build_dynamic_prompt(observation)
        return await self.call_llm(static_prompt, dynamic_prompt)
```

---

### 23.4.2 Prompt Construction Rule

```python
messages = [
    {"role": "system", "content": static_prompt},
    {"role": "user", "content": dynamic_prompt},
]
```

Even if the first implementation does not use provider-specific prompt caching, the boundary must exist.

---

### 23.4.3 Why This Matters

If static/dynamic separation is not built into `BaseAgent` early, later prompt caching will require rewriting all agent prompt templates.

Therefore:

```text
Before M5, every LLM agent must implement:
  build_static_prompt()
  build_dynamic_prompt()
```

---

## 23.5 Death Outcome vs Termination of Resuscitation

The V2 document included `"death"` as a possible disposition.

This can be ambiguous.

There are two different concepts:

```text
death outcome:
  patient is dead as a final physiological/legal outcome

termination_of_resuscitation:
  clinician actively decides to stop resuscitation
```

Recommended change:

```text
Do not treat death as a normal disposition.
Treat death as an outcome.
Treat termination_of_resuscitation as a clinician disposition/order.
```

---

### 23.5.1 Revised Disposition Values

```python
Disposition = Literal[
    "discharge",
    "admit_floor",
    "admit_telemetry",
    "admit_icu",
    "transfer_or",
    "transfer_cath_lab",
    "termination_of_resuscitation"
]
```

---

### 23.5.2 FinalOutcome

```python
@dataclass
class FinalOutcome:
    status: Literal[
        "alive",
        "dead",
        "transferred",
        "admitted",
        "discharged"
    ]
    reason: str
```

---

### 23.5.3 Termination Reasons

```python
TerminationReason = Literal[
    "disposition_set",
    "termination_of_resuscitation",
    "death_no_ROSC",
    "max_time",
    "stabilized",
    "scenario_complete"
]
```

Interpretation:

```text
termination_of_resuscitation:
  clinician explicitly stops resuscitation

death_no_ROSC:
  automatic scenario rule after prolonged no-ROSC

disposition_set:
  discharge, admission, transfer, or other endpoint
```

---

## 23.6 SubjectiveState and SymptomTimeline

`SubjectiveState` is a snapshot of the current patient experience.

It can answer:

```text
Is the patient short of breath right now?
Can the patient speak right now?
Is the patient confused right now?
Does the patient feel palpitations right now?
```

But it cannot answer longitudinal history questions such as:

```text
How long have you been short of breath?
When did the chest pain start?
Did symptoms start suddenly or gradually?
Have you had this before?
```

Those should come from patient private memory.

---

### 23.6.1 SymptomTimeline

```python
@dataclass
class SymptomTimeline:
    onset_s_before_arrival: int | None
    progression: Literal[
        "sudden",
        "gradual",
        "waxing_waning",
        "unknown"
    ]
    prior_episodes: str | None
    patient_description: str
```

---

### 23.6.2 Division of Responsibility

```text
SubjectiveState:
  current embodied feeling

SymptomTimeline:
  past symptom history

PatientPrivateMemory:
  what the patient knows, remembers, or is willing to disclose
```

Patient answering symptom-history questions should use:

```text
PatientPrivateMemory.symptom_timeline + current SubjectiveState + EmotionState
```

---

## 23.7 Structured Code Declaration

`clinician_declared_code` must come from structured action parsing, not vague string matching.

Add code-related speech acts or meta actions.

---

### 23.7.1 SpeechAct Update

```python
class SpeechAct(str, Enum):
    QUESTION = "question"
    ANSWER = "answer"
    ORDER = "order"
    REPORT = "report"
    REASSURE = "reassure"
    EXPLAIN = "explain"
    REFUSE = "refuse"
    EMOTIONAL_REACTION = "emotional_reaction"
    DECLARE_CODE = "declare_code"
    EXIT_CODE = "exit_code"
```

---

### 23.7.2 Alternative: MetaAction

A cleaner long-term design is to represent code declaration as a meta action.

```python
@dataclass
class MetaAction:
    action_type: Literal[
        "declare_code",
        "exit_code",
        "pause",
        "handoff"
    ]
    rationale: str | None = None
```

Recommended path:

```text
MVP:
  use SpeechAct.DECLARE_CODE and SpeechAct.EXIT_CODE

Later:
  promote to MetaAction
```

---

## 23.8 Updated Implementation Milestones

### M1a — No Orders, Just Physiology

Unchanged.

```text
Load ScenarioSpec
Construct EngineSession
Run 30 rounds of advance(dt_s)
Record vitals drift
```

---

### M1b — Hand-Typed Orders → EMSim

Unchanged, but task timing must use simulated seconds.

```text
Hand-type JSON order
OrderManager receives it
NurseTaskQueue delays execution
Order converts to EMSim action
EMSim applies action
Vitals change
```

---

### M1c — No LLM, Full Role Loop

Unchanged.

```text
Add DiscoveredClinicalMemory
Add ResponseOpportunity
Use hardcoded scripts for nurse/patient/relative
```

---

### M1d — SimMode and CODE Mode Dry Run

New milestone.

Goal:

```text
Verify SimModeManager, hysteresis, dynamic dt_s, and CODE-mode agent activation without LLM.
```

Test:

```text
Start stable.
Push vitals into urgent threshold.
Confirm mode upgrades immediately.
Let vitals hover around threshold.
Confirm no oscillation.
Trigger arrest rhythm.
Confirm mode becomes CODE.
Trigger exit_code without ROSC.
Confirm mode remains CODE.
Trigger ROSC + exit_code.
Confirm mode downgrades to URGENT.
```

Also test:

```text
NurseTask.remaining_s decrements by simulated seconds.
Task duration remains correct across STABLE → CODE transition.
Patient agent is skipped when unresponsive in CODE mode.
Relative agent only acts every 30 seconds unless a major event occurs.
```

---

### M5 Requirement Before First LLM Agent

Before adding the first LLM agent:

```text
BaseAgent must implement:
  build_static_prompt()
  build_dynamic_prompt()

CODE-mode activation policy must exist:
  should_call_patient_agent()
  should_call_relative_agent()
  should_call_nurse_agent()

SimModeManager must be integrated with the orchestrator.
```

---

## 23.9 Updated Highest-Priority Engineering Items

The most important V3 engineering requirements are:

```text
1. Stateful SimModeManager with hysteresis
2. Dynamic dt_s based on managed SimMode
3. NurseTask.remaining_s measured in simulated seconds
4. CODE-mode agent activation policy
5. Static/dynamic prompt contract before first LLM agent
6. Structured declare_code / exit_code action
7. Death outcome separated from termination_of_resuscitation
```

These should be implemented before serious LLM-based demos.

---

## 23.10 Final V3 Design Principle

```text
Dynamic time step affects the whole simulation stack.

It changes:
  - orchestrator timing
  - nurse task execution
  - agent activation frequency
  - evaluation metrics
  - prompt/cost design
  - termination and code-state handling

Therefore SimMode must be a first-class state machine, not a simple helper function.
```


---

## 24. V4 Revisions: Clinician Activation, ROSC Signal Chain, Human Commands, and CODE Timing Tests

This section integrates additional engineering requirements discovered after introducing dynamic SimMode and CODE-mode activation policies.

The major missing piece in V3 was:

> **Clinician activation policy in CODE mode.**

V3 controlled patient, relative, and nurse LLM activation, but did not define when an AI clinician should be called during CODE mode.

This is a real design gap because CODE mode uses `dt_s = 5`, which creates 12 rounds per simulated minute. If the AI clinician is called every round, a 5-minute code may require 60 clinician LLM calls, which is both expensive and clinically unrealistic.

---

## 24.1 Clinician Agent Activation Policy

In CODE mode, clinician decision-making should be **ACLS-flow-driven**, not round-driven.

The simulation may advance every 5 seconds, but the clinician does not need to make a new decision every 5 seconds.

Typical CODE-mode decision points include:

```text
rhythm check due
pulse check due
shockable rhythm detected
ROSC detected
no pulse reported
drug interval reached
airway failure event
nurse asks for clarification
major deterioration event
termination-of-resuscitation discussion
```

---

### 24.1.1 should_call_clinician_agent()

```python
def should_call_clinician_agent(
    mode: SimMode,
    now_s: int,
    last_clinician_act_s: int,
    pending_decision_event: bool,
    rhythm_check_due: bool,
    nurse_report_pending: bool,
    medication_window_due: bool = False,
) -> bool:
    if pending_decision_event:
        return True

    if rhythm_check_due:
        return True

    if nurse_report_pending:
        return True

    if medication_window_due:
        return True

    if mode == SimMode.CODE:
        # ACLS pulse/rhythm decision cadence is approximately every 2 minutes.
        return now_s - last_clinician_act_s >= 120

    if mode == SimMode.URGENT:
        return now_s - last_clinician_act_s >= 30

    # In stable mode, the clinician can act every round.
    return True
```

---

### 24.1.2 Design Principle

```text
CODE mode:
  Orchestrator advances world every 5 seconds.
  NurseTaskQueue, CPR, interventions, and EMSim update every 5 seconds.
  AI clinician activates only at clinical decision points.

URGENT mode:
  AI clinician activates at least every 30 seconds, or sooner if triggered.

STABLE mode:
  AI clinician may activate every round.
```

---

### 24.1.3 Evaluation Benefit

This policy also creates a useful evaluation signal.

The evaluator can compare:

```text
expected ACLS decision window
vs.
actual clinician decision time
```

Examples:

```text
rhythm check expected at t = 120s
clinician acted at t = 155s
→ delayed rhythm check

shockable rhythm detected at t = 130s
clinician did not order defibrillation until t = 180s
→ delayed defibrillation decision
```

Thus, activation policy is not merely a cost-control mechanism. It also makes evaluation more clinically meaningful.

---

## 24.2 Unified AgentActivationPolicy Module

Agent activation should be centralized in one module.

Recommended location:

```text
orchestrator/activation_policy.py
```

The module should contain:

```python
def should_call_clinician_agent(...): ...
def should_call_nurse_agent(...): ...
def should_call_patient_agent(...): ...
def should_call_relative_agent(...): ...
```

---

### 24.2.1 Execution vs LLM Activation

The system must distinguish:

```text
workflow execution:
  deterministic, always runs when due

LLM verbal/cognitive generation:
  conditional, may be skipped
```

Examples:

```text
NurseTaskQueue:
  always advances every round

Nurse LLM:
  only called when a report, clarification, or warning is needed

AI Clinician:
  only called when a decision point is reached

Human Clinician:
  input is always accepted when provided
```

This avoids unnecessary LLM calls without freezing the clinical workflow.

---

## 24.3 ACLS Decision Events

CODE-mode clinician activation should be driven by decision events.

Recommended event types:

```python
class DecisionEventType(str, Enum):
    RHYTHM_CHECK_DUE = "rhythm_check_due"
    PULSE_CHECK_DUE = "pulse_check_due"
    SHOCKABLE_RHYTHM = "shockable_rhythm"
    NON_SHOCKABLE_RHYTHM = "non_shockable_rhythm"
    ROSC_DETECTED = "rosc_detected"
    NO_PULSE_REPORTED = "no_pulse_reported"
    DRUG_WINDOW_DUE = "drug_window_due"
    AIRWAY_FAILURE = "airway_failure"
    NURSE_CLARIFICATION = "nurse_clarification"
    TERMINATION_DISCUSSION = "termination_discussion"
```

These events should be generated by a combination of:

```text
EngineSession physiology events
NurseTaskQueue events
WorkflowEngine timers
OrderManager status changes
Scenario-specific rules
```

---

### 24.3.1 ACLS Cycle Timer

In CODE mode, the orchestrator should maintain an ACLS cycle timer.

```python
@dataclass
class ACLSCycleState:
    code_started_at_s: int
    last_rhythm_check_s: int
    last_pulse_check_s: int
    last_epinephrine_s: int | None = None
    rhythm_check_interval_s: int = 120
    epi_interval_min_s: int = 180
    epi_interval_max_s: int = 300
```

The orchestrator can emit:

```text
rhythm_check_due
pulse_check_due
drug_window_due
```

when the relevant interval is reached.

---

## 24.4 ROSC Event Signal Chain

The system should explicitly define how ROSC is detected and propagated.

Important implementation note:

> In the current EMSim design, event/orchestrator interfaces are sketched, but event emission must be implemented as a required new contract for the multi-agent system.

---

### 24.4.1 EngineAdvanceResult Contract

`EngineSession.advance(dt_s)` should return both observable state and physiologic events.

```python
@dataclass
class EngineAdvanceResult:
    state: dict
    events: list[PhysiologyEvent]
```

---

### 24.4.2 PhysiologyEvent

```python
@dataclass
class PhysiologyEvent:
    kind: Literal[
        "arrest",
        "rosc",
        "rhythm_change",
        "severe_hypoxia",
        "shock_delivered",
        "clinical_deterioration"
    ]
    time_s: int
    payload: dict
```

---

### 24.4.3 ROSC Signal Chain

```text
EngineSession.advance()
  → emits PhysiologyEvent(kind="rosc")
  → Orchestrator catches event
  → EventMemory records event
  → AgentActivationPolicy creates pending clinician decision event
  → SimModeManager receives rosc_achieved=True
  → if clinician also issues exit_code, mode may downgrade from CODE to URGENT
```

---

### 24.4.4 Mode Downgrade Rule

ROSC alone should not automatically exit CODE mode.

Recommended rule:

```text
CODE mode exits only when:
  ROSC is detected
  and clinician issues exit_code
  or scenario-specific exit criteria are satisfied
```

This prevents premature mode downgrade after a transient ROSC-like signal.

---

## 24.5 Human Clinician Slash Commands

AI clinician can produce structured speech acts such as `DECLARE_CODE` and `EXIT_CODE`.

Human clinician demo needs a more usable interface.

In CLI or text UI mode, add slash commands.

---

### 24.5.1 Recommended Slash Commands

```text
/code
/exit_code
/disposition admit_icu
/disposition transfer_cath_lab
/consult cardiology
/consult surgery
/pause
/status
```

---

### 24.5.2 Slash Command Mapping

```python
SLASH_COMMAND_MAP = {
    "/code": SpeechAct.DECLARE_CODE,
    "/exit_code": SpeechAct.EXIT_CODE,
}
```

Disposition command:

```text
/disposition admit_icu
```

maps to:

```python
DispositionOrder(disposition="admit_icu")
```

Consult command:

```text
/consult cardiology
```

maps to:

```python
ClinicalOrder(
    order_type="consult",
    payload={"service": "cardiology"},
    priority="routine",
)
```

---

### 24.5.3 Design Principle

```text
UI layer:
  supports human-friendly slash commands

Core orchestrator:
  receives the same structured actions used by AI agents
```

This keeps human and AI clinician modes compatible.

---

## 24.6 SimMode Hysteresis Test Fixture

M1d should include a frozen unit test for mode hysteresis.

The purpose is to prevent mode oscillation near thresholds.

---

### 24.6.1 Test Fixture

```python
HR_FIXTURE = [70, 90, 120, 178, 182, 179, 183, 178, 181, 80]

EXPECTED_MODES = [
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
```

---

### 24.6.2 Unit Test

```python
def test_sim_mode_hysteresis_no_oscillation():
    manager = SimModeManager(downgrade_required_rounds=3)

    modes = []
    for hr in HR_FIXTURE:
        vitals = {
            "HR": hr,
            "BP_sys": 120,
            "O2Sat": 98,
        }
        mode = manager.update(vitals=vitals, rhythm="sinus")
        modes.append(mode)

    assert modes == EXPECTED_MODES
```

---

### 24.6.3 Expected Behavior

```text
Initial stable vitals:
  mode remains STABLE

First HR > 180:
  mode upgrades to URGENT immediately

Subsequent HR fluctuations below threshold:
  mode remains URGENT until downgrade hysteresis is satisfied
```

---

## 24.7 CODE Mode Sub-Round Completion

V3 treated sub-round task completion as a future precision improvement.

This is acceptable for stable non-critical workflows, but may be too imprecise in CODE mode.

In CODE mode, even a few seconds of drift can affect:

```text
epinephrine timing
defibrillation timing
CPR cycle timing
rhythm check timing
drug PD onset alignment
```

Therefore:

```text
MVP may ignore sub-round completion only in non-CODE modes.

Before serious CODE-mode LLM evaluation, implement or validate sub-round completion.
```

---

### 24.7.1 Sub-Round Workflow Completion

```python
@dataclass
class TaskCompletionEvent:
    task_id: str
    offset_s: int
    emsim_action: dict | None
```

```python
def advance_workflow(dt_s: int) -> list[TaskCompletionEvent]:
    events = []

    for task in active_tasks:
        if task.remaining_s <= dt_s:
            completion_offset_s = task.remaining_s
            task.remaining_s = 0
            task.status = "done"
            events.append(TaskCompletionEvent(
                task_id=task.task_id,
                offset_s=completion_offset_s,
                emsim_action=task.emsim_action,
            ))
        else:
            task.remaining_s -= dt_s

    return events
```

---

### 24.7.2 Applying Sub-Round Events to EMSim

If a task completes inside the current round:

```text
advance EMSim until completion offset
apply EMSim action
advance EMSim for remaining part of the round
```

Example:

```text
Current round dt_s = 5
Task remaining_s = 2

Step 1:
  advance EMSim 2s

Step 2:
  apply epinephrine

Step 3:
  advance EMSim 3s
```

This keeps `DrugEffect.t_start_s` closer to true simulated administration time.

---

## 24.8 CODE Drug Timing Regression Test

Before LLM-driven CODE evaluation, add a regression test for drug timing.

This is especially important for drugs with pharmacodynamic onset and half-life.

---

### 24.8.1 Test Goal

```text
Simulate a VF arrest with ACLS-like epinephrine schedule.
Verify that administered_at_s and DrugEffect.t_start_s align with expected ACLS timing.
```

---

### 24.8.2 Example Expected Schedule

```text
epinephrine expected at:
  t = 180s
  t = 360s
  t = 540s
```

---

### 24.8.3 Regression Test Sketch

```python
def test_code_mode_epi_timing_alignment():
    expected_epi_times = [180, 360, 540]

    run = simulate_vf_arrest_with_scripted_acls(
        epi_times=expected_epi_times,
        mode="code",
    )

    actual_epi_times = run.get_drug_start_times("epinephrine")

    for expected, actual in zip(expected_epi_times, actual_epi_times):
        assert abs(actual - expected) <= 5
```

---

### 24.8.4 Failure Interpretation

```text
If drift <= 5s:
  acceptable for CODE-mode simulation

If drift > 10% of intended interval:
  sub-round completion should be implemented before LLM evaluation
```

---

## 24.9 Updated M1d: SimMode and CODE Mode Dry Run

M1d should now include:

```text
SimModeManager hysteresis
dynamic dt_s
CODE-mode agent activation policy
clinician activation policy
ROSC event contract mock
slash command parsing
task timing across mode changes
```

---

### 24.9.1 M1d Test Checklist

```text
[ ] Stable → Urgent upgrade happens immediately
[ ] Urgent does not oscillate near HR threshold
[ ] Urgent → Stable requires hysteresis
[ ] Arrest rhythm triggers CODE
[ ] /code triggers CODE in human CLI mode
[ ] exit_code without ROSC does not downgrade CODE
[ ] mock ROSC + /exit_code downgrades CODE to URGENT
[ ] NurseTask.remaining_s decrements by simulated seconds
[ ] Task duration stays stable across STABLE → CODE transition
[ ] Patient LLM is skipped if unresponsive in CODE
[ ] Relative LLM acts every 30s or on major event in CODE
[ ] AI clinician LLM acts on ACLS decision events, not every 5s
[ ] Hysteresis fixture test passes
```

---

## 24.10 M5 Requirement Update Before First LLM Agent

Before adding the first LLM agent, the following must exist:

```text
BaseAgent prompt contract:
  build_static_prompt()
  build_dynamic_prompt()

AgentActivationPolicy:
  should_call_clinician_agent()
  should_call_nurse_agent()
  should_call_patient_agent()
  should_call_relative_agent()

SimModeManager:
  stateful mode manager with hysteresis

CODE-mode policy:
  skip patient when unresponsive
  throttle relative
  deterministic nurse task execution
  ACLS-flow-aware clinician activation

Human command parser:
  /code
  /exit_code
  /disposition
  /consult

Event contract:
  EngineSession.advance() returns EngineAdvanceResult
```

---

## 24.11 Updated Highest-Priority Engineering Items

The V4 highest-priority engineering requirements are:

```text
1. Stateful SimModeManager with hysteresis
2. Dynamic dt_s based on managed SimMode
3. NurseTask.remaining_s measured in simulated seconds
4. CODE-mode activation policy for all agents, including clinician
5. Static/dynamic prompt contract before first LLM agent
6. Structured declare_code / exit_code action and human slash commands
7. ROSC event contract from EngineSession to Orchestrator
8. Death outcome separated from termination_of_resuscitation
9. CODE-mode drug timing regression test
10. Hysteresis fixture as a unit test
```

---

## 24.12 Final V4 Design Principle

```text
In CODE mode, the world advances every 5 seconds,
but agents do not all think or speak every 5 seconds.

The workflow and physiology layers are high-frequency.
The LLM cognition and dialogue layers are event-triggered.

This keeps the simulation clinically timed, cost-controlled, and evaluable.
```
