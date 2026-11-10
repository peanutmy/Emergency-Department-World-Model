# The Emotion Engine's Role and Educational Value in the Emergency Department World Model

English design document · Based on the current repository implementation · September 14, 2026

This document addresses four questions: what the emotion engine does within the emergency department simulation; what it receives and returns; how its outputs affect the subsequent simulation; and how it could support medical-student training and emergency department communication work. It distinguishes the current implementation, its design rationale, and proposed applications so that intended uses are not presented as validated capabilities.

## 1. Core Purpose: Giving Communication Persistent Consequences in the Simulation

The current emotion engine is a large language model (LLM)-driven emotion state-transition module triggered by received utterances. Given the listener's current emotions, the speaker's identity, the current utterance, and contextual information, it predicts the next four-dimensional emotion state of a patient or family member.

The target of this prediction is the listener's emotional response to the communication. A calmly worded disclosure of risk can still cause fear, and a caring question may leave existing sadness unresolved. The task therefore goes beyond classifying a sentence as having positive or negative sentiment.

Its value within the world model is to preserve the psychological consequences of communication as inspectable state and provide that state to subsequent patient and family-member dialogue generation. The simulation can therefore represent both what clinicians do and how they explain it, together with the listener's subsequent response.

Three responsibilities should be distinguished:

| Module or task | Question it addresses | Current responsibility |
| --- | --- | --- |
| Emotion state transition | How should the listener's emotions change after hearing this utterance? | Emotion engine |
| Role dialogue generation | Given the current emotions and visible information, what might the patient or family member say next? | Patient / relative agent |
| Educational assessment | Is the student's communication accurate, clear, respectful, and consistent with the learning objectives? | Requires separate assessment criteria and faculty review; the emotion engine cannot perform this task alone |

An output such as `fear=present` is consequently a simulated state. It is neither a student's score nor a measurement of a real patient's psychological state. The engine predicts a response to an utterance; it does not observe the real listener's subsequent behavior to confirm that prediction.

## 2. Where It Fits in the Complete Simulation

The system has two state-update paths with different responsibilities:

```text
Clinical intervention -> Physiology adapter and engine -> Physiological state
Communication -> Emotion adapter and engine -> Patient / relative emotion state

Updated states -> Each role's next observation -> Subsequent dialogue and decisions
```

This is a diagram of logical responsibilities, not a claim that the two paths run concurrently. The turn loop determines the actual sequence. If the patient's emotions meet a termination condition, the turn can end early and skip the subsequent physiology update.

When the LLM-based emotion adapter is enabled, the main sequence is:

1. Utterances from clinicians, nurses, patients, or family members are committed as messages with `speaker`, `recipient`, and `content` fields.
2. The turn loop determines whether to invoke each role's emotion adapter, considering conditions such as the patient's ability to perceive the conversation and the family member's presence.
3. The adapter processes messages explicitly addressed to that role, excluding the role's own utterances.
4. The engine predicts a new four-dimensional emotion state, and the adapter converts it into the world model's `AgentEmotion`.
5. `StateManager` writes the result to `GlobalState.psych_state` and can record an `emotion_update` event.
6. Subsequent patient or relative observations include the updated emotions, which the role generator uses when producing its next response.

An emotion update does not regenerate dialogue that has already been committed, and it does not directly determine who gets to speak. The Orchestrator and associated flow control speaking opportunities. Updated emotions take effect when the role is subsequently invoked.

**Activation must be explicit.** When no emotion engine is injected, `IntegrationRunner` uses `NoopEmotionEngine`, which simply copies the existing state. In `generate_utterances.py`, the `--emotion` flag connects the LLM-based emotion adapters for both the patient and the relative. The engine's default prompt strategy is `zero_shot`; this does not mean that every run of the full simulation invokes the LLM-based emotion engine.

Implementation references: [turn_loop.py](../ed_world_model/orchestration/turn_loop.py), [emotion_engine_adapter.py](../ed_world_model/adapters/emotion_engine_adapter.py), [runner.py](../ed_world_model/orchestration/runner.py), [generate_utterances.py](../generate_utterances.py).

## 3. The Emotion State It Represents

The patient and the family member each have a separate emotion state. Clinicians and nurses do not currently carry this four-dimensional state.

| Dimension | Focus of the current definition | Questions for educational discussion |
| --- | --- | --- |
| `fear` | Health, safety, survival, threats, and loss of control | Does the listener understand the risk? Does the communication introduce unnecessary intimidation or unsupported reassurance? |
| `sadness` | Experienced or impending loss, grief, and helplessness | Does the communication allow appropriate sadness and respond to the particular loss? |
| `overwhelm` | Informational or emotional demands exceeding processing capacity | Is too much information delivered at once? Is the explanation broken into manageable parts? Is understanding checked? |
| `shame` | Blame, judgment, exposure, humiliation, and threats to dignity | Is necessary information obtained without making judgments about the person's character? |

Each dimension independently takes one of three ordered values:

| Level | Current meaning |
| --- | --- |
| `absent` | The emotion is not present |
| `present` | The emotion is present, ranging from mild to strong but below crisis level |
| `extreme` | A crisis state severe enough to compromise continued participation; the current engine keeps that dimension at extreme and does not permit it to decrease |

Independent dimensions allow fear and sadness to coexist without requiring a single dominant emotion. These four dimensions are not a complete model of human psychology, however. They do not separately represent trust, comprehension, anger, pain, or willingness to cooperate. `overwhelm=absent` cannot directly be interpreted as successful understanding, and `fear=present` cannot directly be interpreted as treatment refusal.

Within the state design, role profiles and traits describe relatively stable individual characteristics, emotion represents the current response, and recent messages provide conversational context. Facts already known to the clinical team remain in `known_facts`. `StateManager` writes emotion updates to `psych_state`; a model-generated emotional interpretation does not directly become a new diagnosis or medical-history fact.

`init_emotion()` currently makes no LLM call and returns a starting state with all four dimensions absent. The world model uses the same default. This establishes a common simulation baseline; it does not imply that real emergency department patients arrive without fear or sadness. If a caller supplies another current state, the default adapter uses that supplied state. Future teaching scenarios can define different starting states, provided their basis is explicit.

Implementation references: [emotion_engine.py](../emotion_engine.py), [global_state.py](../ed_world_model/state/global_state.py).

## 4. Inputs: Adapter Inputs and What the Model Actually Receives

### 4.1 Inputs Passed from the World Model to the Adapter

| Input | Source and purpose |
| --- | --- |
| Current emotion: `current_patient_emotion` | The current role's `AgentEmotion`; the same parameter name is used for the relative and does not imply a shared patient state |
| `conversation_input` | The list of messages to process, including speaker, recipient, and utterance |
| `recent_messages` | Recent messages used to construct conversational context |
| `patient_profile_context` | The corresponding role's profile, which may include persona, description, and traits |
| Adapter configuration | `agent_role`, scenario, model-call function, and prompt strategy |

The adapter filters out self-utterances and messages with a nonmatching recipient in code. It currently requires `recipient == agent_role`; public messages and messages addressed elsewhere do not automatically trigger a direct update. The visibility boundaries of the dialogue history still need attention, as discussed in Section 10.

The turn loop generally freezes the patient's emotions when the patient is dead, unconscious, or sedated. It includes special handling to preserve utterances heard before sedation within the same turn. Children under three are not currently treated as emotional dialogue agents; the system primarily tracks the present family member instead. These are implementation conventions of this simulation, not conclusions about real children's emotional capacities.

### 4.2 Inputs to `step_emotion()`

A single transition can be expressed as:

```text
Next emotion = F(current emotion, utterance, listener identity, speaker identity,
                 clinical text context, recent dialogue, rules and prompt strategy)
```

| Input | Actual use |
| --- | --- |
| `trajectory.state` | The four current emotion levels, serving as the starting point for the update |
| `trajectory.agent_role` | Identifies whether the patient or the relative is being updated |
| `utterance` | The single utterance being evaluated |
| `speaker_role` | Identifies who spoke; self-utterances are rejected |
| `clinical_context` | Text describing the listener and situation; the adapter combines scenario information and role profile information |
| `dialogue` | Conversation text used to interpret references, prior disclosures, and the meaning of the current utterance |
| `prompt_strategy` | Selects zero-shot, few-shot, or the additional guidance currently named react |
| `llm_call` | A replaceable model-call function; local checks can substitute a mock function |

`EmotionTrajectory` can contain historical states and explanations, but the current prompt builder does not automatically send the entire `trajectory.history` to the model. The prompt explicitly includes the current four-dimensional state and the text inputs described above.

**Dialogue-driven requires a precise interpretation.** The current update function does not directly receive a live vital-sign vector, and the legacy `initial_vitals` argument retained by initialization is no longer used in the computation. However, scenario and clinical-context text may still describe illness, symptoms, or test results. Having no direct live-vitals input therefore does not mean having no clinical information. That background should help interpret the utterance and the listener's ability to understand it; it should not automatically become newly disclosed information that the listener is assumed to know.

Implementation references: [emotion_engine.py](../emotion_engine.py), [emotion_engine_adapter.py](../ed_world_model/adapters/emotion_engine_adapter.py), [turn_loop.py](../ed_world_model/orchestration/turn_loop.py).

## 5. Outputs: Model Response, Trajectory, and World State

The model is asked to return a complete new state, supporting evidence, and an explanation. The following illustrates the interface; it is neither an actual run result nor a human reference label:

```json
{
  "new_state": {
    "fear": "present",
    "sadness": "absent",
    "overwhelm": "absent",
    "shame": "absent"
  },
  "evidence": "The test found a problem that needs prompt attention",
  "reasoning": "The newly disclosed threat moves fear from absent to present; the other dimensions retain their previous levels."
}
```

This example assumes that all four dimensions were initially absent and that the utterance disclosed a new risk without sufficient cues of loss, overload, or humiliation. Actual labels still require the full context and calibration against the assessment criteria.

After parsing the response, `step_emotion()` returns a new `EmotionTrajectory` with the updated `state`, an appended `history`, `last_evidence`, and `last_reasoning`, while preserving the initialization explanation. It forces any dimension already at extreme to remain extreme.

The adapter returns the world model's `AgentEmotion`: four levels plus optional `notes`. It currently concatenates reasoning from the batch of utterances into notes, but does not separately persist every evidence field or the complete utterance-by-utterance state trajectory in `GlobalState`. The evaluation script saves evidence and reasoning separately, and runtime logs can record turn-level states and events. Teaching applications that require utterance-level replay should add dedicated records rather than assume that the current logs preserve every intermediate transition.

`evidence` and `reasoning` are model-generated explanations available for review and discussion. They are not independently verified psychological facts or reliable records of the model's internal reasoning process.

Implementation references: [state_manager.py](../ed_world_model/state/state_manager.py), [emotion_engine_adapter.py](../ed_world_model/adapters/emotion_engine_adapter.py), [evaluate_emotion_engine.py](../tools/evaluate_emotion_engine.py), [generate_utterances.py](../generate_utterances.py).

## 6. How the Outputs Affect the World Model

### 6.1 Continuity Across Patient and Family-Member Responses

`ObservationBuilder` supplies patient_emotion to the patient and relative_emotion to the family member. The patient generator is instructed to respond using the current emotions and visible information. The relative generator is explicitly limited to using emotions to shape wording and tone, without inventing facts.

Fear established in an earlier turn can therefore influence later responses instead of being inferred anew from only the latest utterance. This provides inspectable state memory, but it does not guarantee that the generating model will faithfully follow the state. Consistency between state and dialogue still needs evaluation.

The emotion engine and role generators have separate responsibilities and can be tested or replaced separately. This separation does not mean that they necessarily use different underlying models: the current run script can use the same model across multiple modules. A simulated role's response should therefore not automatically be treated as independent validation of the emotion prediction.

### 6.2 Termination Effects and the Boundary with Treatment

The current `termination.py` triggers emotion-related termination only when any patient emotion reaches extreme. A relative's extreme emotion does not trigger the same encounter termination. This differs from the shared prompt's general description of extreme, and that distinction must be retained when interpreting experimental results.

There is currently no explicit execution rule such as refusing a particular treatment once fear reaches a specified level. Emotions also do not directly modify heart rate, blood pressure, oxygen saturation, or treatment effects; the physiology adapter does not include psych_state as a dedicated engine input. Potential downstream effects mainly occur through role dialogue, subsequent clinical-agent decisions, and early termination caused by patient extreme states.

### 6.3 Locating Events for Debriefing

Turn-level emotions, messages, notes, and events can help faculty locate which simulated changes followed particular communication and then review the original wording and scenario. They provide associations within a model trajectory and model-generated explanations. Studying the causal effect of a particular wording requires controlled replay from the same starting point rather than attributing causality from a single trajectory.

Related implementation: [observation_builder.py](../ed_world_model/orchestration/observation_builder.py), [prompting.py](../ed_world_model/agents/prompting.py), [termination.py](../ed_world_model/state/termination.py), [hybrid_physiology_adapter.py](../ed_world_model/adapters/hybrid_physiology_adapter.py).

## 7. Applications in Medical-Student Training

The most appropriate starting point is formative training with faculty feedback, allowing students to practice, retry, and discuss their decisions. Model labels should not immediately become scores for high-stakes examinations. The current code provides simulation components; the educational workflows below still require a learner interface, curriculum design, and assessment tools.

### 7.1 Communicating Risk Accurately While Supporting the Listener

Using the same case, students can practice direct disclosure, explanations delivered in manageable parts, checks of understanding, and responses to concerns. The emotion engine supplies predicted emotional consequences, the patient agent supplies a subsequent response, and faculty assess whether the information is truthful, sufficiently complete, and appropriate to the situation.

A central teaching principle is that fear or sadness remaining present after an accurate disclosure of serious illness may be appropriate. Good communication should not be defined as making every emotion absent. Such an objective could reward withholding risk information, empty reassurance, or unsupported promises.

A 2025 randomized study involving 221 third-year medical students found that worked-example learning materials improved verbal and nonverbal performance in simulated breaking-bad-news encounters. This supports using examples and simulation exercises in training, but the study did not validate this project, LLM emotion labels, or the educational effects of the current four-dimensional state model. [Bosshard et al., Medical Education](https://pubmed.ncbi.nlm.nih.gov/40074561/)

### 7.2 Organizing Information and Checking Understanding

Overwhelm can provide a starting point for discussing information burden. Faculty can ask students to reorganize a dense explanation into several steps and invite the patient to restate the key points in their own words. AHRQ's teach-back method uses patients' or family members' own explanations to check whether information has been understood. [AHRQ: Teach-Back](https://www.ahrq.gov/teamstepps-program/curriculum/communication/tools/teachback.html)

An emotion label cannot replace a comprehension check. Future educational assessment must independently determine whether the patient correctly restates the risk, next steps, or relevant precautions. A decrease in overwhelm alone cannot establish successful communication.

### 7.3 Asking Sensitive Questions While Protecting Dignity

When discussing medication difficulties, substance use, privacy, or other sensitive issues, students can practice obtaining facts, exploring barriers, and avoiding personal blame. Shame provides one dimension of response for discussion.

The educational objective is to obtain necessary information respectfully, including when the questions are difficult. Assessment must also examine the completeness of the history and the presence of leading questions or discriminatory assumptions. Low shame does not by itself indicate a high-quality interview.

### 7.4 Addressing Different Patient and Family Needs

Separate emotion states allow scenarios in which the patient is relatively calm while the family member remains worried, or the patient cannot participate and explanations are directed mainly to the family. Students must identify the current listener and respond accordingly rather than assume that everyone shares the same information and emotions.

AHRQ's check-back approach emphasizes confirming that information has been received and understood and can also involve patients and family members. These objectives can inform communication tasks, but the current engine does not directly score whether closed-loop communication was completed. [AHRQ: Check-Back](https://www.ahrq.gov/teamstepps-program/curriculum/communication/tools/checkback.html)

### 7.5 Supporting Specific, Repeatable Debriefing

A proposed learning cycle is:

```text
Simulation -> Review key utterances and responses -> Student explains their judgment
-> Faculty checks clinical and communication criteria -> Revise wording -> Retry
```

The retry should use the same starting point. Debriefing should consider the original wording, information accuracy, emotion predictions, actual role responses, and completion of clinical tasks together. AHRQ describes debriefing as a process that supports learning by reviewing events, examining actions and thinking, and identifying improvements for future practice. [AHRQ PSNet: Debriefing for Clinical Learning](https://psnet.ahrq.gov/primer/debriefing-clinical-learning)

Faculty should be able to challenge model labels. For example, a model predicting that sadness disappears while the underlying loss remains is itself a potential model error to discuss, rather than a judgment that students must accept.

## 8. An Illustrative Teaching Scenario

The following is an instructional design example. It is not an actual simulation result and does not prescribe a uniquely correct emotion label.

Scenario: an alert patient has just learned that further urgent assessment is needed and currently has fear=present. There is no evidence that the risk has been resolved. Faculty ask the student to explain what will happen next.

| Student's wording | What to assess educationally | How to interpret the emotion output |
| --- | --- | --- |
| "Don't worry. Everything will definitely be fine." | Is the student making a guarantee unsupported by the available information? | Even if the model predicts less fear, that does not establish good communication |
| "There are several issues we need to assess promptly. I'll explain the immediate step first, then what comes next." | Does the student convey urgency accurately and organize the explanation? | Fear remaining present does not indicate failure; subsequent understanding and responses also matter |
| "I understand that you are frightened. What worries you most right now?" | Does the student acknowledge emotion and identify the specific concern? | The emotion remaining present can be appropriate; identifying the concern has educational value |
| "Could you tell me in your own words what we are planning to do next?" | Does the student actually check comprehension rather than merely ask whether the patient understands? | The content of the patient's explanation must be assessed separately from overwhelm |

The emotion engine places these utterances within a common, recordable simulation feedback process. Faculty judge whether that feedback is reasonable and direct students toward accurate communication, understanding, and dignity rather than the lowest possible emotion values.

## 9. Potential Uses in Emergency Department Work and Current Readiness

| Use | Possible application | What is still needed |
| --- | --- | --- |
| Communication practice for students and newly appointed staff | Repeated practice explaining tests, responding to concerns, disclosing bad news, and asking sensitive questions | Learner input interface, faculty task guides, feedback views, and pilot validation |
| Clinician-nurse-patient-family team exercises | Observe how different roles' wording affects simulated patients and family members | Team-behavior assessment criteria; staff stress, fatigue, and emotions are not currently modeled |
| Teaching cases and communication discussion materials | Generate alternative communication versions for faculty selection and annotation | Human review, checks of information accuracy, and example quality control |
| Offline evaluation of clinical communication AI | Compare emotion predictions across models and prompt strategies on identical inputs | Independent human labels, failure rates, stratified metrics, and controls for same-model self-evaluation bias |
| Early design of communication processes or materials | Explore explanation order, wording, and information density through simulation | User research and real-world validation; simulated differences do not directly establish workflow improvements |

The most defensible current product positioning is a foundational module for emergency department communication simulation and research. It is not yet a validated clinical psychological assessment tool or a complete model for optimizing emergency department operations or throughput.

Applications supporting real work should begin with training, preliminary review of materials, and simulated debriefing. Improvements in patient satisfaction, complaints, treatment adherence, or safety outcomes require additional evidence and cannot be inferred directly from four-dimensional simulated emotions.

## 10. Current Limitations and Their Educational Implications

This section draws on code review and the local simulated checks already completed in this session. The document records these issues; it does not fix the runtime implementation.

| Current limitation | Implication for teaching or research |
| --- | --- |
| The requirements for `present → absent` do not consistently distinguish complete resolution from having a plan or partial relief | The same wording may receive different de-escalation judgments, reducing label consistency |
| The current state does not separately record the triggers of each emotion | Resolving one risk may not establish that every source of fear has disappeared |
| The extreme boundary relies on predicting whether the listener would break down | One prediction cannot establish an actual loss of ability to participate |
| History supplied by the turn loop to the adapter can include the current utterance and later utterances from the same turn | Scoring can be influenced by later information, undermining utterance-level causal interpretation |
| Omitted emotion fields in model output may be filled with absent during parsing | This can create an unsupported reduction in emotion that should not be attributed to the student's communication |
| Patient extreme terminates the encounter, relative extreme does not, and the adapter may continue processing later messages in the same batch | The shared prompt's termination description does not accurately describe every role's runtime behavior |
| Clinical observations include `last_turn_events`, whose emotion-event payloads may expose internal levels | Learner-facing interfaces need explicit filtering; strict concealment of internal labels cannot yet be assumed |
| Inputs are primarily text, without direct measurement of voice, facial expression, or pauses | The system cannot be claimed to cover nonverbal communication comprehensively |
| The default starting state is all absent, and only qualifying utterances trigger emotion updates | Pre-arrival emotions, self-reflection, natural decay, pain, and environmental stimuli are not comprehensively modeled |
| Existing evaluation predicts responses to individual utterances without dialogue history | Single-step metrics do not directly establish the quality of behavior across an entire encounter |

The architecture provides an inspectable simulation update path. Evidence is still needed to establish whether its judgments are appropriate across ages, languages, relationships, and individual characteristics. In particular, a trait should not automatically cast a person as likely to lose control or unwilling to cooperate.

## 11. How to Establish Whether It Is Useful

Three levels should be validated separately because they address different questions.

### 11.1 Engine Level: Agreement with Human Transition Judgments

Use a human-labeled dataset separate from prompt development, with an explicit account of the context available to annotators and to the model. Compare performance by emotion and level, including macro-F1, precision and recall for extreme, change detection, ordinal error, invalid-output rate, and latency. Annotator disagreement should be retained and adjudicated rather than presenting ambiguous cases as uncontested ground truth.

The current evaluation script already reports overall accuracy, MAE, weighted kappa, change detection, and per-emotion statistics; some of the proposed metrics still need to be added. It skips failed predictions and matches reference labels by utterance text. Failure reporting and unique item identifiers should be addressed before formal comparisons.

Zero-shot, few-shot, and the version currently named react should be compared with the same model configuration and inputs. The current react variant adds guidance for a single internal structured review and does not perform tool interactions. The experiment compares these specific prompt implementations; it does not establish that one prompting method is superior across all tasks.

### 11.2 World-Model Level: Appropriate Downstream Behavior

Vary the emotion state while holding role profiles and conversational conditions constant, then assess whether patient and family-member responses change appropriately without altering clinical facts. Also examine continuity across turns, role-specific information visibility, termination semantics, and log completeness.

A comparison between the LLM-based engine and NoopEmotionEngine can be useful, but the absence of an active emotion engine does not imply the absence of emotional dialogue: role generators may still produce emotional expressions from the utterance and scenario. This comparison tests the additional value of the explicit emotion-state module.

### 11.3 Educational Level: Better Student Communication

Begin with faculty acceptability and small usability pilots before designing a study of learning effects. A study could compare training approaches with equivalent cases, practice time, and faculty support, using independent raters blinded where feasible to assess performance on new simulation or standardized-patient tasks. Retention and transfer of learning should also be examined.

Learning measures should include accurate risk explanation, checks of understanding, respect and responsiveness to concerns, and completion of necessary clinical tasks. Student satisfaction, practice frequency, and engine-predicted reductions in emotion are supplementary information, not substitutes for learning outcomes.

## 12. Recommended Next Steps

First resolve inconsistencies in transition semantics and runtime behavior, then have emergency medicine faculty review a small set of teaching cases. Next, add a learner input interface and faculty debriefing view, preserve utterance-level messages, before-and-after states, and model explanations, and allow faculty to correct labels. Finally, evaluate educational effectiveness independently.

A suitable description for external communication is:

> The project's emotion engine translates communication addressed to patients and family members into persistent emotion states and supplies those states to subsequent role interactions in an emergency department simulation. It provides a foundation for communication practice, process debriefing, and model comparison; the validity of its emotion predictions and its educational effectiveness require separate evaluation.

This description preserves its central contribution: giving communication inspectable, discussable consequences within a simulated world while keeping clinical facts, role behavior, and educational assessment subject to their own evaluation criteria.
