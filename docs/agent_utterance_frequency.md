# 为什么 patient / relative 的 utterance 很少

本文解释世界模型里「谁说话、说多少」的决定机制,定位 patient 说得少、relative 几乎不说话的根因,并给出让它们多说的改法。

## 结论速览

「谁说话、说多少」由**两层**串联决定:

1. **Orchestrator** 决定每回合**谁有资格上场**(`select_active_agents`)。
2. 上场后,agent 的 **Stage-1 `should_speak`** 再决定**到底说不说**。

patient 很少上场是因为它**只有被「点名要求回应」时**才被激活;relative **几乎永远上不了场**,因为它只能靠「显式选择」激活,而 demo/脚本从不传。

---

## 第 1 层:Orchestrator 决定「谁有资格上场」

代码:`ed_world_model/orchestration/orchestrator.py:34-65` `select_active_agents`

| Agent | 激活条件 | 代码位置 |
|---|---|---|
| **clinician** | **每回合都激活**(默认始终在线) | orchestrator.py:46 `DEFAULT_CLINICIAN_ACTIVE` |
| **patient** | 只有进了 `required_response_agents`(被点名要求回应)才激活 | orchestrator.py:48-49 |
| **nurse** | 同上;**额外**:有新检查结果 `newly_available_results` 时上场报告 | orchestrator.py:48-52 |
| **relative** | **只有被显式选择** `explicitly_selected_agents` 才激活 | orchestrator.py:21, 54-56 |

其中 `EXPLICITLY_SELECTABLE_AGENTS = (RELATIVE,)`(orchestrator.py:21)——也就是说**只有 relative 能通过「显式选择」上场**,而这个参数 demo 和 `generate_utterances.py` 都没传 → **relative 整局一句话都不说**。

## 第 2 层:agent 自己决定「说不说」

即使被激活,patient / clinician 是**两阶段**生成:
- **Stage-1(decision)** 先判断 `should_speak`(以及 target、key_points 等);`should_speak=false` 就直接沉默,不进入 Stage-2。
- **Stage-2(verbal)** 才真正生成那句话。

所以「被激活」只是有资格,真正说不说还要过 Stage-1 这关。

---

## 「要求回应」是怎么传递的(patient 上场少的关键链路)

1. 某 agent 说话时,其 `verbal_action.requires_response=True` 且有 `target`
   —— `ed_world_model/orchestration/turn_loop.py:385`
2. → 该 `target` 被加入 `runtime_state.required_response_agents`
   —— `ed_world_model/state/state_manager.py:195-196`
3. → **下一回合**,orchestrator 才据此激活它 —— orchestrator.py:48-49
4. → 它的 Stage-1 再决定 `should_speak`

**链路要点:patient 要说话,必须「上一回合被 clinician 用 `requires_response=True` 点名提问」。**

---

## 根因总结

- **patient 说得少**:它只在「clinician 向它提了需要回应的问题」的**下一回合**才上场。而 clinician 大量发言是**对 nurse 下医嘱**或**陈述句**(`requires_response=false`),这些都不会激活 patient → patient 上场机会很少。
- **relative 几乎不说话**:它只能靠 `explicitly_selected_agents` 激活,而运行时从未传入 → 永远不上场。
- 叠加效果:对话被 **clinician 主导**,patient 偶尔蹦一两句短答,relative 缺席;即使情绪引擎已接入、情绪值不低,也因为「人不开口 + 情绪只用于调语气」而在对话里看不出情绪起伏。

### 实测佐证(Acute Respiratory Distress,15 回合)
- clinician 几乎每回合都说;
- patient 全程只有 1–2 句极短回答("Okay, you can do the X-ray.");
- relative **零发言**;
- 接入情绪引擎后,patient 情绪被初始化到 `fear=extreme / overwhelm=extreme` 并保持,但对话中体现不明显——正因为 patient 开口太少。

---

## 建议改法

按「改动从小到大」排列:

### A. 不改源码,只在运行层调度(最快)
- **让 relative 周期性上场**:运行时给 `runner.run_turn(explicitly_selected_agents={"relative"})`(某些回合传)。可在 `generate_utterances.py` 加个开关,如 `--relative-every N`(每 N 回合让 relative 上场一次)。
- 局限:patient 仍受 `requires_response` 链路约束;relative 上场后还要过自己的「说不说」判断。

### B. 引导 clinician 多向 patient 提「需要回应」的问题(prompt 层)
- 在 clinician 的 verbal_decision / prompt 里**鼓励**更多 `target=patient, requires_response=true` 的提问(尤其在病情稳定、适合问诊时)。
- 这样会经由 `requires_response → required_response_agents` 让 patient 下回合上场。
- 局限:依赖模型是否照做;且仍是「下一回合」才生效。

### C. 放宽 orchestrator 的激活规则(改源码,最直接)
在 `select_active_agents` 里增加激活条件,例如:
- **patient**:当上一回合有人「对 patient 说话」(`target=patient`)时就激活,不必非得 `requires_response=true`;
- **relative**:周期性激活,或当 patient 情绪达到某阈值 / 出现坏消息时激活(让家属介入)。
- 优点:从根上提升二者的上场频率;缺点:改变了核心调度语义,需配套测试。

### D. 给 relative 一个「主动开口」的触发器
- 比如:当 `psych_state.patient_emotion` 或 `relative_emotion` 较高、或出现坏消息事件时,自动把 relative 加入 active(类似 nurse 的 `newly_available_results` 规则)。
- 让家属的发言与情绪/病情联动,既增加 relative 戏份,也更真实。

---

## 相关代码索引
- 上场选择:`ed_world_model/orchestration/orchestrator.py:34-65`(`select_active_agents`)
- 始终在线的 clinician:orchestrator.py:46
- `required_response` 激活:orchestrator.py:48-49
- nurse 因新结果上场:orchestrator.py:51-52
- relative 仅显式可选:orchestrator.py:21, 54-56
- `requires_response` → 入队:`turn_loop.py:385`、`state_manager.py:195-196`
- 两阶段 `should_speak`:`ed_world_model/agents/patient.py`、`clinician.py`(decision 阶段)
