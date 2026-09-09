# 情绪 / 特质系统重构 Changelog

本文件记录本次会话对世界模型情绪(emotion)与特质(trait)系统所做的全部修改。

## 设计基线(本次确立)

- 4 个 agent:**patient、relative、clinician(医生)、nurse**。
- **只有 patient 和 relative 是“情绪 agent”**:有 emotion state,也有 trait。
- **clinician 和 nurse 无 emotion、无 trait**,只为让系统跑通而存在。
- **情绪模型**:4 维 `fear / sadness / overwhelm / shame`,4 档 `not at all / low / high / extreme`(`not at all` = 完全不存在,不是“低强度”)。patient 与 relative 共用同一套 schema。
- **情绪的唯一入口**:存于 `GlobalState.psych_state`,经 `ObservationBuilder` 写入 observation,再进入 utterance 生成的 Stage 2 prompt。**没有第二条通道**(`emotion_context` 已删除)。
- **Trait** 存放在 `agent_profiles.<role>.traits`(不搬进 psych_state)。

---

## 1. 随机 Agent Trait(scenario 生成时自动分配)

**目标**:scenario JSON 本身不带 trait;加载时从 trait 库随机给情绪 agent 分配“若干 trait + 每个 trait 的 level”。

### 新增
- `ed_world_model/agents/agent_traits.py`
  - `TraitDefinition`:单个 trait(name、definition、可用 levels —— 直接取自 JSON 的 `trigger` keys)。
  - `AgentTraitLibrary`:从 `ed_world_model/agents/agentTrait/` 加载各角色 trait 文件;`sample(role, rng, min_traits, max_traits)` 随机选 1..N 个不同 trait,每个再随机一个**该 trait 实际定义的** level(单 level 的 trait 强制取那一个,如 patient 的 `Anger` 只有 `high`)。
  - `ROLE_TRAIT_FILES`:**只映射 patient / relative**(见 §4)。

### 修改
- `ed_world_model/scenario_loader.py`
  - 构造参数新增:`randomize_traits=True`、`trait_seed`、`trait_library`、`min_traits_per_role`、`max_traits_per_role`。
  - `_build_agent_profiles`:对每个角色注入随机 trait;**仅限 `TRAIT_BEARING_ROLES = ("patient", "relative")`**,clinician/nurse 恒为 `{}`。
  - `load_scenario(...)` 透传 `randomize_traits` / `trait_seed`。
  - 新增模块级常量 `TRAIT_BEARING_ROLES`。

### Trait 数据文件(`ed_world_model/agents/agentTrait/`)
- 保留:`PatientTrait.json`、`relativeTraits.json`
- **删除**:`ClinicianTrait.json`、`NurseTrait.json`(clinician/nurse 不再有 trait,见 §4)

---

## 2. 端到端情绪轨迹运行脚本

**目标**:用一个 scenario 跑通完整链路 —— scenario → 生理(vitals)轨迹 → 每个 tick 生成 utterance → utterance 喂给情绪引擎 → 情绪状态转移。专为测试 `emotion_engine.py`。

### 新增
- `examples/run_emotion_trajectory.py`
  - 跟踪的是 **patient** 的情绪(引擎禁止 self-utterance,所以由 clinician/nurse 等他人话语驱动)。
  - 流程:`ScenarioLoader`(随机 trait)→ 种子 vitals →`init_emotion(patient)`→ 每个 tick:推进 vitals + 取一句他人 utterance →`step_emotion`。
  - 模式:`--emotion real`(真实 Anthropic 引擎,需 `ANTHROPIC_API_KEY`)/ `--emotion fake`(确定性、无需 key,验证整条 wiring)。
  - vitals 用脚本内确定性 stepper(无 LLM);utterance 用脚本内 `DEFAULT_SCRIPT`(可替换为真实 agent)。
  - 参数:`--scenario`、`--emotion`、`--trait-seed`、`--json`、`--output`。

> 注:`emotion_engine.py` 默认模型 `claude-sonnet-4-20250514` 已弃用(EOL 2026-06-15);real 模式建议 `export EMOTION_ENGINE_MODEL=claude-sonnet-4-6`,并按需 unset 默认的 `ANTHROPIC_BASE_URL` relay。

---

## 3. GlobalState 情绪模型重写(patient + relative)

**修复**:原 `PatientEmotion` 把 `label`/`intensity` 写了 6 遍 → 实际塌缩成 1 个字段的 bug;且与引擎 schema(4 维 / 4 档)、与 adapter(期望 list + 不存在的 `EmotionEntry`)都对不上。`emotion_engine_adapter.py` 当时甚至 import 不进来。

### 修改 — `ed_world_model/state/global_state.py`
- 删除坏掉的 `PatientEmotion`,新增:
  ```python
  EmotionLevel = Literal["not at all", "low", "high", "extreme"]

  class AgentEmotion(StateModel):
      fear / sadness / overwhelm / shame: EmotionLevel = "not at all"
      notes: str | None = None
      def summary() -> dict[str, str]   # 4 维,不含 notes
  ```
- `PsychState` 同时持有 `patient_emotion: AgentEmotion` 与 `relative_emotion: AgentEmotion`。
- `__all__`:移除 `PatientEmotion`,新增 `AgentEmotion`、`EmotionLevel`。
- `AgentEmotion` 与 `emotion_engine.EmotionState` 维度/档位一致 → adapter 转换近乎恒等。

### 修改 — `ed_world_model/state/state_manager.py`
- import `PatientEmotion` → `AgentEmotion`。
- `update_patient_emotion` 改用新字段(`fear/sadness/overwhelm/shame` + `notes`);支持传完整 `AgentEmotion` 或按维度 patch。
- 新增 `update_relative_emotion`;两者共用私有 `_update_emotion(field_name, ...)`。
- `emotion_update` 事件 payload 改为 `{"agent": <field>, **summary}`。

### 修改 — `ed_world_model/adapters/emotion_engine_adapter.py`
- 修复 import:`from emotion_engine import ...`(仓库根模块),删除不存在的 `EmotionEntry`。
- 转换函数 `_patient_emotion_to_state` / `_state_to_patient_emotion` → `_agent_emotion_to_state` / `_state_to_agent_emotion`,改为 `AgentEmotion` ↔ `EmotionState` 的直接字段拷贝。
- 返回类型 `PatientEmotion` → `AgentEmotion`;靠自带 `agent_role` 参数即可分别跟踪 patient 或 relative。

### 修改 — `ed_world_model/orchestration/turn_loop.py`
- `TurnLoop.__init__` 新增 `relative_emotion_engine`(默认 `NoopEmotionEngine`,默认行为不变)。
- 每回合若有 committed_messages,对 **patient 和 relative 各跑一次** `predict` 并更新各自情绪(`recent_messages` 只算一次)。

### 修改 — `ed_world_model/orchestration/runner.py`
- `IntegrationRunner.__init__` 新增 `relative_emotion_engine`,透传给 `TurnLoop`。

### 修改 — `ed_world_model/orchestration/observation_builder.py`
- `build_relative_observation` 新增 `relative_emotion` 字段。

### 测试更新(随 schema 变更)
- `tests/test_global_state.py`、`tests/test_noop_emotion.py`、`tests/test_state_manager.py`、`tests/test_observation_builder.py`、`tests/test_turn_loop_stub.py`:全部改用 `AgentEmotion`,并新增 relative 情绪相关断言。

---

## 4. 删除 clinician / nurse 的 trait

- **删除文件**:`ed_world_model/agents/agentTrait/ClinicianTrait.json`、`NurseTrait.json`。
- `ed_world_model/agents/agent_traits.py`:`ROLE_TRAIT_FILES` 移除 clinician/nurse 映射(否则指向已删文件)。`traits_for("clinician"/"nurse")` 现返回 `[]`。

---

## 5. 删除 `emotion_context` 通道(情绪统一走 observation)+ relative 用情绪

**背景**:`emotion_context` 是一条横跨多文件的死参数(turn_loop 从未设置),情绪一直只经 observation 流动。本次彻底移除该平行通道。

### 删除 `emotion_context`
- `ed_world_model/agents/schemas.py`:`AgentRuntimeInput` 删 `emotion_context` 字段。
- `ed_world_model/agents/prompting.py`:`_compose_prompt` 删 `emotion_label` 参数与情绪渲染段;`build_patient_prompt`/`build_relative_prompt` 指引改指向 observation 里的 `patient_emotion`/`relative_emotion`。
- `ed_world_model/agents/patient.py`:`generate` / `build_verbal_prompt` / `build_prompt` / `build_patient_verbal_prompt` / 兼容包装器全部去掉 `emotion_context`;Stage 2 语气指引改成“用 observation 里的 patient_emotion”;`_PATIENT_DECISION_EXCLUDED_KEYS` 去掉死项 `emotion_context`(保留 `patient_emotion`,Stage 1 仍剥离情绪)。
- `ed_world_model/agents/clinician.py`:同样移除;clinician 无情绪,语气指引简化为只用 traits;`_CLINICIAN_DECISION_EXCLUDED_KEYS` 去掉死项。

### relative 用情绪
- `ed_world_model/agents/relative.py` 与 `prompting.py`:删除过时句 “No relative emotion transition is modeled in v1.3.1.”;新增指引:relative 可用 observation 里的 `relative_emotion` 调语气,但不得据此编造事实(与 patient 对称)。

### 情绪流向(最终)
```
GlobalState.psych_state.{patient_emotion, relative_emotion}
        │  ObservationBuilder
        ▼
observation["patient_emotion" / "relative_emotion"]
        │  Stage 2 (verbal) prompt 中 dump 出来,仅用于调语气
        ▼  （Stage 1 decision 仍剥离情绪,不影响“说什么事实”）
utterance
```

### 测试更新
- `tests/test_patient_agent.py`、`tests/test_clinician_agent.py`、`tests/test_agent_prompting.py`、`tests/test_relative_agent.py`:把基于 `emotion_context` 的断言改为“情绪经 observation 进 Stage 2 / 不进 Stage 1”。
- `tests/test_agent_schemas.py`:无需改(其断言针对 `VerbalDecision`,本就无该字段)。

---

## 受影响文件汇总

### 新增
- `ed_world_model/agents/agent_traits.py`
- `examples/run_emotion_trajectory.py`
- `docs/emotion_trait_refactor_changelog.md`(本文件)

### 修改
- `ed_world_model/scenario_loader.py`
- `ed_world_model/state/global_state.py`
- `ed_world_model/state/state_manager.py`
- `ed_world_model/adapters/emotion_engine_adapter.py`
- `ed_world_model/orchestration/turn_loop.py`
- `ed_world_model/orchestration/runner.py`
- `ed_world_model/orchestration/observation_builder.py`
- `ed_world_model/agents/schemas.py`
- `ed_world_model/agents/prompting.py`
- `ed_world_model/agents/patient.py`
- `ed_world_model/agents/clinician.py`
- `ed_world_model/agents/relative.py`
- 测试:`test_global_state.py`、`test_noop_emotion.py`、`test_state_manager.py`、`test_observation_builder.py`、`test_turn_loop_stub.py`、`test_patient_agent.py`、`test_clinician_agent.py`、`test_agent_prompting.py`、`test_relative_agent.py`

### 删除
- `ed_world_model/agents/agentTrait/ClinicianTrait.json`
- `ed_world_model/agents/agentTrait/NurseTrait.json`

---

## 测试状态

`python -m pytest -q` → **730 passed**。

仅 3 个**预先存在、与本次改动无关**的失败:
- `tests/test_action_registry.py::test_fluid_bolus_has_fluid_type_options_but_no_volume_options`
- `tests/test_action_registry.py::test_oxygen_support_has_device_options_but_no_fixed_fio2_or_peep_options`
  (单独运行也失败,与 emotion/trait 无关)
- `tests/test_scenario_loader.py::test_loader_does_not_import_transition_engines`
  (单独运行通过;是其他 transition-engine 测试污染 `sys.modules` 的套件顺序问题,本次改动文件均未引用 `transition_engines`)

---

# 附录:Per-file diff

> 本仓库不是 git repo,以下 diff 由本次实际改动重建。`-` = 删除行,`+` = 新增行;新增文件给出完整内容,删除文件仅注明。

## 新增文件

### `ed_world_model/agents/agent_traits.py`(新增,完整)

```python
"""Load agent trait definitions and sample random traits/levels per role.

Trait definition files live in ``ed_world_model/agents/agentTrait/`` as JSON,
one file per role. Each file holds a single top-level key (``<role>_traits``)
mapping to a list of trait objects shaped like::

    {
        "trait": "Empathy",
        "definition": "...",
        "trigger": {"high": [...], "low": [...]}
    }

The ``trigger`` keys are the available *levels* for that trait. Some traits only
define a single level (e.g. patient "Anger" is "high" only), so a sampled level
is always drawn from the levels that trait actually defines.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import random
from typing import Any

# Default location of the per-role trait JSON files.
DEFAULT_TRAITS_DIR = Path(__file__).resolve().parent / "agentTrait"

# Maps an AgentProfiles role to (filename, top-level JSON key). Only the patient
# and relative are emotional agents that carry traits; the clinician and nurse
# exist only to drive the simulation and have no trait files.
ROLE_TRAIT_FILES: dict[str, tuple[str, str]] = {
    "patient": ("PatientTrait.json", "patient_traits"),
    "relative": ("relativeTraits.json", "relative_traits"),
}


class TraitDefinition:
    """A single trait with its definition and the levels it defines."""

    def __init__(
        self,
        name: str,
        definition: str,
        levels: Sequence[str],
    ) -> None:
        self.name = name
        self.definition = definition
        # Preserve declaration order while dropping duplicates.
        self.levels: tuple[str, ...] = tuple(dict.fromkeys(levels))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"TraitDefinition(name={self.name!r}, levels={self.levels!r})"


class AgentTraitLibrary:
    """Loads role trait definitions and samples random trait/level sets."""

    def __init__(self, traits_dir: str | Path | None = None) -> None:
        self.traits_dir = Path(traits_dir) if traits_dir is not None else DEFAULT_TRAITS_DIR
        self._by_role: dict[str, list[TraitDefinition]] = {}

    def traits_for(self, role: str) -> list[TraitDefinition]:
        """Return the (cached) trait definitions available for ``role``."""
        if role not in self._by_role:
            self._by_role[role] = self._load_role(role)
        return self._by_role[role]

    def _load_role(self, role: str) -> list[TraitDefinition]:
        spec = ROLE_TRAIT_FILES.get(role)
        if spec is None:
            return []
        filename, top_key = spec
        path = self.traits_dir / filename
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        entries = data.get(top_key, []) if isinstance(data, Mapping) else []
        return [
            definition
            for entry in entries
            if (definition := self._parse_entry(entry)) is not None
        ]

    @staticmethod
    def _parse_entry(entry: Any) -> TraitDefinition | None:
        if not isinstance(entry, Mapping):
            return None
        name = entry.get("trait")
        if not name:
            return None
        trigger = entry.get("trigger")
        levels = list(trigger.keys()) if isinstance(trigger, Mapping) else []
        if not levels:
            return None
        return TraitDefinition(
            name=str(name),
            definition=str(entry.get("definition", "")),
            levels=levels,
        )

    def sample(
        self,
        role: str,
        *,
        rng: random.Random,
        min_traits: int = 1,
        max_traits: int | None = None,
    ) -> dict[str, str]:
        """Randomly assign 1..N distinct traits, each with a random level.

        Returns a mapping of ``trait name -> chosen level`` suitable for the
        ``AgentProfile.traits`` dict. Empty if the role defines no traits.
        """
        available = self.traits_for(role)
        if not available:
            return {}

        upper = len(available) if max_traits is None else min(max_traits, len(available))
        lower = max(1, min(min_traits, upper))
        count = rng.randint(lower, upper)

        chosen = rng.sample(available, count)
        return {
            trait.name: rng.choice(trait.levels)
            for trait in chosen
        }


__all__ = [
    "DEFAULT_TRAITS_DIR",
    "ROLE_TRAIT_FILES",
    "AgentTraitLibrary",
    "TraitDefinition",
]
```

### `examples/run_emotion_trajectory.py`(新增)

完整内容见仓库该路径(约 378 行,本次会话一次性写入,之后未改)。结构:`Tick` / `DEFAULT_SCRIPT`(每 tick 一句他人 utterance + 一个动作)、`seed_vitals` / `step_vitals`(确定性生理 stepper)、`enrich_traits`、`fake_llm_call`(无 key 烟雾测试)、`TickRecord` / `TrajectoryRecord`、`run()`(init_emotion → 逐 tick step_emotion)、`render()`、`main()`(`--scenario/--emotion/--trait-seed/--json/--output`)。

---

## 修改文件

### `ed_world_model/scenario_loader.py`

```diff
 from copy import deepcopy
 import json
+import random
 from collections.abc import Mapping
 from pathlib import Path
 from typing import Any

+from ed_world_model.agents.agent_traits import AgentTraitLibrary
 from ed_world_model.constants import DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS
```
```diff
+# Only these roles are emotional agents that carry traits; the clinician and
+# nurse exist only to drive the simulation and remain trait-free.
+TRAIT_BEARING_ROLES = ("patient", "relative")
+
 DEFAULT_DISCLOSURE_RULES = (
```
```diff
     def __init__(
         self,
         *,
         strict: bool = False,
         default_disclosure_rules: str = DEFAULT_DISCLOSURE_RULES,
+        randomize_traits: bool = True,
+        trait_seed: int | None = None,
+        trait_library: AgentTraitLibrary | None = None,
+        min_traits_per_role: int = 1,
+        max_traits_per_role: int | None = None,
     ) -> None:
         self.strict = strict
         self.default_disclosure_rules = default_disclosure_rules
+        self.randomize_traits = randomize_traits
+        self.trait_seed = trait_seed
+        self.trait_library = trait_library or AgentTraitLibrary()
+        self.min_traits_per_role = min_traits_per_role
+        self.max_traits_per_role = max_traits_per_role
```
```diff
-    def _build_agent_profiles(self, scenario: Mapping[str, Any]) -> AgentProfiles:
-        profiles = scenario.get("agent_profiles")
-        if profiles is None:
-            return AgentProfiles()
-
-        profiles_mapping = self._mapping_or_empty(profiles, "agent_profiles")
-        profile_data: dict[str, AgentProfile] = {}
-        for role in AgentProfiles.model_fields:
-            profile = profiles_mapping.get(role)
-            if profile is None:
-                continue
-            profile_mapping = self._mapping_or_empty(profile, f"agent_profiles.{role}")
-            filtered = {
-                key: profile_mapping[key]
-                for key in AgentProfile.model_fields
-                if key in profile_mapping
-            }
-            supplied_role = filtered.get("role")
-            if supplied_role is not None and supplied_role != role and self.strict:
-                raise ValueError(
-                    f"agent_profiles.{role}.role must be {role!r}, "
-                    f"got {supplied_role!r}."
-                )
-            filtered["role"] = role
-            try:
-                profile_data[role] = AgentProfile.model_validate(filtered)
-            except ValueError as exc:
-                if self.strict:
-                    raise ValueError(f"agent_profiles.{role} is invalid.") from exc
-
-        return AgentProfiles(**profile_data)
+    def _build_agent_profiles(self, scenario: Mapping[str, Any]) -> AgentProfiles:
+        profiles_mapping = self._mapping_or_empty(
+            scenario.get("agent_profiles"),
+            "agent_profiles",
+        )
+        rng = random.Random(self.trait_seed) if self.randomize_traits else None
+
+        profile_data: dict[str, AgentProfile] = {}
+        for role in AgentProfiles.model_fields:
+            profile = profiles_mapping.get(role)
+            if profile is not None:
+                profile_mapping = self._mapping_or_empty(
+                    profile, f"agent_profiles.{role}"
+                )
+                filtered = {
+                    key: profile_mapping[key]
+                    for key in AgentProfile.model_fields
+                    if key in profile_mapping
+                }
+                supplied_role = filtered.get("role")
+                if supplied_role is not None and supplied_role != role and self.strict:
+                    raise ValueError(
+                        f"agent_profiles.{role}.role must be {role!r}, "
+                        f"got {supplied_role!r}."
+                    )
+                filtered["role"] = role
+            else:
+                filtered = {"role": role}
+
+            # Scenarios never carry agent traits; sample them from the trait
+            # library. Only the patient and relative are emotional agents that
+            # carry traits — the clinician and nurse exist only to drive the
+            # simulation and stay trait-free.
+            if (
+                rng is not None
+                and role in TRAIT_BEARING_ROLES
+                and not filtered.get("traits")
+            ):
+                filtered["traits"] = self.trait_library.sample(
+                    role,
+                    rng=rng,
+                    min_traits=self.min_traits_per_role,
+                    max_traits=self.max_traits_per_role,
+                )
+
+            try:
+                profile_data[role] = AgentProfile.model_validate(filtered)
+            except ValueError as exc:
+                if self.strict:
+                    raise ValueError(f"agent_profiles.{role} is invalid.") from exc
+
+        return AgentProfiles(**profile_data)
```
```diff
 def load_scenario(
     source: str | Path | Mapping[str, Any],
     *,
     max_turns: int | None = None,
     strict: bool = False,
+    randomize_traits: bool = True,
+    trait_seed: int | None = None,
 ) -> GlobalState:
-    return ScenarioLoader(strict=strict).load(source, max_turns=max_turns)
+    return ScenarioLoader(
+        strict=strict,
+        randomize_traits=randomize_traits,
+        trait_seed=trait_seed,
+    ).load(source, max_turns=max_turns)
```

### `ed_world_model/state/global_state.py`

```diff
-class PatientEmotion(StateModel):
-    label: str = "Fear"
-    intensity: Literal["low", "medium", "high"] | None = None
-
-    label: str = "Sadness"
-    intensity: Literal["low", "medium", "high"] | None = None
-
-    label: str = "Helplessness"
-    intensity: Literal["low", "medium", "high"] | None = None
-
-    label: str = "Overwhelm"
-    intensity: Literal["low", "medium", "high"] | None = None
-
-    label: str = "Shame"
-    intensity: Literal["low", "medium", "high"] | None = None
-
-    label: str = "Defensiveness"
-    intensity: Literal["low", "medium", "high"] | None = None
-
-
-class PsychState(StateModel):
-    patient_emotion: PatientEmotion = Field(default_factory=PatientEmotion)
+EmotionLevel = Literal["not at all", "low", "high", "extreme"]
+
+
+class AgentEmotion(StateModel):
+    """Four-dimension ordinal emotion state for an emotional agent.
+
+    Only the patient and the relative carry emotion in the world model; the
+    clinician and nurse do not. This mirrors ``emotion_engine.EmotionState``
+    (identical dimensions and levels) ...
+    """
+
+    fear: EmotionLevel = "not at all"
+    sadness: EmotionLevel = "not at all"
+    overwhelm: EmotionLevel = "not at all"
+    shame: EmotionLevel = "not at all"
+    notes: str | None = None
+
+    def summary(self) -> dict[str, str]:
+        """Return the four emotion levels (excluding notes)."""
+        return {
+            "fear": self.fear,
+            "sadness": self.sadness,
+            "overwhelm": self.overwhelm,
+            "shame": self.shame,
+        }
+
+
+class PsychState(StateModel):
+    patient_emotion: AgentEmotion = Field(default_factory=AgentEmotion)
+    relative_emotion: AgentEmotion = Field(default_factory=AgentEmotion)
```
```diff
-    "Message",
-    "PatientEmotion",
+    "AgentEmotion",
+    "EmotionLevel",
+    "Message",
     "PatientInternalState",
```

### `ed_world_model/state/state_manager.py`

```diff
 from ed_world_model.state.global_state import (
+    AgentEmotion,
     DiagnosticResult,
     Event,
     Features,
     GlobalState,
     KnownFacts,
     Message,
-    PatientEmotion,
     PatientState,
```
```diff
-    def update_patient_emotion(
-        self,
-        patient_emotion: PatientEmotion | dict[str, Any] | None = None,
-        *,
-        label: str | None = None,
-        intensity: Literal["low", "medium", "high"] | None = None,
-        notes: str | None = None,
-        record_event: bool = False,
-    ) -> PatientEmotion:
-        if patient_emotion is not None and any(
-            value is not None for value in (label, intensity, notes)
-        ):
-            raise ValueError(
-                "Provide either patient_emotion or individual emotion fields, not both."
-            )
-
-        if patient_emotion is not None:
-            emotion_model = PatientEmotion.model_validate(patient_emotion)
-        else:
-            if label is None and intensity is None and notes is None:
-                raise ValueError("No patient emotion update was provided.")
-            emotion_data = self._state.psych_state.patient_emotion.model_dump()
-            if label is not None:
-                emotion_data["label"] = label
-            if intensity is not None:
-                emotion_data["intensity"] = intensity
-            if notes is not None:
-                emotion_data["notes"] = notes
-            emotion_model = PatientEmotion.model_validate(emotion_data)
-
-        self._state.psych_state.patient_emotion = emotion_model
-
-        if record_event:
-            self.record_event(
-                Event(
-                    type="emotion_update",
-                    turn_index=self._state.runtime_state.turn_index,
-                    payload={"label": emotion_model.label},
-                )
-            )
-
-        return emotion_model
+    def update_patient_emotion(
+        self, emotion=None, *, notes=None, record_event=False, **levels
+    ) -> AgentEmotion:
+        return self._update_emotion(
+            "patient_emotion", emotion, notes=notes,
+            record_event=record_event, levels=levels,
+        )
+
+    def update_relative_emotion(
+        self, emotion=None, *, notes=None, record_event=False, **levels
+    ) -> AgentEmotion:
+        return self._update_emotion(
+            "relative_emotion", emotion, notes=notes,
+            record_event=record_event, levels=levels,
+        )
+
+    def _update_emotion(
+        self, field_name, emotion, *, notes, record_event, levels
+    ) -> AgentEmotion:
+        patch = {**levels}
+        if notes is not None:
+            patch["notes"] = notes
+        if emotion is not None and patch:
+            raise ValueError(
+                "Provide either a full emotion or individual fields, not both."
+            )
+        if emotion is not None:
+            emotion_model = AgentEmotion.model_validate(emotion)
+        else:
+            if not patch:
+                raise ValueError("No emotion update was provided.")
+            emotion_data = getattr(self._state.psych_state, field_name).model_dump()
+            emotion_data.update(patch)
+            emotion_model = AgentEmotion.model_validate(emotion_data)
+        setattr(self._state.psych_state, field_name, emotion_model)
+        if record_event:
+            self.record_event(Event(
+                type="emotion_update",
+                turn_index=self._state.runtime_state.turn_index,
+                payload={"agent": field_name, **emotion_model.summary()},
+            ))
+        return emotion_model
```
> 上面 new 端为可读化精简版,完整签名/注释见源码。

### `ed_world_model/adapters/emotion_engine_adapter.py`

```diff
-from ed_world_model.emotion_engine import (
+from emotion_engine import (
     EMOTION_NAMES,
     EmotionState,
     EmotionTrajectory,
     _call_llm_json,
     init_emotion,
     step_emotion,
 )
-from ed_world_model.state.global_state import (
-    EmotionEntry,
-    PatientEmotion,
-)
+from ed_world_model.state.global_state import AgentEmotion
```
```diff
-- maps the world-model :class:`PatientEmotion` (a list of ``{label, intensity}``
-  entries) to the engine's :class:`EmotionState` (four named fields) and back,
+- maps the world-model :class:`AgentEmotion` (four named fields) to the engine's
+  :class:`EmotionState` (the same four fields) and back,
```
```diff
-    ) -> PatientEmotion:
+    ) -> AgentEmotion:
         self.last_error = None
         if self.scenario and not self._initialized:
             state = self._seed_initial_state(patient_profile_context)
         else:
-            state = _patient_emotion_to_state(current_patient_emotion)
+            state = _agent_emotion_to_state(current_patient_emotion)
```
```diff
-        patient_emotion = _state_to_patient_emotion(trajectory.state)
-        if reasoning_lines:
-            patient_emotion.notes = "\n".join(reasoning_lines)
-        return patient_emotion
+        agent_emotion = _state_to_agent_emotion(trajectory.state)
+        if reasoning_lines:
+            agent_emotion.notes = "\n".join(reasoning_lines)
+        return agent_emotion
```
```diff
-def _patient_emotion_to_state(value: Any) -> EmotionState:
-    try:
-        emotion = (value if isinstance(value, PatientEmotion)
-                   else PatientEmotion.model_validate(value))
-    except (ValueError, TypeError):
-        return EmotionState()
-    levels = {name: "not at all" for name in EMOTION_NAMES}
-    for entry in emotion.emotions:
-        name = entry.label.lower()
-        if name in levels:
-            levels[name] = entry.intensity
-    return EmotionState(**levels)
-
-def _state_to_patient_emotion(state: EmotionState) -> PatientEmotion:
-    emotions = [EmotionEntry(label=name.capitalize(), intensity=state.level_of(name))
-                for name in EMOTION_NAMES]
-    return PatientEmotion(emotions=emotions)
+def _agent_emotion_to_state(value: Any) -> EmotionState:
+    try:
+        emotion = (value if isinstance(value, AgentEmotion)
+                   else AgentEmotion.model_validate(value))
+    except (ValueError, TypeError):
+        return EmotionState()
+    return EmotionState(**{n: emotion.summary()[n] for n in EMOTION_NAMES})
+
+def _state_to_agent_emotion(state: EmotionState) -> AgentEmotion:
+    return AgentEmotion(**{n: state.level_of(n) for n in EMOTION_NAMES})
```

### `ed_world_model/orchestration/turn_loop.py`

```diff
         action_validator: ActionValidator | None = None,
         emotion_engine: Any | None = None,
+        relative_emotion_engine: Any | None = None,
         fact_extractor: FactExtractor | None = None,
     ) -> None:
         ...
         self.emotion_engine = emotion_engine or NoopEmotionEngine()
+        self.relative_emotion_engine = relative_emotion_engine or NoopEmotionEngine()
```
```diff
-        if committed_messages:
-            emotion_output = self.emotion_engine.predict(
-                self.state_manager.state.psych_state.patient_emotion,
-                conversation_input=committed_messages,
-                recent_messages=[_dump(m) for m in ...messages],
-                patient_profile_context=_dump(...agent_profiles.patient),
-            )
-            self.state_manager.update_patient_emotion(emotion_output, record_event=True)
+        if committed_messages:
+            recent_messages = [_dump(m) for m in ...messages]
+            patient_output = self.emotion_engine.predict(
+                self.state_manager.state.psych_state.patient_emotion,
+                conversation_input=committed_messages,
+                recent_messages=recent_messages,
+                patient_profile_context=_dump(...agent_profiles.patient),
+            )
+            self.state_manager.update_patient_emotion(patient_output, record_event=True)
+
+            relative_output = self.relative_emotion_engine.predict(
+                self.state_manager.state.psych_state.relative_emotion,
+                conversation_input=committed_messages,
+                recent_messages=recent_messages,
+                patient_profile_context=_dump(...agent_profiles.relative),
+            )
+            self.state_manager.update_relative_emotion(relative_output, record_event=True)
```

### `ed_world_model/orchestration/runner.py`

```diff
         emotion_engine: Any | None = None,
+        relative_emotion_engine: Any | None = None,
         orchestrator: Orchestrator | None = None,
         ...
         self.emotion_engine = emotion_engine or NoopEmotionEngine()
+        self.relative_emotion_engine = relative_emotion_engine or NoopEmotionEngine()
         ...
         self.turn_loop = TurnLoop(
             ...
             emotion_engine=self.emotion_engine,
+            relative_emotion_engine=self.relative_emotion_engine,
             fact_extractor=fact_extractor,
         )
```

### `ed_world_model/orchestration/observation_builder.py`

```diff
         return {
             "relationship_profile": _dump(global_state.agent_profiles.relative),
+            "relative_emotion": _dump(global_state.psych_state.relative_emotion),
             "recent_messages": _dump_list(
```

### `ed_world_model/agents/schemas.py`

```diff
     role: AgentRole
     observation: dict[str, Any]
     profile: AgentProfile | None = None
-    emotion_context: dict[str, Any] | None = None
     recent_messages: list[dict[str, Any]] | None = None
     turn_index: NonNegativeInt | None = None
```

### `ed_world_model/agents/prompting.py`

```diff
     agent_input = _coerce_runtime_input(runtime_input, expected_role="patient")
     return _compose_prompt(
         agent_input,
-        emotion_label="Patient emotion context",
         role_instructions=[
             "Patient output rules:",
             (
                 "- Use only the patient_internal_state subset present in the "
-                "observation, plus visible messages and patient emotion."
+                "observation, plus visible messages and the patient_emotion in "
+                "the observation."
             ),
```
```diff
             "- Stay silent unless asked or explicitly selected.",
-            "- There is no relative emotion transition in v1.3.1.",
+            (
+                "- You may use the relative_emotion in the observation only to "
+                "shape wording and tone, never to invent facts."
+            ),
             "Relative JSON shape:",
```
```diff
 def _compose_prompt(
     runtime_input: AgentRuntimeInput,
     *,
     role_instructions: list[str],
-    emotion_label: str = "Emotion context",
 ) -> str:
     ...
     if runtime_input.profile is not None:
         ...
-
-    if runtime_input.emotion_context is not None:
-        sections.append(
-            f"{emotion_label}:\n{_format_json(runtime_input.emotion_context)}"
-        )
```

### `ed_world_model/agents/patient.py`

```diff
     def build_verbal_prompt(self, observation, decision, *,
-        emotion_context: Mapping[str, Any] | None = None,
         recent_messages=None, turn_index=None) -> str:
         return build_patient_verbal_prompt(
             observation, decision, profile=self._profile,
-            emotion_context=emotion_context,
             recent_messages=recent_messages, turn_index=turn_index)

     def build_prompt(self, observation, *,
-        emotion_context: Mapping[str, Any] | None = None,
         recent_messages=None, turn_index=None) -> str: ...

     def generate(self, observation, *,
-        emotion_context: Mapping[str, Any] | None = None,
         recent_messages=None, turn_index=None) -> AgentProposal:
         ...
         verbal_prompt = self.build_verbal_prompt(
             observation, decision,
-            emotion_context=emotion_context,
             recent_messages=recent_messages, turn_index=turn_index)
```
```diff
-            "personality, or emotion_context. Do not use style or emotion to "
+            "personality, or patient_emotion. Do not use style or emotion to "
             "decide what facts should be said."
```
```diff
         "Use the VerbalDecision as the source of truth for what to say.",
         (
-            "You may use AgentProfile, traits, and emotion context only to "
-            "shape wording and tone."
+            "You may use AgentProfile, traits, and the patient_emotion in your "
+            "observation only to shape wording and tone."
         ),
```
```diff
 def build_patient_verbal_prompt(observation, decision, *,
     profile=None,
-    emotion_context: Mapping[str, Any] | None = None,
     recent_messages=None, turn_index=None) -> str:
     ...
     if patient_profile is not None:
         ...traits...
-    if emotion_context is not None:
-        sections.append(f"Patient emotion context:\n{_format_json(emotion_context)}")
```
```diff
 def build_patient_prompt(observation, *, profile=None,
-    emotion_context: Mapping[str, Any] | None = None,
     recent_messages=None, turn_index=None) -> str:
     return build_patient_verbal_prompt(observation, VerbalDecision(...),
-        profile=profile, emotion_context=emotion_context,
+        profile=profile,
         recent_messages=recent_messages, turn_index=turn_index)
```
```diff
 _PATIENT_DECISION_EXCLUDED_KEYS = {
     "agent_profile",
     "communication_style",
-    "emotion_context",
     "patient_emotion",
     "profile",
     "traits",
 }
```

### `ed_world_model/agents/clinician.py`

```diff
-            "personality, or emotion_context. Do not use style or emotion to "
+            "or personality. Do not use style to "
             "choose the clinical action or communication content."
```
```diff
 def build_clinician_verbal_prompt(observation, action, decision, *,
     profile=None,
-    emotion_context: Mapping[str, Any] | None = None,
     turn_index=None) -> str:
     ...
-        ("You may use AgentProfile, traits, and emotion context only to "
-         "shape wording and tone."),
-        ("Do not let profile, traits, or emotion change observed facts, "
+        ("You may use AgentProfile and traits only to shape wording and tone."),
+        ("Do not let profile or traits change observed facts, "
          "selected action, available results, or grounded clinical content."),
     ...
-    if emotion_context is not None:
-        sections.append(f"Emotion context:\n{_format_json(emotion_context)}")
```
```diff
 _CLINICIAN_DECISION_EXCLUDED_KEYS = {
     "agent_profile",
     "communication_style",
-    "emotion_context",
     "patient_emotion",
     "patient_internal_state",
     "profile",
     "traits",
     "truth_state",
 }
```

### `ed_world_model/agents/relative.py`

```diff
         "Do not read monitor-level exact vitals unless observation explicitly provides them.",
-        "No relative emotion transition is modeled in v1.3.1.",
         (
             "Before deciding whether to speak, silently reason ..."
         ),
         ANTI_REPETITION_RULE,
         (
             "Your verbal_action.content should reflect your AgentProfile traits ..."
         ),
+        (
+            "You may also use the relative_emotion in your observation "
+            "(fear, sadness, overwhelm, shame) to shape wording and tone."
+        ),
         (
-            "Profile traits shape how you speak; they must not cause you to "
-            "invent hidden clinical facts, exact vitals, or care-blocking behavior."
+            "Profile traits and emotion shape how you speak; they must not cause "
+            "you to invent hidden clinical facts, exact vitals, or care-blocking "
+            "behavior."
         ),
```

---

## 修改文件(测试)

> 以下为要点 hunk;均为把旧 emotion schema / `emotion_context` 断言改为新 schema / observation 通道。

- **`tests/test_global_state.py`** — import `PatientEmotion`→`AgentEmotion`;`patient_emotion` 用例改 `fear/sadness`;`test_patient_emotion_intensity_is_categorical`→`test_agent_emotion_levels_are_categorical`(校验 4 档 + `summary()`,拒绝旧 `medium`)。
- **`tests/test_noop_emotion.py`** — 全量改用 `AgentEmotion(...)`、`psych_state={"patient_emotion": {...}}` 新字段;`result.label`→`result.notes`。
- **`tests/test_state_manager.py`** — import 换 `AgentEmotion`;`psych_state` 初值改新字段;`update_patient_emotion(AgentEmotion(fear=...))`;新增 `test_update_relative_emotion_updates_relative_only`;事件 payload 断言改 `{"agent": "patient_emotion", **summary}`;错误信息断言改 “No emotion update…” / “either a full emotion…”。
- **`tests/test_observation_builder.py`** — `_rich_state()` 的 `patient_emotion` 改新字段 + 加 `relative_emotion`;`test_patient_observation_includes_patient_emotion` 改 `fear/overwhelm`;新增 `test_relative_observation_includes_relative_emotion`。
- **`tests/test_turn_loop_stub.py`** — import 加 `AgentEmotion`;`.label` 断言改为 `current_patient_emotion == AgentEmotion().model_dump()` 且 `psych_state.patient_emotion == AgentEmotion()`。
- **`tests/test_patient_agent.py`** — `_observation()` 的 `patient_emotion` 改新字段;`_prompt()`/`_verbal_prompt()`/`generate()` 去掉 `emotion_context`;decision 排除用例去掉 `emotion_context`;`test_build_prompt_includes_emotion_context_when_provided`→`test_verbal_prompt_includes_patient_emotion_from_observation`。
- **`tests/test_clinician_agent.py`** — decision 排除用例把 `emotion_context` 键换成 `patient_emotion`,断言相应改为 `'"patient_emotion"' not in prompt`。
- **`tests/test_agent_prompting.py`** — 去掉 `emotion_context`/“Emotion context” 断言;patient 用例改用 observation 里的 `patient_emotion`;relative 用例改为断言含 `relative_emotion`、不含旧 “no relative emotion transition”。
- **`tests/test_relative_agent.py`** — 断言旧句 “No relative emotion transition is modeled in v1.3.1.” **不再出现**,改断言含 “relative_emotion in your observation”。

---

## 删除文件

- `ed_world_model/agents/agentTrait/ClinicianTrait.json`(整文件删除)
- `ed_world_model/agents/agentTrait/NurseTrait.json`(整文件删除)
