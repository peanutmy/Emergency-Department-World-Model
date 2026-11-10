# Emotion Engine Methods：作者备注

配套正文：[英文 Methods 草稿](emotion_engine_methods_en.md)。本文件供作者检查，不作为论文 Methods 正文直接粘贴。

## 1. 本次草稿如何组织

正文按论文方法部分组织，依次解释状态表示、输入与触发条件、提示词组成、三种策略、响应处理，以及与世界模型的连接。它描述当前实现，不报告未运行的比较结果，不声称已训练专门的情绪模型，也不声称教育效果已经获得验证。

正文采用 present tense 描述系统构成。正式投稿时可根据期刊风格统一时态、编号和参考文献格式。

## 2. 正式投稿前，必须与实验版本对齐

以下行为目前确实存在，因此草稿没有写成更理想的实现。如果之后修复代码并重新运行实验，应相应修改 Methods；若实验使用旧版本，则不能只删除这些说明而不处理其对结果的影响。

| 当前情况 | 正文采取的写法 | 后续处理 |
| --- | --- | --- |
| recent_messages 可能包含当前和同轮后续消息 | 明确不能保证上下文只包含此前的信息 | 修复时间顺序，并确定哪些实验需要重新运行；离线评估未传 dialogue，应分别说明 |
| 缺少情绪字段时可能自动补 absent | 明确记载当前默认补值行为 | 建议严格要求完整响应，或明确采用保留旧值的其他政策，再更新正文 |
| patient extreme 才触发情绪终止，relative extreme 不会 | 明确区分两个角色 | 确定最终要采用的终止语义 |
| adapter 可能在 extreme 后继续处理当前批次的话语 | 明确终止检查由 turn loop 随后执行 | 若改为逐句立即停止，需要更新方法和实验版本 |
| 部分转换条件只写在 prompt 中 | 区分提示约束与代码强制规则 | 不要写成所有医学／心理语义均由验证器保证 |
| 降至 absent 的完全解决与部分缓解表述有歧义 | 正文概括等级定义及规则，不声称不存在歧义 | 在冻结 prompt 前统一标准；审查 few-shot 示例是否一致 |

这些问题不是这次写作时新增的发现或修复；它们来自前面的代码审查。论文不能以文本润色代替实验实现的一致性检查。

## 3. 还需要填写的实验配置

代码默认值不能证明论文实验实际使用了什么配置。应从真正运行的命令、日志或实验记录中确认以下信息，并放入 Experimental Setup 或 Implementation Details：

| 信息 | 应报告的内容 |
| --- | --- |
| 模型与接口 | 实际 provider、model identifier/version、调用日期、接口或代理服务 |
| 生成参数 | temperature、输出 token 上限、启用的 reasoning/thinking 设置及预算；注明不支持或未使用的参数 |
| 代码和 prompt 版本 | 对应提交或快照、完整定义与规则、few-shot 示例及固定顺序 |
| 运行入口 | 独立 `step_emotion`、完整模拟还是离线评估；确认是否启用了真实 adapter |
| 上下文条件 | 是否包含 scenario、traits、dialogue，以及标注者是否看到相同信息 |
| 错误与重试 | 调用失败、无效 JSON、字段缺失的处理方式，以及实际重试次数 |
| 比较设计 | 相同评估条目、开发／测试划分、重复次数、指标和不确定性 |

`emotion_engine.py` 的内置模型调用当前设置 temperature=0.0、max_tokens=1024，但完整模拟和评估脚本可以注入其他调用函数，因此不能把这两个数值未经核对就写成所有实验的统一配置。同样，仓库中的默认模型名称不应自动被写成论文实际使用的模型。

可在确认后添加如下句式，方括号内容均为待填写信息：

> The transition model was accessed through [PROVIDER/INTERFACE] using [MODEL IDENTIFIER AND VERSION]. Generation used [TEMPERATURE OR NOT APPLICABLE], a maximum output budget of [TOKENS], and [REASONING/THINKING CONFIGURATION]. We evaluated the prompt variants on [DATASET/SPLIT] under matched input conditions, with [REPETITIONS] runs and [ERROR/RETRY POLICY]. The complete prompts and implementation snapshot are provided in [SUPPLEMENT/REPOSITORY LOCATION].

## 4. Prompt 策略的论文命名

建议使用 `zero-shot`、`few-shot`、`structured-review` 作为正文中的明确名称，并在首次出现时说明第三种策略在代码中名为 `react`。它没有外部工具动作或新观察，不宜直接声称实现了原始 ReAct。

目前 four-shot 的四个示例不包含 sadness 的变化或 extreme 输出；草稿如实写明覆盖范围，没有声称四种情绪和所有转换均已被示范。

## 5. 与代码的对应关系

| 正文内容 | 主要依据 |
| --- | --- |
| 状态定义、初始化、模板、策略、解析和 extreme 保持 | [emotion_engine.py](../emotion_engine.py) |
| 消息筛选、上下文拼接、逐句处理、错误行为、notes | [emotion_engine_adapter.py](../ed_world_model/adapters/emotion_engine_adapter.py) |
| 感知条件、角色调用、批次与终止检查顺序 | [turn_loop.py](../ed_world_model/orchestration/turn_loop.py) |
| GlobalState 情绪写入与事件 | [state_manager.py](../ed_world_model/state/state_manager.py) |
| 对患者／家属的后续观察 | [observation_builder.py](../ed_world_model/orchestration/observation_builder.py) |
| patient-only 情绪终止 | [termination.py](../ed_world_model/state/termination.py) |
| Noop 默认路径 | [runner.py](../ed_world_model/orchestration/runner.py) |
| 离线评估的上下文与响应记录 | [evaluate_emotion_engine.py](../tools/evaluate_emotion_engine.py) |

## 6. 外部引用的用途

- [Brown et al. (2020), Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)：仅用于解释通过 prompt 中的示例进行 in-context learning，而非本项目模型训练或性能的证据。
- [Yao et al. (2023), ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)：用于区分原始交替推理与行动框架和本项目的单次内部检查指导。

四维情绪集合与三档等级在正文中被定位为本项目的操作性设计。当前没有补写其临床验证、量表来源或最优性证据。
