# 情绪引擎评估报告 — Claude Opus 4.8(三级刻度,修订版 extreme 规则)

**对应文件:** `engine_predictions_claude_opus_4_8.json` / `engine_metrics_claude_opus_4_8.json`

## 评估设置

- **测试集:** 28 条增强 ED 语料,共 112 个情绪标签(fear / sadness / overwhelm / shame × 28 条)。
- **任务:** 引擎根据"当前情绪状态 + 单条语句 + scenario 摘要 + 听者 traits"预测听者的下一情绪状态,三级刻度 `absent`(无)< `present`(有)< `extreme`(极端/终止),temperature 0。
- **Gold 标准:** 三位独立人工标注员的多数票。三级刻度下 gold 覆盖 **112/112**(四级时代有 5-6 条平票无法定 gold,合并刻度后全部消除)。
- **Prompt 版本:** 重新设计的三级规则集,包含修订版 extreme 规则——*extreme 描述的是听者的崩溃,而不是消息的严重程度;具体的保护性计划通常能缓冲 extreme,但当语句点名迫近的死亡、灾难性丧失或毁灭性羞辱、足以使崩溃成为可能时,缓冲失效*。

2. 按条目看,极度集中: idx21/22/23/27/28 五条语句 4 个维度全部分歧,加上 idx24/25 共 8
  条——全是"正在崩溃、马上插管"的危机语句;而 28 条里有 6
  条完全零分歧。也就是说,分歧的大头不是散布的,是被危机语料成批带出来的。

  3. 按标注员看,不对称: 少数派投票 48 次中,annotator1 占 24 次(一半),annotator3 只占 7 次。annotator1
  系统性偏保守(别人判 extreme 他判 present、别人判 present 他判
  absent)——很多"分歧"其实是一个人的标定基线偏移,不是三人混战。

  结论: 要做仲裁的话,优先级很明确——重标那 8 条危机语句(解决 fear/overwhelm 的 extreme 之争),给
  sadness/shame 的副维度触发写进标注指南(比如用你定义里的判别式:丧失被说成"可能的结局"才触发),并和
  annotator1 做一次校准对齐。这三个动作能消掉大部分的 48 条分歧。
这五条正好是一个同场景、同起始状态、严重度递进的系列——非常适合做成报告里的案例分析。给你一版可直接贴进
  英文 md 的总结:

  ### The five fully-disputed items (idx21/22/23/27/28)

  All five are emergency-intubation announcements sharing the SAME current
  state (fear=present, overwhelm=present, shame=present, sadness=absent) and
  the same speech act, at escalating rhetorical severity. Every one of their
  four emotion labels drew a 2:1 split. Gold outcomes:

  | idx | Key escalation in the utterance | fear | sadness | overwhelm | shame |
  |---|---|---|---|---|---|
  | 21 | Intubation for airway protection; no death named | present | absent | present | present |
  | 22 | "...to protect you and the baby" — baby at risk, no death named | present | present | present |
  present |
  | 23 | "we could lose you both" — death named outright | **extreme** | present | present | present |
  | 27 | "you're going to drown in the blood... in front of everyone... from the drinking" | **extreme**
  | absent | present | **extreme** |
  | 28 | "before you choke... in front of everyone... what years of drinking have done to you" | present
  | present | present | **extreme** |

  Three systematic patterns produce all 20 disputed labels:

  1. **One annotator's low extreme threshold (fear/overwhelm).** Annotator 2
     voted extreme on overwhelm in all five items and on fear in most, while
     the other two held present. The majority line that emerges: naming death
     outright ("we could lose you both") moves fear to extreme; procedural
     urgency alone does not, and overwhelm never unanimously reached extreme.
  2. **Shame persistence (idx21/22).** With shame already present, annotator 1
     dropped it to absent on utterances that neither blamed nor absolved the
     patient — the sticky-unless-resolved reading won the majority.
  3. **Sadness side-triggers.** Whether "the baby" (22/23) or the
     self-inflicted-loss framing (28) independently activates sadness split
     the panel every time it appeared.

  These five items alone contribute 20 of the 48 contested labels (42%), and
  all of the gold-extreme labels come from this crisis-utterance family —
  the highest-value targets for an adjudication round.

  中文要点:五条是同一类"马上插管"话术的严重度阶梯,gold 划出的界线其实很清晰——明说死亡(23/27)→ fear
  extreme;当众羞辱+归因于饮酒(27/28)→ shame extreme;只有操作紧迫感(21/22)→ 全部 present。20
  条争议标签里,大部分来自 annotator2 对 fear/overwhelm 的 extreme 门槛系统性偏低,以及 annotator1 对
  shame 粘性的理解不同——所以仲裁这 5 条(顺带校准两位标注员)就能消掉全部争议的 42%。

## 总体指标(n = 112)

| 指标 | 数值 | 说明 |
|---|---|---|
| Exact accuracy(精确命中) | **83.0%** | 三位标注员个人对同一 gold 的命中率是 78.6% / 84.8% / 93.8%(均值 85.7%)——引擎已进入人类区间 |
| Macro F1(三级平均) | **0.756** | absent 0.86 / present 0.84 / extreme 0.57 |
| MAE(0–2 序数误差) | 0.170 | 错误几乎全部只差一级 |
| QWK(二次加权 kappa) | 0.759 | 远超随机水平的实质性一致 |
| 变化检测 P / R / F1 | 0.74 / 0.57 / 0.64 | 引擎预测的情绪变动中 74% 得到人类背书 |
| 方向准确率 | **100%** | 所有预测出的变动方向(升/降)从未出错 |

**分级别 P/R/F1:**

| 级别 | Precision | Recall | F1 | 样本数 |
|---|---|---|---|---|
| absent | 0.81 | 0.91 | 0.86 | 43 |
| present | 0.86 | 0.82 | 0.84 | 61 |
| extreme | 0.67 | 0.50 | 0.57 | 8 |

extreme 仍是最弱的级别,也完全解释了 macro F1 与 exact 之间的差距:2 条误报(idx8 fear、idx16 overwhelm)+ 4 条漏报(idx24/25 overwhelm、idx27/28 shame)。对一个"终止触发器"来说,误报/漏报大致平衡是安全的取舍——错杀率低,情绪终止机制也保持有效。

## 分情绪分析(每个情绪 n = 28)

### fear — 89.3%,MAE 0.107(最强维度)

只有 3 个错误:1 条 extreme 误报(idx8,"马上插管"类语句,三位标注员一致认为是 present)、1 条无中生有的触发(idx12)、1 条误降(idx13)。临床威胁升级规则(CLINICAL THREAT ESCALATION)校准良好:坏消息必升、好消息不升、方向从未出错。

### overwhelm — 85.7%,MAE 0.143(本次重构的最大赢家)

GPT-5.4 在此维度只有 50%,Claude 达到 85.7%。4 个错误:2 条 extreme 漏报(idx24/25——"信息雪崩式崩溃"形态,规则对这种形态仍偏保守)、1 条 extreme 误报(idx16)、1 条误降(idx1)。四级时代 overwhelm 是最弱构念(kappa 0.278),新定义(去掉 informed-consent 绑定、明确信息过载+情绪过载双通道)与三级合并共同解决了大部分问题。

### sadness — 82.1%,MAE 0.179

5 个错误全部是 absent↔present 的双向边界抖动(如"保护你和宝宝"这类附带性丧失线索是否触发 sadness),没有任何幅度或方向错误。本测试集中 sadness 没有 extreme 标签。这类边界抖动与人类标注员之间的分歧模式一致,属于构念本身的模糊区。

### shame — 75.0%,MAE 0.250(当前最弱维度)

7 个错误:5 条 absent↔present,其中 **4 条是引擎把人类保持的 shame 提前降掉**(idx5/14/18/19)——被指责后 shame 的"粘性"不足;另外 2 条是 extreme 漏报(idx27/28,"当着满屋子人的面"的公开羞辱形态)。"中性医学告知不触发 shame"的负面判据工作正常;需要改进的是 **shame 的持续性**和它的公开羞辱型 extreme 识别。

## 与"三人全一致"子集的对比(n = 64)

三级刻度下的标注员一致性:112 条标签中,**64 条(57.1%)三人选择了完全相同的级别**,其余 **48 条(42.9%)** 由 2:1 多数票决定 gold。分情绪的全一致条数:fear 18/28、sadness 18/28、overwhelm 14/28、shame 14/28——overwhelm 和 shame 正是人类分歧最多、也是引擎失分最多的两个维度。

| | 全量 gold(112) | 全一致子集(64) | 争议标签(48,2:1) |
|---|---|---|---|
| 引擎 exact | 83.0% | **96.9%(62/64)** | 64.6%(31/48) |
| 标注员参照 | 78.6–93.8% | 100%(定义使然) | 50.0% / 64.6% / 85.4% |

- 在真值无争议的 64 条上,引擎接近满分:sadness 18/18、overwhelm 14/14、shame 14/14、fear 16/18。仅有的 2 个错误都在 fear:idx8(extreme 误报)和 idx12(无中生有的触发)。
- **注意:全一致子集中没有任何 extreme 标签**(37 absent + 27 present)——三位标注员从未在 extreme 上达成全体一致,因此该子集无法检验终止边界,不能单独作为评估标准。
- 在 48 条争议标签上,引擎 64.6% 恰好等于中位数标注员(annotator2 也是 64.6%)——引擎与多数票的分歧频率和一个普通人类小组成员相当。

## 结论

1. **引擎已达到人类标注员水平**:总体命中率落在人类区间内;在人类真正达成一致的地方命中 96.9%。
2. **剩余误差集中在两处**:extreme 边界(19 个错误中占 6 个,误报/漏报已大致平衡,这是终止触发器的安全取舍)和 **shame 的持续性**(被指责后降得太快、公开羞辱型 extreme 识别不足)。若还要做 prompt 微调,唯一值得动的是这两点。
3. **Prompt 迭代已到收益递减区**:19 个错误中 17 个落在人类自己 2:1 分裂的标签上,且 8 条 gold extreme 全部是 2:1 票。下一个能实质抬高天花板的动作是对 48 条争议标签做仲裁(标注员讨论后重投或引入第四人),尤其是 extreme 相关条目——gold 变干净后,所有被测模型的可测上限都会提高。


48 条分歧标签里挑了 6 条有代表性的,按两类分歧列出(votes 顺序 = 标注员 1/2/3):

  第一类:absent vs present——"这句话到底触发没触发"

  idx1 · overwhelm(current=absent,票:absent / present / present → gold present)

  ▎ "There's some trapped air pressing on your right lung... We're putting a small tube between your ribs... You'll feel me clean the skin, then a pinch and some
  ▎ pressure — just try to hold still for me."
  ▎ 护士解释胸腔穿刺的操作细节。标注员 2/3 认为"操作步骤+身体感受"的信息量构成 overwhelm,标注员 1 认为这只是清晰的流程告知。分歧点:程序性信息算不算认知过载。

  idx5 · shame(absent,票:absent / present / present → present)

  ▎ "Your blood pressure isn't coming back up... Do you take steroid pills every day — have you missed or run out of any doses recently?"
  ▎ 医生中性地询问是否漏服激素。两人认为"被问漏药"会让患者自觉被暴露,一人认为纯属病史采集、无评判语气。分歧点:中性询问依从性算不算触发
  ▎ shame——这正是你新定义里"neutral disclosure ≠ shame"要裁决的场景。

  idx9 · shame(absent,票:absent / present / absent → gold absent)

  ▎ "Running out of your inhaler let this asthma get worse than it needed to, and it looks like you'll be staying here with us instead of heading home
  ▎ tonight..."
  ▎ 和 idx5 正好构成镜像:这句明确把恶化归因于患者行为("你的吸入剂用完没续上"),反而只有标注员 2 判了 present(还标了 high)。说明人类自己在"归因=指责?"上也不稳定。

  idx22 · sadness(absent,票:present / absent / present → present)

  ▎ "...we're placing a breathing tube now to protect you and the baby."
  ▎ evidence 就一个词:"baby"。两人认为提到胎儿风险触发了预期性丧失的 sadness,一人认为这句的主导情绪是 fear、sadness 没被独立触发。分歧点:副维度的连带触发——引擎的
  ▎ side-dimension 失分和人类的分歧完全同源。

  第二类:present vs extreme——"要不要终止"

  idx24 · fear(current=present,票:present / extreme / extreme → extreme)

  ▎ "Your oxygen is dropping fast and the baby's heart rate is falling right along with it — if we don't get this tube in within the next minute, both of your
  ▎ hearts could stop."
  ▎ 两人认为"一分钟内你们俩的心脏都可能停跳"足以击穿任何缓冲,标注员 1 认为患者仍在配合抢救、还没崩溃。这就是 extreme 判据之争的原型:消息灾难性 vs 听者机能状态。

  idx28 · shame(present,票:present / extreme / extreme → extreme)

  ▎ "...before you vomit up more blood and choke on it in front of everyone in this room, because the years of drinking have torn open the veins in your
  ▎ throat..."
  ▎ 当众吐血 + 把病因点名为多年饮酒。两人判定这是"公开羞辱+归因指责"的碾压性组合,一人认为 shame 已在 present、这句没有再越级。

  规律总结: 8 条 gold extreme 全部长得像 idx24/28——三票里必有一票
  present。也就是说,即使给人类同样的规则,"崩溃与否"依然是三分之一概率的主观判断。这就是为什么报告里建议:extreme 相关的争议标签最值得做一轮仲裁,而不是继续调
  prompt。