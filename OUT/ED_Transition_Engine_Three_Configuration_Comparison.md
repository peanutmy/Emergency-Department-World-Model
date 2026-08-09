# Three-Configuration Evaluation Report

## 1. Executive Summary

This report compares the best former hybrid configuration, the best pure-LLM configuration, and the new two-stage direction-first hybrid.

The evaluation contains:

- 57 transition pairs
- 159 evaluated vital-sign observations
- 58 ground-truth decreases
- 56 ground-truth increases
- 45 ground-truth stable observations

The primary metric is vital-level direction accuracy. 

| Configuration | Correct directions | Vital-level accuracy | Normalized RMSE |
|---|---:|---:|---:|
| Former hybrid: GPT-5, 3-shot static | 102/159 | 64.15% | 3.520 |
| Pure LLM: GPT-5.5, 3-shot kind-hint matched | **113/159** | **71.07%** | 3.474 |
| Two-stage hybrid: GPT-5, 3-shot static | 106/159 | 66.67% | **3.373** |
| Two-stage hybrid calibrated v2: GPT-5, 3-shot static | 90/159 | 56.60% | 3.542 |
| Two-stage hybrid calibrated v3: GPT-5, 3-shot static | 110/159 | 69.18% | 3.430 |

The pure LLM remains the best overall configuration. It has the highest direction accuracy and the best normalized MAE while maintaining substantially better balance across increase, decrease, and stable categories.

The two-stage hybrid improves over the former hybrid by 4 correct observations, or 2.52 percentage points. It is also the best configuration for `no_action`and for recognizing true increases and decreases. However, it **overcorrects** the former hybrid's conservative behavior and frequently predicts a change when
the ground truth is stable.

## 2. Compared Configurations

### 2.1 Former Hybrid

- Model: GPT-5
- Prompting: 3-shot static
- Structure: rule-based prediction plus an LLM-supplied signed residual

### 2.2 Pure LLM

- Model: GPT-5.5
- Prompting: 3-shot kind-hint matched
- Structure: LLM directly predicts absolute post-transition vital values

### 2.3 Two-Stage Hybrid

- Model: GPT-5
- Prompting: 3-shot static with fixed, role-based demonstrations
- Stage 1: LLM predicts `increase`, `decrease`, or `stable`
- Stage 2: LLM predicts an unsigned magnitude after the direction is fixed
- Rule engine: used as a magnitude reference and fallback

## 3. Evaluation Method

Direction categories use the repository's vital-specific dead zones:

| Vital | Stable dead zone |
|---|---:|
| HR | +/-5 bpm |
| BP_sys | +/-5 mmHg |
| BP_dia | +/-5 mmHg |
| RR | +/-2 breaths/min |
| O2Sat | +/-2 percentage points |
| T | +/-0.2 degrees |

## 4. Overall Direction Performance

### 4.1 Class Precision and Recall

| Configuration | Class | Precision | Recall | Support | Predicted |
|---|---|---:|---:|---:|---:|
| Former hybrid | Decrease | **84.38%** | 46.55% | 58 | 32 |
| Former hybrid | Increase | 76.92% | 71.43% | 56 | 52 |
| Former hybrid | Stable | 46.67% | **77.78%** | 45 | 75 |
| Pure LLM | Decrease | 84.21% | 55.17% | 58 | 38 |
| Pure LLM | Increase | **79.31%** | 82.14% | 56 | 58 |
| Pure LLM | Stable | 55.56% | **77.78%** | 45 | 63 |
| Two-stage hybrid | Decrease | 66.67% | **75.86%** | 58 | 66 |
| Two-stage hybrid | Increase | 66.67% | **85.71%** | 56 | 72 |
| Two-stage hybrid | Stable | **66.67%** | 31.11% | 45 | 21 |
| Two-stage hybrid v2 | Decrease | 95.00% | 32.76% | 58 | 20 |
| Two-stage hybrid v2 | Increase | 78.72% | 66.07% | 56 | 47 |
| Two-stage hybrid v2 | Stable | 36.96% | 75.56% | 45 | 92 |
| Two-stage hybrid v3 | Decrease | 74.07% | 68.97% | 58 | 54 |
| Two-stage hybrid v3 | Increase | 73.13% | **87.50%** | 56 | 67 |
| Two-stage hybrid v3 | Stable | 55.26% | 46.67% | 45 | 38 |

The former hybrid is conservative. It predicts stable 75 times even though stable appears only 45 times. This gives it high stable recall but low decrease recall.

The pure LLM is the most balanced configuration. Its predicted class distribution is closer to the ground-truth distribution, and it has the best increase precision while retaining strong stable recall.

The two-stage hybrid predicts non-stable directions much more often. This substantially improves increase and decrease recall, but stable recall falls to 31.11%.

## 5. Confusion Matrices

Rows are ground truth. Columns are engine predictions.

### 5.1 Former Hybrid

| Ground truth | Predicted decrease | Predicted increase | Predicted stable | Support |
|---|---:|---:|---:|---:|
| Decrease | 27 | 5 | 26 | 58 |
| Increase | 2 | 40 | 14 | 56 |
| Stable | 3 | 7 | 35 | 45 |

### 5.2 Pure LLM

| Ground truth | Predicted decrease | Predicted increase | Predicted stable | Support |
|---|---:|---:|---:|---:|
| Decrease | 32 | 6 | 20 | 58 |
| Increase | 2 | 46 | 8 | 56 |
| Stable | 4 | 6 | 35 | 45 |

### 5.3 Two-Stage Hybrid

| Ground truth | Predicted decrease | Predicted increase | Predicted stable | Support |
|---|---:|---:|---:|---:|
| Decrease | 44 | 8 | 6 | 58 |
| Increase | 7 | 48 | 1 | 56 |
| Stable | 15 | 16 | 14 | 45 |

## 6. Main Direction Failures

| Error type | Former hybrid | Pure LLM | Two-stage hybrid | Two-stage v2 | Two-stage v3 |
|---|---:|---:|---:|---:|---:|
| Decrease -> stable | 26 | 20 | **6** | 39 | 15 |
| Increase -> stable | 14 | 8 | **1** | 19 | 2 |
| Stable -> decrease | 3 | 4 | 15 | **1** | 9 |
| Stable -> increase | 7 | **6** | 16 | 10 | 15 |
| Decrease -> increase | 5 | 6 | 8 | **0** | 3 |
| Increase -> decrease | 2 | 2 | 7 | **0** | 5 |
| Total direction errors | 57 | **46** | 53 | 69 | 49 |

### 6.1 Former Hybrid

The former hybrid's main problem is conservative stable prediction:

- 40 of 57 errors are non-stable observations predicted as stable.
- Decrease-to-stable alone accounts for 26 errors.
- Decrease recall is only 46.55%.

This supports the earlier conclusion that the rule-plus-residual architecture anchors the LLM too strongly to small or stable rule-based changes.

### 6.2 Pure LLM

The pure LLM still exhibits some stable bias:

- 20 decreases are predicted as stable.
- 8 increases are predicted as stable.

However, it makes only 10 stable-to-non-stable errors and therefore maintains the best overall balance. Its principal remaining direction failure is decrease-to-stable.

### 6.3 Two-Stage Hybrid

The two-stage design solves most of the former conservative failure:

- Decrease-to-stable falls from 26 to 6.
- Increase-to-stable falls from 14 to 1.

The new dominant error is the reverse:

- 31 of 53 errors are stable observations predicted as changing.
- 16 stable observations are predicted as increases.
- 15 stable observations are predicted as decreases.

The design therefore changes the bias from "too stable" to "too active." It also increases complete direction reversals: 15 increase/decrease reversals, compared with 7 for the former hybrid and 8 for the pure LLM.

## 7. Performance by Vital

| Vital | Support | Former hybrid | Pure LLM | Two-stage hybrid | Two-stage v2 | Two-stage v3 |
|---|---:|---:|---:|---:|---:|---:|
| HR | 33 | 42.42% | 69.70% | 75.76% | 42.42% | **78.79%** |
| BP_sys | 37 | **83.78%** | **83.78%** | 70.27% | 59.46% | 72.97% |
| BP_dia | 35 | 60.00% | **68.57%** | 62.86% | 62.86% | 62.86% |
| RR | 22 | **54.55%** | 45.45% | 45.45% | **54.55%** | 50.00% |
| O2Sat | 30 | 73.33% | **76.67%** | 73.33% | 63.33% | **76.67%** |
| T | 2 | **100.00%** | **100.00%** | 50.00% | 50.00% | 50.00% |

The two-stage design produces a large HR improvement but loses performance on BP_sys and RR. 

## 8. Performance by Common Action Type

| Action type | Support | Former hybrid | Pure LLM | Two-stage hybrid | Two-stage v2 | Two-stage v3 |
|---|---:|---:|---:|---:|---:|---:|
| `no_action` | 42 | 47.62% | 54.76% | 59.52% | 33.33% | **61.90%** |
| `fluid_bolus` | 30 | 63.33% | **86.67%** | 56.67% | 53.33% | 60.00% |
| `oxygen_support` | 22 | 77.27% | 81.82% | 63.64% | **86.36%** | 77.27% |
| `airway_management` | 19 | 63.16% | 57.89% | 63.16% | 42.11% | **68.42%** |

The two-stage hybrid is the strongest configuration for `no_action`, improving by 11.90 percentage points over the former hybrid. This is evidence that an explicit direction decision helps when the rule engine would otherwise favor no change.

The pure LLM is substantially stronger for `fluid_bolus` and `oxygen_support`. The two-stage direction prompt appears to infer clinically plausible change too readily in cases whose evaluated ground truth remains within the stable dead zone.

## 9. Interpretation

### Best overall configuration

The pure LLM is currently the strongest general-purpose configuration:

- Highest vital-level direction accuracy: 71.07%
- Strongest balance among all three direction categories

### Value of the two-stage hybrid

The two-stage hybrid demonstrates that direction-first prediction can address
the former hybrid's conservative bias:

- Best decrease recall
- Best increase recall
- Best `no_action` accuracy
- Lowest normalized RMSE
- Complete reasoning for both stages

However, its current direction stage is insufficiently calibrated for stable observations. Its lower overall accuracy
