# Transition Engine Evaluation Report

**Scope:** 11 reviewed cases, 57 transition pairs

## 1. Evaluation Scope

| Dimension | Evaluated values |
|---|---|
| Dataset | 11 reviewed scenario files containing 57 transition pairs |
| Models | `gpt-4o-mini`, `gpt-4o`, `gpt-4.1`, `gpt-5`, `gpt-5.5` |
| Engines | Hybrid and pure LLM |
| Prompt settings | Zero-shot, 3-shot static, 3-shot kind hint matched |
| Rule baseline | Evaluated once over all 57 pairs |

### Metrics

- **Direction accuracy:** Whether each requested vital moved up, down, or remained stable in agreement with the labeled target. Higher is better.
- **Normalized L2:** Scaled numeric distance between predicted and labeled target vitals. Lower is better.

Direction accuracy and normalized L2 must be interpreted together. A prediction can correctly indicate that blood pressure decreases while remaining numerically far from a target representing arrest.

## 2. Rule-Based Baseline

| Direction accuracy | Mean normalized L2 | Median normalized L2 | Pairs |
|---:|---:|---:|---:|
| 0.4784 | 3.4143 | 1.7404 | 57 |

## 3. Model Comparison

The following values are averages across the three prompting settings for each model.

| Engine | Model | Average direction | Average mean L2 | Best direction setting |
|---|---|---:|---:|---|
| Hybrid | `gpt-4o-mini` | 0.5284 | 3.4456 | Matched: 0.5582 |
| Hybrid | `gpt-4o` | 0.6384 | 3.2109 | Matched: 0.6453 |
| Hybrid | `gpt-4.1` | 0.6073 | 3.2052 | Static: 0.6442 |
| Hybrid | `gpt-5` | **0.6443** | **3.1475** | Static: **0.6807** |
| Hybrid | `gpt-5.5` | 0.6162 | 3.1880 | Static: 0.6383 |
| Pure LLM | `gpt-4o-mini` | 0.4882 | 3.4645 | Matched: 0.4991 |
| Pure LLM | `gpt-4o` | 0.6396 | 3.0928 | Zero-shot: 0.6550 |
| Pure LLM | `gpt-4.1` | 0.6602 | 3.0088 | Static: 0.6737 |
| Pure LLM | `gpt-5` | **0.7143** | **2.9311** | Zero-shot: 0.7219 |
| Pure LLM | `gpt-5.5` | 0.7048 | 3.0008 | Matched: **0.7360** |

### Winning Configurations

| Engine | Criterion | Configuration | Score | Delta from rule baseline |
|---|---|---|---:|---:|
| Hybrid | Direction accuracy | `gpt-5`, 3-shot static | 0.6807 | +0.2023 |
| Hybrid | Mean L2 | `gpt-4o`, zero-shot | 3.1095 | -0.3048 |
| Pure LLM | Direction accuracy | `gpt-5.5`, 3-shot matched | 0.7360 | +0.2576 |
| Pure LLM | Mean L2 | `gpt-5`, 3-shot matched | 2.8075 | -0.6068 |
| Pure LLM | Median L2 | `gpt-5`, zero-shot | 1.2164 | -0.5240 |

### Comparison Findings

- Pure LLM beat the corresponding hybrid configuration on direction accuracy in 11 of 15 comparisons.
- Pure LLM beat hybrid on mean normalized L2 in 14 of 15 comparisons.
- Averaged across settings, `gpt-5` was the strongest model for both engines.
- `gpt-4o-mini` remained near or below the rule baseline and was the weakest model overall.

### Direction Confusion Matrices

#### Hybrid: `gpt-5`, 3-Shot Static

| Ground truth / prediction | Increase | Decrease | Stable | Support |
|---|---:|---:|---:|---:|
| Increase | **40** | 2 | 14 | 56 |
| Decrease | 5 | **27** | 25 | 57 |
| Stable | 7 | 3 | **35** | 45 |
| Predicted total | 52 | 32 | 74 | 158 |

| Ground-truth class | Recall | Precision |
|---|---:|---:|
| Increase | 0.7143 | 0.7692 |
| Decrease | **0.4737** | 0.8438 |
| Stable | 0.7778 | **0.4730** |

Pooled vital-level accuracy is `102 / 158 = 0.6456`. The largest error is decrease predicted as stable: 25 of 57 true decreases (`43.9%`). The model generated 74 stable predictions despite only 45 stable labels, confirming that hybrid residual adjustments are often too conservative to cross the evaluator's direction threshold.

#### Pure LLM: `gpt-5.5`, 3-Shot `kind_hint` Matched

| Ground truth / prediction | Increase | Decrease | Stable | Support |
|---|---:|---:|---:|---:|
| Increase | **46** | 2 | 8 | 56 |
| Decrease | 6 | **32** | 19 | 57 |
| Stable | 6 | 4 | **35** | 45 |
| Predicted total | 58 | 38 | 62 | 158 |

| Ground-truth class | Recall | Precision |
|---|---:|---:|
| Increase | 0.8214 | 0.7931 |
| Decrease | **0.5614** | 0.8421 |
| Stable | 0.7778 | 0.5645 |

Pooled vital-level accuracy is `113 / 158 = 0.7152`. Relative to the hybrid configuration, the pure LLM correctly classified six more increases and five more decreases while preserving the same number of correct stable predictions. Its largest remaining error is also decrease predicted as stable: 19 of 57 true decreases (`33.3%`).

## 4. Hybrid Engine Diagnostic

The detailed hybrid failure analysis uses `gpt-5` with 3-shot static examples because it achieved the highest hybrid direction accuracy.

| Metric | Result |
|---|---:|
| Direction accuracy | 0.6807 |
| Mean normalized L2 | 3.1772 |
| Median normalized L2 | 1.4624 |
| Pairs evaluated | 57 |

### Performance by Pair Type

| Pair type | Count | Direction accuracy | Mean L2 | Median L2 |
|---|---:|---:|---:|---:|
| Physiology response | 43 | 0.7469 | 2.2935 | 1.3416 |
| Time progression / `no_action` | 14 | **0.4774** | **6.1001** | **6.1106** |

### Performance by Vital

| Vital | Count | Direction accuracy | Mean normalized absolute error |
|---|---:|---:|---:|
| BP_sys | 37 | 0.8378 | 2.3541 |
| O2Sat | 29 | 0.7586 | 1.4138 |
| BP_dia | 35 | 0.6000 | 1.4371 |
| RR | 23 | 0.5217 | 1.5091 |
| HR | 33 | **0.4242** | 1.9939 |
| T | 2 | 1.0000 | 0.0000 |

### Effect of the LLM Residual Layer

| Comparison with rule baseline | Improved | Worsened | Unchanged |
|---|---:|---:|---:|
| Direction accuracy | 22 pairs | 3 pairs | 32 pairs |
| Normalized L2 | 24 pairs | 11 pairs | 21 pairs |

The correction layer materially improved the rule baseline, particularly for systolic blood pressure and oxygen saturation. Heart-rate and respiratory-rate phase changes frequently remained incorrect.

## 5. Failure Analysis

### 5.1 `no_action` Is Semantically Overloaded

The single `kind_hint="no_action"` represents several distinct mechanisms:

- ordinary time progression
- continuing hemorrhage
- respiratory fatigue
- aspiration
- seizure onset
- unstable bradycardia
- PEA, VF, or asystolic arrest
- transition to the next scripted scenario phase

Only 3 of the 14 `no_action` pairs achieved perfect direction accuracy in the selected hybrid configuration. Across all hybrid models and settings, the best `no_action` direction accuracy was only `0.5595`. This suggests an input and representation limitation rather than a single-model problem.

so right now, engines receive information:

  - category
  - scenario_description
  - case_context
  - Current vitals and features
  - Structured action and parameters, including elapsed_min

Additional information options:

  1. Recent vital history

  Include the previous one or two measurements with timestamps:

  "recent_vitals": [
    {"minutes_ago": 5, "HR": 120, "BP_sys": 105, "RR": 34, "O2Sat": 88},
    {"minutes_ago": 0, "HR": 125, "BP_sys": 92, "RR": 24, "O2Sat": 82}
  ]

  This lets the model distinguish compensation from fatigue or accelerating deterioration. but the transition pairs are not continuous

### 5.2 The Model Selects the Wrong Physiological Phase

The generated reasoning is usually clinically coherent, but it often describes early compensation while the target represents late decompensation.

| Case / pair | Model reasoning pattern | Labeled trajectory | Result |
|---|---|---|---|
| Pulmonary edema p2 | Distress should increase RR | Patient tires; RR falls from 34 to 24 | Wrong RR direction |
| Aortic dissection p3 | Pain and catecholamines raise BP and HR | BP and HR collapse within four minutes | All target directions wrong |
| SAH p1 | Cushing response and stress raise BP | Scripted BP fall to 80/50 | All target directions wrong |
| Adrenal crisis p5 | Shock and acidosis cause tachypnea | VF arrest; RR becomes 0 | Wrong phase and magnitude |
| Pediatric asthma p6 | Compensatory tachycardia persists | Bradycardic collapse; HR becomes 35 | Wrong HR direction |
| GI bleed p4 | Continued blood loss gradually lowers BP | PEA arrest; BP becomes 0/0 | Direction passes; L2 remains 8.24 |

A useful contrast occurs within the pulmonary edema case. In the earlier state, RR is 34 and the model predicts continued tachypnea, missing the fatigue transition. In the later state, RR is already 24 with worsening hypoxemia, and the model correctly reasons that fatigue can reduce respiratory drive. Explicit phase information in the current state therefore improves performance.

### 5.3 Determining Branch Information Is Hidden

The expected transition is frequently specified only in `source.modifier_text`, which is intentionally excluded from engine input. Examples include:

- `Patient Tires`
- `VF arrest`
- `No MgSO4 by 3 min -> Arrest`
- `BP progressively drops to 80/50`

The engine generally receives only `no_action`, elapsed time, current vitals, and sanitized case context. Multiple future trajectories can be clinically plausible from that information.

This should not be fixed by exposing source labels to clinical agents. Instead, physiology-only hidden state should identify the current trajectory phase, such as `compensating`, `fatiguing`, `peri_arrest`, or `arrest`.

### 5.4 Residual Adjustments Are Too Conservative

The rule engine often predicts no change or modest deterioration. The LLM remains anchored to this baseline and usually returns small corrections.

Among the 21 scorable `no_action` direction mistakes, 16 produced changes inside the evaluator's dead zone and were therefore classified as stable. In several cases, the reasoning explicitly described deterioration, but the numeric adjustment was too small to count as directional movement.

For example, changing HR from 50 to 45 is described as worsening bradycardia, but the HR dead zone is 5. A change of exactly -5 is classified as stable, while the labeled change from 50 to 40 is classified as down.

### 5.5 Continuous Residuals Do Not Represent Discrete Events Well

Seizure, apnea, PEA, VF, and asystole are discrete transitions. A continuous residual model tends to predict ordinary physiological movement instead of selecting one of these terminal states.

The largest outlier was the aortic-dissection heparin/asystole pair:

- Target: HR, BP, RR, and O2Sat all become 0.
- Hybrid prediction: mild hemodynamic deterioration.
- Normalized L2: 32.15.

Removing this single pair lowers the selected hybrid mean L2 from `3.1772` to `2.6503`.

### 5.6 HR and RR Are the Weakest Vitals

Heart rate had direction accuracy `0.4242`, and respiratory rate had `0.5217`.

- HR frequently remained stable when the target required a decrease.
- RR often increased because the model predicted compensation, while the target represented fatigue or apnea.
- Diastolic BP also suffered from corrections that remained inside the direction dead zone.

### 5.7 Why Hybrid Engine Perform worse than Pure LLM

The rule baseline is conservative, and the residual hybrid does not override it strongly enough. 158 vitals  transition, 

  - Ground truth has 45 stable vital transitions.
  - Rule engine predicts 106 stable transitions. hybrid model rule based (anchor)+ residual 
  - Hybrid reduces this to 74 stable predictions.
  - Pure LLM predicts 55 stable transitions, much closer to the ground truth.

### 5.8 Why both two engines predict decrease to stable?

| Configuration        | Total Error | no change | small decrease | small increase |
| -------------------- | ----------- | --------- | -------------- | -------------- |
| Hybrid gpt-5 static  | 25          | 6         | 10             | 9              |
| Pure gpt-5.5 matched | 19          | 7         | 9              | 3              |

