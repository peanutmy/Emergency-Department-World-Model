# Failure Case Report

 **Evaluation unit:** individual vital-sign direction 
**Comparable observations:** 158 scored vital directions per configuration

## Executive Summary

This report compares vital-level direction errors from the strongest direction configuration for each evaluated LLM-based engine:

- **Hybrid:** `gpt-5`, 3-shot static
- **Pure LLM:** `gpt-5.5`, 3-shot `kind_hint` matched

| Vital-level result | Hybrid | Pure LLM | Difference |
|---|---:|---:|---:|
| Correct directions | 102 | **113** | +11 |
| Incorrect directions | 56 | **45** | -11 |
| Vital-level direction accuracy | 0.6456 | **0.7152** | +0.0696 |

Pure LLM reduces the number of misclassified vital directions by `11/56 = 19.6%`. Its clearest advantage is reducing conservative or insufficient-magnitude errors from 14 to 7. Ambiguous or script-dependent errors remain almost unchanged: 25 for hybrid and 24 for pure LLM.

## 1. Method

The four cause categories are:

1. **Script-dependent or ambiguous:** the exact target branch, event, or timing is not uniquely selected by the allowed standalone input.
2. **Conservative or insufficient magnitude:** the prediction moves in the expected direction or remains unchanged, but the change is too small to reach the labeled direction.
3. **Dead-zone or point-target boundary:** the target or prediction lies near the categorical direction threshold, converting a plausible numeric difference into a direction mismatch.
4. **Action semantics or clinical reasoning:** the engine and label interpret the meaning or immediate physiological effect of an action differently.

Some observations could reasonably fit more than one category. The assignment represents the primary cause indicated by the target, prediction, allowed input, and model reasoning.

## 2. Cause Distribution

![](/Users/saygoforit/Desktop/Emergency-Department-World-Model/OUT/ED_Transition_Engine_Failure_Distribution.png)

| Primary cause | Hybrid errors | Share of hybrid errors | Pure errors | Share of pure errors |
|---|---:|---:|---:|---:|
| Script-dependent or ambiguous | 25 | 44.6% | 24 | **53.3%** |
| Conservative or insufficient magnitude | 14 | 25.0% | **7** | 15.6% |
| Dead-zone or point-target boundary | 10 | 17.9% | **9** | 20.0% |
| Action semantics or clinical reasoning | 7 | 12.5% | **5** | 11.1% |
| **Total errors** | **56** | **100.0%** | **45** | **100.0%** |

Pure LLM removes seven under-scaled errors, one boundary error, and two action-semantics errors. It removes only one ambiguous error. Consequently, ambiguity accounts for a larger percentage of the smaller pure-LLM error set.

## 3. Direction-Error Distribution

| Ground truth to prediction | Hybrid errors | Hybrid share | Pure errors | Pure share |
|---|---:|---:|---:|---:|
| Decrease to stable | 25 | **44.6%** | 19 | **42.2%** |
| Increase to stable | 14 | 25.0% | 8 | 17.8% |
| Stable to increase | 7 | 12.5% | 6 | 13.3% |
| Decrease to increase | 5 | 8.9% | 6 | 13.3% |
| Stable to decrease | 3 | 5.4% | 4 | 8.9% |
| Increase to decrease | 2 | 3.6% | 2 | 4.4% |
| **Total** | **56** | **100.0%** | **45** | **100.0%** |

Hybrid predicts stable instead of a true increase or decrease in `39/56 = 69.6%` of its errors. Pure LLM reduces this pattern to `27/45 = 60.0%`, but decrease-to-stable remains the largest error for both engines.

## 4. Errors by Vital Sign

| Vital | Scored observations | Hybrid errors | Hybrid error rate | Pure errors | Pure error rate | Error change |
|---|---:|---:|---:|---:|---:|---:|
| HR | 33 | 19 | 57.6% | **10** | **30.3%** | -9 |
| BP_sys | 37 | 6 | 16.2% | 6 | 16.2% | 0 |
| BP_dia | 35 | 14 | 40.0% | **11** | **31.4%** | -3 |
| RR | 22 | **10** | **45.5%** | 12 | 54.5% | +2 |
| O2Sat | 29 | 7 | 24.1% | **6** | **20.7%** | -1 |
| T | 2 | 0 | 0.0% | 0 | 0.0% | 0 |
| **Total** | **158** | **56** | **35.4%** | **45** | **28.5%** | **-11** |

Pure LLM's largest improvement is HR, where errors fall from 19 to 10. RR is the only vital that worsens, increasing from 10 to 12 errors. This reflects direct LLM predictions that more often change RR in cases where the target remains stable or selects a different respiratory phase.

## 5. Cause Analysis

### 5.1 Script-Dependent or Ambiguous

This category contains 25 hybrid errors and 24 pure-LLM errors. It is the dominant residual failure source after removing the rule anchor.

Representative vital-level failures include:

- `aortic_dissection:p2`: HR, BP, RR, and O2Sat targets select asystolic arrest, while both engines predict little or no immediate change after heparin.
- `subarachnoid_hemorrhage:p1`: BP and HR targets select rapid hypotension and reduced HR, while both engines predict a continued stress or Cushing response.

Pure LLM improves overall accuracy without materially reducing this category. This indicates an input-identifiability limitation rather than primarily a residual-versus-direct prediction problem.

### 5.2 Conservative or Insufficient Magnitude

Hybrid has 14 under-scaled vital errors, while pure LLM has 7.

Examples include:

- Fluid treatment lowers HR too little in `adrenal_crisis:p1`, `adrenal_crisis:p2`, and `massive_upper_gi_bleed:p3`.
- Pacing raises BP_dia by only five points in `aortic_dissection:p6`.

The halving of this category is the strongest evidence that the hybrid rule baseline anchors residual adjustments toward small changes.

### 5.3 Dead-Zone or Point-Target Boundary

Hybrid has 10 boundary errors and pure LLM has 9.

Examples include:

- `subarachnoid_hemorrhage:p4`: target O2Sat changes from 87 to 88 and is classified stable, while both engines predict a larger oxygen response.
- `severe_pediatric_asthma_exacerbation:p1`: target BP changes are at the five-point dead-zone boundary, while predictions cross into increase.

These errors are sensitive to the categorical threshold and exact point target. A strict-sign analysis should be reported alongside the dead-zone analysis to distinguish correct-sign but threshold-crossing disagreements.

### 5.4 Action Semantics or Clinical Reasoning

Hybrid has 7 action-semantics errors and pure LLM has 5.

Examples include:

- `opioid_overdose_with_ards:p6`: target HR rises after intubation, while both engines leave opioid-related bradycardia stable.
- `elderly_psychosis_agitation:p3`: target BP improves after intubation, while both engines predict no direct circulatory improvement.

These errors require an explicit contract for action timing and for the meaning of observed vitals such as spontaneous versus assisted RR.

## 6. Conclusions

1. **Pure LLM improves vital-level direction accuracy from 64.6% to 71.5%.** It produces 11 fewer misclassified vital directions.

2. **The largest architecture-specific improvement is calibration magnitude.** Under-scaled errors fall from 14 to 7 when the LLM predicts final values directly.

3. **Ambiguity becomes the dominant residual limitation.** Absolute ambiguous errors remain nearly unchanged at 25 versus 24 and account for more than half of pure LLM errors.

4. **Decrease-to-stable remains the most frequent error.** It accounts for 25 hybrid errors and 19 pure errors.

5. **HR improves substantially under direct prediction.** Errors fall from 19 to 10, while RR errors increase from 10 to 12.

6. **Boundary-sensitive results should be supplemented with strict-sign scoring.** This would separate wrong direction from correct-sign changes that disagree only because of the dead zone.

   
