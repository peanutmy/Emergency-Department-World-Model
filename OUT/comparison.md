# Model comparison

Rule-based baseline (shared): direction_accuracy=0.4533, normalized_l2_mean=3.4460, normalized_l2_median=1.8028 (n=235)

## hybrid

| model | setting | dir_acc | L2_mean | L2_median | api_err | parse_err | n |
|---|---|---|---|---|---|---|---|
| gpt-4.1 | 0shot | 0.5494 | 3.2877 | 1.5811 | n/a | n/a | 235 |
| gpt-4.1 | 3shot-kind_hint_matched | 0.5830 | 3.1544 | 1.5000 | n/a | n/a | 235 |
| gpt-4.1 | 3shot-static | 0.5909 | 3.2464 | 1.5000 | n/a | n/a | 235 |
| gpt-4o | 0shot | 0.5850 | 3.2637 | 1.6763 | n/a | n/a | 235 |
| gpt-4o | 3shot-kind_hint_matched | 0.6091 | 3.1744 | 1.5811 | n/a | n/a | 235 |
| gpt-4o | 3shot-static | 0.5784 | 3.3196 | 1.8028 | n/a | n/a | 235 |
| gpt-4o-mini | 0shot | 0.5035 | 3.5007 | 1.8439 | n/a | n/a | 235 |
| gpt-4o-mini | 3shot-kind_hint_matched | 0.5330 | 3.4827 | 2.0322 | n/a | n/a | 235 |
| gpt-4o-mini | 3shot-static | 0.5447 | 3.4956 | 1.8868 | n/a | n/a | 235 |
| gpt-5 | 0shot | 0.6104 | 3.1127 | 1.5000 | n/a | n/a | 235 |
| gpt-5 | 3shot-kind_hint_matched | 0.6298 | 2.9471 | 1.5000 | n/a | n/a | 235 |
| gpt-5 | 3shot-static | 0.6343 | 3.0014 | 1.4832 | n/a | n/a | 235 |
| gpt-5.5 | 0shot | 0.6089 | 2.9863 | 1.5000 | n/a | n/a | 235 |
| gpt-5.5 | 3shot-kind_hint_matched | 0.6019 | 3.0575 | 1.5000 | n/a | n/a | 235 |
| gpt-5.5 | 3shot-static | 0.6502 | 3.0383 | 1.4000 | n/a | n/a | 235 |

## pure_llm

| model | setting | dir_acc | L2_mean | L2_median | api_err | parse_err | n |
|---|---|---|---|---|---|---|---|
| gpt-4.1 | 0shot | 0.6559 | 2.9548 | 1.4142 | 0 | 0 | 235 |
| gpt-4.1 | 3shot-kind_hint_matched | 0.6588 | 2.9093 | 1.4142 | 0 | 0 | 235 |
| gpt-4.1 | 3shot-static | 0.6511 | 2.8579 | 1.4142 | 0 | 0 | 235 |
| gpt-4o | 0shot | 0.5699 | 3.0708 | 1.5000 | 0 | 0 | 235 |
| gpt-4o | 3shot-kind_hint_matched | 0.6456 | 2.9854 | 1.4036 | 0 | 0 | 235 |
| gpt-4o | 3shot-static | 0.6030 | 3.1456 | 1.6583 | 0 | 0 | 235 |
| gpt-4o-mini | 0shot | 0.3891 | 3.5736 | 2.0000 | 0 | 0 | 235 |
| gpt-4o-mini | 3shot-kind_hint_matched | 0.4872 | 3.4064 | 1.6000 | 0 | 0 | 235 |
| gpt-4o-mini | 3shot-static | 0.4344 | 3.5303 | 1.8193 | 0 | 0 | 235 |
| gpt-5 | 0shot | 0.6633 | 2.8430 | 1.4000 | 0 | 1 | 235 |
| gpt-5 | 3shot-kind_hint_matched | 0.7042 | 2.7747 | 1.4000 | 0 | 0 | 235 |
| gpt-5 | 3shot-static | 0.6848 | 2.7966 | 1.4142 | 0 | 0 | 235 |
| gpt-5.5 | 0shot | 0.6679 | 2.9171 | 1.4000 | 0 | 0 | 235 |
| gpt-5.5 | 3shot-kind_hint_matched | 0.6836 | 2.8094 | 1.3748 | 0 | 0 | 235 |
| gpt-5.5 | 3shot-static | 0.6886 | 2.9506 | 1.4000 | 0 | 0 | 235 |

