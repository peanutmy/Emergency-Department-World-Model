# Model comparison

Rule-based baseline (shared): direction_accuracy=0.4784, normalized_l2_mean=3.4143, normalized_l2_median=1.7404 (n=57)

## hybrid

| model | setting | dir_acc | L2_mean | L2_median | api_err | parse_err | n |
|---|---|---|---|---|---|---|---|
| gpt-4o-mini | 0shot | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-4o-mini | 3shot-static | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-4o-mini | 3shot-kind_hint_matched | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-4o | 0shot | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-4o | 3shot-static | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-4o | 3shot-kind_hint_matched | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-4.1 | 0shot | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-4.1 | 3shot-static | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-4.1 | 3shot-kind_hint_matched | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-5 | 0shot | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-5 | 3shot-static | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-5 | 3shot-kind_hint_matched | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-5.5 | 0shot | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-5.5 | 3shot-static | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |
| gpt-5.5 | 3shot-kind_hint_matched | 1.0000 | 0.6000 | 0.4000 | n/a | n/a | 3 |

## pure_llm

| model | setting | dir_acc | L2_mean | L2_median | api_err | parse_err | n |
|---|---|---|---|---|---|---|---|
| gpt-4o-mini | 0shot | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-4o-mini | 3shot-static | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-4o-mini | 3shot-kind_hint_matched | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-4o | 0shot | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-4o | 3shot-static | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-4o | 3shot-kind_hint_matched | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-4.1 | 0shot | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-4.1 | 3shot-static | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-4.1 | 3shot-kind_hint_matched | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-5 | 0shot | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-5 | 3shot-static | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-5 | 3shot-kind_hint_matched | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-5.5 | 0shot | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-5.5 | 3shot-static | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |
| gpt-5.5 | 3shot-kind_hint_matched | 0.3333 | 1.4294 | 1.6000 | 3 | 0 | 3 |

