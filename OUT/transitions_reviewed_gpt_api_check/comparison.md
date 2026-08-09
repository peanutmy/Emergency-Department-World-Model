# Model comparison

Rule-based baseline (shared): direction_accuracy=0.4784, normalized_l2_mean=3.4143, normalized_l2_median=1.7404 (n=57)

## hybrid

| model | setting | dir_acc | L2_mean | L2_median | api_err | parse_err | n |
|---|---|---|---|---|---|---|---|
| gpt-4o-mini | 0shot | 0.5000 | 0.8944 | 0.8944 | n/a | n/a | 1 |
| gpt-4o-mini | 3shot-static | 0.5000 | 1.7889 | 1.7889 | n/a | n/a | 1 |
| gpt-4o-mini | 3shot-kind_hint_matched | 0.5000 | 0.8944 | 0.8944 | n/a | n/a | 1 |
| gpt-4o | 0shot | 0.5000 | 0.8944 | 0.8944 | n/a | n/a | 1 |
| gpt-4o | 3shot-static | 0.5000 | 1.2806 | 1.2806 | n/a | n/a | 1 |
| gpt-4o | 3shot-kind_hint_matched | 1.0000 | 0.8944 | 0.8944 | n/a | n/a | 1 |
| gpt-4.1 | 0shot | 1.0000 | 0.4000 | 0.4000 | n/a | n/a | 1 |
| gpt-4.1 | 3shot-static | 1.0000 | 0.4000 | 0.4000 | n/a | n/a | 1 |
| gpt-4.1 | 3shot-kind_hint_matched | 1.0000 | 0.4000 | 0.4000 | n/a | n/a | 1 |
| gpt-5 | 0shot | 1.0000 | 0.4000 | 0.4000 | n/a | n/a | 1 |
| gpt-5 | 3shot-static | 1.0000 | 0.4000 | 0.4000 | n/a | n/a | 1 |
| gpt-5 | 3shot-kind_hint_matched | 1.0000 | 0.4000 | 0.4000 | n/a | n/a | 1 |
| gpt-5.5 | 0shot | 1.0000 | 0.4000 | 0.4000 | n/a | n/a | 1 |
| gpt-5.5 | 3shot-static | 1.0000 | 0.4000 | 0.4000 | n/a | n/a | 1 |
| gpt-5.5 | 3shot-kind_hint_matched | 1.0000 | 0.4000 | 0.4000 | n/a | n/a | 1 |

## pure_llm

| model | setting | dir_acc | L2_mean | L2_median | api_err | parse_err | n |
|---|---|---|---|---|---|---|---|
| gpt-4o-mini | 0shot | 0.5000 | 0.8944 | 0.8944 | 0 | 0 | 1 |
| gpt-4o-mini | 3shot-static | 0.5000 | 0.8944 | 0.8944 | 0 | 0 | 1 |
| gpt-4o-mini | 3shot-kind_hint_matched | 0.5000 | 0.8000 | 0.8000 | 0 | 0 | 1 |
| gpt-4o | 0shot | 0.5000 | 0.8944 | 0.8944 | 0 | 0 | 1 |
| gpt-4o | 3shot-static | 0.5000 | 1.4422 | 1.4422 | 0 | 0 | 1 |
| gpt-4o | 3shot-kind_hint_matched | 0.5000 | 1.4422 | 1.4422 | 0 | 0 | 1 |
| gpt-4.1 | 0shot | 0.5000 | 0.8000 | 0.8000 | 0 | 0 | 1 |
| gpt-4.1 | 3shot-static | 0.5000 | 0.8000 | 0.8000 | 0 | 0 | 1 |
| gpt-4.1 | 3shot-kind_hint_matched | 1.0000 | 0.8000 | 0.8000 | 0 | 0 | 1 |
| gpt-5 | 0shot | 1.0000 | 0.5657 | 0.5657 | 0 | 0 | 1 |
| gpt-5 | 3shot-static | 0.5000 | 0.8944 | 0.8944 | 0 | 0 | 1 |
| gpt-5 | 3shot-kind_hint_matched | 1.0000 | 0.8944 | 0.8944 | 0 | 0 | 1 |
| gpt-5.5 | 0shot | 0.5000 | 0.8944 | 0.8944 | 0 | 0 | 1 |
| gpt-5.5 | 3shot-static | 0.5000 | 0.8000 | 0.8000 | 0 | 0 | 1 |
| gpt-5.5 | 3shot-kind_hint_matched | 0.5000 | 0.8944 | 0.8944 | 0 | 0 | 1 |

