# Model comparison

Rule-based baseline (shared): direction_accuracy=0.4533, normalized_l2_mean=3.4460, normalized_l2_median=1.8028 (n=235)

## hybrid

| model | setting | dir_acc | L2_mean | L2_median | api_err | parse_err | n |
|---|---|---|---|---|---|---|---|
| gpt-4o | 0shot | 0.8667 | 0.8761 | 0.8000 | n/a | n/a | 5 |
| gpt-4o | 3shot-static | 0.8667 | 0.9961 | 0.8000 | n/a | n/a | 5 |
| gpt-4o | 3shot-kind_hint_matched | 0.8667 | 0.9561 | 0.8000 | n/a | n/a | 5 |

## pure_llm

| model | setting | dir_acc | L2_mean | L2_median | api_err | parse_err | n |
|---|---|---|---|---|---|---|---|
| gpt-4o | 0shot | 0.8667 | 0.6919 | 0.4000 | 0 | 0 | 5 |
| gpt-4o | 3shot-static | 0.8667 | 0.9703 | 0.6000 | 0 | 0 | 5 |
| gpt-4o | 3shot-kind_hint_matched | 0.9333 | 0.9614 | 0.4000 | 0 | 0 | 5 |

