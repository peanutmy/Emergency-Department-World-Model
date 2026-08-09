# Model comparison

Rule-based baseline (shared): direction_accuracy=0.4784, normalized_l2_mean=3.4143, normalized_l2_median=1.7404 (n=57)

## hybrid

| model | setting | dir_acc | L2_mean | L2_median | api_err | parse_err | n |
|---|---|---|---|---|---|---|---|
| gpt-4o-mini | 0shot | 0.4830 | 3.5218 | 1.8566 | n/a | n/a | 57 |
| gpt-4o-mini | 3shot-static | 0.5439 | 3.4713 | 1.8945 | n/a | n/a | 57 |
| gpt-4o-mini | 3shot-kind_hint_matched | 0.5582 | 3.3438 | 1.6429 | n/a | n/a | 57 |
| gpt-4o | 0shot | 0.6442 | 3.1095 | 1.6683 | n/a | n/a | 57 |
| gpt-4o | 3shot-static | 0.6257 | 3.2673 | 1.6391 | n/a | n/a | 57 |
| gpt-4o | 3shot-kind_hint_matched | 0.6453 | 3.2559 | 1.5214 | n/a | n/a | 57 |
| gpt-4.1 | 0shot | 0.5930 | 3.1964 | 1.5294 | n/a | n/a | 57 |
| gpt-4.1 | 3shot-static | 0.6442 | 3.2003 | 1.5469 | n/a | n/a | 57 |
| gpt-4.1 | 3shot-kind_hint_matched | 0.5848 | 3.2188 | 1.4160 | n/a | n/a | 57 |
| gpt-5 | 0shot | 0.6310 | 3.1283 | 1.4805 | n/a | n/a | 57 |
| gpt-5 | 3shot-static | 0.6807 | 3.1772 | 1.4624 | n/a | n/a | 57 |
| gpt-5 | 3shot-kind_hint_matched | 0.6211 | 3.1370 | 1.4772 | n/a | n/a | 57 |
| gpt-5.5 | 0shot | 0.6193 | 3.1706 | 1.5262 | n/a | n/a | 57 |
| gpt-5.5 | 3shot-static | 0.6383 | 3.2304 | 1.5811 | n/a | n/a | 57 |
| gpt-5.5 | 3shot-kind_hint_matched | 0.5909 | 3.1628 | 1.5406 | n/a | n/a | 57 |

## pure_llm

| model | setting | dir_acc | L2_mean | L2_median | api_err | parse_err | n |
|---|---|---|---|---|---|---|---|
| gpt-4o-mini | 0shot | 0.4719 | 3.5611 | 2.2422 | 0 | 0 | 57 |
| gpt-4o-mini | 3shot-static | 0.4936 | 3.3883 | 1.6514 | 0 | 0 | 57 |
| gpt-4o-mini | 3shot-kind_hint_matched | 0.4991 | 3.4442 | 1.5000 | 0 | 0 | 57 |
| gpt-4o | 0shot | 0.6550 | 3.0326 | 1.3396 | 0 | 0 | 57 |
| gpt-4o | 3shot-static | 0.6360 | 3.0921 | 1.4711 | 0 | 0 | 57 |
| gpt-4o | 3shot-kind_hint_matched | 0.6278 | 3.1536 | 1.2977 | 0 | 0 | 57 |
| gpt-4.1 | 0shot | 0.6567 | 3.0487 | 1.2247 | 0 | 0 | 57 |
| gpt-4.1 | 3shot-static | 0.6737 | 2.8935 | 1.4071 | 0 | 0 | 57 |
| gpt-4.1 | 3shot-kind_hint_matched | 0.6503 | 3.0843 | 1.4644 | 0 | 0 | 57 |
| gpt-5 | 0shot | 0.7219 | 2.9978 | 1.2164 | 0 | 0 | 57 |
| gpt-5 | 3shot-static | 0.7050 | 2.9878 | 1.3964 | 0 | 0 | 57 |
| gpt-5 | 3shot-kind_hint_matched | 0.7161 | 2.8075 | 1.4282 | 0 | 0 | 57 |
| gpt-5.5 | 0shot | 0.6830 | 3.0614 | 1.2590 | 0 | 0 | 57 |
| gpt-5.5 | 3shot-static | 0.6953 | 2.9411 | 1.2288 | 0 | 0 | 57 |
| gpt-5.5 | 3shot-kind_hint_matched | 0.7360 | 3.0000 | 1.2325 | 0 | 0 | 57 |

