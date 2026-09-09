# Model comparison

Rule-based baseline (shared): direction_accuracy=0.4533, normalized_l2_mean=3.4460, normalized_l2_median=1.8028 (n=235)

## pure_llm

| model | setting | dir_acc | L2_mean | L2_median | api_err | parse_err | n |
|---|---|---|---|---|---|---|---|
| gpt-5 | 0shot | 0.7700 | 4.9197 | 1.7797 | 0 | 0 | 10 |
| gpt-5 | 3shot-static | 0.7783 | 5.0763 | 2.1179 | 0 | 0 | 10 |
| gpt-5 | 3shot-kind_hint_matched | 0.8483 | 4.8276 | 1.7673 | 0 | 0 | 10 |
| gpt-5.5 | 0shot | 0.7250 | 4.7215 | 2.3345 | 0 | 0 | 10 |
| gpt-5.5 | 3shot-static | 0.7250 | 4.8770 | 2.3075 | 0 | 0 | 10 |
| gpt-5.5 | 3shot-kind_hint_matched | 0.7250 | 4.9149 | 2.0427 | 0 | 0 | 10 |

