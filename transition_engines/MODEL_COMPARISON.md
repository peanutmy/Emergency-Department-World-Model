# Multi-model + few-shot comparison

This adds (1) multi-provider LLM clients and (2) optional few-shot prompting to
the transition-pair engines, plus a driver that benchmarks models × few-shot
settings across both LLM engines. The deterministic `RuleBasedEngine` is the
shared baseline.

**Invariant:** the ED world model's use of `HybridEngine` is unchanged. Few-shot
is an additive `examples=None` parameter (default = the exact zero-shot prompt),
and multi-model routing lives only in the comparison driver, not in the engines
or the runtime path. The single-model runners (`run_hybrid_eval`,
`run_pure_llm_eval`) keep their original `--llm-mode real` behavior (OpenAI via
`OpenAILLMClient`, which still raises on a missing key).

## Models

Configured in `clients/factory.py::MODEL_REGISTRY`:

| model id | provider | API | temperature |
|---|---|---|---|
| `gpt-4o-mini`, `gpt-4o`, `gpt-4.1` | OpenAI | Chat Completions (JSON mode) | tunable |
| `gpt-5` | OpenAI | Responses | fixed (omitted) |
| `gpt-5.5` | OpenAI | Responses | tunable |
| `gemini-2.5-flash` | Google | google-genai | tunable |
| `claude-sonnet-4-6` | Anthropic | Messages | tunable |

(`gpt-4-turbo` is intentionally excluded.)

## API keys / SDKs

Keys are read from the environment; **a missing key degrades gracefully** to
empty JSON (and records `last_api_error`) instead of crashing, so the whole
matrix runs in `--dry-run` today with no keys.

- `OPENAI_API_KEY` (SDK `openai`, already installed)
- `ANTHROPIC_API_KEY` (SDK `anthropic` — `pip install anthropic`)
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` (SDK `google-genai` — `pip install google-genai`)

Provider SDKs are imported lazily (only when a real call is made), so the
package imports fine without `anthropic` / `google-genai` installed.

## Few-shot

Leave-one-case-out, deterministic by `seed`. Two selection strategies:
`static` (same global order for every prediction) and `kind_hint_matched`
(prefer examples whose action `kind_hint` matches the current pair). Hybrid
examples show the rule baseline + ideal residual; pure-LLM examples show the
stated post-action target vitals.

Single-engine runs:

```bash
# zero-shot (default) is unchanged
python -m transition_engines.run_hybrid_eval transitions OUT -r --llm-mode real --model gpt-5.5
# 3-shot, kind_hint matched
python -m transition_engines.run_hybrid_eval transitions OUT -r \
    --llm-mode real --model gpt-5.5 --shots 3 --example-strategy kind_hint_matched
```

`--shots` accepts `0` or `3`; `--example-strategy` ∈ {`static`, `kind_hint_matched`}.

## Full comparison

```bash
# keyless wiring/sanity check (empty predictions, no network)
python -m transition_engines.run_model_comparison transitions OUT_ROOT -r --dry-run --limit-pairs 3

# real run, all registry models, both engines
python -m transition_engines.run_model_comparison transitions OUT_ROOT -r
```

For each engine ∈ {hybrid, pure_llm} × model × setting ∈ {`0shot`,
`3shot-static`, `3shot-kind_hint_matched`} it runs the existing evaluator and
writes per-run CSV/JSON under `OUT_ROOT/<engine>/<model>/<setting>/`. Rule-based
runs once under `OUT_ROOT/rule_based/`. Aggregates land in
`OUT_ROOT/comparison.csv` and `OUT_ROOT/comparison.md`. Metrics are the existing
target-vital **direction accuracy** and masked **normalized L2**.

Note: hybrid per-run rows show `n/a` for api/parse error counts because
`run_hybrid_eval`'s summary (shared with the rule-based summary builder) does not
tally LLM errors; pure-LLM rows do. Direction/L2 metrics are unaffected.
