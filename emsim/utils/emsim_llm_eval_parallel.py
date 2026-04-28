"""
Parallel cache prefill for LLMEngine.

Runs every (before, actions, duration_s) triple across the transitions
corpus through `LLMEngine.step()` with a thread pool, populating the
on-disk cache. The subsequent `emsim_eval.py --engine ...:LLMEngine`
then reads from cache and scores deterministically.

Usage:
    python emsim_llm_eval_parallel.py
    python emsim_llm_eval_parallel.py --workers 12 --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from emsim_llm_engine import LLMEngine
from emsim_eval import iter_case_files


def collect_tasks(transitions_dir: str):
    tasks = []
    for cat, fname, case in iter_case_files(transitions_dir):
        for pair in case["pairs"]:
            tasks.append((
                case["case_id"], pair["id"],
                pair["before"], pair["actions"], pair["duration_s"],
            ))
    return tasks


def prefill(transitions_dir: str, workers: int, model: str | None) -> None:
    engine = LLMEngine(model=model, cache_enabled=True)
    tasks = collect_tasks(transitions_dir)
    n = len(tasks)
    print(f"Collected {n} pairs; prefilling with {workers} workers (model={engine.model})")

    t0 = time.time()
    done = 0
    errors: list[tuple[str, str, str]] = []

    def run_one(task):
        case_id, pair_id, before, actions, duration_s = task
        try:
            engine.step(before, actions, duration_s)
            return (case_id, pair_id, None)
        except Exception as e:
            return (case_id, pair_id, f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        for f in as_completed(futures):
            case_id, pair_id, err = f.result()
            done += 1
            if err:
                errors.append((case_id, pair_id, err))
            if done % 50 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                stats = engine.stats()
                print(
                    f"  {done:4d}/{n}  elapsed={elapsed:6.1f}s  "
                    f"rate={rate:4.1f}/s  calls={stats['calls']}  "
                    f"cache_hits={stats['cache_hits']}  failures={stats['failures']}"
                )

    stats = engine.stats()
    elapsed = time.time() - t0
    print(f"\nDone. {done}/{n} pairs in {elapsed:.1f}s")
    print(f"  model         : {stats['model']}")
    print(f"  calls         : {stats['calls']}")
    print(f"  cache_hits    : {stats['cache_hits']}")
    print(f"  failures      : {stats['failures']}")
    print(f"  tokens_in     : {stats['tokens_in']}")
    print(f"  tokens_out    : {stats['tokens_out']}")

    # Rough cost (gpt-4o-mini): $0.15 / MTok input, $0.60 / MTok output.
    cost = (stats["tokens_in"] / 1e6) * 0.15 + (stats["tokens_out"] / 1e6) * 0.60
    print(f"  est. cost     : ${cost:.4f}")

    if errors:
        print(f"\n{len(errors)} task errors (first 10):")
        for case_id, pair_id, err in errors[:10]:
            print(f"  {case_id}/{pair_id}: {err}")

    print("\nNext:")
    print("  python emsim_eval.py --transitions transitions "
          "--engine emsim_llm_engine:LLMEngine --export llm_phase0.csv --worst 20")


def main():
    ap = argparse.ArgumentParser(description="Parallel prefill for LLMEngine cache")
    ap.add_argument("--transitions", default="transitions")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    prefill(args.transitions, args.workers, args.model)


if __name__ == "__main__":
    main()
