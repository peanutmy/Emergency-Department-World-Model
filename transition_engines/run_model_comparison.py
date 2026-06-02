"""Drive a model x few-shot comparison over both LLM engines.

For each engine in {hybrid, pure_llm}, each model in the registry, and each
shot setting in {0-shot, 3-shot static, 3-shot kind_hint_matched}, this runs the
existing per-engine evaluator and collects the standard metrics (target-vital
direction accuracy + masked normalized L2). The deterministic rule-based engine
is run exactly once as a shared baseline.

Pass ``api_key=""`` (CLI ``--dry-run``) to exercise the full matrix with no keys
and no network: every provider client degrades to empty predictions, which is
useful for wiring/sanity checks before real keys are available.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from transition_engines.clients import MODEL_REGISTRY, build_llm_client
from transition_engines.few_shot import ExampleBank, ExampleSelector
from transition_engines.run_hybrid_eval import run_hybrid_eval
from transition_engines.run_pure_llm_eval import run_pure_llm_eval
from transition_engines.run_rule_based_eval import run_rule_based_eval


DEFAULT_MODELS: list[str] = list(MODEL_REGISTRY)

# (shots, strategy): 0-shot once, then 3-shot under each selection strategy.
DEFAULT_SHOT_SETTINGS: list[tuple[int, str | None]] = [
    (0, None),
    (3, "static"),
    (3, "kind_hint_matched"),
]

_ENGINES = {
    "hybrid": {"run": run_hybrid_eval, "schema": "adjustments"},
    "pure_llm": {"run": run_pure_llm_eval, "schema": "vitals"},
}

_COMPARISON_CSV = "comparison.csv"
_COMPARISON_MD = "comparison.md"
_CSV_COLUMNS = [
    "engine",
    "model",
    "shots",
    "strategy",
    "setting",
    "direction_accuracy_mean",
    "normalized_l2_mean",
    "normalized_l2_median",
    "num_api_errors",
    "num_parse_errors",
    "num_pairs",
    "output_dir",
]


def setting_label(shots: int, strategy: str | None) -> str:
    return "0shot" if shots <= 0 else f"{shots}shot-{strategy}"


def run_model_comparison(
    input_path: str | Path,
    output_root: str | Path,
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    engines: Sequence[str] = ("hybrid", "pure_llm"),
    shot_settings: Sequence[tuple[int, str | None]] = DEFAULT_SHOT_SETTINGS,
    recursive: bool = True,
    limit_pairs: int | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 300,
    seed: int = 0,
    include_rule_based: bool = True,
    emit_progress: bool = True,
) -> dict[str, Any]:
    """Run the full matrix and write a comparison table; return its structure."""

    input_path = Path(input_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    unknown = [model for model in models if model not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Known: {sorted(MODEL_REGISTRY)}")
    unknown_engines = [engine for engine in engines if engine not in _ENGINES]
    if unknown_engines:
        raise ValueError(f"Unknown engines: {unknown_engines}")

    total_runs = len(engines) * len(models) * len(shot_settings)
    if emit_progress:
        print(
            f"Model comparison: {len(engines)} engines x {len(models)} models x "
            f"{len(shot_settings)} settings = {total_runs} LLM runs"
            + (" + 1 rule-based baseline" if include_rule_based else "")
        )

    rows: list[dict[str, Any]] = []
    run_index = 0
    for engine in engines:
        schema = _ENGINES[engine]["schema"]
        run_eval = _ENGINES[engine]["run"]
        for model in models:
            for shots, strategy in shot_settings:
                run_index += 1
                label = setting_label(shots, strategy)
                out_dir = output_root / engine / model / label
                if emit_progress:
                    print(f"  [{run_index}/{total_runs}] {engine} | {model} | {label}")

                client = build_llm_client(
                    model,
                    response_schema_type=schema,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    api_key=api_key,
                )
                selector = _build_selector(
                    input_path,
                    shots=shots,
                    strategy=strategy,
                    seed=seed,
                    engine_kind=engine,
                )
                summary = run_eval(
                    input_path,
                    out_dir,
                    recursive=recursive,
                    llm_client=client,
                    example_selector=selector,
                    limit_pairs=limit_pairs,
                    emit_console_summary=False,
                )
                rows.append(
                    _row_from_summary(
                        engine=engine,
                        model=model,
                        shots=shots,
                        strategy=strategy,
                        label=label,
                        summary=summary,
                        out_dir=out_dir,
                    )
                )

    rule_based: dict[str, Any] | None = None
    if include_rule_based:
        rb_summary = run_rule_based_eval(
            input_path,
            output_root / "rule_based",
            recursive=recursive,
            emit_console_summary=False,
        )
        rule_based = _baseline_row(rb_summary, output_root / "rule_based")

    comparison = {
        "rows": rows,
        "rule_based": rule_based,
        "settings": [setting_label(s, st) for s, st in shot_settings],
        "models": list(models),
        "engines": list(engines),
    }

    _write_comparison_csv(output_root / _COMPARISON_CSV, rows, rule_based)
    _write_comparison_md(output_root / _COMPARISON_MD, comparison)
    return comparison


def _parse_setting_label(label: str) -> tuple[int, str | None]:
    if label == "0shot":
        return 0, None
    # e.g. "3shot-kind_hint_matched" -> (3, "kind_hint_matched")
    shots_part, _, strategy = label.partition("-")
    shots = int(shots_part.replace("shot", "") or 0)
    return shots, (strategy or None)


def aggregate_comparison(
    output_root: str | Path,
    *,
    engines: Sequence[str] = ("hybrid", "pure_llm"),
) -> dict[str, Any]:
    """Rebuild comparison.{csv,md} from per-run summaries already on disk.

    Lets you re-run only some models (e.g. the reasoning models) into an
    existing output root, then stitch every model's results back into one table
    without re-running the models that already finished.
    """

    output_root = Path(output_root)
    rows: list[dict[str, Any]] = []
    discovered_models: list[str] = []
    discovered_engines: list[str] = []

    for engine in engines:
        engine_dir = output_root / engine
        if not engine_dir.is_dir():
            continue
        discovered_engines.append(engine)
        summary_name = f"{engine}_summary.json"
        for model_dir in sorted(p for p in engine_dir.iterdir() if p.is_dir()):
            model = model_dir.name
            if model not in discovered_models:
                discovered_models.append(model)
            for setting_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                summary_path = setting_dir / summary_name
                if not summary_path.exists():
                    continue
                summary = _load_summary(summary_path)
                shots, strategy = _parse_setting_label(setting_dir.name)
                rows.append(
                    _row_from_summary(
                        engine=engine,
                        model=model,
                        shots=shots,
                        strategy=strategy,
                        label=setting_dir.name,
                        summary=summary,
                        out_dir=setting_dir,
                    )
                )

    rule_based = None
    rb_path = output_root / "rule_based" / "rule_based_summary.json"
    if rb_path.exists():
        rule_based = _baseline_row(_load_summary(rb_path), output_root / "rule_based")

    comparison = {
        "rows": rows,
        "rule_based": rule_based,
        "settings": sorted({row["setting"] for row in rows}),
        "models": discovered_models,
        "engines": discovered_engines,
    }
    _write_comparison_csv(output_root / _COMPARISON_CSV, rows, rule_based)
    _write_comparison_md(output_root / _COMPARISON_MD, comparison)
    return comparison


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_selector(
    input_path: Path,
    *,
    shots: int,
    strategy: str | None,
    seed: int,
    engine_kind: str,
) -> ExampleSelector | None:
    if shots <= 0 or strategy is None:
        return None
    bank = ExampleBank.from_dataset(input_path, recursive=True)
    return ExampleSelector(
        bank,
        k=shots,
        strategy=strategy,
        seed=seed,
        engine_kind=engine_kind,
    )


def _row_from_summary(
    *,
    engine: str,
    model: str,
    shots: int,
    strategy: str | None,
    label: str,
    summary: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    overall = summary.get("overall", {}) if isinstance(summary, dict) else {}
    return {
        "engine": engine,
        "model": model,
        "shots": shots,
        "strategy": strategy or "-",
        "setting": label,
        "direction_accuracy_mean": overall.get("direction_accuracy_mean"),
        "normalized_l2_mean": overall.get("normalized_l2_mean"),
        "normalized_l2_median": overall.get("normalized_l2_median"),
        "num_api_errors": overall.get("num_api_errors"),
        "num_parse_errors": overall.get("num_parse_errors"),
        "num_pairs": overall.get("num_pairs"),
        "output_dir": str(out_dir),
    }


def _baseline_row(summary: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    overall = summary.get("overall", {}) if isinstance(summary, dict) else {}
    return {
        "engine": "rule_based",
        "model": "rule_based",
        "shots": 0,
        "strategy": "-",
        "setting": "-",
        "direction_accuracy_mean": overall.get("direction_accuracy_mean"),
        "normalized_l2_mean": overall.get("normalized_l2_mean"),
        "normalized_l2_median": overall.get("normalized_l2_median"),
        "num_api_errors": None,
        "num_parse_errors": None,
        "num_pairs": overall.get("num_pairs"),
        "output_dir": str(out_dir),
    }


def _write_comparison_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    rule_based: dict[str, Any] | None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        if rule_based is not None:
            writer.writerow(rule_based)
        for row in rows:
            writer.writerow(row)


def _write_comparison_md(path: Path, comparison: dict[str, Any]) -> None:
    lines: list[str] = ["# Model comparison", ""]
    rule_based = comparison.get("rule_based")
    if rule_based is not None:
        lines.append(
            "Rule-based baseline (shared): "
            f"direction_accuracy={_fmt(rule_based['direction_accuracy_mean'])}, "
            f"normalized_l2_mean={_fmt(rule_based['normalized_l2_mean'])}, "
            f"normalized_l2_median={_fmt(rule_based['normalized_l2_median'])} "
            f"(n={rule_based['num_pairs']})"
        )
        lines.append("")

    for engine in comparison["engines"]:
        lines.append(f"## {engine}")
        lines.append("")
        lines.append(
            "| model | setting | dir_acc | L2_mean | L2_median | "
            "api_err | parse_err | n |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in comparison["rows"]:
            if row["engine"] != engine:
                continue
            lines.append(
                f"| {row['model']} | {row['setting']} | "
                f"{_fmt(row['direction_accuracy_mean'])} | "
                f"{_fmt(row['normalized_l2_mean'])} | "
                f"{_fmt(row['normalized_l2_median'])} | "
                f"{_na(row['num_api_errors'])} | {_na(row['num_parse_errors'])} | "
                f"{_na(row['num_pairs'])} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.4f}"
    return "n/a"


def _na(value: Any) -> str:
    return "n/a" if value is None else str(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LLM models and few-shot settings across both engines."
    )
    parser.add_argument("input_path", type=Path, help="Dataset file or directory.")
    parser.add_argument("output_root", type=Path, help="Root output directory.")
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Recurse into directories."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model ids to compare (default: all registry models).",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["hybrid", "pure_llm"],
        choices=("hybrid", "pure_llm"),
        help="Engines to evaluate.",
    )
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-rule-based",
        action="store_true",
        help="Skip the shared rule-based baseline run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force keyless degrade (empty predictions, no network).",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip running; rebuild comparison.{csv,md} from existing summaries.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.aggregate_only:
        aggregate_comparison(args.output_root, engines=tuple(args.engines))
        return 0
    run_model_comparison(
        args.input_path,
        args.output_root,
        models=args.models,
        engines=tuple(args.engines),
        recursive=args.recursive,
        limit_pairs=args.limit_pairs,
        api_key="" if args.dry_run else None,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        include_rule_based=not args.no_rule_based,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
