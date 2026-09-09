"""Run PureLLMEngine evaluation over transition-pair JSON files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from transition_engines.common import (
    CANONICAL_VITAL_KEYS,
    NORMALIZED_L2_SCALES,
    build_engine_input,
    evaluate_pair,
    iter_engine_inputs,
)
from transition_engines.few_shot import ExampleBank, ExampleSelector
from transition_engines.openai_llm_client import OpenAILLMClient
from transition_engines.pure_llm_engine import LLMClient, PureLLMEngine
from transition_engines.run_rule_based_eval import discover_case_files


PAIR_RESULTS_FILENAME = "pure_llm_pair_results.csv"
SUMMARY_FILENAME = "pure_llm_summary.json"
CSV_COLUMNS = [
    "case_id",
    "source_file",
    "pair_id",
    "pair_type",
    "kind_hint",
    "target_vitals",
    "direction_accuracy",
    "normalized_l2",
    "num_target_vitals",
    "num_missing_predictions",
    "missing_predictions",
    "direction_match",
    "before_vitals",
    "target_vitals_values",
    "predicted_vitals",
    "llm_reasoning",
    "llm_raw_response",
    "llm_parse_error",
    "llm_api_error",
    "model",
]
VITAL_SCALES = NORMALIZED_L2_SCALES


class FakeLLMClient:
    """Deterministic no-op LLM client for runner integration tests."""

    def __init__(self) -> None:
        self.model = "fake"
        self.prompts: list[str] = []
        self.responses: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        response = json.dumps({"vitals": {}}, ensure_ascii=True, sort_keys=True)
        self.responses.append(response)
        return response


def run_pure_llm_eval(
    input_path: str | Path,
    output_dir: str | Path,
    recursive: bool = False,
    *,
    llm_mode: str = "fake",
    model: str = "gpt-5.5",
    temperature: float = 0.0,
    max_output_tokens: int = 300,
    limit_pairs: int | None = None,
    llm_client: LLMClient | None = None,
    example_selector: Any | None = None,
    emit_console_summary: bool = True,
) -> dict[str, Any]:
    """Evaluate PureLLMEngine and write pair-level CSV plus aggregate JSON.

    example_selector=None (default) runs zero-shot, identical to prior behavior.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if limit_pairs is not None and limit_pairs < 0:
        raise ValueError("limit_pairs must be non-negative")
    source_files = discover_case_files(input_path, recursive=recursive)
    client = (
        llm_client
        if llm_client is not None
        else _make_llm_client(
            llm_mode,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    )
    records = _evaluate_files(
        source_files,
        client,
        limit_pairs=limit_pairs,
        example_selector=example_selector,
    )
    summary = build_summary(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_pair_results_csv(output_dir / PAIR_RESULTS_FILENAME, records)
    _write_summary_json(output_dir / SUMMARY_FILENAME, summary)

    if emit_console_summary:
        print_console_summary(len(source_files), records, summary)

    return summary


def _make_llm_client(
    llm_mode: str,
    *,
    model: str = "gpt-5.5",
    temperature: float = 0.0,
    max_output_tokens: int = 300,
) -> LLMClient:
    if llm_mode == "fake":
        return FakeLLMClient()
    if llm_mode == "real":
        return OpenAILLMClient(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_schema_type="vitals",
        )
    raise ValueError(f"Unsupported llm_mode: {llm_mode!r}")


def _evaluate_files(
    source_files: Iterable[Path],
    llm_client: LLMClient,
    *,
    limit_pairs: int | None = None,
    example_selector: Any | None = None,
) -> list[dict[str, Any]]:
    engine = PureLLMEngine(llm_client)
    records: list[dict[str, Any]] = []

    for source_file in source_files:
        case_doc = _load_json(source_file)
        case_id = case_doc.get("case_id")
        for pair, _ in iter_engine_inputs(case_doc):
            if limit_pairs is not None and len(records) >= limit_pairs:
                return records
            engine_input = build_engine_input(case_doc, pair)
            target_vital_names = pair["label"]["evaluation"]["target_vitals"]
            # Zero-shot keeps the original predict() call signature exactly; only
            # supply the additive examples kwarg when a selector is configured.
            if example_selector is None:
                engine_output = engine.predict(
                    engine_input,
                    target_vital_names=target_vital_names,
                )
            else:
                engine_output = engine.predict(
                    engine_input,
                    target_vital_names=target_vital_names,
                    examples=example_selector.select(engine_input),
                )
            eval_result = evaluate_pair(pair, engine_output)
            records.append(
                _record_from_result(
                    case_id=case_id,
                    source_file=source_file,
                    pair=pair,
                    engine_input=engine_input,
                    engine_output=engine_output,
                    eval_result=eval_result,
                )
            )

    return records


def _record_from_result(
    *,
    case_id: str | None,
    source_file: Path,
    pair: dict[str, Any],
    engine_input: dict[str, Any],
    engine_output: dict[str, Any],
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    target_vitals = list(eval_result["target_vitals"])
    all_target_values = pair.get("label", {}).get("target", {}).get("vitals", {})
    target_vital_values = {
        vital: all_target_values.get(vital)
        for vital in target_vitals
        if vital in all_target_values
    }
    predicted_vitals = engine_output.get("prediction", {}).get("vitals", {})
    metadata = _mapping(engine_output.get("metadata"))
    action = _mapping(engine_input.get("action"))

    return {
        "case_id": case_id,
        "source_file": str(source_file),
        "pair_id": eval_result["pair_id"],
        "pair_type": pair.get("pair_type"),
        "kind_hint": action.get("kind_hint"),
        "target_vitals": target_vitals,
        "direction_accuracy": eval_result["direction_accuracy"],
        "normalized_l2": eval_result["normalized_l2"],
        "num_target_vitals": eval_result["num_target_vitals"],
        "num_missing_predictions": eval_result["num_missing_predictions"],
        "missing_predictions": eval_result["missing_predictions"],
        "direction_match": eval_result["direction_match"],
        "before_vitals": _mapping(engine_input.get("before")).get("vitals", {}),
        "target_vitals_values": target_vital_values,
        "predicted_vitals": predicted_vitals,
        "llm_reasoning": metadata.get("llm_reasoning", {}),
        "llm_raw_response": metadata.get("llm_raw_response"),
        "llm_parse_error": metadata.get("llm_parse_error"),
        "llm_api_error": metadata.get("llm_api_error"),
        "model": metadata.get("model"),
    }


def build_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build overall and grouped aggregate metrics from pair records."""

    direction_values = [
        record["direction_accuracy"]
        for record in records
        if record["direction_accuracy"] is not None
    ]
    normalized_l2_values = [
        record["normalized_l2"]
        for record in records
        if record["normalized_l2"] is not None
    ]

    return {
        "overall": {
            "num_pairs": len(records),
            "num_evaluated_pairs": sum(
                1 for record in records if _is_evaluated(record)
            ),
            "direction_accuracy_mean": _mean_or_none(direction_values),
            "normalized_l2_mean": _mean_or_none(normalized_l2_values),
            "normalized_l2_median": _median_or_none(normalized_l2_values),
            "num_api_errors": _count_present(records, "llm_api_error"),
            "num_parse_errors": _count_present(records, "llm_parse_error"),
        },
        "by_kind_hint": _group_summary(
            records,
            key="kind_hint",
            include_error_counts=True,
        ),
        "by_pair_type": _group_summary(records, key="pair_type"),
        "by_target_vital": _target_vital_summary(records),
    }


def print_console_summary(
    total_files: int,
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Print a concise deterministic run summary."""

    overall = summary["overall"]
    print(f"Total files: {total_files}")
    print(f"Total pairs: {overall['num_pairs']}")
    print(
        "Overall direction accuracy: "
        f"{_format_metric(overall['direction_accuracy_mean'])}"
    )
    print(
        "Overall normalized L2 mean/median: "
        f"{_format_metric(overall['normalized_l2_mean'])} / "
        f"{_format_metric(overall['normalized_l2_median'])}"
    )
    print(f"Num API errors: {overall['num_api_errors']}")
    print(f"Num parse errors: {overall['num_parse_errors']}")

    print("Top 10 worst pairs by normalized_l2:")
    worst_pairs = sorted(
        (record for record in records if record["normalized_l2"] is not None),
        key=lambda record: (
            -record["normalized_l2"],
            str(record["source_file"]),
            str(record["pair_id"]),
        ),
    )[:10]
    if not worst_pairs:
        print("  (none)")
    for record in worst_pairs:
        print(
            "  "
            f"{record['source_file']}:{record['pair_id']} "
            f"kind_hint={_group_key(record['kind_hint'])} "
            f"normalized_l2={_format_metric(record['normalized_l2'])}"
        )

    print("Metrics by kind_hint:")
    kind_rows = sorted(
        summary["by_kind_hint"].items(),
        key=lambda item: (-item[1]["count"], item[0]),
    )
    if not kind_rows:
        print("  (none)")
    for kind_hint, metrics in kind_rows:
        print(
            "  "
            f"{kind_hint}: count={metrics['count']} "
            f"direction_accuracy_mean="
            f"{_format_metric(metrics['direction_accuracy_mean'])} "
            f"normalized_l2_mean={_format_metric(metrics['normalized_l2_mean'])} "
            f"normalized_l2_median={_format_metric(metrics['normalized_l2_median'])} "
            f"num_api_errors={metrics['num_api_errors']} "
            f"num_parse_errors={metrics['num_parse_errors']}"
        )


def _group_summary(
    records: Sequence[dict[str, Any]],
    *,
    key: str,
    include_error_counts: bool = False,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_group_key(record.get(key))].append(record)

    summary: dict[str, dict[str, Any]] = {}
    for group_key, group_records in sorted(grouped.items()):
        direction_values = [
            record["direction_accuracy"]
            for record in group_records
            if record["direction_accuracy"] is not None
        ]
        normalized_l2_values = [
            record["normalized_l2"]
            for record in group_records
            if record["normalized_l2"] is not None
        ]
        metrics = {
            "count": len(group_records),
            "direction_accuracy_mean": _mean_or_none(direction_values),
            "normalized_l2_mean": _mean_or_none(normalized_l2_values),
            "normalized_l2_median": _median_or_none(normalized_l2_values),
        }
        if include_error_counts:
            metrics["num_api_errors"] = _count_present(group_records, "llm_api_error")
            metrics["num_parse_errors"] = _count_present(
                group_records,
                "llm_parse_error",
            )
        summary[group_key] = metrics
    return summary


def _target_vital_summary(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for vital in CANONICAL_VITAL_KEYS:
        count = 0
        direction_values: list[float] = []
        normalized_abs_errors: list[float] = []

        for record in records:
            if vital not in record["target_vitals"]:
                continue
            count += 1

            if vital in record["direction_match"]:
                direction_values.append(
                    1.0 if record["direction_match"][vital] else 0.0
                )

            target_value = record["target_vitals_values"].get(vital)
            predicted_value = record["predicted_vitals"].get(vital)
            if _is_number(target_value) and _is_number(predicted_value):
                normalized_abs_errors.append(
                    abs(float(predicted_value) - float(target_value))
                    / VITAL_SCALES[vital]
                )

        summary[vital] = {
            "count": count,
            "direction_accuracy_mean_for_that_vital": _mean_or_none(
                direction_values
            ),
            "normalized_abs_error_mean_for_that_vital": _mean_or_none(
                normalized_abs_errors
            ),
        }

    return summary


def _write_pair_results_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(_csv_record(record))


def _write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _csv_record(record: dict[str, Any]) -> dict[str, Any]:
    csv_record = dict(record)
    for key in (
        "target_vitals",
        "missing_predictions",
        "direction_match",
        "before_vitals",
        "target_vitals_values",
        "predicted_vitals",
        "llm_reasoning",
    ):
        csv_record[key] = json.dumps(
            csv_record[key],
            ensure_ascii=False,
            sort_keys=True,
        )
    return csv_record


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_evaluated(record: dict[str, Any]) -> bool:
    return record["direction_accuracy"] is not None or record["normalized_l2"] is not None


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return median(values)


def _count_present(records: Sequence[dict[str, Any]], key: str) -> int:
    return sum(1 for record in records if record.get(key))


def _group_key(value: Any) -> str:
    if value is None:
        return "null"
    return str(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PureLLMEngine over transition-pair JSON files."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Single transition-pair JSON file or directory of JSON files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for pure_llm_pair_results.csv and pure_llm_summary.json.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search input directories for JSON case files.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=("fake", "real"),
        default="fake",
        help="LLM backend mode. Fake mode requires no API key.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.5",
        help="OpenAI model name used with --llm-mode real.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="OpenAI sampling temperature used with --llm-mode real.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=300,
        help="OpenAI max output tokens used with --llm-mode real.",
    )
    parser.add_argument(
        "--limit-pairs",
        type=int,
        help="Evaluate at most this many pairs; useful for real-mode smoke tests.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        choices=(0, 3),
        default=0,
        help="Few-shot demonstrations per pair (0 = zero-shot).",
    )
    parser.add_argument(
        "--example-strategy",
        choices=("static", "kind_hint_matched"),
        default="static",
        help="Few-shot example selection strategy (used when --shots > 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic few-shot example selection.",
    )
    parser.add_argument(
        "--example-source",
        type=Path,
        default=None,
        help="Dataset path for the few-shot example bank (defaults to input_path).",
    )
    return parser.parse_args(argv)


def build_example_selector(
    *,
    shots: int,
    strategy: str,
    seed: int,
    example_source: Path,
    engine_kind: str = "pure_llm",
) -> ExampleSelector | None:
    """Build a leave-one-case-out selector, or None for zero-shot."""

    if shots <= 0:
        return None
    bank = ExampleBank.from_dataset(example_source, recursive=True)
    return ExampleSelector(
        bank,
        k=shots,
        strategy=strategy,
        seed=seed,
        engine_kind=engine_kind,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selector = build_example_selector(
        shots=args.shots,
        strategy=args.example_strategy,
        seed=args.seed,
        example_source=args.example_source or args.input_path,
        engine_kind="pure_llm",
    )
    run_pure_llm_eval(
        args.input_path,
        args.output_dir,
        recursive=args.recursive,
        llm_mode=args.llm_mode,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        limit_pairs=args.limit_pairs,
        example_selector=selector,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
