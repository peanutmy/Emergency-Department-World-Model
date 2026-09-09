"""Run deterministic rule-based evaluation over transition-pair JSON files."""
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
from transition_engines.rule_based import RuleBasedEngine


PAIR_RESULTS_FILENAME = "rule_based_pair_results.csv"
SUMMARY_FILENAME = "rule_based_summary.json"
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
]
VITAL_SCALES = NORMALIZED_L2_SCALES


def run_rule_based_eval(
    input_path: str | Path,
    output_dir: str | Path,
    recursive: bool = False,
    *,
    emit_console_summary: bool = True,
) -> dict[str, Any]:
    """Evaluate RuleBasedEngine and write pair-level CSV plus aggregate JSON."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    source_files = discover_case_files(input_path, recursive=recursive)
    records = _evaluate_files(source_files)
    summary = build_summary(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_pair_results_csv(output_dir / PAIR_RESULTS_FILENAME, records)
    _write_summary_json(output_dir / SUMMARY_FILENAME, summary)

    if emit_console_summary:
        print_console_summary(len(source_files), records, summary)

    return summary


def discover_case_files(input_path: str | Path, recursive: bool = False) -> list[Path]:
    """Return deterministic case JSON files from a file or directory input."""

    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Input file is not a JSON file: {input_path}")
        case_doc = _load_json(input_path)
        if not _is_case_doc(case_doc):
            raise ValueError(f"Input JSON is not a transition case file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*.json" if recursive else "*.json"
    candidates = sorted(input_path.glob(pattern), key=_path_sort_key)
    case_files: list[Path] = []
    for candidate in candidates:
        case_doc = _load_json(candidate)
        if _is_case_doc(case_doc):
            case_files.append(candidate)
    return case_files


def build_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build overall and grouped aggregate metrics from pair records."""

    by_kind_hint = _group_summary(records, key="kind_hint")
    by_pair_type = _group_summary(records, key="pair_type")
    return {
        "overall": {
            "num_pairs": len(records),
            "num_evaluated_pairs": sum(1 for record in records if _is_evaluated(record)),
            "direction_accuracy_mean": _mean_or_none(
                [
                    record["direction_accuracy"]
                    for record in records
                    if record["direction_accuracy"] is not None
                ]
            ),
            "normalized_l2_mean": _mean_or_none(
                [
                    record["normalized_l2"]
                    for record in records
                    if record["normalized_l2"] is not None
                ]
            ),
            "normalized_l2_median": _median_or_none(
                [
                    record["normalized_l2"]
                    for record in records
                    if record["normalized_l2"] is not None
                ]
            ),
        },
        "by_kind_hint": by_kind_hint,
        "by_pair_type": by_pair_type,
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
            f"normalized_l2_median={_format_metric(metrics['normalized_l2_median'])}"
        )


def _evaluate_files(source_files: Iterable[Path]) -> list[dict[str, Any]]:
    engine = RuleBasedEngine()
    records: list[dict[str, Any]] = []

    for source_file in source_files:
        case_doc = _load_json(source_file)
        case_id = case_doc.get("case_id")
        for pair, _ in iter_engine_inputs(case_doc):
            engine_input = build_engine_input(case_doc, pair)
            engine_output = engine.predict(engine_input)
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
    action = engine_input.get("action", {})
    if not isinstance(action, dict):
        action = {}

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
        "before_vitals": engine_input.get("before", {}).get("vitals", {}),
        "target_vitals_values": target_vital_values,
        "predicted_vitals": predicted_vitals,
    }


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
    ):
        csv_record[key] = json.dumps(
            csv_record[key],
            ensure_ascii=False,
            sort_keys=True,
        )
    return csv_record


def _group_summary(
    records: Sequence[dict[str, Any]],
    *,
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_group_key(record.get(key))].append(record)

    return {
        group_key: {
            "count": len(group_records),
            "direction_accuracy_mean": _mean_or_none(
                [
                    record["direction_accuracy"]
                    for record in group_records
                    if record["direction_accuracy"] is not None
                ]
            ),
            "normalized_l2_mean": _mean_or_none(
                [
                    record["normalized_l2"]
                    for record in group_records
                    if record["normalized_l2"] is not None
                ]
            ),
            "normalized_l2_median": _median_or_none(
                [
                    record["normalized_l2"]
                    for record in group_records
                    if record["normalized_l2"] is not None
                ]
            ),
        }
        for group_key, group_records in sorted(grouped.items())
    }


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _is_case_doc(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("pairs"), list)


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


def _group_key(value: Any) -> str:
    if value is None:
        return "null"
    return str(value)


def _path_sort_key(path: Path) -> str:
    return path.as_posix()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RuleBasedEngine over transition-pair JSON files."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Single transition-pair JSON file or directory of JSON files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for rule_based_pair_results.csv and rule_based_summary.json.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search input directories for JSON case files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_rule_based_eval(args.input_path, args.output_dir, recursive=args.recursive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
