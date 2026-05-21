"""Compare rule-based, hybrid, and pure LLM pair-level evaluation CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Sequence

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


PAIR_RESULTS_FILENAME = "comparison_pair_results.csv"
SUMMARY_FILENAME = "comparison_summary.json"
JOIN_KEYS = ("source_file", "case_id", "pair_id")
L2_TOLERANCE = 1e-9
ENGINE_LABELS = ("rule", "hybrid", "pure_llm")
SUCCESS_CRITERION = "direction_accuracy == 1.0 for every requested vital"
PAIRWISE_LABELS = (
    ("hybrid_vs_rule", "hybrid", "rule"),
    ("pure_llm_vs_rule", "pure_llm", "rule"),
    ("pure_llm_vs_hybrid", "pure_llm", "hybrid"),
)
CSV_COLUMNS = [
    "source_file",
    "case_id",
    "pair_id",
    "pair_type",
    "kind_hint",
    "target_vitals",
    "rule_direction_accuracy",
    "hybrid_direction_accuracy",
    "pure_llm_direction_accuracy",
    "rule_passed",
    "hybrid_passed",
    "pure_llm_passed",
    "failed_engines",
    "passed_engines",
    "failure_pattern",
    "hybrid_delta_direction_accuracy_vs_rule",
    "pure_llm_delta_direction_accuracy_vs_rule",
    "pure_llm_delta_direction_accuracy_vs_hybrid",
    "rule_normalized_l2",
    "hybrid_normalized_l2",
    "pure_llm_normalized_l2",
    "hybrid_delta_normalized_l2_vs_rule",
    "pure_llm_delta_normalized_l2_vs_rule",
    "pure_llm_delta_normalized_l2_vs_hybrid",
    "best_engine_by_l2",
    "best_normalized_l2",
    "rule_predicted_vitals",
    "hybrid_predicted_vitals",
    "pure_llm_predicted_vitals",
    "hybrid_llm_adjustments",
    "hybrid_llm_reasoning",
    "pure_llm_reasoning",
    "hybrid_llm_api_error",
    "pure_llm_llm_api_error",
]


def compare_three_eval_results(
    rule_csv_path: str | Path,
    hybrid_csv_path: str | Path,
    pure_llm_csv_path: str | Path,
    output_dir: str | Path,
    *,
    emit_console_summary: bool = True,
) -> dict[str, Any]:
    """Compare existing pair-level CSVs without recomputing engine metrics."""

    rule_rows = _read_rows(Path(rule_csv_path))
    hybrid_rows = _read_rows(Path(hybrid_csv_path))
    pure_llm_rows = _read_rows(Path(pure_llm_csv_path))
    records = build_comparison_records(rule_rows, hybrid_rows, pure_llm_rows)
    summary = build_summary(records)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_pair_results_csv(output_dir / PAIR_RESULTS_FILENAME, records)
    _write_summary_json(output_dir / SUMMARY_FILENAME, summary)

    if emit_console_summary:
        print_console_summary(records, summary)

    return summary


def build_comparison_records(
    rule_rows: Sequence[dict[str, str]],
    hybrid_rows: Sequence[dict[str, str]],
    pure_llm_rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    rule_by_key = _index_rows(rule_rows, label="rule")
    hybrid_by_key = _index_rows(hybrid_rows, label="hybrid")
    pure_llm_by_key = _index_rows(pure_llm_rows, label="pure_llm")
    joined_keys = sorted(set(rule_by_key) & set(hybrid_by_key) & set(pure_llm_by_key))

    records: list[dict[str, Any]] = []
    for key in joined_keys:
        rule = rule_by_key[key]
        hybrid = hybrid_by_key[key]
        pure_llm = pure_llm_by_key[key]

        l2 = {
            "rule": _parse_float(rule.get("normalized_l2")),
            "hybrid": _parse_float(hybrid.get("normalized_l2")),
            "pure_llm": _parse_float(pure_llm.get("normalized_l2")),
        }
        direction = {
            "rule": _parse_float(rule.get("direction_accuracy")),
            "hybrid": _parse_float(hybrid.get("direction_accuracy")),
            "pure_llm": _parse_float(pure_llm.get("direction_accuracy")),
        }
        passed = {
            engine: _engine_passed(direction[engine])
            for engine in ENGINE_LABELS
        }
        best_engine, best_l2 = _best_engine_by_l2(l2)

        records.append(
            {
                "source_file": key[0],
                "case_id": key[1],
                "pair_id": key[2],
                "pair_type": _first_present(
                    pure_llm.get("pair_type"),
                    hybrid.get("pair_type"),
                    rule.get("pair_type"),
                ),
                "kind_hint": _first_present(
                    pure_llm.get("kind_hint"),
                    hybrid.get("kind_hint"),
                    rule.get("kind_hint"),
                ),
                "target_vitals": _parse_json_field(
                    _first_present(
                        pure_llm.get("target_vitals"),
                        hybrid.get("target_vitals"),
                        rule.get("target_vitals"),
                    )
                ),
                "rule_direction_accuracy": direction["rule"],
                "hybrid_direction_accuracy": direction["hybrid"],
                "pure_llm_direction_accuracy": direction["pure_llm"],
                "rule_passed": passed["rule"],
                "hybrid_passed": passed["hybrid"],
                "pure_llm_passed": passed["pure_llm"],
                "failed_engines": _engines_with_status(passed, passed_status=False),
                "passed_engines": _engines_with_status(passed, passed_status=True),
                "failure_pattern": _failure_pattern(passed),
                "hybrid_delta_direction_accuracy_vs_rule": _delta(
                    direction["hybrid"],
                    direction["rule"],
                ),
                "pure_llm_delta_direction_accuracy_vs_rule": _delta(
                    direction["pure_llm"],
                    direction["rule"],
                ),
                "pure_llm_delta_direction_accuracy_vs_hybrid": _delta(
                    direction["pure_llm"],
                    direction["hybrid"],
                ),
                "rule_normalized_l2": l2["rule"],
                "hybrid_normalized_l2": l2["hybrid"],
                "pure_llm_normalized_l2": l2["pure_llm"],
                "hybrid_delta_normalized_l2_vs_rule": _delta(
                    l2["hybrid"],
                    l2["rule"],
                ),
                "pure_llm_delta_normalized_l2_vs_rule": _delta(
                    l2["pure_llm"],
                    l2["rule"],
                ),
                "pure_llm_delta_normalized_l2_vs_hybrid": _delta(
                    l2["pure_llm"],
                    l2["hybrid"],
                ),
                "best_engine_by_l2": best_engine,
                "best_normalized_l2": best_l2,
                "rule_predicted_vitals": _parse_json_field(
                    rule.get("predicted_vitals")
                ),
                "hybrid_predicted_vitals": _parse_json_field(
                    hybrid.get("predicted_vitals")
                ),
                "pure_llm_predicted_vitals": _parse_json_field(
                    pure_llm.get("predicted_vitals")
                ),
                "hybrid_llm_adjustments": _parse_json_field(
                    hybrid.get("llm_adjustments")
                ),
                "hybrid_llm_reasoning": _parse_json_field(
                    hybrid.get("llm_reasoning")
                ),
                "pure_llm_reasoning": _parse_json_field(
                    pure_llm.get("llm_reasoning")
                ),
                "hybrid_llm_api_error": hybrid.get("llm_api_error"),
                "pure_llm_llm_api_error": pure_llm.get("llm_api_error"),
            }
        )

    return records


def build_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _summary_for_records(records),
        "by_kind_hint": _group_summary(records, key="kind_hint"),
    }


def print_console_summary(
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Print ranked pairwise comparison lists and grouped metrics."""

    overall = summary["overall"]
    print(f"Compared pairs: {overall['num_pairs']}")
    print("Normalized L2 mean by engine:")
    for engine in ENGINE_LABELS:
        metrics = overall["engines"][engine]
        print(f"  {engine}: {_format_metric(metrics['normalized_l2_mean'])}")

    for pairwise_key, new_label, old_label in PAIRWISE_LABELS:
        label = pairwise_key.replace("_", " ")
        delta_key = f"{new_label}_delta_normalized_l2_vs_{old_label}"
        print(f"Top 20 most improved pairs: {label}")
        _print_ranked_pairs(
            sorted(
                (
                    record
                    for record in records
                    if record[delta_key] is not None
                    and _is_improved_l2(record[delta_key])
                ),
                key=lambda record: (
                    record[delta_key],
                    str(record["source_file"]),
                    str(record["pair_id"]),
                ),
            )[:20],
            delta_key=delta_key,
        )

        print(f"Top 20 most worsened pairs: {label}")
        _print_ranked_pairs(
            sorted(
                (
                    record
                    for record in records
                    if record[delta_key] is not None
                    and _is_worsened_l2(record[delta_key])
                ),
                key=lambda record: (
                    -record[delta_key],
                    str(record["source_file"]),
                    str(record["pair_id"]),
                ),
            )[:20],
            delta_key=delta_key,
        )

    print("Best engine by normalized_l2:")
    for engine, count in sorted(overall["best_l2_counts"].items()):
        print(f"  {engine}: {count}")

    failure_analysis = overall["failure_analysis"]
    print(f"Failure analysis ({failure_analysis['success_criterion']}):")
    for bucket in (
        "rule_failed_hybrid_passed",
        "rule_failed_hybrid_failed_pure_llm_passed",
        "all_three_failed",
    ):
        print(f"  {bucket}: {failure_analysis[bucket]['count']}")

    print("Metrics by kind_hint:")
    kind_rows = sorted(
        summary["by_kind_hint"].items(),
        key=lambda item: (-item[1]["count"], item[0]),
    )
    if not kind_rows:
        print("  (none)")
    for kind_hint, metrics in kind_rows:
        engine_metrics = metrics["engines"]
        print(
            "  "
            f"{kind_hint}: count={metrics['count']} "
            f"rule_l2_mean={_format_metric(engine_metrics['rule']['normalized_l2_mean'])} "
            f"hybrid_l2_mean={_format_metric(engine_metrics['hybrid']['normalized_l2_mean'])} "
            f"pure_llm_l2_mean="
            f"{_format_metric(engine_metrics['pure_llm']['normalized_l2_mean'])}"
        )


def _summary_for_records(
    records: Sequence[dict[str, Any]],
    *,
    include_failure_pairs: bool = True,
) -> dict[str, Any]:
    return {
        "num_pairs": len(records),
        "engines": _engine_summaries(records),
        "pairwise": {
            pairwise_key: _pairwise_summary(
                records,
                new_label=new_label,
                old_label=old_label,
            )
            for pairwise_key, new_label, old_label in PAIRWISE_LABELS
        },
        "best_l2_counts": _best_l2_counts(records),
        "failure_analysis": _failure_analysis(
            records,
            include_pair_refs=include_failure_pairs,
        ),
    }


def _engine_summaries(
    records: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for engine in ENGINE_LABELS:
        direction_values = [
            record[f"{engine}_direction_accuracy"]
            for record in records
            if record[f"{engine}_direction_accuracy"] is not None
        ]
        normalized_l2_values = [
            record[f"{engine}_normalized_l2"]
            for record in records
            if record[f"{engine}_normalized_l2"] is not None
        ]
        summaries[engine] = {
            "direction_accuracy_mean": _mean_or_none(direction_values),
            "normalized_l2_mean": _mean_or_none(normalized_l2_values),
            "normalized_l2_median": _median_or_none(normalized_l2_values),
            "num_l2_pairs": len(normalized_l2_values),
        }
    return summaries


def _pairwise_summary(
    records: Sequence[dict[str, Any]],
    *,
    new_label: str,
    old_label: str,
) -> dict[str, Any]:
    delta_l2_key = f"{new_label}_delta_normalized_l2_vs_{old_label}"
    delta_direction_key = f"{new_label}_delta_direction_accuracy_vs_{old_label}"
    delta_l2_values = [
        record[delta_l2_key]
        for record in records
        if record[delta_l2_key] is not None
    ]
    delta_direction_values = [
        record[delta_direction_key]
        for record in records
        if record[delta_direction_key] is not None
    ]
    compared_count = len(delta_l2_values)
    improved_count = sum(1 for value in delta_l2_values if _is_improved_l2(value))
    worsened_count = sum(1 for value in delta_l2_values if _is_worsened_l2(value))
    unchanged_count = sum(1 for value in delta_l2_values if _is_unchanged_l2(value))

    return {
        "num_l2_compared_pairs": compared_count,
        "mean_delta_normalized_l2": _mean_or_none(delta_l2_values),
        "median_delta_normalized_l2": _median_or_none(delta_l2_values),
        "mean_delta_direction_accuracy": _mean_or_none(delta_direction_values),
        "improved_count": improved_count,
        "improved_percentage": _percentage(improved_count, compared_count),
        "worsened_count": worsened_count,
        "worsened_percentage": _percentage(worsened_count, compared_count),
        "unchanged_count": unchanged_count,
        "unchanged_percentage": _percentage(unchanged_count, compared_count),
    }


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
            **_summary_for_records(group_records, include_failure_pairs=False),
        }
        for group_key, group_records in sorted(grouped.items())
    }


def _best_l2_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {engine: 0 for engine in ENGINE_LABELS}
    counts["tie"] = 0
    counts["none"] = 0
    for record in records:
        best_engine = record["best_engine_by_l2"]
        if best_engine in counts:
            counts[best_engine] += 1
        else:
            counts["none"] += 1
    return counts


def _failure_analysis(
    records: Sequence[dict[str, Any]],
    *,
    include_pair_refs: bool = True,
) -> dict[str, Any]:
    grouped_by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_by_pattern[str(record.get("failure_pattern") or "unknown")].append(
            record
        )

    return {
        "success_criterion": SUCCESS_CRITERION,
        "engine_failure_counts": {
            engine: sum(
                1 for record in records if not bool(record.get(f"{engine}_passed"))
            )
            for engine in ENGINE_LABELS
        },
        "failure_pattern_counts": {
            pattern: len(pattern_records)
            for pattern, pattern_records in sorted(grouped_by_pattern.items())
        },
        "failure_patterns": {
            pattern: _failure_bucket_summary(
                pattern_records,
                include_pair_refs=include_pair_refs,
            )
            for pattern, pattern_records in sorted(grouped_by_pattern.items())
        },
        "rule_failed_hybrid_passed": _failure_bucket_summary(
            [
                record
                for record in records
                if not bool(record.get("rule_passed"))
                and bool(record.get("hybrid_passed"))
            ],
            include_pair_refs=include_pair_refs,
        ),
        "rule_failed_hybrid_passed_pure_llm_failed": _failure_bucket_summary(
            [
                record
                for record in records
                if not bool(record.get("rule_passed"))
                and bool(record.get("hybrid_passed"))
                and not bool(record.get("pure_llm_passed"))
            ],
            include_pair_refs=include_pair_refs,
        ),
        "rule_failed_hybrid_failed_pure_llm_passed": _failure_bucket_summary(
            [
                record
                for record in records
                if not bool(record.get("rule_passed"))
                and not bool(record.get("hybrid_passed"))
                and bool(record.get("pure_llm_passed"))
            ],
            include_pair_refs=include_pair_refs,
        ),
        "all_three_failed": _failure_bucket_summary(
            [
                record
                for record in records
                if not bool(record.get("rule_passed"))
                and not bool(record.get("hybrid_passed"))
                and not bool(record.get("pure_llm_passed"))
            ],
            include_pair_refs=include_pair_refs,
        ),
        "all_three_passed": _failure_bucket_summary(
            [
                record
                for record in records
                if bool(record.get("rule_passed"))
                and bool(record.get("hybrid_passed"))
                and bool(record.get("pure_llm_passed"))
            ],
            include_pair_refs=include_pair_refs,
        ),
    }


def _failure_bucket_summary(
    records: Sequence[dict[str, Any]],
    *,
    include_pair_refs: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(records)}
    if include_pair_refs:
        summary["pairs"] = [_pair_ref(record) for record in records]
    return summary


def _pair_ref(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_file": record.get("source_file"),
        "case_id": record.get("case_id"),
        "pair_id": record.get("pair_id"),
        "kind_hint": record.get("kind_hint"),
        "target_vitals": record.get("target_vitals"),
        "failure_pattern": record.get("failure_pattern"),
    }


def _engine_passed(direction_accuracy: float | None) -> bool:
    return (
        direction_accuracy is not None
        and math.isfinite(direction_accuracy)
        and direction_accuracy >= 1.0 - L2_TOLERANCE
    )


def _engines_with_status(
    passed_by_engine: dict[str, bool],
    *,
    passed_status: bool,
) -> list[str]:
    return [
        engine
        for engine in ENGINE_LABELS
        if bool(passed_by_engine.get(engine)) is passed_status
    ]


def _failure_pattern(passed_by_engine: dict[str, bool]) -> str:
    return "_".join(
        f"{engine}_{'pass' if bool(passed_by_engine.get(engine)) else 'fail'}"
        for engine in ENGINE_LABELS
    )


def _best_engine_by_l2(l2: dict[str, float | None]) -> tuple[str | None, float | None]:
    numeric = {
        engine: value
        for engine, value in l2.items()
        if value is not None and math.isfinite(value)
    }
    if not numeric:
        return None, None

    best_value = min(numeric.values())
    best_engines = [
        engine
        for engine, value in numeric.items()
        if abs(value - best_value) <= L2_TOLERANCE
    ]
    if len(best_engines) > 1:
        return "tie", best_value
    return best_engines[0], best_value


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _index_rows(
    rows: Sequence[dict[str, str]],
    *,
    label: str,
) -> dict[tuple[str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(join_key, "") for join_key in JOIN_KEYS)
        if key in indexed:
            raise ValueError(f"Duplicate {label} row for join key: {key}")
        indexed[key] = row
    return indexed


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
        "rule_predicted_vitals",
        "hybrid_predicted_vitals",
        "pure_llm_predicted_vitals",
        "hybrid_llm_adjustments",
        "hybrid_llm_reasoning",
        "pure_llm_reasoning",
        "failed_engines",
        "passed_engines",
    ):
        csv_record[key] = json.dumps(
            csv_record[key],
            ensure_ascii=False,
            sort_keys=True,
        )
    return csv_record


def _print_ranked_pairs(
    records: Sequence[dict[str, Any]],
    *,
    delta_key: str,
) -> None:
    if not records:
        print("  (none)")
        return

    for record in records:
        print(
            "  "
            f"{record['source_file']}:{record['pair_id']} "
            f"kind_hint={_group_key(record['kind_hint'])} "
            f"delta_l2={_format_metric(record[delta_key])}"
        )


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_json_field(value: str | None) -> Any:
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _first_present(*values: str | None) -> str | None:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _delta(new_value: float | None, old_value: float | None) -> float | None:
    if new_value is None or old_value is None:
        return None
    return new_value - old_value


def _is_improved_l2(delta_l2: float | None) -> bool:
    return delta_l2 is not None and delta_l2 < -L2_TOLERANCE


def _is_worsened_l2(delta_l2: float | None) -> bool:
    return delta_l2 is not None and delta_l2 > L2_TOLERANCE


def _is_unchanged_l2(delta_l2: float | None) -> bool:
    return delta_l2 is not None and abs(delta_l2) <= L2_TOLERANCE


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return median(values)


def _percentage(count: int, total: int) -> float | None:
    if total == 0:
        return None
    return (count / total) * 100.0


def _group_key(value: Any) -> str:
    if value is None or value == "":
        return "null"
    return str(value)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare rule-based, hybrid, and pure LLM pair-level evaluation CSVs."
        )
    )
    parser.add_argument(
        "rule_based_csv",
        type=Path,
        help="Path to rule_based_pair_results.csv.",
    )
    parser.add_argument(
        "hybrid_csv",
        type=Path,
        help="Path to hybrid_pair_results.csv.",
    )
    parser.add_argument(
        "pure_llm_csv",
        type=Path,
        help="Path to pure_llm_pair_results.csv.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for comparison_pair_results.csv and comparison_summary.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    compare_three_eval_results(
        args.rule_based_csv,
        args.hybrid_csv,
        args.pure_llm_csv,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
