"""Compare rule-based and hybrid pair-level evaluation CSVs."""
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
CSV_COLUMNS = [
    "source_file",
    "case_id",
    "pair_id",
    "pair_type",
    "kind_hint",
    "target_vitals",
    "rule_direction_accuracy",
    "hybrid_direction_accuracy",
    "delta_direction_accuracy",
    "rule_normalized_l2",
    "hybrid_normalized_l2",
    "delta_normalized_l2",
    "rule_predicted_vitals",
    "hybrid_predicted_vitals",
    "hybrid_llm_adjustments",
]


def compare_eval_results(
    rule_csv_path: str | Path,
    hybrid_csv_path: str | Path,
    output_dir: str | Path,
    *,
    emit_console_summary: bool = True,
) -> dict[str, Any]:
    """Compare existing pair-level evaluation CSVs without recomputing metrics."""

    rule_rows = _read_rows(Path(rule_csv_path))
    hybrid_rows = _read_rows(Path(hybrid_csv_path))
    records = build_comparison_records(rule_rows, hybrid_rows)
    summary = build_summary(records)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_pair_results_csv(output_dir / PAIR_RESULTS_FILENAME, records)
    _write_summary_json(output_dir / SUMMARY_FILENAME, summary)

    if emit_console_summary:
        print_console_summary(records)

    return summary


def build_comparison_records(
    rule_rows: Sequence[dict[str, str]],
    hybrid_rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    rule_by_key = _index_rows(rule_rows, label="rule")
    hybrid_by_key = _index_rows(hybrid_rows, label="hybrid")
    joined_keys = sorted(set(rule_by_key) & set(hybrid_by_key))

    records: list[dict[str, Any]] = []
    for key in joined_keys:
        rule = rule_by_key[key]
        hybrid = hybrid_by_key[key]
        rule_l2 = _parse_float(rule.get("normalized_l2"))
        hybrid_l2 = _parse_float(hybrid.get("normalized_l2"))
        rule_direction = _parse_float(rule.get("direction_accuracy"))
        hybrid_direction = _parse_float(hybrid.get("direction_accuracy"))
        delta_l2 = _delta(hybrid_l2, rule_l2)
        delta_direction = _delta(hybrid_direction, rule_direction)

        records.append(
            {
                "source_file": key[0],
                "case_id": key[1],
                "pair_id": key[2],
                "pair_type": hybrid.get("pair_type") or rule.get("pair_type"),
                "kind_hint": hybrid.get("kind_hint") or rule.get("kind_hint"),
                "target_vitals": _parse_json_field(
                    hybrid.get("target_vitals") or rule.get("target_vitals")
                ),
                "rule_direction_accuracy": rule_direction,
                "hybrid_direction_accuracy": hybrid_direction,
                "delta_direction_accuracy": delta_direction,
                "rule_normalized_l2": rule_l2,
                "hybrid_normalized_l2": hybrid_l2,
                "delta_normalized_l2": delta_l2,
                "improved_l2": _is_improved_l2(delta_l2),
                "worsened_l2": _is_worsened_l2(delta_l2),
                "unchanged_l2": _is_unchanged_l2(delta_l2),
                "rule_predicted_vitals": _parse_json_field(
                    rule.get("predicted_vitals")
                ),
                "hybrid_predicted_vitals": _parse_json_field(
                    hybrid.get("predicted_vitals")
                ),
                "hybrid_llm_adjustments": _parse_json_field(
                    hybrid.get("llm_adjustments")
                ),
            }
        )

    return records


def build_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    delta_l2_values = [
        record["delta_normalized_l2"]
        for record in records
        if record["delta_normalized_l2"] is not None
    ]
    compared_count = len(delta_l2_values)
    improved_count = sum(1 for record in records if record["improved_l2"])
    worsened_count = sum(1 for record in records if record["worsened_l2"])
    unchanged_count = sum(1 for record in records if record["unchanged_l2"])

    return {
        "overall": {
            "num_pairs": len(records),
            "num_l2_compared_pairs": compared_count,
            "mean_delta_normalized_l2": _mean_or_none(delta_l2_values),
            "median_delta_normalized_l2": _median_or_none(delta_l2_values),
            "improved_count": improved_count,
            "improved_percentage": _percentage(improved_count, compared_count),
            "worsened_count": worsened_count,
            "worsened_percentage": _percentage(worsened_count, compared_count),
            "unchanged_count": unchanged_count,
            "unchanged_percentage": _percentage(unchanged_count, compared_count),
        },
        "by_kind_hint": _group_summary(records),
    }


def print_console_summary(records: Sequence[dict[str, Any]]) -> None:
    """Print requested ranked comparison lists."""

    print("Top 20 most improved pairs:")
    _print_ranked_pairs(
        sorted(
            (
                record
                for record in records
                if record["delta_normalized_l2"] is not None
                and record["improved_l2"]
            ),
            key=lambda record: (
                record["delta_normalized_l2"],
                str(record["source_file"]),
                str(record["pair_id"]),
            ),
        )[:20]
    )

    print("Top 20 most worsened pairs:")
    _print_ranked_pairs(
        sorted(
            (
                record
                for record in records
                if record["delta_normalized_l2"] is not None
                and record["worsened_l2"]
            ),
            key=lambda record: (
                -record["delta_normalized_l2"],
                str(record["source_file"]),
                str(record["pair_id"]),
            ),
        )[:20]
    )

    print("Top 20 high-error unchanged pairs:")
    _print_ranked_pairs(
        sorted(
            (
                record
                for record in records
                if record["unchanged_l2"]
                and record["hybrid_normalized_l2"] is not None
            ),
            key=lambda record: (
                -record["hybrid_normalized_l2"],
                str(record["source_file"]),
                str(record["pair_id"]),
            ),
        )[:20],
        include_hybrid_l2=True,
    )


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
        "hybrid_llm_adjustments",
    ):
        csv_record[key] = json.dumps(
            csv_record[key],
            ensure_ascii=False,
            sort_keys=True,
        )
    return csv_record


def _group_summary(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_group_key(record.get("kind_hint"))].append(record)

    summary: dict[str, dict[str, Any]] = {}
    for kind_hint, group_records in sorted(grouped.items()):
        delta_l2_values = [
            record["delta_normalized_l2"]
            for record in group_records
            if record["delta_normalized_l2"] is not None
        ]
        delta_direction_values = [
            record["delta_direction_accuracy"]
            for record in group_records
            if record["delta_direction_accuracy"] is not None
        ]
        summary[kind_hint] = {
            "count": len(group_records),
            "mean_delta_l2": _mean_or_none(delta_l2_values),
            "median_delta_l2": _median_or_none(delta_l2_values),
            "improved_count": sum(
                1 for record in group_records if record["improved_l2"]
            ),
            "worsened_count": sum(
                1 for record in group_records if record["worsened_l2"]
            ),
            "unchanged_count": sum(
                1 for record in group_records if record["unchanged_l2"]
            ),
            "mean_delta_direction_accuracy": _mean_or_none(delta_direction_values),
        }
    return summary


def _print_ranked_pairs(
    records: Sequence[dict[str, Any]],
    *,
    include_hybrid_l2: bool = False,
) -> None:
    if not records:
        print("  (none)")
        return

    for record in records:
        extra = ""
        if include_hybrid_l2:
            extra = f" hybrid_l2={_format_metric(record['hybrid_normalized_l2'])}"
        print(
            "  "
            f"{record['source_file']}:{record['pair_id']} "
            f"kind_hint={_group_key(record['kind_hint'])} "
            f"delta_l2={_format_metric(record['delta_normalized_l2'])}"
            f"{extra}"
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
        description="Compare rule-based and hybrid pair-level evaluation CSVs."
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
        "output_dir",
        type=Path,
        help="Directory for comparison_pair_results.csv and comparison_summary.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    compare_eval_results(args.rule_based_csv, args.hybrid_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
