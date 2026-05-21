"""Build target-vital pair analysis JSON from three engine result CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


JOIN_KEYS = ("source_file", "case_id", "pair_id")
DEFAULT_OUTPUT_FILENAME = "pair_engine_analysis.json"
PASS_TOLERANCE = 1e-9
SUCCESS_CRITERION = "direction_accuracy == 1.0 for every requested target vital"


def build_pair_analysis_json(
    rule_csv_path: str | Path,
    hybrid_csv_path: str | Path,
    pure_llm_csv_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build and write pair-level JSON focused on target vitals only."""

    rule_rows = _read_rows(Path(rule_csv_path))
    hybrid_rows = _read_rows(Path(hybrid_csv_path))
    pure_llm_rows = _read_rows(Path(pure_llm_csv_path))
    pairs = build_pair_records(rule_rows, hybrid_rows, pure_llm_rows)
    document = {
        "metadata": {
            "num_pairs": len(pairs),
            "success_criterion": SUCCESS_CRITERION,
            "source_csvs": {
                "rule_based": str(rule_csv_path),
                "hybrid": str(hybrid_csv_path),
                "pure_llm": str(pure_llm_csv_path),
            },
        },
        "pairs": pairs,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return document


def build_pair_records(
    rule_rows: Sequence[dict[str, str]],
    hybrid_rows: Sequence[dict[str, str]],
    pure_llm_rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    """Join engine CSV rows and shape each pair into the requested JSON form."""

    rule_by_key = _index_rows(rule_rows, label="rule")
    hybrid_by_key = _index_rows(hybrid_rows, label="hybrid")
    pure_llm_by_key = _index_rows(pure_llm_rows, label="pure_llm")
    joined_keys = sorted(set(rule_by_key) & set(hybrid_by_key) & set(pure_llm_by_key))

    records: list[dict[str, Any]] = []
    for key in joined_keys:
        rule = rule_by_key[key]
        hybrid = hybrid_by_key[key]
        pure_llm = pure_llm_by_key[key]

        target_vitals = _parse_json_list(
            _first_present(
                rule.get("target_vitals"),
                hybrid.get("target_vitals"),
                pure_llm.get("target_vitals"),
            )
        )
        before_vitals = _filter_to_targets(
            _parse_json_mapping(
                _first_present(
                    rule.get("before_vitals"),
                    hybrid.get("before_vitals"),
                    pure_llm.get("before_vitals"),
                )
            ),
            target_vitals,
        )
        post_vitals = _filter_to_targets(
            _parse_json_mapping(
                _first_present(
                    rule.get("target_vitals_values"),
                    hybrid.get("target_vitals_values"),
                    pure_llm.get("target_vitals_values"),
                )
            ),
            target_vitals,
        )

        records.append(
            {
                "source_file": key[0],
                "pair_id": key[2],
                "before_vitals": before_vitals,
                "kind_hint": _first_present(
                    rule.get("kind_hint"),
                    hybrid.get("kind_hint"),
                    pure_llm.get("kind_hint"),
                ),
                "post_vitals": post_vitals,
                "target_vitals": target_vitals,
                "rule_based_engine": {
                    "predicted_vitals": _filter_to_targets(
                        _parse_json_mapping(rule.get("predicted_vitals")),
                        target_vitals,
                    ),
                    "passed": _passed(rule.get("direction_accuracy")),
                },
                "hybrid_engine": {
                    "adjustments": _filter_to_targets(
                        _parse_json_mapping(hybrid.get("llm_adjustments")),
                        target_vitals,
                    ),
                    "predicted_vitals": _filter_to_targets(
                        _parse_json_mapping(hybrid.get("predicted_vitals")),
                        target_vitals,
                    ),
                    "passed": _passed(hybrid.get("direction_accuracy")),
                    "reasoning": _filter_to_targets(
                        _parse_json_mapping(hybrid.get("llm_reasoning")),
                        target_vitals,
                    ),
                },
                "pure_llm_engine": {
                    "predicted_vitals": _filter_to_targets(
                        _parse_json_mapping(pure_llm.get("predicted_vitals")),
                        target_vitals,
                    ),
                    "passed": _passed(pure_llm.get("direction_accuracy")),
                    "reasoning": _filter_to_targets(
                        _parse_json_mapping(pure_llm.get("llm_reasoning")),
                        target_vitals,
                    ),
                },
            }
        )
    return records


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


def _filter_to_targets(
    values: dict[str, Any],
    target_vitals: Sequence[str],
) -> dict[str, Any]:
    return {vital: values.get(vital) for vital in target_vitals}


def _parse_json_mapping(value: str | None) -> dict[str, Any]:
    parsed = _parse_json(value)
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_list(value: str | None) -> list[str]:
    parsed = _parse_json(value)
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _parse_json(value: str | None) -> Any:
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _first_present(*values: str | None) -> str | None:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _passed(direction_accuracy: str | None) -> bool:
    if direction_accuracy in (None, ""):
        return False
    try:
        value = float(direction_accuracy)
    except ValueError:
        return False
    return math.isfinite(value) and value >= 1.0 - PASS_TOLERANCE


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build target-vital pair analysis JSON from engine CSV outputs."
    )
    parser.add_argument("rule_based_csv", type=Path)
    parser.add_argument("hybrid_csv", type=Path)
    parser.add_argument("pure_llm_csv", type=Path)
    parser.add_argument("output_path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    document = build_pair_analysis_json(
        args.rule_based_csv,
        args.hybrid_csv,
        args.pure_llm_csv,
        args.output_path,
    )
    print(f"Wrote {document['metadata']['num_pairs']} pairs to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
