"""Evaluate the two-stage direction-first hybrid over transition-pair JSON."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from transition_engines.clients import build_llm_client
from transition_engines.common import (
    CANONICAL_VITAL_KEYS,
    NORMALIZED_L2_SCALES,
    build_engine_input,
    direction,
    evaluate_pair,
)
from transition_engines.direction_first_hybrid_engine import (
    DirectionFirstHybridEngine,
    LLMClient,
    STAGE1_PROMPT_PROFILES,
)
from transition_engines.few_shot import (
    ExampleBank,
    render_direction_first_example,
)
from transition_engines.rule_based import RuleBasedEngine
from transition_engines.run_rule_based_eval import (
    build_summary,
    discover_case_files,
    print_console_summary,
)


PAIR_RESULTS_FILENAME = "direction_first_hybrid_pair_results.csv"
SUMMARY_FILENAME = "direction_first_hybrid_summary.json"

# Each role has deterministic fallbacks for leave-one-case-out evaluation.
STATIC_REFERENCE_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {
    "untreated_deterioration": (
        ("massive_upper_gi_bleed", "p1"),
        ("severe_pediatric_asthma_exacerbation", "p2"),
        ("adrenal_crisis", "p3"),
    ),
    "effective_treatment": (
        ("opioid_overdose_with_ards", "p3"),
        ("aortic_dissection", "p6"),
        ("subarachnoid_hemorrhage", "p7"),
    ),
    "stable_or_limited_response": (
        ("serotonin_syndrome", "p1"),
        ("opioid_overdose_with_ards", "p5"),
        ("subarachnoid_hemorrhage", "p4"),
    ),
}

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
    "rule_prediction_vitals",
    "selected_directions",
    "direction_sources",
    "effective_directions",
    "resolved_magnitudes",
    "magnitude_sources",
    "computed_residuals",
    "few_shot_examples",
    "direction_llm_directions",
    "direction_llm_reasoning",
    "direction_llm_raw_response",
    "direction_llm_parse_error",
    "direction_llm_api_error",
    "magnitude_llm_magnitudes",
    "magnitude_llm_reasoning",
    "magnitude_llm_raw_response",
    "magnitude_llm_parse_error",
    "magnitude_llm_api_error",
    "stage1_prompt_profile",
    "model",
]


class FakeDirectionClient:
    """Deterministic Stage 1 client for local wiring checks."""

    model = "fake-direction"
    last_api_error: str | None = None

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {
                "directions": {
                    vital: "stable" for vital in CANONICAL_VITAL_KEYS
                },
                "reasoning": {
                    vital: "Fake stable direction."
                    for vital in CANONICAL_VITAL_KEYS
                },
            },
            sort_keys=True,
        )


class FakeMagnitudeClient:
    """Deterministic Stage 2 client for local wiring checks."""

    model = "fake-magnitude"
    last_api_error: str | None = None

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {
                "magnitudes": {
                    vital: None for vital in CANONICAL_VITAL_KEYS
                },
                "reasoning": {
                    vital: "Fake stable magnitude."
                    for vital in CANONICAL_VITAL_KEYS
                },
            },
            sort_keys=True,
        )


class DirectionFirstStaticSelector:
    """Select one fixed demonstration per role with case-level exclusion."""

    def __init__(
        self,
        bank: ExampleBank,
        reference_groups: dict[
            str, tuple[tuple[str, str], ...]
        ] = STATIC_REFERENCE_GROUPS,
    ) -> None:
        self.reference_groups = reference_groups
        self._entries = {
            (entry.get("case_id"), entry.get("pair_id")): entry
            for entry in bank.entries
        }
        missing = [
            f"{case_id}:{pair_id}"
            for references in reference_groups.values()
            for case_id, pair_id in references
            if (case_id, pair_id) not in self._entries
        ]
        if missing:
            raise ValueError(
                "Static direction-first examples missing from bank: "
                + ", ".join(missing)
            )

    def select(self, engine_input: dict[str, Any]) -> list[dict[str, Any]]:
        current_case = engine_input.get("case_id")
        selected_case_ids: set[Any] = set()
        rendered: list[dict[str, Any]] = []

        for role, references in self.reference_groups.items():
            entry = self._select_reference(
                references,
                current_case=current_case,
                selected_case_ids=selected_case_ids,
            )
            selected_case_ids.add(entry.get("case_id"))
            rendered.append(
                {
                    "_case_id": entry.get("case_id"),
                    "_pair_id": entry.get("pair_id"),
                    "_kind_hint": entry.get("kind_hint"),
                    "_role": role,
                    **render_direction_first_example(entry),
                }
            )
        return rendered

    def _select_reference(
        self,
        references: tuple[tuple[str, str], ...],
        *,
        current_case: Any,
        selected_case_ids: set[Any],
    ) -> dict[str, Any]:
        for reference in references:
            entry = self._entries[reference]
            case_id = entry.get("case_id")
            if case_id != current_case and case_id not in selected_case_ids:
                return entry
        raise ValueError(
            "No leave-one-case-out static example is available for "
            f"current_case={current_case!r}"
        )


def run_direction_first_hybrid_eval(
    input_path: str | Path,
    output_dir: str | Path,
    recursive: bool = False,
    *,
    llm_mode: str = "fake",
    model: str = "gpt-5",
    temperature: float = 0.0,
    max_output_tokens: int = 800,
    limit_pairs: int | None = None,
    direction_client: LLMClient | None = None,
    magnitude_client: LLMClient | None = None,
    example_selector: Any | None = None,
    stage1_prompt_profile: str = "original",
    emit_console_summary: bool = True,
    emit_progress: bool = True,
) -> dict[str, Any]:
    """Run the direction-first engine and write detailed evaluation outputs."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if limit_pairs is not None and limit_pairs < 0:
        raise ValueError("limit_pairs must be non-negative")

    source_files = discover_case_files(input_path, recursive=recursive)
    if direction_client is None or magnitude_client is None:
        default_direction, default_magnitude = _make_clients(
            llm_mode,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        direction_client = direction_client or default_direction
        magnitude_client = magnitude_client or default_magnitude

    records = _evaluate_files(
        source_files,
        direction_client=direction_client,
        magnitude_client=magnitude_client,
        limit_pairs=limit_pairs,
        example_selector=example_selector,
        stage1_prompt_profile=stage1_prompt_profile,
        emit_progress=emit_progress,
    )
    summary = build_summary(records)
    summary["vital_level"] = build_vital_level_summary(records)
    summary["llm_diagnostics"] = build_llm_diagnostics(records)
    summary["configuration"] = {
        "engine": "direction_first_hybrid",
        "model": model,
        "shots": 3 if example_selector is not None else 0,
        "strategy": "static" if example_selector is not None else None,
        "stage1_prompt_profile": stage1_prompt_profile,
        "two_stage_calls_per_pair": 2,
        "static_reference_groups": {
            role: [
                {"case_id": case_id, "pair_id": pair_id}
                for case_id, pair_id in references
            ]
            for role, references in STATIC_REFERENCE_GROUPS.items()
        }
        if example_selector is not None
        else {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_pair_results_csv(output_dir / PAIR_RESULTS_FILENAME, records)
    _write_summary_json(output_dir / SUMMARY_FILENAME, summary)

    if emit_console_summary:
        print_console_summary(len(source_files), records, summary)
        vital_level = summary["vital_level"]
        diagnostics = summary["llm_diagnostics"]
        print(
            "Vital-level direction accuracy: "
            f"{_format_metric(vital_level['direction_accuracy'])} "
            f"({vital_level['num_correct_directions']}/"
            f"{vital_level['num_vital_observations']})"
        )
        print(
            "Stage API errors (direction/magnitude): "
            f"{diagnostics['direction_api_errors']}/"
            f"{diagnostics['magnitude_api_errors']}"
        )

    return summary


def build_vital_level_summary(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build micro-averaged direction metrics over target vital observations."""

    labels = ("increase", "decrease", "stable")
    confusion = {
        true_label: {pred_label: 0 for pred_label in labels}
        for true_label in labels
    }
    by_vital: dict[str, dict[str, int]] = defaultdict(
        lambda: {"support": 0, "correct": 0}
    )
    normalized_squares: list[float] = []

    for record in records:
        before = _mapping(record.get("before_vitals"))
        target = _mapping(record.get("target_vitals_values"))
        predicted = _mapping(record.get("predicted_vitals"))
        for vital in record.get("target_vitals", []):
            if vital not in CANONICAL_VITAL_KEYS:
                continue
            before_value = before.get(vital)
            target_value = target.get(vital)
            predicted_value = predicted.get(vital)
            if not all(
                _is_number(value)
                for value in (before_value, target_value, predicted_value)
            ):
                continue
            true_label = _direction_label(
                direction(before_value, target_value, vital)
            )
            predicted_label = _direction_label(
                direction(before_value, predicted_value, vital)
            )
            if true_label is None or predicted_label is None:
                continue
            confusion[true_label][predicted_label] += 1
            by_vital[vital]["support"] += 1
            if true_label == predicted_label:
                by_vital[vital]["correct"] += 1
            normalized_squares.append(
                (
                    (float(predicted_value) - float(target_value))
                    / NORMALIZED_L2_SCALES[vital]
                )
                ** 2
            )

    support = {
        label: sum(confusion[label].values())
        for label in labels
    }
    predicted_counts = {
        label: sum(confusion[true][label] for true in labels)
        for label in labels
    }
    per_class = {
        label: {
            "support": support[label],
            "predicted_count": predicted_counts[label],
            "precision": _divide(
                confusion[label][label],
                predicted_counts[label],
            ),
            "recall": _divide(confusion[label][label], support[label]),
        }
        for label in labels
    }
    total = sum(support.values())
    correct = sum(confusion[label][label] for label in labels)
    return {
        "num_vital_observations": total,
        "num_correct_directions": correct,
        "direction_accuracy": _divide(correct, total),
        "normalized_rmse": (
            math.sqrt(sum(normalized_squares) / len(normalized_squares))
            if normalized_squares
            else None
        ),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "by_vital": {
            vital: {
                **counts,
                "direction_accuracy": _divide(
                    counts["correct"],
                    counts["support"],
                ),
            }
            for vital, counts in sorted(by_vital.items())
        },
    }


def build_llm_diagnostics(
    records: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """Count per-stage errors and fallback use."""

    return {
        "direction_api_errors": sum(
            bool(record.get("direction_llm_api_error")) for record in records
        ),
        "direction_parse_errors": sum(
            bool(record.get("direction_llm_parse_error")) for record in records
        ),
        "magnitude_api_errors": sum(
            bool(record.get("magnitude_llm_api_error")) for record in records
        ),
        "magnitude_parse_errors": sum(
            bool(record.get("magnitude_llm_parse_error")) for record in records
        ),
        "direction_rule_fallback_vitals": sum(
            value == "rule_fallback"
            for record in records
            for value in _mapping(record.get("direction_sources")).values()
        ),
        "magnitude_rule_fallback_vitals": sum(
            str(value).startswith("rule_absolute_delta")
            for record in records
            for value in _mapping(record.get("magnitude_sources")).values()
        ),
    }


def _make_clients(
    llm_mode: str,
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> tuple[LLMClient, LLMClient]:
    if llm_mode == "fake":
        return FakeDirectionClient(), FakeMagnitudeClient()
    if llm_mode == "real":
        if model.startswith("gpt-") and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set; refusing to run a real GPT "
                "evaluation with fallback responses."
            )
        return (
            build_llm_client(
                model,
                response_schema_type="directions",
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
            build_llm_client(
                model,
                response_schema_type="magnitudes",
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
    raise ValueError(f"Unsupported llm_mode: {llm_mode!r}")


def _evaluate_files(
    source_files: Iterable[Path],
    *,
    direction_client: LLMClient,
    magnitude_client: LLMClient,
    limit_pairs: int | None,
    example_selector: Any | None,
    stage1_prompt_profile: str,
    emit_progress: bool,
) -> list[dict[str, Any]]:
    files = list(source_files)
    total_pairs = _count_pairs(files)
    if limit_pairs is not None:
        total_pairs = min(total_pairs, limit_pairs)

    engine = DirectionFirstHybridEngine(
        RuleBasedEngine(),
        direction_client,
        magnitude_client,
        stage1_prompt_profile=stage1_prompt_profile,
    )
    records: list[dict[str, Any]] = []
    for source_file in files:
        case_doc = _load_json(source_file)
        case_id = case_doc.get("case_id")
        for pair in case_doc.get("pairs", []):
            if limit_pairs is not None and len(records) >= limit_pairs:
                return records
            engine_input = build_engine_input(case_doc, pair)
            examples = (
                example_selector.select(engine_input)
                if example_selector is not None
                else None
            )
            engine_output = engine.predict(
                engine_input,
                target_vital_names=pair["label"]["evaluation"]["target_vitals"],
                examples=examples,
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
            if emit_progress:
                print(
                    f"[{len(records)}/{total_pairs}] "
                    f"{case_id}:{pair.get('id')}",
                    flush=True,
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
    all_target_values = _mapping(
        _mapping(pair.get("label")).get("target")
    ).get("vitals", {})
    target_values = {
        vital: all_target_values.get(vital)
        for vital in target_vitals
        if isinstance(all_target_values, dict) and vital in all_target_values
    }
    metadata = _mapping(engine_output.get("metadata"))
    rule_prediction = _mapping(metadata.get("rule_prediction"))
    direction_stage = _mapping(metadata.get("direction_stage"))
    magnitude_stage = _mapping(metadata.get("magnitude_stage"))
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
        "target_vitals_values": target_values,
        "predicted_vitals": _mapping(
            _mapping(engine_output.get("prediction")).get("vitals")
        ),
        "rule_prediction_vitals": _mapping(rule_prediction.get("vitals")),
        "selected_directions": metadata.get("selected_directions", {}),
        "direction_sources": metadata.get("direction_sources", {}),
        "effective_directions": metadata.get("effective_directions", {}),
        "resolved_magnitudes": metadata.get("resolved_magnitudes", {}),
        "magnitude_sources": metadata.get("magnitude_sources", {}),
        "computed_residuals": metadata.get("computed_residuals", {}),
        "few_shot_examples": metadata.get("few_shot_examples", []),
        "direction_llm_directions": direction_stage.get("llm_directions", {}),
        "direction_llm_reasoning": direction_stage.get("llm_reasoning", {}),
        "direction_llm_raw_response": direction_stage.get("llm_raw_response"),
        "direction_llm_parse_error": direction_stage.get("llm_parse_error"),
        "direction_llm_api_error": direction_stage.get("llm_api_error"),
        "magnitude_llm_magnitudes": magnitude_stage.get("llm_magnitudes", {}),
        "magnitude_llm_reasoning": magnitude_stage.get("llm_reasoning", {}),
        "magnitude_llm_raw_response": magnitude_stage.get("llm_raw_response"),
        "magnitude_llm_parse_error": magnitude_stage.get("llm_parse_error"),
        "magnitude_llm_api_error": magnitude_stage.get("llm_api_error"),
        "stage1_prompt_profile": metadata.get("stage1_prompt_profile"),
        "model": direction_stage.get("model"),
    }


def _write_pair_results_csv(
    path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
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
    result = dict(record)
    structured_fields = (
        "target_vitals",
        "missing_predictions",
        "direction_match",
        "before_vitals",
        "target_vitals_values",
        "predicted_vitals",
        "rule_prediction_vitals",
        "selected_directions",
        "direction_sources",
        "effective_directions",
        "resolved_magnitudes",
        "magnitude_sources",
        "computed_residuals",
        "few_shot_examples",
        "direction_llm_directions",
        "direction_llm_reasoning",
        "magnitude_llm_magnitudes",
        "magnitude_llm_reasoning",
    )
    for key in structured_fields:
        result[key] = json.dumps(
            result.get(key),
            ensure_ascii=False,
            sort_keys=True,
        )
    return result


def build_static_selector(example_source: Path) -> DirectionFirstStaticSelector:
    bank = ExampleBank.from_dataset(example_source, recursive=True)
    return DirectionFirstStaticSelector(bank)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the two-stage direction-first hybrid engine."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument(
        "--llm-mode",
        choices=("fake", "real"),
        default="fake",
    )
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=800)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument(
        "--stage1-prompt-profile",
        choices=STAGE1_PROMPT_PROFILES,
        default="original",
        help="Versioned Stage 1 direction prompt; Stage 2 is unchanged.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        choices=(0, 3),
        default=3,
        help="Use zero-shot or the fixed three-example direction-first policy.",
    )
    parser.add_argument(
        "--example-source",
        type=Path,
        default=None,
        help="Example bank path; defaults to input_path.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Suppress per-pair progress messages.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selector = (
        build_static_selector(args.example_source or args.input_path)
        if args.shots == 3
        else None
    )
    run_direction_first_hybrid_eval(
        args.input_path,
        args.output_dir,
        recursive=args.recursive,
        llm_mode=args.llm_mode,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        limit_pairs=args.limit_pairs,
        example_selector=selector,
        stage1_prompt_profile=args.stage1_prompt_profile,
        emit_progress=not args.quiet_progress,
    )
    return 0


def _count_pairs(source_files: Sequence[Path]) -> int:
    return sum(
        len(_load_json(source_file).get("pairs", []))
        for source_file in source_files
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _direction_label(value: str | None) -> str | None:
    return {
        "up": "increase",
        "down": "decrease",
        "stable": "stable",
        None: None,
    }[value]


def _divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
