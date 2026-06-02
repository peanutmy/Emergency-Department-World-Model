"""End-to-end test for the model comparison driver (keyless dry-run)."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transition_engines.run_model_comparison import (
    DEFAULT_SHOT_SETTINGS,
    aggregate_comparison,
    run_model_comparison,
)


DATASET = ROOT / "transitions"


def test_default_shot_settings_are_zero_and_three_only() -> None:
    shots = {shots for shots, _ in DEFAULT_SHOT_SETTINGS}
    assert shots == {0, 3}
    # 0-shot once, 3-shot under each of the two strategies.
    assert DEFAULT_SHOT_SETTINGS == [
        (0, None),
        (3, "static"),
        (3, "kind_hint_matched"),
    ]


def test_matrix_runs_keyless_and_writes_comparison(tmp_path: Path) -> None:
    result = run_model_comparison(
        DATASET,
        tmp_path,
        models=["gpt-4o", "claude-sonnet-4-6"],
        engines=("hybrid", "pure_llm"),
        recursive=True,
        limit_pairs=2,
        api_key="",  # force graceful degrade, no network
    )

    rows = result["rows"]
    # 2 engines x 2 models x 3 settings.
    assert len(rows) == 12
    assert {row["engine"] for row in rows} == {"hybrid", "pure_llm"}
    assert {row["model"] for row in rows} == {"gpt-4o", "claude-sonnet-4-6"}

    settings = {row["setting"] for row in rows}
    assert settings == {"0shot", "3shot-static", "3shot-kind_hint_matched"}

    # Each row carries the standard metrics keys.
    for row in rows:
        assert "direction_accuracy_mean" in row
        assert "normalized_l2_mean" in row
        assert "normalized_l2_median" in row


def test_rule_based_runs_once_as_baseline(tmp_path: Path) -> None:
    result = run_model_comparison(
        DATASET,
        tmp_path,
        models=["gpt-4o"],
        engines=("hybrid",),
        recursive=True,
        limit_pairs=2,
        api_key="",
    )
    baseline = result["rule_based"]
    # A single dict (run once), not a per-model/per-engine list.
    assert isinstance(baseline, dict)
    assert baseline["num_pairs"] > 0
    assert isinstance(baseline["direction_accuracy_mean"], float)


def test_writes_comparison_csv_and_md_and_run_dirs(tmp_path: Path) -> None:
    run_model_comparison(
        DATASET,
        tmp_path,
        models=["gpt-4o"],
        engines=("hybrid",),
        recursive=True,
        limit_pairs=2,
        api_key="",
    )
    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "comparison.md").exists()
    # Per-run output directory + summary file from the underlying runner.
    assert (tmp_path / "hybrid" / "gpt-4o" / "0shot" / "hybrid_summary.json").exists()


def test_aggregate_only_rebuilds_table_from_existing_summaries(tmp_path: Path) -> None:
    # Build a real on-disk tree (keyless) for two models across both engines.
    run_model_comparison(
        DATASET,
        tmp_path,
        models=["gpt-4o", "gpt-5"],
        engines=("hybrid", "pure_llm"),
        recursive=True,
        limit_pairs=2,
        api_key="",
    )
    # Remove the aggregated files to prove aggregation rebuilds them from disk.
    (tmp_path / "comparison.csv").unlink()
    (tmp_path / "comparison.md").unlink()

    result = aggregate_comparison(tmp_path)

    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "comparison.md").exists()
    # All per-run summaries on disk are discovered (2 models x 2 engines x 3).
    assert len(result["rows"]) == 12
    assert {r["model"] for r in result["rows"]} == {"gpt-4o", "gpt-5"}
    assert {r["engine"] for r in result["rows"]} == {"hybrid", "pure_llm"}
    assert {r["setting"] for r in result["rows"]} == {
        "0shot",
        "3shot-static",
        "3shot-kind_hint_matched",
    }
    assert result["rule_based"]["num_pairs"] > 0
