#!/usr/bin/env python
"""Run the emotion engine as a judge over augmented eval items and score it.

For each item in ``augmented_drafts.jsonl`` the engine predicts the
listener's NEXT emotion state from scenario_summary + listener traits +
current_state + the single augmented utterance (no dialogue history). Use
``--no-scenario`` and/or ``--no-traits`` for no-context ablations.

Predictions are written to ``engine_predictions_<model>.json`` inside the
input dir. If ``human_labels_merged.json`` exists, metrics are computed
against the human majority-vote gold and against the generation-time draft
labels.

Example:

    env OPENAI_API_KEY=... \
    python tools/evaluate_emotion_engine.py \
      --provider openai \
      --model gpt-5.4 \
      --input-dir eval_data/emotion_aug_v3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_utterances as gu
from emotion_engine import (
    EMOTION_LEVELS,
    EMOTION_NAMES,
    EmotionState,
    EmotionTrajectory,
    PROMPT_STRATEGIES,
    step_emotion,
)


LEVEL_INDEX = {level: i for i, level in enumerate(EMOTION_LEVELS)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", value).strip("_")


def predict(
    drafts: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    attempts: int,
    include_scenario: bool = True,
    include_traits: bool = True,
    prompt_strategy: str | None = None,
) -> list[dict[str, Any]]:
    llm = gu._make_json_call(provider, model)
    predictions: list[dict[str, Any]] = []
    for i, draft in enumerate(drafts, start=1):
        context_parts = []
        if include_scenario:
            context_parts.append(str(draft["scenario_summary"]))
        if include_traits:
            context_parts.append(
                "Listener traits: "
                + json.dumps(draft.get("traits") or {}, ensure_ascii=False)
            )
        context = "\n".join(context_parts)
        trajectory = EmotionTrajectory(
            agent_role=draft["listener"],
            state=EmotionState(**draft["current_state"]),
        )
        record: dict[str, Any] = {
            "idx": i,
            "item_id": draft.get("item_id"),
            "target_emotion": draft.get("target_emotion"),
            "target_level": draft.get("target_level"),
            "current_state": draft["current_state"],
            "augmented_utterance": draft["augmented_utterance"],
            "prompt_strategy": prompt_strategy or "zero_shot",
        }
        error: str | None = None
        for attempt in range(attempts):
            try:
                out = step_emotion(
                    trajectory,
                    utterance=draft["augmented_utterance"],
                    speaker_role=draft["speaker"],
                    clinical_context=context,
                    prompt_strategy=prompt_strategy,
                    llm_call=llm,
                )
                record["predicted_next_state"] = out.state.summary()
                record["evidence"] = out.last_evidence
                record["reasoning"] = out.last_reasoning
                error = None
                break
            except Exception as exc:  # noqa: BLE001
                error = str(exc)[:200]
                print(f"[predict] item {i} attempt {attempt + 1} failed: {error}")
        if error is not None:
            record["error"] = error
        predictions.append(record)
        state = record.get("predicted_next_state")
        print(f"[predict] {i}/{len(drafts)} {json.dumps(state)}")
    return predictions


def _gold_map(merged: list[dict[str, Any]]) -> dict[str, dict[str, str | None]]:
    """Map augmented_utterance -> {emotion: gold or None}."""
    return {
        cell["augmented_utterance"]: {
            emotion: cell["labels"][emotion]["gold"] for emotion in EMOTION_NAMES
        }
        for cell in merged
    }


def _quadratic_weighted_kappa(pairs: list[tuple[int, int]], n_classes: int) -> float:
    if not pairs:
        return float("nan")
    observed = [[0.0] * n_classes for _ in range(n_classes)]
    for a, b in pairs:
        observed[a][b] += 1
    total = len(pairs)
    row = [sum(observed[i]) for i in range(n_classes)]
    col = [sum(observed[i][j] for i in range(n_classes)) for j in range(n_classes)]
    num = 0.0
    den = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            weight = ((i - j) ** 2) / ((n_classes - 1) ** 2)
            expected = row[i] * col[j] / total
            num += weight * observed[i][j]
            den += weight * expected
    return 1.0 - (num / den) if den else float("nan")


def score(
    predictions: list[dict[str, Any]],
    reference: dict[str, dict[str, str | None]],
    *,
    name: str,
) -> dict[str, Any]:
    per_emotion: dict[str, dict[str, Any]] = {}
    all_pairs: list[tuple[int, int]] = []
    change_tp = change_fp = change_fn = 0
    direction_correct = direction_total = 0
    target_hits = target_total = 0
    side_hits = side_total = 0

    for pred in predictions:
        predicted = pred.get("predicted_next_state")
        ref = reference.get(pred["augmented_utterance"])
        if not predicted or not ref:
            continue
        current = pred["current_state"]
        for emotion in EMOTION_NAMES:
            gold = ref.get(emotion)
            if gold is None:
                continue  # unresolved adjudication tie
            got = predicted[emotion]
            gi, pi, ci = LEVEL_INDEX[gold], LEVEL_INDEX[got], LEVEL_INDEX[current[emotion]]
            stats = per_emotion.setdefault(
                emotion, {"n": 0, "exact": 0, "abs_err": 0}
            )
            stats["n"] += 1
            stats["exact"] += int(pi == gi)
            stats["abs_err"] += abs(pi - gi)
            all_pairs.append((gi, pi))

            gold_changed = gi != ci
            pred_changed = pi != ci
            if gold_changed and pred_changed:
                change_tp += 1
                direction_total += 1
                direction_correct += int((gi > ci) == (pi > ci))
            elif pred_changed:
                change_fp += 1
            elif gold_changed:
                change_fn += 1

            if emotion == pred.get("target_emotion"):
                target_total += 1
                target_hits += int(pi == gi)
            else:
                side_total += 1
                side_hits += int(pi == gi)

    n = sum(s["n"] for s in per_emotion.values())
    exact = sum(s["exact"] for s in per_emotion.values())
    abs_err = sum(s["abs_err"] for s in per_emotion.values())
    precision = change_tp / (change_tp + change_fp) if (change_tp + change_fp) else float("nan")
    recall = change_tp / (change_tp + change_fn) if (change_tp + change_fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall)
        else float("nan")
    )

    report = {
        "comparison": name,
        "labels_scored": n,
        "exact_accuracy": exact / n if n else float("nan"),
        "mae": abs_err / n if n else float("nan"),
        "quadratic_weighted_kappa": _quadratic_weighted_kappa(
            all_pairs, len(EMOTION_LEVELS)
        ),
        "change_detection": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "direction_accuracy": (
                direction_correct / direction_total if direction_total else float("nan")
            ),
        },
        "target_dimension_accuracy": target_hits / target_total if target_total else float("nan"),
        "side_dimension_accuracy": side_hits / side_total if side_total else float("nan"),
        "per_emotion": {
            emotion: {
                "n": s["n"],
                "exact_accuracy": s["exact"] / s["n"],
                "mae": s["abs_err"] / s["n"],
            }
            for emotion, s in per_emotion.items()
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-dir", default="eval_data/emotion_aug_v3")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--no-scenario",
        action="store_true",
        help="Ablation: withhold scenario_summary from the engine.",
    )
    parser.add_argument(
        "--no-traits",
        action="store_true",
        help="Ablation: withhold listener traits from the engine.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip prediction; score an existing predictions file.",
    )
    parser.add_argument(
        "--prompt-strategy",
        choices=PROMPT_STRATEGIES,
        default="zero_shot",
        help=(
            "Emotion prompt variant: zero_shot is the deployed default; "
            "few_shot adds calibration examples; react adds a single-call "
            "ReAct-style internal review."
        ),
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    drafts = _read_jsonl(input_dir / "augmented_drafts.jsonl")
    variant = ""
    if args.no_scenario:
        variant += "_noscenario"
    if args.no_traits:
        variant += "_notraits"
    if args.prompt_strategy != "zero_shot":
        variant += f"_{_safe_tag(args.prompt_strategy)}"
    predictions_path = (
        input_dir / f"engine_predictions_{_safe_tag(args.model)}{variant}.json"
    )

    if args.metrics_only:
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    else:
        predictions = predict(
            drafts,
            provider=args.provider,
            model=args.model,
            attempts=args.attempts,
            include_scenario=not args.no_scenario,
            include_traits=not args.no_traits,
            prompt_strategy=args.prompt_strategy,
        )
        predictions_path.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[write] predictions -> {predictions_path}")

    reports = []

    merged_path = input_dir / "human_labels_merged.json"
    if merged_path.exists():
        gold = _gold_map(json.loads(merged_path.read_text(encoding="utf-8")))
        reports.append(score(predictions, gold, name="engine_vs_human_gold"))
    else:
        print("[metrics] no human_labels_merged.json - skipping human comparison")

    draft_reference = {
        d["augmented_utterance"]: dict(d.get("draft_label_next_state") or {})
        for d in drafts
        if d.get("draft_label_next_state")
    }
    reports.append(score(predictions, draft_reference, name="engine_vs_draft_labels"))

    for report in reports:
        report["variant"] = variant or "_full"

    metrics_path = input_dir / f"engine_metrics_{_safe_tag(args.model)}{variant}.json"
    metrics_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[write] metrics -> {metrics_path}")
    for report in reports:
        print(f"\n=== {report['comparison']} ({report['labels_scored']} labels) ===")
        print(f"  exact accuracy : {report['exact_accuracy']:.1%}")
        print(f"  MAE            : {report['mae']:.3f}")
        print(f"  QWK            : {report['quadratic_weighted_kappa']:.3f}")
        cd = report["change_detection"]
        print(
            f"  change P/R/F1  : {cd['precision']:.2f}/{cd['recall']:.2f}/{cd['f1']:.2f}"
            f"  direction {cd['direction_accuracy']:.1%}"
        )
        print(
            f"  target-dim acc : {report['target_dimension_accuracy']:.1%}"
            f"   side-dim acc: {report['side_dimension_accuracy']:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
