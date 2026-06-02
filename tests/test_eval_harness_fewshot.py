"""Tests for additive few-shot + factory wiring in the eval runners."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transition_engines import run_hybrid_eval as hybrid_runner
from transition_engines import run_pure_llm_eval as pure_runner


SAMPLE = ROOT / "transitions" / "Cardiology" / "Acute Respiratory Distress.json"


class CapturingClient:
    def __init__(self) -> None:
        self.model = "capturing"
        self.last_api_error = None
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return '{"adjustments": {}, "vitals": {}}'


class StubSelector:
    def select(self, engine_input: dict) -> list[dict]:
        return [{"marker": "WIRED_EXAMPLE_MARKER"}]


def test_hybrid_runner_threads_examples_into_prompt(tmp_path: Path) -> None:
    client = CapturingClient()
    hybrid_runner.run_hybrid_eval(
        SAMPLE,
        tmp_path,
        llm_client=client,
        example_selector=StubSelector(),
        limit_pairs=1,
        emit_console_summary=False,
    )
    assert client.prompts
    assert "WIRED_EXAMPLE_MARKER" in client.prompts[0]


def test_hybrid_runner_zero_shot_has_no_example_block(tmp_path: Path) -> None:
    client = CapturingClient()
    hybrid_runner.run_hybrid_eval(
        SAMPLE,
        tmp_path,
        llm_client=client,
        limit_pairs=1,
        emit_console_summary=False,
    )
    assert client.prompts
    assert "Few-shot examples" not in client.prompts[0]


def test_pure_runner_threads_examples_into_prompt(tmp_path: Path) -> None:
    client = CapturingClient()
    pure_runner.run_pure_llm_eval(
        SAMPLE,
        tmp_path,
        llm_client=client,
        example_selector=StubSelector(),
        limit_pairs=1,
        emit_console_summary=False,
    )
    assert client.prompts
    assert "WIRED_EXAMPLE_MARKER" in client.prompts[0]
