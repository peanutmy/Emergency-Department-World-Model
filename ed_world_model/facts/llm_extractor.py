"""LLM-backed extraction for explicit patient/relative disclosures."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import re
from typing import Any

from pydantic import ValidationError

from ed_world_model.facts.extractor import (
    FactExtractionContext,
    FactExtractionError,
)
from ed_world_model.facts.schemas import FactExtractionResult


ALLOWED_FACT_SPEAKERS = {"patient", "relative"}
DEFAULT_MAX_RECENT_MESSAGES = 10

_KNOWN_FACT_KEYS = (
    "chief_complaint",
    "known_history",
    "known_allergies",
    "known_medications",
    "known_symptoms",
)
_MESSAGE_KEYS = ("speaker", "recipient", "content", "turn_index")
_FORBIDDEN_OUTPUT_KEYS = {
    "raw_text",
    "chain_of_thought",
    "internal_reasoning",
    "notes",
    "course",
    "ontology",
    "ontology_id",
    "synonym",
    "synonyms",
    "diagnosis",
    "diagnoses",
    "differential",
    "differential_diagnosis",
}


class LLMFactExtractor:
    """Extract structured known facts with an explicit LLM client."""

    def __init__(
        self,
        llm_client: Any,
        *,
        model: str | None = None,
        config: Mapping[str, Any] | None = None,
        max_recent_messages: int = DEFAULT_MAX_RECENT_MESSAGES,
    ) -> None:
        if max_recent_messages < 0:
            raise ValueError("max_recent_messages must be non-negative.")
        self.llm_client = llm_client
        self.model = model
        self.config = dict(config or {})
        self.max_recent_messages = max_recent_messages

    def extract(
        self,
        content: str,
        speaker: str,
        context: FactExtractionContext | None = None,
    ) -> FactExtractionResult:
        """Extract explicitly spoken patient/relative facts from one utterance."""

        normalized_content = content.strip()
        if speaker not in ALLOWED_FACT_SPEAKERS:
            return FactExtractionResult(
                ignored=True,
                reason="Fact extraction only supports patient or relative speakers.",
            )
        if normalized_content == "":
            return FactExtractionResult(
                ignored=True,
                reason="Blank utterance.",
            )

        prompt = self.build_prompt(
            content=normalized_content,
            speaker=speaker,
            context=context or FactExtractionContext(),
        )
        response_text = self._generate(prompt)
        return parse_fact_extraction_response(response_text)

    def build_prompt(
        self,
        *,
        content: str,
        speaker: str,
        context: FactExtractionContext,
    ) -> str:
        """Build the strict extraction prompt for tests and runtime use."""

        safe_context = {
            "speaker": speaker,
            "utterance": content,
            "last_question": context.last_question,
            "recent_messages": _safe_recent_messages(
                context.recent_messages,
                max_messages=self.max_recent_messages,
            ),
            "existing_known_facts": _safe_known_facts(context.known_facts),
        }
        context_json = json.dumps(safe_context, indent=2, sort_keys=True)
        schema_json = json.dumps(_output_schema_template(), indent=2, sort_keys=True)

        return (
            "You are mapping a patient/relative utterance into structured known "
            "facts.\n\n"
            "Rules:\n"
            "- Extract only facts explicitly stated by the current speaker.\n"
            "- Do not infer diagnosis.\n"
            "- Do not infer hidden facts.\n"
            "- Do not use medical knowledge to add plausible symptoms.\n"
            "- Do not copy facts from scenario hidden truth.\n"
            "- Do not extract from clinician/nurse messages.\n"
            "- Do not treat clinician/nurse assumptions as patient facts.\n"
            "- Preserve negation.\n"
            '- If the speaker denies a symptom, output that symptom with '
            'status=\"absent\".\n'
            '- If the speaker says they are unsure, use status=\"uncertain\".\n'
            "- If no extractable fact is present, return ignored=true with a "
            "short reason.\n"
            "- The current utterance is the grounding source for extracted "
            "facts.\n"
            "- Use recent messages, last_question, and existing known_facts only "
            "to resolve references such as it, that, or a time-only answer.\n"
            "- Example: if the clinician asked when shortness of breath started, "
            "the patient says \"A few hours ago.\", and existing known_facts has "
            "shortness of breath, update symptom name=\"shortness of breath\" "
            "with onset=\"a few hours ago\".\n"
            "- If the referent is unclear, return ignored=true or empty facts.\n"
            '- Do not create \"unknown symptom\".\n'
            "- Do not use ontology or synonym mapping.\n"
            "- Do not merge SOB and shortness of breath unless the utterance or "
            "existing context clearly provides the same name.\n"
            "- Do not include notes or course fields.\n"
            "- Do not include ontology fields, synonym fields, raw_text, "
            "chain_of_thought, internal_reasoning, diagnosis, or differential "
            "diagnosis.\n"
            "- Return JSON only.\n\n"
            "Output JSON schema:\n"
            f"{schema_json}\n\n"
            "Safe context supplied to you:\n"
            f"{context_json}\n"
        )

    def _generate(self, prompt: str) -> str:
        try:
            if hasattr(self.llm_client, "generate"):
                response = self.llm_client.generate(prompt)
            elif callable(self.llm_client):
                response = self.llm_client(prompt)
            else:
                raise FactExtractionError(
                    "llm_client must provide generate(prompt) or be callable."
                )
        except FactExtractionError as exc:
            redacted_message = _redact_sensitive_text(str(exc))
            if redacted_message == str(exc):
                raise
            raise FactExtractionError(redacted_message) from exc
        except Exception as exc:
            raise FactExtractionError(
                "LLM fact extraction request failed: "
                f"{type(exc).__name__}: {_redact_sensitive_text(str(exc))}"
            ) from exc

        if not isinstance(response, str):
            raise FactExtractionError(
                "LLM fact extraction response must be a JSON string."
            )
        return response


def parse_fact_extraction_response(response_text: str) -> FactExtractionResult:
    """Parse and validate strict JSON from the LLM fact extractor."""

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise FactExtractionError("LLM fact extraction returned invalid JSON.") from exc

    if not isinstance(data, Mapping):
        raise FactExtractionError(
            "LLM fact extraction output must be a JSON object."
        )

    forbidden_path = _find_forbidden_key(data)
    if forbidden_path is not None:
        raise FactExtractionError(
            f"LLM fact extraction output contains forbidden key: {forbidden_path}"
        )

    try:
        return FactExtractionResult.model_validate(data)
    except ValidationError as exc:
        raise FactExtractionError(
            f"LLM fact extraction output did not match schema: {exc}"
        ) from exc


def _safe_recent_messages(
    messages: Sequence[Any],
    *,
    max_messages: int,
) -> list[dict[str, Any]]:
    if max_messages == 0:
        return []
    recent = list(messages)[-max_messages:]
    safe_messages: list[dict[str, Any]] = []
    for message in recent:
        data = _to_mapping(message)
        if data is None:
            continue
        safe_messages.append(
            {
                key: deepcopy(data.get(key))
                for key in _MESSAGE_KEYS
                if key in data
            }
        )
    return safe_messages


def _safe_known_facts(known_facts: Any) -> dict[str, Any]:
    data = _to_mapping(known_facts)
    if data is None:
        return {}
    return {
        key: deepcopy(data.get(key))
        for key in _KNOWN_FACT_KEYS
        if key in data
    }


def _to_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, Mapping):
        return value
    return None


def _find_forbidden_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            current_path = f"{path}.{key}"
            if str(key) in _FORBIDDEN_OUTPUT_KEYS:
                return current_path
            nested = _find_forbidden_key(item, current_path)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _find_forbidden_key(item, f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def _redact_sensitive_text(message: str) -> str:
    return re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "[REDACTED_API_KEY]", message)


def _output_schema_template() -> dict[str, Any]:
    return {
        "symptoms": [
            {
                "name": "...",
                "status": "present | absent | uncertain",
                "onset": None,
                "severity": None,
                "source_texts": ["..."],
            }
        ],
        "history": [
            {
                "item": "...",
                "status": "present | absent | uncertain",
                "source_texts": ["..."],
            }
        ],
        "allergies": [
            {
                "substance": None,
                "status": "present | absent | uncertain",
                "reaction": None,
                "source_texts": ["..."],
            }
        ],
        "medications": [
            {
                "name": None,
                "status": "current | not_taking | unknown",
                "source_texts": ["..."],
            }
        ],
        "ignored": False,
        "reason": None,
    }


__all__ = [
    "LLMFactExtractor",
    "parse_fact_extraction_response",
]
