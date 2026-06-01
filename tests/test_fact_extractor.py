from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ed_world_model.facts import (
    FactExtractionContext,
    FactExtractionError,
    FactExtractionResult,
    FakeFactExtractor,
    KnownAllergy,
    KnownHistory,
    KnownMedication,
    KnownSymptom,
    LLMFactExtractor,
    NoopFactExtractor,
)
from ed_world_model.agents.llm_client import FakeLLMClient


def test_known_symptom_supports_required_structured_fields() -> None:
    symptom = KnownSymptom(
        name="shortness of breath",
        status="present",
        onset="sudden onset a few hours ago",
        severity="severe",
        source_texts=["I am very short of breath."],
    )

    assert symptom.model_dump() == {
        "name": "shortness of breath",
        "status": "present",
        "onset": "sudden onset a few hours ago",
        "severity": "severe",
        "source_texts": ["I am very short of breath."],
    }


@pytest.mark.parametrize(
    "field_name",
    ["course", "notes", "ontology", "ontology_id", "synonym", "synonyms"],
)
def test_known_symptom_excludes_disallowed_fields(field_name: str) -> None:
    assert field_name not in KnownSymptom.model_fields
    with pytest.raises(ValueError):
        KnownSymptom(
            name="dyspnea",
            status="present",
            source_texts=["I am short of breath."],
            **{field_name: "not allowed"},
        )


def test_non_symptom_fact_schemas_exclude_notes() -> None:
    assert "notes" not in KnownHistory.model_fields
    assert "notes" not in KnownAllergy.model_fields
    assert "notes" not in KnownMedication.model_fields

    with pytest.raises(ValueError):
        KnownHistory(
            item="asthma",
            status="present",
            source_texts=["I have asthma."],
            notes="not allowed",
        )


def test_fact_extraction_result_serializes_structured_facts() -> None:
    result = FactExtractionResult(
        symptoms=[
            KnownSymptom(
                name="chest pain",
                status="absent",
                source_texts=["I do not have chest pain."],
            )
        ],
        history=[
            KnownHistory(
                item="diabetes",
                status="uncertain",
                source_texts=["I am not sure if he has diabetes."],
            )
        ],
        allergies=[
            KnownAllergy(
                substance=None,
                status="absent",
                source_texts=["No allergies."],
            )
        ],
        medications=[
            KnownMedication(
                name="metformin",
                status="current",
                source_texts=["I take metformin."],
            )
        ],
    )

    dumped = result.model_dump()
    assert dumped["symptoms"][0]["name"] == "chest pain"
    assert dumped["history"][0]["item"] == "diabetes"
    assert dumped["allergies"][0]["substance"] is None
    assert dumped["medications"][0]["status"] == "current"
    assert dumped["ignored"] is False


def test_noop_fact_extractor_returns_ignored_result() -> None:
    result = NoopFactExtractor().extract(
        "I feel short of breath.",
        "patient",
        FactExtractionContext(),
    )

    assert result.ignored is True
    assert result.symptoms == []


def test_fake_fact_extractor_returns_scripted_results_and_records_calls() -> None:
    extractor = FakeFactExtractor(
        [
            {
                "symptoms": [
                    {
                        "name": "shortness of breath",
                        "status": "present",
                        "source_texts": ["I feel short of breath."],
                    }
                ]
            }
        ]
    )
    context = FactExtractionContext(last_question="How are you feeling?")

    result = extractor.extract("I feel short of breath.", "patient", context)

    assert result.symptoms[0].name == "shortness of breath"
    assert extractor.calls == [
        {
            "content": "I feel short of breath.",
            "speaker": "patient",
            "context": context,
        }
    ]


def test_llm_fact_extractor_parses_valid_symptom_present_output() -> None:
    client = FakeLLMClient(
        [
            {
                "symptoms": [
                    {
                        "name": "shortness of breath",
                        "status": "present",
                        "onset": None,
                        "severity": "severe",
                        "source_texts": ["I feel very short of breath."],
                    }
                ],
                "ignored": False,
                "reason": None,
            }
        ]
    )

    result = LLMFactExtractor(client).extract(
        "I feel very short of breath.",
        "patient",
    )

    assert result.symptoms == [
        KnownSymptom(
            name="shortness of breath",
            status="present",
            severity="severe",
            source_texts=["I feel very short of breath."],
        )
    ]
    assert client.prompts


def test_llm_fact_extractor_parses_symptom_absent_output() -> None:
    client = FakeLLMClient(
        [
            {
                "symptoms": [
                    {
                        "name": "chest pain",
                        "status": "absent",
                        "onset": None,
                        "severity": None,
                        "source_texts": ["No chest pain."],
                    }
                ]
            }
        ]
    )

    result = LLMFactExtractor(client).extract("No chest pain.", "patient")

    assert result.symptoms == [
        KnownSymptom(
            name="chest pain",
            status="absent",
            source_texts=["No chest pain."],
        )
    ]


def test_llm_fact_extractor_parses_onset_update_output() -> None:
    client = FakeLLMClient(
        [
            {
                "symptoms": [
                    {
                        "name": "shortness of breath",
                        "status": "present",
                        "onset": "a few hours ago",
                        "severity": None,
                        "source_texts": ["A few hours ago."],
                    }
                ]
            }
        ]
    )
    context = FactExtractionContext(
        last_question="When did the shortness of breath start?",
        known_facts={
            "known_symptoms": [
                {
                    "name": "shortness of breath",
                    "status": "present",
                    "onset": None,
                    "severity": None,
                    "source_texts": ["I am short of breath."],
                }
            ]
        },
    )

    result = LLMFactExtractor(client).extract(
        "A few hours ago.",
        "patient",
        context,
    )

    assert result.symptoms[0].name == "shortness of breath"
    assert result.symptoms[0].onset == "a few hours ago"


def test_llm_fact_extractor_parses_allergy_output() -> None:
    client = FakeLLMClient(
        [
            {
                "allergies": [
                    {
                        "substance": "penicillin",
                        "status": "present",
                        "reaction": "hives",
                        "source_texts": ["I get hives from penicillin."],
                    }
                ]
            }
        ]
    )

    result = LLMFactExtractor(client).extract(
        "I get hives from penicillin.",
        "patient",
    )

    assert result.allergies == [
        KnownAllergy(
            substance="penicillin",
            status="present",
            reaction="hives",
            source_texts=["I get hives from penicillin."],
        )
    ]


def test_llm_fact_extractor_parses_medication_output() -> None:
    client = FakeLLMClient(
        [
            {
                "medications": [
                    {
                        "name": "metformin",
                        "status": "current",
                        "source_texts": ["I take metformin."],
                    }
                ]
            }
        ]
    )

    result = LLMFactExtractor(client).extract("I take metformin.", "patient")

    assert result.medications == [
        KnownMedication(
            name="metformin",
            status="current",
            source_texts=["I take metformin."],
        )
    ]


def test_llm_fact_extractor_returns_ignored_result_from_llm_output() -> None:
    client = FakeLLMClient([{"ignored": True, "reason": "No disclosure."}])

    result = LLMFactExtractor(client).extract("Okay.", "relative")

    assert result.ignored is True
    assert result.reason == "No disclosure."


def test_llm_fact_extractor_ignores_blank_or_unsupported_speaker_without_call() -> None:
    client = FakeLLMClient([{"ignored": False}])
    extractor = LLMFactExtractor(client)

    blank = extractor.extract("   ", "patient")
    unsupported = extractor.extract("The patient has chest pain.", "nurse")

    assert blank.ignored is True
    assert unsupported.ignored is True
    assert client.prompts == []


def test_llm_fact_extractor_rejects_invalid_json_with_specific_error() -> None:
    client = FakeLLMClient(["not json"])

    with pytest.raises(FactExtractionError, match="invalid JSON"):
        LLMFactExtractor(client).extract("I feel short of breath.", "patient")


@pytest.mark.parametrize("field_name", ["notes", "course"])
def test_llm_fact_extractor_rejects_notes_or_course_fields(field_name: str) -> None:
    client = FakeLLMClient(
        [
            {
                "symptoms": [
                    {
                        "name": "shortness of breath",
                        "status": "present",
                        "source_texts": ["I am short of breath."],
                        field_name: "not allowed",
                    }
                ]
            }
        ]
    )

    with pytest.raises(FactExtractionError, match=field_name):
        LLMFactExtractor(client).extract("I am short of breath.", "patient")


@pytest.mark.parametrize(
    "field_name",
    [
        "raw_text",
        "chain_of_thought",
        "internal_reasoning",
        "diagnosis",
        "ontology",
        "synonyms",
    ],
)
def test_llm_fact_extractor_rejects_forbidden_fields(field_name: str) -> None:
    client = FakeLLMClient(
        [
            {
                "history": [
                    {
                        "item": "asthma",
                        "status": "present",
                        "source_texts": ["I have asthma."],
                    }
                ],
                field_name: "not allowed",
            }
        ]
    )

    with pytest.raises(FactExtractionError, match=field_name):
        LLMFactExtractor(client).extract("I have asthma.", "patient")


def test_llm_fact_extractor_uses_fake_client_only_in_tests() -> None:
    client = FakeLLMClient([{"ignored": True, "reason": "No fact."}])

    LLMFactExtractor(client).extract("Okay.", "patient")

    assert client.prompts
    assert client.provider == "fake"


def test_llm_fact_extractor_prompt_contains_required_rules() -> None:
    client = FakeLLMClient([{"ignored": True, "reason": "No fact."}])

    LLMFactExtractor(client).extract("A few hours ago.", "patient")

    prompt = client.prompts[0]
    assert "Extract only facts explicitly stated by the current speaker." in prompt
    assert "Do not infer diagnosis." in prompt
    assert "Preserve negation." in prompt
    assert "Use recent messages, last_question, and existing known_facts only" in prompt
    assert "If the referent is unclear, return ignored=true" in prompt
    assert "Do not use ontology or synonym mapping." in prompt
    assert "Do not include notes or course fields." in prompt
    assert "Return JSON only." in prompt


def test_llm_fact_extractor_prompt_omits_hidden_context_keys() -> None:
    client = FakeLLMClient([{"ignored": True, "reason": "No fact."}])
    context = FactExtractionContext(
        recent_messages=[
            {
                "speaker": "clinician",
                "content": "When did it start?",
                "truth_state": {"secret": "hidden fever"},
            }
        ],
        known_facts={
            "known_symptoms": [],
            "available_results": [
                {"name": "Chest X-ray", "result": "pulmonary edema"}
            ],
            "truth_state": {"secret": "hidden diabetes"},
            "patient_internal_state": {"symptoms": ["hidden cough"]},
        },
        last_question="When did it start?",
    )

    LLMFactExtractor(client).extract("A few hours ago.", "patient", context)

    prompt = client.prompts[0]
    assert "truth_state" not in prompt
    assert "patient_internal_state" not in prompt
    assert "hidden fever" not in prompt
    assert "hidden diabetes" not in prompt
    assert "hidden cough" not in prompt
    assert "available_results" not in prompt
    assert "pulmonary edema" not in prompt
