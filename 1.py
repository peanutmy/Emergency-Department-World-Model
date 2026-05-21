import json
from pathlib import Path
from transition_engines.common import build_engine_input
from transition_engines.hybrid_engine import HybridEngine
from transition_engines.openai_llm_client import OpenAILLMClient
from transition_engines.pure_llm_engine import PureLLMEngine
from transition_engines.rule_based import RuleBasedEngine

case_path = Path("transitions/Cardiology/Pregnant Cardiomyopathy.json")
pair_id = "p2"

case_doc = json.loads(case_path.read_text(encoding="utf-8"))
pair = next(pair for pair in case_doc["pairs"] if pair["id"] == pair_id)

engine_input = build_engine_input(case_doc, pair)
target_vitals = pair["label"]["evaluation"]["target_vitals"]

hybrid = HybridEngine(
      RuleBasedEngine(),
      OpenAILLMClient(
          model="gpt-5.5",
          temperature=0.0,
          max_output_tokens=700,
          response_schema_type="adjustments",
      ),
  )

pure_llm = PureLLMEngine(
      OpenAILLMClient(
          model="gpt-5.5",
          temperature=0.0,
          max_output_tokens=700,
          response_schema_type="vitals",
      ),
  )

# hybrid_output = hybrid.predict(engine_input, target_vital_names=target_vitals)
pure_output = pure_llm.predict(engine_input, target_vital_names=target_vitals)

def only_targets(values):
      return {vital: values.get(vital) for vital in target_vitals}

result = {
      "source_file": str(case_path),
      "target_vitals": target_vitals,
      "before_vitals": only_targets(engine_input["before"].get("vitals", {})),
      "ground_truth_post_vitals": only_targets(pair["label"]["target"].get("vitals", {})),
    #   "hybrid_engine": {
    #       "predicted_vitals": only_targets(hybrid_output["prediction"]["vitals"]),
    #       "adjustments": only_targets(hybrid_output["metadata"].get("llm_adjustments", {})),
    #       "reasoning": only_targets(hybrid_output["metadata"].get("llm_reasoning", {})),
    #       "raw_response": hybrid_output["metadata"].get("llm_raw_response"),
    #       "parse_error": hybrid_output["metadata"].get("llm_parse_error"),
    #       "api_error": hybrid_output["metadata"].get("llm_api_error"),
    #   },
      "pure_llm_engine": {
          "predicted_vitals": only_targets(pure_output["prediction"]["vitals"]),
          "reasoning": only_targets(pure_output["metadata"].get("llm_reasoning", {})),
          "raw_response": pure_output["metadata"].get("llm_raw_response"),
          "parse_error": pure_output["metadata"].get("llm_parse_error"),
          "api_error": pure_output["metadata"].get("llm_api_error"),
      },
  }

print(json.dumps(result, indent=2, ensure_ascii=False))