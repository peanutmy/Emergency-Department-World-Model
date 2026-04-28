"""
EMSim Physiology Engine — pure LLM backend (zero-shot).

Implements the PhysiologyEngine protocol (emsim_engine.py) using a chat LLM
as the transition function. The LLM returns a delta JSON over `before`
which the engine merges into a schema-valid `after`.

Usage:
    python emsim_eval.py --engine emsim_llm_engine:LLMEngine --export llm.csv

Environment:
    OPENAI_BASE_URL  proxy / direct OpenAI endpoint
    OPENAI_API_KEY   API key

Cache: responses are keyed by a sha256 of (model, before, actions, duration_s)
and stored under .cache/llm/. Re-runs are free.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from copy import deepcopy
from typing import Any

from openai import OpenAI


State = dict[str, Any]
Action = dict[str, Any]


# ----------------------------------------------------------------------
# Paths / defaults
# ----------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_PROJECT_ROOT, ".cache", "llm")
os.makedirs(CACHE_DIR, exist_ok=True)

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_BASE_URL = "http://148.113.224.153:3000/v1"
_DEFAULT_API_KEY = "sk-TNKdCnAgOrhx4b8Y3KgAiDYW5zZbA1pkRrmbUD4BnDyUtqxu"

# Bump when SYSTEM_PROMPT changes so prior cache entries (built against the
# old prompt) are not returned. v2 added optional `mechanism_rhythm` output.
_PROMPT_VERSION = "v2"


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """You are an emergency-department physiology engine. Given a patient's before-state, a list of clinical actions taken at t=0, and an observation window duration_s, predict the after-state as a JSON delta over before.

Rules:
1. Output ONLY valid JSON, no code fences, no prose outside JSON.
2. Use the exact schema below.
3. vitals_delta: only include vitals that change. Values are absolute deltas (e.g. {"HR": 15} means HR rises by 15; negative means falls).
4. interventions_changed: only include intervention fields whose value changes from before. Use null for fields set to null. Allowed numeric/categorical fields: airway (bool), O2_device (null|"nasal"|"NRB"|"BVM"|"HFNC"|"BiPAP"|"vent"), PEEP (number), FiO2 (number 0..1), vent_rate (number|null), vent_TV_ml (number|null), intubated (bool), CPR_active (bool), defib_last_J (number|null), pacing_active (bool), pacing_rate (number|null), fluids_rate_ml_hr (number), fluid_type (null|"crystalloid"|"colloid"|"blood"), warming_active (bool), cooling_active (bool), needle_decompression (bool), chest_tube (bool), pericardiocentesis (bool).
5. mechanism_severity: only include if severity changes (mild/moderate/severe). Omit the field entirely if unchanged. Do NOT change pathology.name.
6. mechanism_rhythm: only include if cardiac rhythm changes from before. Allowed: "sinus", "SVT", "VT", "VF", "asystole", "PEA", "bradycardia". Most pairs DO NOT change rhythm — omit this field by default. Change rhythm only when there is a clear clinical trigger:
   - Successful defibrillation on VF/pulseless VT → may convert to sinus (ROSC).
   - Failed defibrillation, severe deterioration, prolonged arrest → asystole.
   - New severe bradyarrhythmia (beta-blocker / CCB / digoxin tox; high-grade AV block) → bradycardia.
   - Severe hyperkalemia, exsanguinating hemorrhage, massive PE → PEA.
   - Decompensating ischemia / electrical storm → VT or VF.
   - Adenosine on SVT, vagal maneuver → may convert to sinus.
   When rhythm did not change, OMIT this field rather than echoing the before-rhythm.
7. Be clinically reasonable:
   - apply_NRB / apply_nasal / apply_BVM / intubate: raise O2Sat toward 98-100 unless pathology is severe shunt; set O2_device and FiO2 accordingly.
   - start_CPR during arrest: HR stays 0 (monitor shows no pulse); set CPR_active true.
   - defibrillate on VF/pulseless VT: may convert rhythm but monitor HR stays 0 until ROSC.
   - Vasopressors (epinephrine, norepinephrine): raise BP and HR in shock states.
   - Atropine: raise HR in bradycardia; little effect if HR already normal.
   - Sedatives (propofol, midazolam) + intubate: drop BP modestly; no effect on conscious monitor vitals beyond that.
   - Pure-wait (empty actions) pairs: vitals usually drift only slightly unless pathology is severe and uncompensated.
8. Keep vitals in physiologic range: HR 0-250, BP_sys 0-250, BP_dia 0-150, RR 0-60, O2Sat 0-100, T 30-42.
9. Drug actions (type="drug" in the actions list) already capture what was administered — do not echo them anywhere in the output; just let their physiologic effects show up via vitals_delta / interventions_changed.

Output schema (JSON only, no prose):
{
  "vitals_delta": { "HR"?: number, "BP_sys"?: number, "BP_dia"?: number, "RR"?: number, "O2Sat"?: number, "T"?: number },
  "interventions_changed": { <subset of intervention fields>: <value> },
  "mechanism_severity"?: "mild"|"moderate"|"severe",
  "mechanism_rhythm"?: "sinus"|"SVT"|"VT"|"VF"|"asystole"|"PEA"|"bradycardia",
  "reasoning": "<one sentence>"
}"""

USER_TEMPLATE = """## Before state
{before_json}

## Actions taken at t=0
{actions_json}

## Observation window
{duration_s} seconds

Predict the after state as a delta JSON."""


_ALLOWED_INTERVENTION_KEYS = {
    "airway", "O2_device", "PEEP", "FiO2",
    "vent_rate", "vent_TV_ml", "intubated",
    "CPR_active", "defib_last_J",
    "pacing_active", "pacing_rate",
    "fluids_rate_ml_hr", "fluid_type",
    "warming_active", "cooling_active",
    "needle_decompression", "chest_tube", "pericardiocentesis",
}
_VITAL_KEYS = {"HR", "BP_sys", "BP_dia", "RR", "O2Sat", "T"}
_RHYTHM_VALUES = {"sinus", "SVT", "VT", "VF", "asystole", "PEA", "bradycardia"}
_VITAL_CLIP = {
    "HR": (0.0, 250.0),
    "BP_sys": (0.0, 250.0),
    "BP_dia": (0.0, 150.0),
    "RR": (0.0, 60.0),
    "O2Sat": (0.0, 100.0),
    "T": (25.0, 43.0),
}


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------
class LLMEngine:
    """Pure LLM physiology engine. zero-shot; output cached to disk."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        cache_enabled: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model or os.environ.get("EMSIM_LLM_MODEL", _DEFAULT_MODEL)
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled

        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", _DEFAULT_API_KEY),
        )

        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.tokens_in = 0
        self.tokens_out = 0

    # ---- public step() ----
    def step(self, before: State, actions: list[Action], duration_s: float) -> State:
        key = self._cache_key(before, actions, duration_s)

        delta: dict[str, Any] | None = None
        if self.cache_enabled:
            delta = self._load_cache(key)
            if delta is not None:
                with self._lock:
                    self.cache_hits += 1

        if delta is None:
            delta = self._call_llm_with_retry(before, actions, duration_s)
            if self.cache_enabled and not delta.get("__failed__"):
                self._save_cache(key, delta)

        return self._apply_delta(before, delta, actions, duration_s)

    # ---- caching ----
    def _cache_key(self, before: State, actions, duration_s) -> str:
        payload = json.dumps(
            {"m": self.model, "v": _PROMPT_VERSION,
             "b": before, "a": actions, "d": duration_s},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, key: str) -> str:
        return os.path.join(CACHE_DIR, f"{key}.json")

    def _load_cache(self, key: str):
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, key: str, delta: dict) -> None:
        path = self._cache_path(key)
        tmp = path + f".tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(delta, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    # ---- LLM call ----
    def _call_llm_with_retry(self, before: State, actions, duration_s) -> dict:
        user = USER_TEMPLATE.format(
            before_json=json.dumps(before, ensure_ascii=False),
            actions_json=json.dumps(actions, ensure_ascii=False),
            duration_s=duration_s,
        )
        last_err: str | None = None
        for attempt in range(self.max_retries):
            try:
                with self._lock:
                    self.calls += 1
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or "{}"
                delta = json.loads(raw)
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    with self._lock:
                        self.tokens_in += getattr(usage, "prompt_tokens", 0) or 0
                        self.tokens_out += getattr(usage, "completion_tokens", 0) or 0
                return delta
            except json.JSONDecodeError as e:
                last_err = f"JSONDecodeError: {e}"
                if attempt == self.max_retries - 1:
                    break
                time.sleep(0.5 + random.random())
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt == self.max_retries - 1:
                    break
                time.sleep(min(2 ** attempt, 8) + random.random())
        with self._lock:
            self.failures += 1
        return {
            "vitals_delta": {},
            "interventions_changed": {},
            "reasoning": f"LLM call failed: {last_err}",
            "__failed__": True,
        }

    # ---- merge ----
    def _apply_delta(
        self,
        before: State,
        delta: dict,
        actions: list[Action],
        duration_s: float,
    ) -> State:
        after = deepcopy(before)

        # Vitals (absolute deltas, clipped to physiologic range).
        vd = delta.get("vitals_delta") or {}
        if isinstance(vd, dict):
            for k, v in vd.items():
                if k not in _VITAL_KEYS:
                    continue
                if not isinstance(v, (int, float)):
                    continue
                new_val = float(after["vitals"][k]) + float(v)
                lo, hi = _VITAL_CLIP[k]
                after["vitals"][k] = max(lo, min(hi, new_val))

        # Interventions (overwrite to provided value).
        ic = delta.get("interventions_changed") or {}
        if isinstance(ic, dict):
            for k, v in ic.items():
                if k in _ALLOWED_INTERVENTION_KEYS and k in after["interventions"]:
                    after["interventions"][k] = v

        # Drugs are no longer part of state. They appear only in pair.actions[]
        # and are consumed by the engine's physiology path — nothing to merge here.

        # Mechanism severity (never rename pathology).
        sev = delta.get("mechanism_severity")
        if isinstance(sev, str) and sev in ("mild", "moderate", "severe"):
            after["mechanism"]["pathology"]["severity"] = sev

        # Mechanism rhythm (optional; defaults to before via deepcopy).
        rhy = delta.get("mechanism_rhythm")
        if isinstance(rhy, str) and rhy in _RHYTHM_VALUES:
            after["mechanism"]["rhythm"] = rhy

        return after

    # ---- diagnostics ----
    def stats(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "cache_hits": self.cache_hits,
                "failures": self.failures,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "model": self.model,
            }
