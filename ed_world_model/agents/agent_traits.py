"""Load agent trait definitions and sample random traits/levels per role.

Trait definition files live in ``ed_world_model/agents/agentTrait/`` as JSON,
one file per role. Each file holds a single top-level key (``<role>_traits``)
mapping to a list of trait objects shaped like::

    {
        "trait": "Empathy",
        "definition": "...",
        "trigger": {"high": [...], "low": [...]}
    }

The ``trigger`` keys are the available *levels* for that trait. Some traits only
define a single level (e.g. patient "Anger" is "high" only), so a sampled level
is always drawn from the levels that trait actually defines.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import random
from typing import Any

# Default location of the per-role trait JSON files.
DEFAULT_TRAITS_DIR = Path(__file__).resolve().parent / "agentTrait"

# Maps an AgentProfiles role to (filename, top-level JSON key). Only the patient
# and relative are emotional agents that carry traits; the clinician and nurse
# exist only to drive the simulation and have no trait files.
ROLE_TRAIT_FILES: dict[str, tuple[str, str]] = {
    "patient": ("PatientTrait.json", "patient_traits"),
    "relative": ("relativeTraits.json", "relative_traits"),
}


class TraitDefinition:
    """A single trait with its definition and the levels it defines."""

    def __init__(
        self,
        name: str,
        definition: str,
        levels: Sequence[str],
    ) -> None:
        self.name = name
        self.definition = definition
        # Preserve declaration order while dropping duplicates.
        self.levels: tuple[str, ...] = tuple(dict.fromkeys(levels))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"TraitDefinition(name={self.name!r}, levels={self.levels!r})"


class AgentTraitLibrary:
    """Loads role trait definitions and samples random trait/level sets."""

    def __init__(self, traits_dir: str | Path | None = None) -> None:
        self.traits_dir = Path(traits_dir) if traits_dir is not None else DEFAULT_TRAITS_DIR
        self._by_role: dict[str, list[TraitDefinition]] = {}

    def traits_for(self, role: str) -> list[TraitDefinition]:
        """Return the (cached) trait definitions available for ``role``."""
        if role not in self._by_role:
            self._by_role[role] = self._load_role(role)
        return self._by_role[role]

    def _load_role(self, role: str) -> list[TraitDefinition]:
        spec = ROLE_TRAIT_FILES.get(role)
        if spec is None:
            return []
        filename, top_key = spec
        path = self.traits_dir / filename
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        entries = data.get(top_key, []) if isinstance(data, Mapping) else []
        return [
            definition
            for entry in entries
            if (definition := self._parse_entry(entry)) is not None
        ]

    @staticmethod
    def _parse_entry(entry: Any) -> TraitDefinition | None:
        if not isinstance(entry, Mapping):
            return None
        name = entry.get("trait")
        if not name:
            return None
        trigger = entry.get("trigger")
        levels = list(trigger.keys()) if isinstance(trigger, Mapping) else []
        if not levels:
            return None
        return TraitDefinition(
            name=str(name),
            definition=str(entry.get("definition", "")),
            levels=levels,
        )

    def sample(
        self,
        role: str,
        *,
        rng: random.Random,
        min_traits: int = 1,
        max_traits: int | None = None,
    ) -> dict[str, str]:
        """Randomly assign 1..N distinct traits, each with a random level.

        Returns a mapping of ``trait name -> chosen level`` suitable for the
        ``AgentProfile.traits`` dict. Empty if the role defines no traits.
        """
        available = self.traits_for(role)
        if not available:
            return {}

        upper = len(available) if max_traits is None else min(max_traits, len(available))
        lower = max(1, min(min_traits, upper))
        count = rng.randint(lower, upper)

        chosen = rng.sample(available, count)
        return {
            trait.name: rng.choice(trait.levels)
            for trait in chosen
        }


__all__ = [
    "DEFAULT_TRAITS_DIR",
    "ROLE_TRAIT_FILES",
    "AgentTraitLibrary",
    "TraitDefinition",
]
