"""
Migrate all transition JSONs from old schema (action: object) to new schema
(actions: [] array + duration_s at pair level).

Old pair:
  { "id": ..., "before": ..., "action": {type, name, ..., duration_s}, "after": ... }

New pair:
  { "id": ..., "before": ..., "actions": [{type, name, ...}], "duration_s": N, "after": ... }

Null actions (type=null) become empty actions array.
"""
import json
from pathlib import Path

ROOT = Path(r"D:\wmed\emsim\transitions")


def migrate_pair(pair: dict) -> dict:
    old_action = pair.pop("action")
    duration = old_action.pop("duration_s", 0)

    if old_action.get("type") is None:
        new_actions = []
    else:
        # Strip duration_s from the action object if it snuck in
        old_action.pop("duration_s", None)
        new_actions = [old_action]

    pair["actions"] = new_actions
    pair["duration_s"] = duration
    # Reorder: id, comment, before, actions, duration_s, after
    ordered = {}
    for k in ("id", "comment", "before", "actions", "duration_s", "after"):
        if k in pair:
            ordered[k] = pair[k]
    for k, v in pair.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def migrate_file(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    new_pairs = [migrate_pair(p) for p in data.get("pairs", [])]
    data["pairs"] = new_pairs
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    files = [p for p in ROOT.rglob("*.json") if p.name != "schema.json"]
    for f in files:
        try:
            migrate_file(f)
        except Exception as e:
            print(f"FAIL {f}: {e}")
    print(f"migrated {len(files)} files")


if __name__ == "__main__":
    main()
