#!/usr/bin/env python
"""Run the controlled v1.3.1 fake-agent scenario demo."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ed_world_model.demo import main


if __name__ == "__main__":
    raise SystemExit(main())
