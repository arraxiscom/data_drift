#!/usr/bin/env python3
"""CLI wrapper for story figure export."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from drift_lab.export_figures import main  # noqa: E402

if __name__ == "__main__":
    main()
