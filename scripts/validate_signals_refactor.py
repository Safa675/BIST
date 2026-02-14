#!/usr/bin/env python3
"""
Wrapper entrypoint for signal refactor parity validation.

Delegates to Models/validate_signal_outputs.py to preserve the existing
comparison logic while providing a stable script path for Phase 4 checks.
"""

from __future__ import annotations

import logging
import runpy
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    models_script = project_root / "Models" / "validate_signal_outputs.py"

    if not models_script.exists():
        logger.info(f"Missing validator script: {models_script}")
        return 1

    sys.argv[0] = str(models_script)
    runpy.run_path(str(models_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
