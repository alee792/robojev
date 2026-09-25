"""E13: a fast LLM prepares the plan's branches, Jev picks among them on a mid-task correction, and low
Jev confidence escalates back to the LLM. Open loop, scored against a hand-written oracle.

The code lives in the e13_branches/ package next to this file; see e13_branches/cli.py for the run
commands and context/07-experiments-to-run.md for the hypothesis and cost.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # common.py and the package

from e13_branches.cli import main  # noqa: E402  (the package directory shadows this file on import)

if __name__ == "__main__":
    main()
