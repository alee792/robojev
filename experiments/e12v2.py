"""E12 update (e12v2): the v2 design (docs/v2.md) as a text-only closed-loop episode simulator, with
controls. The code lives in the e12v2/ package next to this file (core/ = design logic, sim/ = text
world, eval/ = scenarios, oracle, controls, metrics, CLI); see e12v2/eval/cli.py for run commands and
context/07-experiments-to-run.md for the hypothesis, pass criteria and cost.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # common.py, e12_blocksworld, e13_branches, the package

from e12v2.eval.cli import main  # noqa: E402  (the package directory shadows this file on import)

if __name__ == "__main__":
    main()
