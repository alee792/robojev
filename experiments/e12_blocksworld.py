"""E12: text-only blocks world, closed loop over v2's fast layers (Sequencer, Spotter, Listener).

The code lives in the e12_blocksworld/ package next to this file; see e12_blocksworld/cli.py for the
run commands and context/07-experiments-to-run.md for the hypothesis and cost.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # common.py, e11_reorder.py, the package

from e12_blocksworld.cli import main  # noqa: E402  (the package directory shadows this file on import)

if __name__ == "__main__":
    main()
