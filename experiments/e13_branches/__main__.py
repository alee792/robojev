import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # common.py

from e13_branches.cli import main  # noqa: E402

main()
