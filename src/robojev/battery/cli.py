"""robojev-battery: the spelling battery from the command line (docs/battery.md).

    uv run --frozen robojev-battery --n 20 --seed 0 --world stub --decider mock --planner mock
    uv run --frozen robojev-battery --n 6 --seed 0 --world stub --compare
    uv run --frozen robojev-battery --world sim --n 10 --seed 1 --decider jev --planner openai --paraphrase
    uv run --frozen robojev-battery --world stub --arms 2 --n 5 --decider mock

Phrasings: `train` offline, `held_out` whenever a live decider or planner runs or --paraphrase is on
(the Mac), unless --phrasings says otherwise. Episode files go under --out (default
experiments/results/battery/<stamp>-<world>-<decider>/).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .generate import Grammar
from .models import DECIDERS, PLANNERS, is_live
from .paraphrase import hook
from .phrasings import SPLITS
from .run import CONTROLS, WORLDS, Options, compare, run, table

RESULTS = Path(__file__).resolve().parents[3] / "experiments" / "results" / "battery"


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", choices=WORLDS, default="stub", help="the physics-free stub (fast) or the MuJoCo WidowX")
    ap.add_argument("--decider", choices=DECIDERS, default="mock",
                    help="mock: the offline stand-in; rules, always_llm: the controls; jev, decisions: live (keys)")
    ap.add_argument("--planner", choices=PLANNERS, default="mock")
    ap.add_argument("--arms", type=int, choices=(1, 2), default=1, help="2: the stub pair, letters in arm_0's half, positions in arm_1's")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare", action="store_true", help=f"run {', '.join(CONTROLS)} on the same episodes, side by side")
    ap.add_argument("--paraphrase", action="store_true", help="rewrite every line with a model first (OPENAI_API_KEY)")
    ap.add_argument("--phrasings", choices=SPLITS, default=None, help="default: held_out when live or paraphrasing, else train")
    ap.add_argument("--opening", default=None, help="the word the opening ask spells (default OPENAI)")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    live = is_live(a.decider, a.planner) or a.paraphrase
    split = a.phrasings or ("held_out" if live else "train")
    out = a.out or RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.world}-{'compare' if a.compare else a.decider}"
    grammar = Grammar(arms=a.arms) if a.opening is None else Grammar(opening=a.opening, arms=a.arms)
    opts = Options(a.n, a.seed, a.world, a.decider, a.planner, a.arms, split, out, hook(a.paraphrase), grammar, quiet=a.quiet)
    print(f"battery: {a.n} episodes, seed {a.seed}, {a.world}, {a.arms} arm(s), {split} phrasings"
          f"{', paraphrased' if a.paraphrase else ''}; files under {out}")
    summaries = compare(opts) if a.compare else [run(opts)]
    print()
    print(table(summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
