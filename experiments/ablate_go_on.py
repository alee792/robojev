"""The Decisions API on the slice's "ok, go on" (scenario e), with and without rule 5's text
(docs/live-results.md, "Rules added after the re-run": `user_words_in_task` and the two sentences
added to `about_the_task` / the route's `ask_about`). Takes the request recorded in a run after the
rules and puts back the older text field by field; 5 calls per variant.

    uv run --frozen python experiments/ablate_go_on.py OLD.jsonl.gz NEW.jsonl [--decider decisions|jev]
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json


def request(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    for line in op(path, "rt"):
        r = json.loads(line)
        if r.get("type") == "note" and r["kind"] == "decision_request" and r["rec"].get("event") == "user_text" \
                and "ok, go on" in (r["rec"].get("event_text") or ""):
            return r["rec"]
    raise SystemExit(f"no 'ok, go on' request in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("--decider", default="decisions"); ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    old, new = request(a.old), request(a.new)
    if a.decider == "jev":
        from robojev.brain.backends.jev import JevHTTP
        be = JevHTTP(max_requests=200)
    else:
        from robojev.brain.backends.decisions_api import DecisionsAPI
        be = DecisionsAPI(max_requests=200)

    diffs = list(_diff_paths(old["questions"], new["questions"]))   # leaf paths whose text rule 5 changed
    print("changed question fields:", [".".join(map(str, path)) for path in diffs])

    def variant(state_field: bool, about: bool, ask: bool):
        s, q = copy.deepcopy(new["state"]), copy.deepcopy(new["questions"])
        if not state_field:
            s.pop("user_words_in_task", None)
        for path in diffs:
            if (path[-1] == "about_the_task" and not about) or (path[-1] == "ask_about" and not ask):
                _set(q, path, _at(old["questions"], path))
        return s, q

    variants = {
        "as recorded after the rules": (True, True, True),
        "without user_words_in_task": (False, True, True),
        "route ask_about as before": (True, True, False),
        "about_the_task as before": (True, False, True),
        "all three as before": (False, False, False),
    }
    for name, flags in variants.items():
        s, q = variant(*flags)
        outs = []
        for _ in range(a.n):
            r = be.answer(s, q)["answers"]["right_now"]
            outs.append(f"{r.get('choice')} {r.get('confidence', 0):.2f} (resume p {r.get('probabilities', {}).get('resume', 0):.2f})")
        print(f"{name:32s} " + "; ".join(outs), flush=True)


def _diff_paths(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() & b.keys():
            yield from _diff_paths(a[k], b[k], path + (k,))
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from _diff_paths(x, y, path + (i,))
    elif a != b:
        yield path


def _at(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def _set(obj, path, value) -> None:
    _at(obj, path[:-1])[path[-1]] = value


if __name__ == "__main__":
    main()
