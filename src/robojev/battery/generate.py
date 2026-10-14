"""The grammar: seeded episodes of the spelling demo (docs/showcase-demo.md), generated, not listed.

An episode is an opening ask ("spell OPENAI"), a set of letter cubes on the table, a row of positions,
and 0-3 events, each at a time or on a condition, each with a phrasing drawn from phrasings.py (the
`train` half offline, `held_out` on the Mac). The event types:

    new_word        a different word, spellable from the letters on the table
    other_letter    "use the other A": said while the arm reaches for a letter that has a duplicate free
                    on the table; the word, and so the end state, is unchanged
    letter_taken    a letter is taken off the table, and comes back later or not (if a spare covers it)
    letter_added    a letter that was not on the table appears
    letter_nudged   a letter is moved a few cm where it lies
    letter_placed   the person puts a letter into a position themselves
    chatter         talk that asks for nothing
    wait_go_on      a wait, then a go-on a couple of seconds later
    hand_near       a hand near the gripper, briefly
    stop            the STOP button (rare; `Grammar.p_stop`), always the last event

An episode carries NO expected action for any event: no "hold here", no "replan there". It carries
the events, when they happen, and the expected END STATE: the latest word asked for. The brain reacts
as it likes and judge.py scores the row it leaves and the time it took.

Times are in PICKS (one pick_and_place's worth of time), so the same episode runs on the stub (a pick
is under a second) and on the physics sim (about seven): the runner converts with the world's pick
time. Conditions ("after 2 letters of the current word are in place", "while the arm carries a
letter") are judged by the runner from the true world; each has a fallback time so an episode whose
condition never comes still fires its events.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from .phrasings import OPENING, PHRASINGS, REMARK_EVENTS, SPLITS

EVENT_KINDS = ("new_word", "other_letter", "letter_taken", "letter_added", "letter_nudged", "letter_placed", "chatter", "wait_go_on",
               "hand_near", "stop")
LINE_KINDS = ("opening", "new_word", "other_letter", "chatter", "wait", "go_on", "remark", "reply")

# Words spellable from the demo's letters O P E N A A I D (docs/showcase-demo.md), as corrections.
WORDS = ("PANDA", "PAID", "IDEA", "OPEN", "DEAN", "NOPE", "PINE", "PAIN", "PIANO", "DINE", "AIDE", "NAPE", "DOPE", "PANE",
         "PAD", "DIP", "END", "ONE", "PEN", "NAP")


def spellable(word: str, letters: Sequence[str]) -> bool:
    """Every letter of `word` is among `letters`, counting duplicates."""
    have = Counter(ch.upper() for ch in letters)
    return all(have[ch] >= n for ch, n in Counter(word.upper()).items())


@dataclass(frozen=True)
class When:
    """When an event fires. `kind` "at": `k` picks after the start; "placed": once `k` letters of the
    word in force read right in the positions; "carrying": once that holds and the arm is carrying a
    letter; "reaching": once that holds and an arm is reaching for the event's letter (not grasped
    yet) with another cube of that letter free on the table. `by_picks`: fire anyway this many picks
    after the start."""
    kind: str
    k: float
    by_picks: float

    def __post_init__(self) -> None:
        if self.kind not in ("at", "placed", "carrying", "reaching"):
            raise ValueError(f"unknown When kind {self.kind!r}")


@dataclass(frozen=True)
class Event:
    """One thing that happens. Only what happens and when; never what the robot should do about it."""
    index: int
    kind: str
    when: When
    text: str | None = None                  # the line typed (text events) or the remark that goes with a world event
    word: str | None = None                  # new_word: the word now asked for
    label: str | None = None                 # letter_taken / letter_added / letter_nudged / other_letter: the letter concerned
    position: int | None = None              # letter_placed: the position (index into the word in force)
    returns: str | None = None               # letter_taken: "if_needed" (back only if no spare covers it) | "always"
    return_after_picks: float | None = None  # letter_taken: how long after the taking it comes back
    return_text: str | None = None           # letter_taken: the remark on returning it
    go_on_text: str | None = None            # wait_go_on: the go-on line
    go_on_after_s: float = 2.0
    nudge_m: float = 0.03
    hand_for_s: float = 1.5
    hand_dist_m: float = 0.15

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}")


@dataclass(frozen=True)
class Line:
    """What a line the person says is, for whoever needs to know (the offline mock decider does)."""
    kind: str
    word: str | None = None


@dataclass(frozen=True)
class Episode:
    seed: int
    index: int
    split: str
    opening: str                             # the task text
    word: str                                # the word it asks for
    pool: str                                # letters on the table at the start, in layout order
    parked: str                              # letters not on the table yet (letter_added brings one)
    positions: int
    events: tuple[Event, ...]
    reply: str                               # what the person answers if asked what the end result should be
    expected_word: str                       # the END STATE: the latest word asked for
    arms: int = 1

    @property
    def words(self) -> list[str]:
        """Every word asked for, in order: the opening and each new_word."""
        return [self.word] + [e.word for e in self.events if e.kind == "new_word" and e.word]

    @property
    def lines(self) -> dict[str, Line]:
        """text -> what it is, for every line the person may say in this episode."""
        out = {self.opening: Line("opening", self.word), self.reply: Line("reply")}
        for e in self.events:
            if e.kind == "new_word":
                out[e.text] = Line("new_word", e.word)
            elif e.kind == "other_letter":
                out[e.text] = Line("other_letter")
            elif e.kind == "chatter":
                out[e.text] = Line("chatter")
            elif e.kind == "wait_go_on":
                out[e.text] = Line("wait")
                out[e.go_on_text] = Line("go_on")
            elif e.text:
                out[e.text] = Line("remark")
            if e.return_text:
                out[e.return_text] = Line("remark")
        return out

    def letters(self) -> str:
        """Every letter that is ever on the table."""
        return self.pool + self.parked

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Grammar:
    """What the generator draws from. The defaults are the demo's."""
    opening: str = "OPENAI"
    pool: str = "OPENAAID"                               # the demo's cubes
    words: tuple[str, ...] = WORDS                       # corrections, each spellable from the pool
    spares: tuple[str, ...] = ("N", "E", "P", "A")       # a spare cube or two may be added, drawn from these
    extras: tuple[str, ...] = ("N", "E", "A", "D")       # a parked cube (for letter_added) carries one of these
    positions: int = 6
    n_events: tuple[float, ...] = (0.2, 0.4, 0.3, 0.1)   # weights for 0, 1, 2, 3 events
    kinds: dict[str, float] = field(default_factory=lambda: {
        "new_word": 3.0, "other_letter": 1.0, "letter_taken": 2.0, "letter_added": 1.0, "letter_nudged": 1.5, "letter_placed": 1.5,
        "chatter": 2.0, "wait_go_on": 1.5, "hand_near": 1.5})
    caps: dict[str, int] = field(default_factory=lambda: {"new_word": 2})   # per episode; others at most once
    p_remark: float = 0.5
    p_stop: float = 0.1
    fallback_picks: float = 8.0                           # a condition's fallback: this long after its nominal time
    arms: int = 1

    def __post_init__(self) -> None:
        bad = [w for w in self.words if not spellable(w, self.pool)]
        if bad or not spellable(self.opening, self.pool):
            raise ValueError(f"words not spellable from {self.pool!r}: {bad or [self.opening]}")
        if self.positions < max(len(w) for w in (self.opening, *self.words)):
            raise ValueError("fewer positions than the longest word")


Paraphrase = Callable[[str, str], str]      # (text, line kind) -> the text rewritten; see paraphrase.py


def _nominal(w: When) -> float:
    return {"at": w.k, "placed": w.k + 0.5, "reaching": w.k + 0.6, "carrying": w.k + 0.7}[w.kind]


def duplicated(word: str, letters: Sequence[str]) -> list[str]:
    """The letters of `word` that two or more of `letters` carry, in the word's order, once each."""
    have = Counter(ch.upper() for ch in letters)
    out: list[str] = []
    for ch in word.upper():
        if have[ch] >= 2 and ch not in out:
            out.append(ch)
    return out


class Generator:
    def __init__(self, grammar: Grammar | None = None):
        self.g = grammar or Grammar()

    def episodes(self, n: int, seed: int, split: str = "train", paraphrase: Paraphrase | None = None) -> list[Episode]:
        """`n` episodes for `seed`, the same every time for the same arguments. `paraphrase`, if given,
        rewrites every line and remark after generation (a new_word line must keep its word, else the
        original stands)."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
        return [self.episode(seed, i, split, paraphrase) for i in range(n)]

    def episode(self, seed: int, index: int, split: str = "train", paraphrase: Paraphrase | None = None) -> Episode:
        g = self.g
        rng = random.Random(f"{seed}:{index}")
        words_table = PHRASINGS

        def say(kind: str, word: str | None = None) -> str:
            t = rng.choice(words_table[kind][split])
            return t.replace("{W}", word) if word else t

        pool = list(g.pool)
        for _ in range(rng.choices((0, 1, 2), weights=(0.3, 0.5, 0.2))[0]):
            pool.append(rng.choice(g.spares))
        rng.shuffle(pool)
        parked = [rng.choice(g.extras)] if rng.random() < 0.5 else []
        n_events = rng.choices(range(len(g.n_events)), weights=g.n_events)[0]
        kinds: list[str] = []
        counts: Counter = Counter()
        names, weights = zip(*g.kinds.items())
        for _ in range(n_events * 4):
            if len(kinds) == n_events:
                break
            k = rng.choices(names, weights=weights)[0]
            if counts[k] >= g.caps.get(k, 1):
                continue
            if k == "letter_added" and not parked:
                parked.append(rng.choice(g.extras))
            counts[k] += 1
            kinds.append(k)
        whens = [self._when(rng, k, g) for k in kinds]
        order = sorted(range(len(kinds)), key=lambda i: _nominal(whens[i]))
        events: list[Event] = []
        current = g.opening
        available = pool + []                           # letters on the table so far (parked ones join when added)
        parked_left = list(parked)
        for i in order:
            k, w = kinds[i], whens[i]
            remark = say(k) if k in REMARK_EVENTS and rng.random() < g.p_remark else None
            e = Event(len(events), k, w, text=remark)
            if k == "new_word":
                cands = [x for x in g.words if x != current and spellable(x, available)]
                word = rng.choice(cands) if cands else current
                e = Event(len(events), k, w, text=say("new_word", word), word=word)
                current = word
            elif k == "other_letter":
                dups = duplicated(current, available)
                if not dups:
                    continue                                # no letter of the word has a twin on the table: nothing to swap to
                lab = rng.choice(dups)
                e = Event(len(events), k, w, text=say("other_letter").replace("{L}", lab), label=lab)
            elif k == "chatter":
                e = Event(len(events), k, w, text=say("chatter"))
            elif k == "wait_go_on":
                e = Event(len(events), k, w, text=say("wait"), go_on_text=say("go_on"))
            elif k == "letter_taken":
                rest = current[int(w.k):] or current
                e = Event(len(events), k, w, text=remark, label=rng.choice(rest),
                          returns="if_needed" if rng.random() < 0.7 else "always",
                          return_after_picks=round(rng.uniform(1.0, 2.0), 2),
                          return_text=say("letter_returned") if rng.random() < g.p_remark else None)
            elif k == "letter_added":
                lab = parked_left.pop(0)
                available.append(lab)
                e = Event(len(events), k, w, text=remark, label=lab)
            elif k == "letter_nudged":
                rest = current[int(w.k):] or current
                e = Event(len(events), k, w, text=remark, label=rng.choice(rest))
            elif k == "letter_placed":
                lo = min(int(w.k), len(current) - 1)
                e = Event(len(events), k, w, text=remark, position=rng.randrange(lo, len(current)))
            events.append(e)
        if rng.random() < g.p_stop:
            last = _nominal(events[-1].when) if events else 0.0
            k = int(last) + 1
            events.append(Event(len(events), "stop", When("carrying", k, k + g.fallback_picks)))
        ep = Episode(seed, index, split, say_opening(rng, split, g.opening), g.opening, "".join(pool), "".join(parked),
                     g.positions, tuple(events), say("reply"), current, g.arms)
        return _paraphrased(ep, paraphrase) if paraphrase is not None else ep

    def _when(self, rng: random.Random, kind: str, g: Grammar) -> When:
        if kind in ("new_word", "chatter"):
            if rng.random() < 0.4:
                k = round(rng.uniform(1.2, 3.0), 2)
                return When("at", k, k + g.fallback_picks)
            k = rng.randrange(0, 3)
            return When("carrying", k, k + g.fallback_picks)
        if kind in ("wait_go_on", "hand_near"):
            k = rng.randrange(0, 3)
            return When("carrying", k, k + g.fallback_picks)
        if kind == "other_letter":
            k = rng.randrange(0, 3)
            return When("reaching", k, k + g.fallback_picks)
        if kind == "letter_added" and rng.random() < 0.5:
            k = round(rng.uniform(1.0, 2.5), 2)
            return When("at", k, k + g.fallback_picks)
        k = rng.randrange(0, 3)                                    # letter_taken / nudged / placed / added
        return When("placed", k, k + g.fallback_picks)


def say_opening(rng: random.Random, split: str, word: str) -> str:
    return rng.choice(OPENING[split]).replace("{W}", word)


def _paraphrased(ep: Episode, para: Paraphrase) -> Episode:
    """Every line and remark rewritten. A new_word line that lost its word keeps the original."""
    def text(t: str | None, kind: str, word: str | None = None) -> str | None:
        if t is None:
            return None
        new = para(t, kind)
        if word and word not in new:
            return t
        return new or t
    events = []
    for e in ep.events:
        kind = {"new_word": "new_word", "other_letter": "other_letter", "chatter": "chatter", "wait_go_on": "wait"}.get(e.kind, "remark")
        keep = e.word or (e.label if e.kind == "other_letter" and e.text and e.label in e.text else None)   # a named letter must survive the rewrite
        events.append(Event(e.index, e.kind, e.when, text(e.text, kind, keep), e.word, e.label, e.position, e.returns,
                            e.return_after_picks, text(e.return_text, "remark"), text(e.go_on_text, "go_on"),
                            e.go_on_after_s, e.nudge_m, e.hand_for_s, e.hand_dist_m))
    return Episode(ep.seed, ep.index, ep.split, text(ep.opening, "opening", ep.word), ep.word, ep.pool, ep.parked, ep.positions,
                   tuple(events), text(ep.reply, "reply"), ep.expected_word, ep.arms)


def episode_from_dict(d: dict) -> Episode:
    """An Episode back from its asdict() form (an episodes file, or the `episode` line of an episode JSONL)."""
    d = {k: v for k, v in d.items() if k != "type"}
    events = tuple(Event(**{**e, "when": When(**e["when"])}) for e in d["events"])
    return Episode(**{**d, "events": events})


def episodes(n: int, seed: int, split: str = "train", grammar: Grammar | None = None,
             paraphrase: Paraphrase | None = None) -> list[Episode]:
    return Generator(grammar).episodes(n, seed, split, paraphrase)


def first_with(kinds: Sequence[str], seed: int = 0, split: str = "train", grammar: Grammar | None = None,
               limit: int = 500) -> Episode:
    """The first generated episode (seed, index 0, 1, ...) whose events include every kind in `kinds`
    and no stop. For tests that need a particular shape without listing it by hand."""
    gen = Generator(grammar)
    for i in range(limit):
        ep = gen.episode(seed, i, split)
        have = {e.kind for e in ep.events}
        if set(kinds) <= have and "stop" not in have:
            return ep
    raise LookupError(f"no episode with {kinds} in the first {limit} of seed {seed}")
