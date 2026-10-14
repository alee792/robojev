"""What the person says, by event type, in two halves: `train` phrasings are the ones the offline
runs use and the ones the rules control was written from (controls.py); `held_out` phrasings are
kept for the live runs on the Mac, so a decider meets wording nobody tuned for. Every type has at
least four of each. `{W}` is the word; `{L}` the letter an `other_letter` line is about.

Nothing here says what the robot should do about a line: that is the brain's, and the battery
judges only what the row reads afterwards and how long things took (docs/battery.md).
"""
from __future__ import annotations

SPLITS = ("train", "held_out")

# The opening ask, the task text the episode starts with.
OPENING = {
    "train": ("spell {W}", "please spell {W} with the letters", "can you spell {W} in the positions", "write {W}, left to right"),
    "held_out": ("I'd like {W} spelled out", "lay out the word {W}", "put the letters down so they read {W}", "{W}, please"),
}

PHRASINGS: dict[str, dict[str, tuple[str, ...]]] = {
    # a different word (a correction; the end result changes)
    "new_word": {
        "train": ("actually, spell {W}", "scrap that, spell {W} instead", "change of plan: make it {W}",
                  "no wait, I meant {W}", "let's do {W} rather than that"),
        "held_out": ("hmm, on second thought, {W}", "can you switch to {W}?", "forget the first word, write {W}",
                     "new word: {W}", "do {W} instead please"),
    },
    # the other cube with the same letter, said while the arm reaches for one of two (the end result is
    # unchanged; which cube spells the letter is the only thing that changes)
    "other_letter": {
        "train": ("use the other {L}", "take the other one instead", "not that one, the other {L}", "grab the other {L} please",
                  "the other one, not that"),
        "held_out": ("switch to the other {L}", "no, the other one", "I'd rather you took the other {L}", "leave that {L}, use the other",
                     "the other {L} please"),
    },
    # talk that asks for nothing
    "chatter": {
        "train": ("nice day for it", "my coffee's gone cold", "you're doing great", "the camera's a bit crooked",
                  "I like this little robot"),
        "held_out": ("did anyone feed the cat", "that's a lovely shade of red", "I should call my mum later",
                     "what a tidy desk", "good job so far"),
    },
    # a momentary wait, and the go-on that ends it
    "wait": {
        "train": ("hang on a second", "wait wait", "hold on a moment", "one sec", "pause for a moment"),
        "held_out": ("give me a moment", "just a tick", "hold your horses", "freeze for a second", "hmm, wait up"),
    },
    "go_on": {
        "train": ("ok, go on", "carry on", "alright, continue", "you can keep going", "go ahead"),
        "held_out": ("right, as you were", "proceed", "ok resume", "all good, go", "keep at it"),
    },
    # remarks a person makes while doing something to the table (the scene change is the event; the
    # remark asks for nothing)
    "letter_taken": {
        "train": ("I'm borrowing this one", "let me just take this", "I need this letter a moment", "mine now"),
        "held_out": ("this one's coming with me", "I'll have that", "just nabbing this", "ooh, I want this letter"),
    },
    "letter_returned": {
        "train": ("here, you can have it back", "putting it back now", "there, returned", "ok it's back"),
        "held_out": ("sorry, here it is again", "back it goes", "you can use this again", "returning your letter"),
    },
    "letter_added": {
        "train": ("here's another one for you", "I found one more letter", "have an extra", "adding this to the pile"),
        "held_out": ("there's a spare, take it", "one more for the table", "look what I found", "this might help"),
    },
    "letter_nudged": {
        "train": ("oops, bumped it", "sorry, knocked that one", "that slid a bit", "my elbow caught it"),
        "held_out": ("whoops, that moved", "didn't mean to shift that", "clumsy me", "it skidded a little"),
    },
    "letter_placed": {
        "train": ("I'll put this one in myself", "let me do this letter", "there, that one's done for you", "I'll help with this one"),
        "held_out": ("saving you a trip", "I'll pop that in", "done that one for you", "let me place this"),
    },
    "hand_near": {
        "train": ("let me just reach past", "excuse my hand", "just grabbing my pen", "mind me a second"),
        "held_out": ("coming through", "reaching over, sorry", "don't mind my arm", "let me get that"),
    },
    # the person's answer when the robot asks what the end result should be
    "reply": {
        "train": ("just do what you can with what's there", "spell what you can for now", "carry on with the letters you have",
                  "whatever you can manage"),
        "held_out": ("use what's on the table", "do your best with those", "as much of it as you can", "the rest can wait"),
    },
}

TEXT_EVENTS = ("new_word", "other_letter", "chatter", "wait")    # events that are a line the person types or says
REMARK_EVENTS = ("letter_taken", "letter_added", "letter_nudged", "letter_placed", "hand_near")   # a remark may go with these


def check_tables() -> list[str]:
    """Problems with the tables: a type with fewer than four phrasings in a half, a phrasing in both
    halves, a word slot missing where one is needed. Empty = fine."""
    out = []
    for kind, halves in {"opening": OPENING, **PHRASINGS}.items():
        for split in SPLITS:
            if len(halves.get(split, ())) < 4:
                out.append(f"{kind}: fewer than four {split} phrasings")
        both = set(halves["train"]) & set(halves["held_out"])
        if both:
            out.append(f"{kind}: in both halves: {sorted(both)}")
        if kind in ("opening", "new_word") and any("{W}" not in t for s in SPLITS for t in halves[s]):
            out.append(f"{kind}: a phrasing has no {{W}}")
        if kind == "other_letter" and not any("{L}" in t for s in SPLITS for t in halves[s]):
            out.append(f"{kind}: no phrasing names the letter")
    return out
