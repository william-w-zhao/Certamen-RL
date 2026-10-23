"""
Answer-Line Parser: turns a raw answer field into structured accept/prompt/reject lists.

parse_answer("Tiberius (Sempronius) Gracchus [prompt on “gracchus”]")
-> {"main": "Tiberius Gracchus",
    "accept": ["Tiberius Gracchus", "Tiberius Sempronius Gracchus"],
    "prompt": ["gracchus"], 
    "reject": [], 
    "notes": [],
    "kind": "short", 
    "flags": []}
"""

import itertools
import re
import json
import random
from collections import Counter
from pathlib import Path

# words that mark a bracket/parenthesis as an instruction to the moderator, not answer text
DIRECTIVE_WORDS = re.compile(
    r"\b(accept|prompt|do not|don[’']t|reject|n\.\s?b|moderator|reveal|read|respectively"
    r"|i\.e|e\.g|equivalent|either order|see b|any one|literal)", re.I
)

# Inside a directive: each keyword starts a clause that runs until the next keyword.
CLAUSE = re.compile(
    r"(do not accept|don[’']t accept|do not prompt|reject|also accept|accept|prompt(?: on)?)"
    r"\s*:?\s*(.*?)(?=do not accept|don[’']t accept|do not prompt|reject|also accept|accept|prompt|$)",
    re.I | re.S,
)

# Clause contents that are a rule, not a specific answer ("accept equivalents").
GENERIC = {
    "equivalents": "accept_equivalents",
    "literal translation": "accept_equivalents",
    "either order": "either_order",
    "any order": "either_order",
}

# A prompt clause often says *how* to prompt: 'prompt on “geryon” with “can you be more specific?”'
PROMPT_TAIL = re.compile(r"\s+(with|by|before|if|until|and ask)\b.*$", re.I | re.S)

# "Student should rise and say..." -- a performed answer, not text a model can give.
ACTION = re.compile(r"^\s*(the\s+)?(students?|players?)\b|\b(students?|players?) (should|must|will)\b", re.I)

# Word-and-meaning pairs: "Dūcō - Lead", "Moneo = I Warn", "Carō – Flesh; Carbō – Coal"
MULTI = re.compile(r"\s[-–—=]{1,2}\s|;")

MAX_EXPANSIONS = 32
LONG_WORDS = 6 

def split_top(text, sep="/"):
    """
    Split on `sep` only where it isn't inside (), [] or {}.
    """
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]

def squash(text):
    """
    Collapse whitespace and punctuation
    """
    return re.sub(r"\s+", " ", text).strip(" ,.;:")

def pull_directives(raw):
    """
    Remove [..] blocks and instruction-like (..) blocks; return (rest, directives, notes).
    """
    directives, notes = [], []

    def take_bracket(m):
        directives.append(m.group(1))
        return " "

    text = re.sub(r"\[([^\]]*)\]?", take_bracket, raw)  # tolerate a missing closing ]

    def take_paren(m):
        inner = m.group(1)
        if DIRECTIVE_WORDS.search(inner):
            directives.append(inner)
            return " "
        # Long explanations or quoted glosses: "(all others have 1 meaning...)", ("Health")
        if len(inner.split()) > 4 or re.search(r"[\"“”]", inner):
            notes.append(inner)
            return " "
        return m.group(0)  # short parenthetical: an optional part of the answer, keep it

    text = re.sub(r"\(([^()]*)\)", take_paren, text)
    return text, directives, notes

def parse_directive(text, out):
    """
    Fill out['accept'|'prompt'|'reject'|'flags'] from one directive's text.
    """
    found = False
    for keyword, body in CLAUSE.findall(text):
        found = True
        kw = keyword.lower()
        bucket = "reject" if ("not" in kw or "n't" in kw or "’t" in kw or "reject" in kw) \
            else "prompt" if "prompt" in kw else "accept"
        for phrase, flag in GENERIC.items():
            if phrase in body.lower():
                out["flags"].append(flag)
                body = ""
        if bucket == "prompt":
            body = PROMPT_TAIL.sub("", body)
        quoted = re.findall(r"[\"“]([^\"”]+)[\"”]", body)
        items = quoted or re.split(r"\s+or\s+|,|/", body)
        out[bucket] += [squash(i) for i in items if squash(i)]
    if not found:
        out["notes"].append(squash(text))

def expand(alt):
    """
    All spellings of one alternative.

    {a / b}  -> required choice: a or b
    (a)      -> optional: with or without a
    (a / b)  -> optional choice: nothing, a, or b
    """
    pieces = re.split(r"(\{[^{}]*\}|\([^()]*\))", alt)
    options = []
    for p in pieces:
        if p.startswith("{"):
            options.append(split_top(p[1:-1]) or [""])
        elif p.startswith("("):
            options.append([""] + split_top(p[1:-1]))
        else:
            options.append([p])
    out = []
    for combo in itertools.islice(itertools.product(*options), MAX_EXPANSIONS):
        s = squash("".join(combo))
        s = re.sub(r"\s+([,.;:!?])", r"\1", s)
        if s and s not in out:
            out.append(s)
    return out

def parse_answer(raw):
    out = {"main": "", "accept": [], "prompt": [], "reject": [], "notes": [],
           "kind": "short", "flags": []}
    text = raw.strip()

    # Leftover question text: '“Nobody?” A: Polyphemus' -> keep what follows "A:"
    m = re.search(r"(?:^|\s)(?:ans(?:wer)?|a)\s*:\s*(.+)$", text, re.I | re.S)
    if m:
        if m.start() > 0:
            out["flags"].append("question_text_in_answer")
        text = m.group(1)

    if ACTION.search(text):
        out["kind"] = "action"

    text, directives, notes = pull_directives(text)
    if re.search(r"\bsee (b\d|below|bonus)", text, re.I):
        out["kind"] = "unusable"
    out["notes"] += notes
    for d in directives:
        parse_directive(d, out)

    m = re.search(r"\b(also accept|accept)\b.*$", text, re.I | re.S)
    if m:
        parse_directive(m.group(0), out)
        text = text[: m.start()]

    text = squash(text.replace("//", "/"))
    text = re.sub(r"(?<![\w{(/])[^\s/{}()\[\]]+(?:/[^\s/{}()\[\]]+)+",
                  lambda m: "{" + m.group(0).replace("/", " / ") + "}", text)
    alternatives = split_top(text)
    if not alternatives:
        out["kind"] = "unusable"
        return out

    if out["kind"] == "short":
        if MULTI.search(text):
            out["kind"] = "multi"
        elif max(len(a.split()) for a in alternatives) > LONG_WORDS:
            out["kind"] = "long"

    if out["kind"] != "short":
        out["main"] = text
        out["accept"] = [text] + [a for a in out["accept"] if a != text]
    else:
        line_accepts = []
        for alt in alternatives:
            line_accepts += [s for s in expand(alt) if s not in line_accepts]
        out["accept"] = line_accepts + [a for a in out["accept"] if a not in line_accepts]
        out["main"] = min(expand(alternatives[0]) or [alternatives[0]], key=len)

        lens = [len(a.split()) for a in alternatives]
        if len(alternatives) > 1 and min(lens) == 1 and max(lens) > 1:
            out["flags"].append("slash_scope_ambiguous")

    out["flags"] = sorted(set(out["flags"]))
    return out


if __name__ == "__main__":
    data = Path(__file__).resolve().parent.parent / "data" / "tossups_clean.json"
    records = json.load(open(data))
    parsed = [(r["answer"], parse_answer(r["answer"])) for r in records]

    print("kind:")
    for k, n in Counter(p["kind"] for _, p in parsed).most_common():
        print(f"  {n:6d}  {k}")
    print("flags:")
    for k, n in Counter(f for _, p in parsed for f in p["flags"]).most_common():
        print(f"  {n:6d}  {k}")
    print("has prompt:", sum(bool(p["prompt"]) for _, p in parsed),
          " has reject:", sum(bool(p["reject"]) for _, p in parsed))

    random.seed(0)
    for kind in ["short", "multi", "long", "action", "unusable"]:
        group = [x for x in parsed if x[1]["kind"] == kind]
        print(f"\n=== sample: {kind} ===")
        for raw, p in random.sample(group, min(4, len(group))):
            print(f"  RAW    {raw[:100]!r}")
            print(f"  ACCEPT {p['accept'][:5]}  PROMPT {p['prompt']}  REJECT {p['reject']}  FLAGS {p['flags']}")
