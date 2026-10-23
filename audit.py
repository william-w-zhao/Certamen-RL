import json
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

def section(title):
    print(f"\n=== {title} ===")

def top(counter, n=15):
    for value, count in counter.most_common(n):
        print(f"  {count:6d}  {value!r}")
    if len(counter) > n:
        print(f"  ... {len(counter) - n} more distinct values")

def main():
    raw = json.load(open(DATA / "tossups.json"))
    tok = json.load(open(DATA / "tossups_tokenized.json"))

    # --- question identity ---
    section("Files")
    print(f"tossups.json:           {len(raw)} records")
    print(f"tossups_tokenized.json: {len(tok)} records")

    raw_by_id = {r["_id"]["$oid"]: r for r in raw}
    same_text = sum(
        raw_by_id.get(t["_id"]["$oid"], {}).get("question") == t["question"] for t in tok
    )
    print(f"tokenized records whose question text matches tossups.json: {same_text}")

    # --- category distribution ---
    section("category");      
    top(Counter(t["category"] for t in tok))

    # --- level distribution ---
    section("difficulty");    
    top(Counter(t["difficulty"] for t in tok))

    # --- set distribution ---
    section("sets")
    sets = Counter(t["setName"] for t in tok)
    print(f"{len(sets)} distinct sets; largest:")
    top(sets, n=10)

    # --- clues ---
    section("Clues")
    # clues per question
    n_clues = Counter(len(t.get("clues") or []) for t in tok)
    print("clues per question (# of clues - # of questions):")
    for k in sorted(n_clues):
        print(f"  {k:3d} - {n_clues[k]}")

    # clue type distribution
    print("clue types:")
    types = []
    for t in tok:
        for c in t.get("clues") or []:
            types.append(c["type"])
    top(Counter(types))

    # clue position check 
    ok_q, bad = 0, 0
    bad_examples = []
    for t in tok:
        for c in t.get("clues") or []:
            in_q = t["question"][c["start"]:c["end"]] == c["text"]
            ok_q += in_q
            if not (in_q):
                bad += 1
                if len(bad_examples) < 5:
                    bad_examples.append((c, t["question"][c["start"]:c["end"]]))
    print(f"offsets match question:           {ok_q}")
    print(f"offsets do not match:              {bad}")
    for c, actual in bad_examples:
        print(f"  clue {c['text']!r} vs text at offsets {actual!r}")

    # --- duplicates ---
    section("duplicates")

    # normalize
    def norm(s):
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    
    groups = Counter(norm(t["question"]) for t in tok)
    duplicates = {k: n for k, n in groups.items() if n > 1}
    print(f"{len(duplicates)} texts appear more than once, covering "
          f"{sum(duplicates.values())} records")

    sets_per_text = {}
    for t in tok:
        k = norm(t["question"])
        if k in duplicates:
            sets_per_text.setdefault(k, set()).add(t["setName"])
    print(f"of those, {sum(len(s) > 1 for s in sets_per_text.values())} span more than one set")

    # --- answer formats ---
    section("Answer formats (answer)")
    patterns = {
        "slash /":         r"/",
        "parentheses ( )": r"\(",
        "brackets [ ]":    r"\[",
        "'accept'":        r"\baccept",
        "'prompt'":        r"\bprompt",
        "' or '":          r"\bor\b",
        "semicolon":       r";",
        "digits":          r"\d",
        "macron/diacritic": r"[āēīōūȳĀĒĪŌŪáéíóú]",
    }

    answers = [t["answer"] for t in tok]
    for name, pat in patterns.items():
        hits = [a for a in answers if re.search(pat, a, re.I)]
        print(f"  {len(hits):6d}  {name:18s} e.g. {hits[0][:80]!r}" if hits else f"       0  {name}")
    lengths = sorted(len(a) for a in answers)
    print(f"answer length (chars): median {lengths[len(lengths)//2]}, "
          f"90th pct {lengths[int(len(lengths)*.9)]}, max {lengths[-1]}")

    # --- no answers ---
    print(f"empty answers: {sum(not a.strip() for a in answers)}")

if __name__ == "__main__":
    main()
