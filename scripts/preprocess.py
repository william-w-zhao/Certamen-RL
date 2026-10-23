"""
Cleaning pipeline: tossups_tokenized.json -> tossups_clean.json

Outputs:
  data/tossups_clean.json    the records we keep, in the slimmed-down schema
  data/tossups_dropped.json  every dropped record with the reason(s), for spot-checking
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from answers import parse_answer

DATA = Path(__file__).resolve().parent.parent / "data"

# width-less characters
INVISIBLE = re.compile("[​‌‍⁠﻿­]")

# answer merged onto tossup
ANSWER_MARKER = re.compile(r"\?\s+[A-Z][A-Z0-9'’()/,.\- ]{2,}(?:\s+(?:[A-Z][a-z]|\()|\s*$)")

# bonus merged into tossup
BONUS_MARKER = re.compile(r"\b(?:BO\d|Bonus\s*\d)", re.I)

# leftover bonus labels ("TELEMACHUS B1+", "WALLS B1: Translate") and packet headers ("...? Novice — Semis")
LEFTOVER_MARKER = re.compile(r"\bB\d\s*[+:]|(?:Novice|Intermediate|Advanced|Elite)\s+[—–]\s+\w+\s*$")

# questions that need a handout or shared screen ("examine this visual", "In which picture...")
VISUAL = re.compile(r"\b(?:visuals?|pictured|pictures?|photograph|hand(?:ed)? ?out)\b|https?://", re.I)

# "which of the following" with nothing after the "?": the options ended up in the answer field
MISSING_OPTIONS = re.compile(r"\b(?:which|what) of the following\b(?:\s+\w+)?(?:,?\s*if any,?)?[^:;,“\"‘?]*\?\s*$", re.I)

# sets checked by hand: ~40% of records are citations only ("Conte pp. 156, 159 /"), with misaligned answers
BROKEN_SETS = {"Eel Elite 2021", "Premier Elite 2020"}

# answer kinds a text model can't give: performed answers and "See Below"
UNPLAYABLE_KINDS = {"action", "unusable"}

# OCR garbage with abnormal capitalization/letter-digit mix
OCR_WORD = re.compile(r"\b(?=\w*[a-z][A-Z])(?!Mc|Mac)\w+\b|\b\w*[A-Za-z]\d\w*\b|\b\d+[a-z]+\w*\b")

# OCR correct (dates, ordinals)
OCR_OK = re.compile(r"\d+(st|nd|rd|th|s|bc|ad)", re.I)

# Mojibak macrons decoded with the wrong font encoding
MOJIBAKE = re.compile(r"[A-Za-z][§Ç¦æã]|[A-Za-z][~}][A-Za-z]")

# entire packets check
MAX_QUESTION_CHARS = 1000

# max OCR garbage rate
SET_OCR_RATE = 0.10

# year cutoff
MIN_YEAR = 2008

LEVELS = {1: "novice", 5: "intermediate", 10: "advanced"}

# stray category consolidation
CATEGORY_FIXES = {"Literature": "Latin Literature", "History": "Roman History"}

def strip_invisible(record):
    """
    Remove invisible characters from question and answer, shifting clue offsets to match.
    """
    q = record["question"]
    # new_pos[i] = position of old character i in the stripped string
    new_pos, kept = [], 0
    for c in q:
        new_pos.append(kept)
        if not INVISIBLE.match(c):
            kept += 1
    new_pos.append(kept)

    record["question"] = INVISIBLE.sub("", q)
    record["answer"] = INVISIBLE.sub("", record["answer"])
    for c in record.get("clues") or []:
        c["start"], c["end"] = new_pos[c["start"]], new_pos[c["end"]]
        c["text"] = INVISIBLE.sub("", c["text"])
    return record

def ocr_words(text):
    """
    Check for OCR
    """
    text = BONUS_MARKER.sub(" ", text)
    return [w for w in OCR_WORD.findall(text) if not OCR_OK.fullmatch(w)]

def problems(record):
    """
    List of reasons this record should be dropped (empty list = keep).
    """
    q, a = record["question"], record["answer"]
    reasons = []
    if MIN_YEAR and record["setYear"] < MIN_YEAR:
        reasons.append("before_min_year")
    if not a.strip():
        reasons.append("empty_answer")
    if len(q) > MAX_QUESTION_CHARS:
        reasons.append("too_long")
    if ANSWER_MARKER.search(q):
        reasons.append("merged_questions")
    if BONUS_MARKER.search(q):
        reasons.append("merged_bonuses")
    if len(ocr_words(q)) >= 2:
        reasons.append("ocr_garbage")
    if MOJIBAKE.search(q) or MOJIBAKE.search(a):
        reasons.append("mojibake")
    if q.count("!") >= 3 and "! " not in q:
        reasons.append("bang_separators")  # "!Of!what!Gallic!tribe!..."
    if record["setName"] in BROKEN_SETS:
        reasons.append("broken_set")
    if LEFTOVER_MARKER.search(q):
        reasons.append("leftover_marker")
    if VISUAL.search(q):
        reasons.append("visual")
    if MISSING_OPTIONS.search(q):
        reasons.append("missing_options")
    if a.strip() and parse_answer(a)["kind"] in UNPLAYABLE_KINDS:
        reasons.append("unplayable_answer")
    return reasons

def bad_sets(records, reasons_by_id):
    """
    Sets where at least SET_OCR_RATE of records are OCR garbage.
    """
    total, ocr = Counter(), Counter()
    for r in records:
        total[r["setName"]] += 1
        ocr[r["setName"]] += "ocr_garbage" in reasons_by_id[r["_id"]["$oid"]]
    return {s for s in total if ocr[s] / total[s] >= SET_OCR_RATE}

def reformat(record):
    """
    Reformat json entry
    """
    return {
        "id": record["_id"]["$oid"],
        "set": record["setName"],
        "year": record["setYear"],
        "level": LEVELS.get(record["difficulty"]),
        "category": CATEGORY_FIXES.get(record["category"], record["category"]),
        "round": record["packet"]["name"],
        "packet_number": record["packetNumber"],
        "question_number": record["questionNumber"],
        "question": record["question"],
        "answer": record["answer"],
        "answer_parsed": parse_answer(record["answer"]),
        "clues": [
            {k: c[k] for k in ("text", "type", "start", "end")}
            for c in record.get("clues") or []
        ],
    }

def check_offsets(record):
    """
    Check clue offsets
    """
    for c in record["clues"]:
        assert record["question"][c["start"]:c["end"]] == c["text"], (record["id"], c)

def main():
    records = json.load(open(DATA / "tossups_tokenized.json"))
    records = [strip_invisible(r) for r in records]

    reasons_by_id = {r["_id"]["$oid"]: problems(r) for r in records}
    dropped_sets = bad_sets(records, reasons_by_id)
    for r in records:
        if r["setName"] in dropped_sets:
            reasons_by_id[r["_id"]["$oid"]].append("bad_set")

    kept, dropped = [], []
    for r in records:
        reasons = reasons_by_id[r["_id"]["$oid"]]
        out = reformat(r)
        if reasons:
            dropped.append({"reasons": reasons, **out})
        else:
            check_offsets(out)
            kept.append(out)

    json.dump(kept, open(DATA / "tossups_clean.json", "w"), ensure_ascii=False, indent=1)
    json.dump(dropped, open(DATA / "tossups_dropped.json", "w"), ensure_ascii=False, indent=1)

    # --- Report ---
    print(f"input:   {len(records)}")
    print(f"kept:    {len(kept)}")
    print(f"dropped: {len(dropped)}")
    print("\nrecords flagged per reason (a record can have several):")
    for reason, n in Counter(x for d in dropped for x in d["reasons"]).most_common():
        print(f"  {n:6d}  {reason}")
    print(f"\nwhole sets dropped ({len(dropped_sets)}):")
    for s in sorted(dropped_sets):
        print(f"  {s}")

    by_year = defaultdict(lambda: [0, 0])
    for r in kept:
        by_year[r["year"] // 5 * 5][0] += 1
    for r in dropped:
        by_year[r["year"] // 5 * 5][1] += 1
    print("\nkept / dropped by 5-year bucket:")
    for y in sorted(by_year):
        k, d = by_year[y]
        print(f"  {y}-{y + 4}: {k:6d} kept  {d:5d} dropped")

if __name__ == "__main__":
    main()
