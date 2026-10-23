"""
Training, Validation, Test Set Split: tossups_clean.json -> splits.json

Splits by set (whole tournaments), holding out the newest years so the base model is
least likely to have seen them. Near-duplicate questions are kept out of train when a
copy is in val/test, so eval scores aren't inflated by memorized questions.

Outputs:
  data/splits.json  {"train": [ids], "val": [ids], "test": [ids], "removed": [{id, dup_of, similarity}]}
"""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# held-out years (all sets from these years)
TEST_YEARS = {2025}
VAL_YEARS = {2024}

# near-duplicate question and answer rule: questions share this much of their word 3-grams AND an accepted answer
DUP_SIMILARITY = 0.5

# near-duplicate question rule: questions share this much of their word 3-grams regardless of answer
DUP_SIMILARITY_ANY_ANSWER = 0.9

# 3-grams in more than this many questions ("what is the") are too common to signal a duplicate
MAX_SHINGLE_FREQ = 50

def norm(text):
    """
    Lowercase words with macrons/diacritics removed.
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", text)

def get_shingles(text, n=3):
    """
    Set of word n-grams ("what author greatly", "author greatly imitated", ...).
    """
    w = norm(text)
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}

def answer_keys(record):
    """
    Normalized accepted answers, so "Catullus" and "(Gaius Valerius) Catullus" overlap.
    """
    return {" ".join(norm(a)) for a in record["answer_parsed"]["accept"]} - {""}

def label(record):
    if record["year"] in TEST_YEARS:
        return "test"
    if record["year"] in VAL_YEARS:
        return "val"
    return "train"

def find_duplicates(records):
    """
    Return [(similarity, i, j)] for near-duplicate pairs.

    Comparing all ~400M pairs is too slow, so only compare questions that share
    at least one uncommon 3-gram (an inverted index).
    """
    sh = [get_shingles(r["question"]) for r in records]
    freq = Counter(s for shingles in sh for s in shingles)
    # locations for uncommon shingles
    index = defaultdict(list)
    for i, shingles in enumerate(sh):
        for s in shingles:
            if freq[s] <= MAX_SHINGLE_FREQ:
                index[s].append(i)

    candidates = set()
    for ids in index.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                candidates.add((ids[a], ids[b]))

    keys = [answer_keys(r) for r in records]
    dups = []
    for i, j in candidates:
        # compute Jaccard similarity
        sim = len(sh[i] & sh[j]) / len(sh[i] | sh[j])
        if sim >= DUP_SIMILARITY_ANY_ANSWER or (sim >= DUP_SIMILARITY and keys[i] & keys[j]):
            dups.append((sim, i, j))
    return dups

def main():
    records = json.load(open(DATA / "tossups_clean.json"))
    split = [label(r) for r in records]
    dups = find_duplicates(records)

    # a train record with a copy in val/test is removed; a val record with a copy in test too
    rank = {"train": 0, "val": 1, "test": 2}
    removed = {}
    for sim, i, j in dups:
        if rank[split[i]] == rank[split[j]]:
            continue  # same side: no leakage
        lo, hi = (i, j) if rank[split[i]] < rank[split[j]] else (j, i)
        if lo not in removed or sim > removed[lo]["similarity"]:
            removed[lo] = {"id": records[lo]["id"], "dup_of": records[hi]["id"],
                           "similarity": round(sim, 2)}

    out = {"train": [], "val": [], "test": [], "removed": list(removed.values())}
    for i, r in enumerate(records):
        if i not in removed:
            out[split[i]].append(r["id"])

    json.dump(out, open(DATA / "splits.json", "w"), indent=1)

    print(f"near-duplicate pairs found: {len(dups)}")
    print(f"  across splits (leaks):    {sum(rank[split[i]] != rank[split[j]] for _, i, j in dups)}")
    print(f"records removed for leaking: {len(removed)} "
          f"({Counter(split[i] for i in removed)})")
    print()
    by_id = {r["id"]: r for r in records}
    for name in ("train", "val", "test"):
        rs = [by_id[i] for i in out[name]]
        print(f"{name:5s} {len(rs):6d} records, {len({r['set'] for r in rs})} sets")
        print(f"      level    {dict(Counter(r['level'] for r in rs).most_common())}")
        print(f"      category {dict(Counter(r['category'] for r in rs).most_common())}")

if __name__ == "__main__":
    main()
