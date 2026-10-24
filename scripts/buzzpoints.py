"""
Buzzpoint Marker: where in each question the model may buzz

A buzzpoint is a character offset into the question: at offset k the model has heard
question[:k]. For questions with tokenized clues, the buzzpoints are:
  - after every syllable of every word inside a clue span (players often buzz mid-word or after the first syllable of a word)
  - the end of the question

For questions without tokenized clues, we use a fallback of allowing the model to buzz every k words

Outputs:
  data/buzzpoints.json  {question id: [offsets]}
"""

import json
import random
import re
import unicodedata
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# questions without clue spans get a buzzpoint every this many words
FALLBACK_K = 3

VOWELS = frozenset("aeiouy")

# Latin diphthongs
LAT_DIPHTHONGS = frozenset({"ae", "au", "oe", "eu", "ai"})

# English multi-vowel monophthongs
ENG_MONOPHTHONGS = frozenset({"ee", "oo", "ou", "oa", "ay", "ey", "oy"})

# words where "ui" is one vowel sound: cui, huic, cuius
UI_DIPHTHONGS = frozenset({
    "cuius", "huius", "cui", "huic", "alicuius", "alicui", "cuiusdam", "cuidam",
    "cuiusque", "cuique", "cuiusvis", "cuivis", "cuiuslibet", "cuilibet",
    "cuiuscumque", "cuicumque", "cuiusquam", "cuiquam", "uniuscuiusque", "unicuique",
})

# a plosive followed by a liquid/nasal starts a syllable together: "pa|tris", "a|gros"
PLOSIVES = frozenset("pbtdcg")
LIQUIDS_NASALS = frozenset("lmnr")

# capitalized Roman numerals ("III", "XIV"); not split into syllables
ROMAN_NUMERAL = re.compile(r"[IVXLCDM]{2,}")

def base_char(c):
    """
    Lowercase letter without diacritics: "Ā" -> "a".
    """
    return unicodedata.normalize("NFD", c)[0].lower()

def phon_units(word):
    """
    Split a word into vowel (V) and consonant (C) units, each as (kind, base, start, end)
    with start/end offsets into the word. "h" is skipped: it never affects where a
    syllable splits ("A|chil|les", "pul|cher").
    """
    b = "".join(base_char(c) for c in word)
    ui_at = -1
    if b in UI_DIPHTHONGS:
        ui_at = b.find("ui")

    units, i, n = [], 0, len(b)
    while i < n:
        c = b[i]
        nxt = b[i + 1] if i + 1 < n else ""
        after = b[i + 2] if i + 2 < n else ""

        if i == ui_at:
            units.append(("V", "ui", i, i + 2))
            i += 2
        # ignore h and non-alphabetic characters
        elif c == "h" or not ("a" <= c <= "z"):
            i += 1 
        # let qu be one consonant 
        elif c == "q" and nxt == "u":
            units.append(("C", "qu", i, i + 2))
            i += 2
        # let gu be one consonant
        elif c == "g" and nxt == "u" and after in VOWELS and i > 0 and b[i - 1] == "n":
            units.append(("C", "gu", i, i + 2))
            i += 2
        # handle consonantal i between vowels (e.g. maior)
        elif c == "i" and i > 0 and b[i - 1] in VOWELS and nxt in VOWELS:
            units.append(("C", "i", i, i + 1))
            i += 1
        # handle word-initial i before vowels pronounced as a consonant (e.g. iulius)
        elif c == "i" and i == 0 and nxt in VOWELS and nxt != "i":
            units.append(("C", "i", i, i + 1))
            i += 1
        # diphthong, unless its "i" is a consonant before another vowel ("ma|ior")
        elif c in VOWELS and c + nxt in LAT_DIPHTHONGS | ENG_MONOPHTHONGS and not (nxt == "i" and after in VOWELS):
            units.append(("V", c + nxt, i, i + 2))
            i += 2
        elif c in VOWELS:
            units.append(("V", c, i, i + 1))
            i += 1
        else:
            units.append(("C", c, i, i + 1))
            i += 1
    return units

def onset_size(cluster):
    """
    How many of the consonants between two vowels start the next syllable
    """
    # plosive + liquid pair
    if len(cluster) >= 2 and cluster[-2][1] in PLOSIVES and cluster[-1][1] in LIQUIDS_NASALS:
        return 2
    return min(1, len(cluster))

def syllable_starts(word):
    """
    Find where each syllable starts
    """
    if ROMAN_NUMERAL.fullmatch(word):
        return []
    units = phon_units(word)
    vowels = [k for k, u in enumerate(units) if u[0] == "V"]
    starts = []
    for prev, cur in zip(vowels, vowels[1:]):
        cluster = units[prev + 1:cur]
        size = onset_size(cluster)
        if size:
            starts.append(cluster[-size][2])
        else:
            starts.append(units[prev][3])
    return starts

def words_in(text, start=0, end=None):
    """
    (start, end) offsets of each run of letters in text[start:end].
    """
    end = len(text) if end is None else end
    spans, i = [], start
    while i < end:
        if text[i].isalpha():
            j = i
            while j < end and (text[j].isalpha() or unicodedata.combining(text[j])):
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans

def buzzpoints(record):
    """
    Sorted offsets where the model may buzz on this question.
    """
    q = record["question"]
    points = {len(q.rstrip())}

    if record["clues"]:
        for clue in record["clues"]:
            for ws, we in words_in(q, clue["start"], clue["end"]):
                points.add(we)
                points.update(ws + s for s in syllable_starts(q[ws:we]))
    else:
        words = words_in(q)
        points.update(we for k, (_, we) in enumerate(words, 1) if k % FALLBACK_K == 0)

    end = len(q.rstrip())
    return sorted(p for p in points if p > 0 and (p == end or any(c.isalnum() for c in q[p:end])))

def show(question, points):
    """
    Return the question with "|" at every buzzpoint for verification
    """
    out, last = [], 0
    for p in points:
        out.append(question[last:p])
        out.append("|")
        last = p
    out.append(question[last:])
    return "".join(out)

def main():
    records = json.load(open(DATA / "tossups_clean.json"))
    out = {r["id"]: buzzpoints(r) for r in records}
    json.dump(out, open(DATA / "buzzpoints.json", "w"))

    counts = sorted(len(p) for p in out.values())
    n = len(counts)
    print(f"buzzpoints per question: median {counts[n // 2]}, "
          f"10th pct {counts[n // 10]}, 90th pct {counts[n * 9 // 10]}, max {counts[-1]}")
    print(f"total buzzpoints: {sum(counts)}")

    first = sorted(out[r["id"]][0] / max(1, len(r["question"].rstrip())) for r in records)
    print(f"first buzzpoint at fraction of question: median {first[n // 2]:.2f}")

    rng = random.Random(0)
    print("\nsamples:")
    for r in rng.sample(records, 6):
        print(f"  {show(r['question'], out[r['id']])}\n")

if __name__ == "__main__":
    main()
