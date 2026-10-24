"""
Tests for the syllabifier and buzzpoints. Run:  python3 -m pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from buzzpoints import buzzpoints, show, syllable_starts

def split(word):
    """
    "Agamemnon" -> "A|ga|mem|non"
    """
    s = [0] + syllable_starts(word) + [len(word)]
    return "|".join(word[a:b] for a, b in zip(s, s[1:]))

# --- syllables ---

def test_single_consonant_starts_next_syllable():
    assert split("Agamemnon") == "A|ga|mem|non"

def test_plosive_plus_liquid_stay_together():
    assert split("patris") == "pa|tris"

def test_h_is_skipped():
    assert split("Achilles") == "A|chil|les"
    assert split("Abraham") == "A|bra|ham"

def test_diphthongs():
    assert split("Caesar") == "Cae|sar"
    assert split("praetorian") == "prae|to|ri|an"
    assert split("cuius") == "cui|us"

def test_consonantal_i():
    assert split("maior") == "ma|ior"
    assert split("Troia") == "Tro|ia"
    assert split("iam") == "iam"

def test_qu_and_gu():
    assert split("aqua") == "a|qua"
    assert split("lingua") == "lin|gua"
    assert split("Augustus") == "Au|gus|tus"

def test_macrons_dont_change_splits():
    assert split("Fīliae") == "Fī|li|ae"
    assert split("Hēraclēs") == "Hē|ra|clēs"

def test_y_is_a_vowel():
    assert split("Tyre") == "Ty|re"
    assert split("rhythm") == "rhythm"

# --- buzzpoints ---

def record(question, clues):
    """
    A minimal record; clues given as substrings of the question.
    """
    spans = []
    for text in clues:
        start = question.index(text)
        spans.append({"text": text, "start": start, "end": start + len(text)})
    return {"question": question, "clues": spans}

def test_points_only_inside_clues_plus_end():
    r = record("Who killed the Minotaur in Crete?", ["killed the Minotaur"])
    assert show(r["question"], buzzpoints(r)) == "Who kil|led| the| Mi|no|taur| in Crete?|"

def test_no_clues_falls_back_to_every_k_words():
    r = record("one two three four five six seven eight?", [])
    assert show(r["question"], buzzpoints(r)) == "one two three| four five six| seven eight?|"

def test_no_point_right_before_final_punctuation():
    r = record("What is the meaning?", ["meaning"])
    assert show(r["question"], buzzpoints(r)) == "What is the me|a|ning?|"  # not "ning|?|"

def test_offsets_are_exact():
    q = "Quis Rōmam condidit?"
    r = record(q, ["Rōmam condidit"])
    for p in buzzpoints(r):
        assert 0 < p <= len(q)

def test_roman_numerals_not_split():
    assert split("III") == "III"
    assert split("XIV") == "XIV"
