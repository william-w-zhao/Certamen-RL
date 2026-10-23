"""
Tests for the answer matcher. Run:  python3 -m pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from answers import parse_answer
from matcher import judge, norm

# known words for fuzzy tests, so they don't depend on the data file
VOCAB = frozenset({"marcius", "marius", "gaius", "ptolemy", "xii", "xiii", "pollux"})

# --- normalize ---

def test_macrons_and_case():
    assert norm("Duae Fīliae") == "duae filiae"

def test_ligatures():
    assert norm("Cæsar") == norm("Caesar")

def test_punctuation_and_quotes():
    assert norm("“Pholus”") == "pholus"
    assert norm("Ovid’s Heroides") == norm("ovids heroides")

def test_leading_article():
    assert norm("The Aeneid") == "aeneid"
    assert norm("The") == "the"  # don't empty a one-word answer

def test_numbers_and_eras():
    assert norm("44 B.C.") == norm("44 BC") == norm("44 BCE") == "44 bc"
    assert norm("A.D. 79") == norm("79 CE") == "79 ad"
    assert norm("Twelve") == "12"
    assert norm("Second Punic War") == norm("2nd Punic War")

def test_spelling_variants():
    assert norm("Iuno") == norm("Juno")
    assert norm("Iocasta") == norm("Jocasta")
    assert norm("Herakles") == norm("Heracles")
    assert norm("Kronos") == norm("Cronos")
    assert norm("Kalends") == norm("Calends")

def test_k_kept_where_it_changes_the_answer():
    assert norm("Keres") != norm("Ceres")
    assert norm("Nike") != norm("Nice")

# --- judge ---

def test_exact_and_optional_parts():
    p = parse_answer("(P. Cornelius) Scipio Africanus (prompt on “scipio”)")
    assert judge("Scipio Africanus", p, "Roman History", VOCAB) == "correct"
    assert judge("Scipio", p, "Roman History", VOCAB) == "prompt"

def test_initial_in_answer_line_matches_full_name():
    p = parse_answer("(P. Cornelius) Scipio Africanus")
    assert judge("Publius Cornelius Scipio Africanus", p, "Roman History", VOCAB) == "correct"

def test_reject_beats_accept():
    p = parse_answer('First Peloponnesian War [prompt on "Peloponnesian War,” '
                     'do not accept "2nd Peloponnesian War"]')
    assert judge("First Peloponnesian War", p, "Culture", VOCAB) == "correct"
    assert judge("1st peloponnesian war", p, "Culture", VOCAB) == "correct"
    assert judge("Peloponnesian War", p, "Culture", VOCAB) == "prompt"
    assert judge("Second Peloponnesian War", p, "Culture", VOCAB) == "wrong"

def test_fuzzy_spelling_slip():
    assert judge("Agamemnom", parse_answer("Agamemnon"), "Mythology", VOCAB) == "correct"
    assert judge("Gaius Julius Ceasar", parse_answer("Gaius Julius Caesar"), "Roman History", VOCAB) == "correct"

def test_no_fuzzy_on_language():
    assert judge("Amet", parse_answer("Amat"), "Language", VOCAB) == "wrong"

def test_no_fuzzy_on_short_or_numeric_answers():
    assert judge("Hara", parse_answer("Hera"), "Mythology", VOCAB) == "wrong"
    assert judge("146 bc", parse_answer("149 B.C."), "Roman History", VOCAB) == "wrong"

def test_a_real_word_is_not_a_typo():
    # "Marcius" is a different Roman, not a misspelling of "Marius"
    assert judge("Gaius Marcius", parse_answer("Gaius Marius"), "Roman History", VOCAB) == "wrong"
    assert judge("Ptolemy XII", parse_answer("Ptolemy XIII"), "Roman History", VOCAB) == "wrong"
    assert judge("Pollux", parse_answer("Pollio"), "Roman History", VOCAB) == "wrong"

def test_spelling_variants_are_correct_even_for_known_words():
    # both spellings are real words in the data, so fuzzy matching can't help; normalize does
    vocab = VOCAB | {"herakles", "heracles", "jocasta", "iocasta"}
    assert judge("Herakles", parse_answer("Heracles"), "Mythology", vocab) == "correct"
    assert judge("Jocasta", parse_answer("Iocasta"), "Mythology", vocab) == "correct"

def test_lone_initial_is_not_a_name():
    assert judge("Domitian’s", parse_answer("D.?"), "Roman History", VOCAB) == "wrong"

def test_greek():
    assert norm("Ἀλήθεια") == "αληθεια"
    assert judge("χρημάτων", parse_answer("χρημάτων"), "Language", VOCAB) == "correct"

def test_word_count_must_match():
    p = parse_answer("Marcus Aurelius Antoninus")
    assert judge("Marcus Aurelius", p, "Roman History", VOCAB) == "wrong"

def test_long_answers_are_unsure_unless_exact():
    p = parse_answer("I fear that Romulus was a king of barbarians.")
    assert p["kind"] == "long"
    assert judge("I fear that Romulus was a king of barbarians", p, "Language", VOCAB) == "correct"
    assert judge("I am afraid Romulus was king of the barbarians", p, "Language", VOCAB) == "unsure"

def test_empty_response():
    assert judge("", parse_answer("Hera"), "Mythology", VOCAB) == "wrong"
