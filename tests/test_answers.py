"""Regression tests for the answer-line parser. Run:  python3 -m pytest tests/

Each case is a real answer line from the data. When you change a rule in answers.py,
add the answer line that motivated it here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from answers import parse_answer

def test_optional_parenthetical():
    p = parse_answer("(Publius) Valerius Publicola")
    assert p["main"] == "Valerius Publicola"
    assert set(p["accept"]) == {"Valerius Publicola", "Publius Valerius Publicola"}


def test_spaced_slash_separates_answers():
    p = parse_answer("Colosseum / Flavian Amphitheater")
    assert p["accept"] == ["Colosseum", "Flavian Amphitheater"]


def test_double_slash_same_as_slash():
    p = parse_answer("Augustus / Octavian // (Gaius) Octavius")
    assert set(p["accept"]) == {"Augustus", "Octavian", "Octavius", "Gaius Octavius"}


def test_tight_slash_swaps_one_word():
    p = parse_answer("(Publius) Terentius/Terence (Afer)")
    assert "Terence" in p["accept"]
    assert "Publius Terentius Afer" in p["accept"]


def test_braces_are_required_choice():
    p = parse_answer("(Lucius) Tarquinius {Superbus // The Proud}")
    assert "Tarquinius Superbus" in p["accept"]
    assert "Tarquinius The Proud" in p["accept"]
    assert "Tarquinius" not in p["accept"]


def test_prompt_and_reject():
    p = parse_answer('First Peloponnesian War [prompt on "Peloponnesian War,” '
                     'do not accept "2nd Peloponnesian War"]')
    assert p["accept"] == ["First Peloponnesian War"]
    assert p["prompt"] == ["Peloponnesian War"]
    assert p["reject"] == ["2nd Peloponnesian War"]


def test_prompt_drops_how_to_prompt():
    p = parse_answer('Donkey / Ass [prompt on “burro” by asking “what type of animal is a burro?”]')
    assert p["prompt"] == ["burro"]


def test_accept_equivalents_is_a_flag_not_an_answer():
    p = parse_answer("Modesty / Shame / Shyness / Bashfulness [Accept Equivalents]")
    assert "accept_equivalents" in p["flags"]
    assert "Equivalents" not in p["accept"]


def test_unbracketed_also_accept():
    p = parse_answer("Numidia (Also Accept Jugurtha)")
    assert p["accept"] == ["Numidia", "Jugurtha"]


def test_long_explanation_is_a_note():
    p = parse_answer("Impetus (all others have 1 meaning in the singular and a different meaning in the plural)")
    assert p["accept"] == ["Impetus"]
    assert p["notes"]


def test_question_text_leaked_into_answer():
    p = parse_answer("“Nobody?” A: Polyphemus")
    assert p["accept"] == ["Polyphemus"]
    assert "question_text_in_answer" in p["flags"]


def test_kinds():
    assert parse_answer("Laudo – Praise / Claudo - Close")["kind"] == "multi"
    assert parse_answer("I fear that Romulus was a king of barbarians.")["kind"] == "long"
    assert parse_answer("Student should stand and sing “Happy birthday to you”.")["kind"] == "action"
    assert parse_answer("See Below")["kind"] == "unusable"
    assert parse_answer("Quirinal, Viminal, Esquiline [do not read list of hills, see below]")["kind"] == "short"
