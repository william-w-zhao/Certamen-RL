"""
Answer Matcher: is a model's response correct for a question's parsed answer line?

    vocab = build_vocab(records)  # once, from the clean data
    judge("Hera", parse_answer("Hera"), "Mythology", vocab)  -> "correct"
    judge("scipio", parse_answer("Scipio Aemilianus [prompt on “Scipio”]"), "Roman History", vocab) -> "prompt"

Verdicts:
  correct  matches an accepted answer
  prompt   matches a "prompt on" answer (the moderator would ask for more)
  wrong    matches a rejected answer, or nothing
  unsure   long/multi answer line (translations, word-meaning pairs) with no exact
           match; exact matching can't decide these, the LLM judge will
"""

import difflib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

# letters that NFKD doesn't split into base letter + accent
LIGATURES = {"æ": "ae", "œ": "oe", "ß": "ss", "ø": "o", "ð": "d", "þ": "th"}

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}
ORDINAL_WORDS = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
}

ARTICLES = {"the"}

# words whose K must stay a K: changing it makes a different answer that's also in the data
# (Keres, Greek death spirits, aren't Ceres; Nike, the goddess, isn't "nice", from scio)
KEEP_K = {"keres", "nike"}

# accepted spellings of the same name, matched at the start of a word so endings carry over
# ("Virgil" = "Vergil", "Virgilius" = "Vergilius"); both are real words, so fuzzy matching can't help
SPELLING_VARIANTS = {"virgil": "vergil"}

# fuzzy matching is per word: how similar a misspelled word must be, and how long
# ("Agamemnom" vs "Agamemnon" is 0.89; "Pollux" vs "Pollio" is 0.67)
FUZZY_THRESHOLD = 0.85
FUZZY_MIN_LENGTH = 5

# no fuzzy matching here: one letter changes a Latin form ("amat" vs "amet")
NO_FUZZY_CATEGORIES = {"Language"}

def norm(text):
    """
    Reduce an answer to a comparable form: lowercase, no macrons/accents, no punctuation,
    no leading article, "&" as "and", number words as digits, j -> i and k -> c,
    known spelling variants ("Virgil" -> "Vergil"), eras as "44 bc" / "79 ad".
    """
    text = text.lower()
    for ligature, plain in LIGATURES.items():
        text = text.replace(ligature, plain)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # "&" means "and" ("Colō & Collīnus" = "Colō and Collīnus")
    text = text.replace("&", " and ")

    # erase periods and apostrophes
    text = re.sub(r"[.'’‘`]", "", text)
    text = re.sub(r"[^a-z0-9α-ω]+", " ", text)

    words = text.split()
    words = [str(NUMBER_WORDS.get(w, w)) for w in words]
    words = [ORDINAL_WORDS.get(w, w) for w in words]
    if len(words) > 1 and words[0] in ARTICLES:
        words = words[1:]
    # spelling variants: Latin has no J ("Iuno" = "Juno"), and K is the Greek spelling
    # of a Latin C ("Herakles" = "Heracles", "Kalends" = "Calends")
    words = [w.replace("j", "i") for w in words]
    words = [w if w in KEEP_K else w.replace("k", "c") for w in words]
    for variant, standard in SPELLING_VARIANTS.items():
        words = [standard + w[len(variant):] if w.startswith(variant) else w for w in words]
    text = " ".join(words)

    # handle dates
    text = re.sub(r"\b(\d+) bce\b", r"\1 bc", text)
    text = re.sub(r"\b(\d+) ce\b", r"\1 ad", text)
    text = re.sub(r"\b(?:ad|ce) (\d+)\b", r"\1 ad", text)
    return text

def equal(a, b):
    """
    Check if equal after normalizing, also ignoring spaces ("mountain-top" vs "mountaintop").
    """
    return a == b or a.replace(" ", "") == b.replace(" ", "")

def build_vocab(records):
    """
    Create a vocabulary of normalized words from tossup texts, so we can differentiate a simple spelling error from
    an actual wrong answer (e.g. Agamennon is clearly trying to be Agamemnon, but Marcus is not the same as Marcius)
    """
    vocab = set()
    for r in records:
        vocab.update(norm(r["question"]).split())
        for a in r["answer_parsed"]["accept"]:
            vocab.update(norm(a).split())
    return frozenset(vocab)

def judge(response, parsed, category, vocab):
    """
    Verdict for one response: "correct", "prompt", "wrong" or "unsure."
    """
    r = norm(response)
    if not r:
        return "wrong"

    accept = [norm(a) for a in parsed["accept"]]
    prompt = [norm(p) for p in parsed["prompt"]]
    reject = [norm(x) for x in parsed["reject"]]

    # check for explicit rejects first
    if any(equal(r, x) for x in reject):
        return "wrong"
    # check then for correct answers
    if any(equal(r, a) for a in accept):
        return "correct"
    # check for prompts
    if any(equal(r, p) for p in prompt):
        return "prompt"

    # if a longer answer that needs more judging
    if parsed["kind"] != "short":
        return "unsure"

    # if category matching has to be exact and no previous case triggers, return wrong
    if category in NO_FUZZY_CATEGORIES:
        return "wrong"
    # check for close answers
    if any(close_enough(r, x, vocab) for x in reject):
        return "wrong"
    if any(close_enough(r, a, vocab) for a in accept):
        return "correct"
    return "wrong"

def close_enough(response, answer, vocab):
    """
    Same words in the same order, allowing per word:
      - an initial in the answer line for a full name ("p cornelius" ~ "publius cornelius")
      - a spelling slip in a long word, if the slip isn't itself a known word
    """
    rw, aw = response.split(), answer.split()
    if len(rw) != len(aw):
        return False
    # for initial-only answer
    allow_initials = len(aw) > 1 and rw[-1] == aw[-1]
    return all(word_close(r, a, vocab, allow_initials) for r, a in zip(rw, aw))

def word_close(r, a, vocab, allow_initials=False):
    """
    Is response word r the same as answer word a, or a safe misspelling of it?
    """
    if r == a:
        return True
    # check for abbreviated praenomina
    if allow_initials and len(a) == 1 and a.isalpha() and r.startswith(a):
        return True
    # check for wrong answers that are superficially close (e.g. dates)
    if len(a) < FUZZY_MIN_LENGTH or not a.isalpha() or not r.isalpha() or r in vocab:
        return False
    # return if the similarity ratio is above our threshold
    return ratio(r, a) >= FUZZY_THRESHOLD or swapped_letters(r, a)

def swapped_letters(r, a):
    """
    Check for neighboring letter swaps
    """
    diff = [i for i in range(len(a)) if len(r) == len(a) and r[i] != a[i]]
    return len(diff) == 2 and diff[1] == diff[0] + 1 and r[diff[0]] == a[diff[1]] and r[diff[1]] == a[diff[0]]

def ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

if __name__ == "__main__":
    data = Path(__file__).resolve().parent.parent / "data" / "tossups_clean.json"
    records = json.load(open(data))
    vocab = build_vocab(records)

    # check that each question's answer is correct evaluated
    expected = {"accept": "correct", "prompt": "prompt", "reject": "wrong"}
    failures = Counter()
    examples = []
    for r in records:
        p = r["answer_parsed"]
        for field, verdict in expected.items():
            for item in p[field]:
                got = judge(item, p, r["category"], vocab)
                if got != verdict:
                    # record where our answer failed
                    failures[(field, got)] += 1
                    if len(examples) < 10:
                        examples.append((r["answer"], field, item, got))
    print("1. own answer items that got the wrong verdict:")
    for (field, got), n in failures.most_common():
        print(f"  {n:6d}  {field} item -> {got}")
    for raw, field, item, got in examples:
        print(f"    {raw[:70]!r}: {field} {item!r} -> {got}")

    # check for false positives by giving randomized answers in the same question category
    rng = random.Random(42)
    by_category = {}
    for r in records:
        by_category.setdefault(r["category"], []).append(r)
    trials, false_accepts = 0, []
    for r in records:
        # only check questions with short answers, so longer LLM judging is not required
        if r["answer_parsed"]["kind"] != "short":
            continue
        other = rng.choice(by_category[r["category"]])
        response = norm(other["answer_parsed"]["main"])
        # skip if the randomly chosen answer happens to be the same as the question's actual answer
        if other["id"] == r["id"] or any(equal(response, norm(a)) for a in r["answer_parsed"]["accept"]):
            continue
        trials += 1
        # check for false positive
        if judge(other["answer_parsed"]["main"], r["answer_parsed"], r["category"], vocab) == "correct":
            false_accepts.append((other["answer_parsed"]["main"], r["answer"]))
    print(f"\n2. swapped answers judged correct: {len(false_accepts)} of {trials} "
          f"({len(false_accepts) / trials:.3%})")
    for response, answer_line in false_accepts[:15]:
        print(f"    {response!r} accepted for {answer_line[:70]!r}")

    # check for accepts by answers that do exist in the vocabulary that look very similar
    mains = sorted({norm(r["answer_parsed"]["main"]) for r in records
                    if r["answer_parsed"]["kind"] == "short"})
    fuzzy_ok = [r for r in records
                if r["answer_parsed"]["kind"] == "short" and r["category"] not in NO_FUZZY_CATEGORIES]
    trials, false_accepts = 0, []
    # test on 1000 random questions
    for r in rng.sample(fuzzy_ok, 1000):
        accept = [norm(a) for a in r["answer_parsed"]["accept"]]
        # get answers that look the most similar
        for lookalike in difflib.get_close_matches(norm(r["answer_parsed"]["main"]), mains, n=6, cutoff=0.6):
            # skip if the answers are just equal
            if any(equal(lookalike, a) for a in accept):
                continue
            trials += 1
            # check for false positive
            if judge(lookalike, r["answer_parsed"], r["category"], vocab) == "correct":
                false_accepts.append((lookalike, r["answer"]))
    print(f"\n3. look-alike answers judged correct: {len(false_accepts)} of {trials} "
          f"(check by hand: some are the same person written differently)")
    for response, answer_line in false_accepts[:15]:
        print(f"    {response!r} accepted for {answer_line[:70]!r}")
