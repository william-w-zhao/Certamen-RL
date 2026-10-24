# Certamen Bot

## Reproducing the clean data and train/val/test split

Requires Python 3.10+ (no other packages; `pytest` only for tests).

1. Put `tossups.json` and `tossups_tokenized.json` in `data/`.
2. Run the cleaning pipeline:

   ```bash
   python3 scripts/preprocess.py
   ```

   This reads `tossups_tokenized.json`, drops broken records (pre-2008 sets, OCR garbage, mojibake, merged questions/bonuses, visual questions, multiple-choice questions missing their options, unplayable answers, two hand-checked broken sets), parses each answer line into `answer_parsed`, and writes `tossups_clean.json` and `tossups_dropped.json`. It prints how many records were dropped for each reason.

3. Make the train/val/test split:

   ```bash
   python3 scripts/split.py
   ```

   This reads `tossups_clean.json` and writes `splits.json` with the question IDs in each split. Splits are by year: 2025 sets are test, 2024 sets are val, everything earlier is train (`TEST_YEARS` and `VAL_YEARS` in the script). A train question that is a near-verbatim copy of a val/test question is removed and listed under `removed`. The split is deterministic, so rerunning it on the same clean data gives the same `splits.json`.

4. Compute the buzzpoints (where the model may buzz):

   ```bash
   python3 scripts/buzzpoints.py
   ```

   This reads `tossups_clean.json` and writes `buzzpoints.json`: for each question ID, the character offsets where the model gets a chance to buzz. That's after every syllable inside a clue span plus the end of the question, or every `FALLBACK_K` (3) words for questions without clues.

## Other scripts

- `python3 scripts/audit.py` prints a read-only report on the raw data (categories, clues, duplicates, answer formats).
- `python3 scripts/answers.py` runs the answer-line parser over the clean data and prints stats and samples.
- `python3 scripts/matcher.py` self-checks the answer matcher on the clean data: each question's own accepted/prompt/rejected answers, swapped answers from other questions, and look-alike answers (~20s).
- `python3 -m pytest tests/` runs the answer-parser and matcher tests.
