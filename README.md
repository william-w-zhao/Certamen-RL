# Certamen Bot

## Reproducing the clean data

Requires Python 3.10+ (no other packages; `pytest` only for tests).

1. Put `tossups.json` and `tossups_tokenized.json` in `data/`.
2. Run the cleaning pipeline:

   ```bash
   python3 scripts/preprocess.py
   ```

   This reads `tossups_tokenized.json`, drops broken records (pre-2008 sets, OCR garbage, mojibake, merged questions/bonuses, visual questions, unplayable answers, two hand-checked broken sets), parses each answer line into `answer_parsed`, and writes `tossups_clean.json` and `tossups_dropped.json`. It prints how many records were dropped for each reason.

## Other scripts

- `python3 scripts/audit.py` prints a read-only report on the raw data (categories, clues, duplicates, answer formats).
- `python3 scripts/answers.py` runs the answer-line parser over the clean data and prints stats and samples.
- `python3 -m pytest tests/` runs the answer-parser tests.
