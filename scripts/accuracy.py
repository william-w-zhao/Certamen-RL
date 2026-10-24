"""
Accuracy Test: how often does a base model know the answer, at each point in the question?

For each question in a split, the model sees the question up to the buzzpoints nearest
25%, 50%, 75% and 100% of the way through, and must answer (no buzz/wait decision yet).
Answers are scored with the matcher.

Run:
  python3 scripts/accuracy.py --backend oracle --limit 50          # no model: checks the pipeline
  python3 scripts/accuracy.py --backend mlx --model mlx-community/Qwen3-8B-4bit --limit 20
  python3 scripts/accuracy.py --backend vllm --model Qwen/Qwen3.6-27B                # GPU VM

Outputs:
  results/accuracy_<backend>_<model>_<split>.jsonl   one line per (question, position)
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from matcher import build_vocab, judge

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# where in the question to test, as a fraction of its length
FRACTIONS = (0.25, 0.5, 0.75, 1.0)

# example questions shown before each test question, to set the answer format
N_EXAMPLES = 3

# answers are short; stop generating after this many tokens
MAX_TOKENS = 32

SYSTEM = (
    "You are an expert Certamen player. Certamen is a quiz game about Latin, classical "
    "mythology, Roman history and culture, and Latin literature. Questions are read aloud, "
    "and you may only hear part of one: a question that stops early ends with \"…\", "
    "possibly in the middle of a word. Never continue, repeat or describe the question. "
    "Work out what the full question is most likely asking, and reply with your best guess "
    "at its answer: only the answer, with no explanation."
)

# where to cut the example questions, so the model sees how to answer an unfinished question
# (the last example is shown whole)
EXAMPLE_CUTS = (0.5, 0.75, 1.0)

def nearest_points(question, points):
    """
    For each fraction in FRACTIONS, the buzzpoint closest to that fraction of the question.
    Returns [(fraction, offset)].
    """
    length = max(1, len(question.rstrip()))
    return [(f, min(points, key=lambda p: abs(p / length - f))) for f in FRACTIONS]

def pick_examples(train, rng):
    """
    N_EXAMPLES short-answer train questions, one per category where possible.
    """
    by_category = defaultdict(list)
    for r in train:
        if r["answer_parsed"]["kind"] == "short" and len(r["question"]) < 200:
            by_category[r["category"]].append(r)
    categories = sorted(by_category)
    rng.shuffle(categories)
    return [rng.choice(by_category[c]) for c in categories[:N_EXAMPLES]]

def heard(question, offset):
    """
    What the player has heard at this offset: the start of the question, with "…" if it stops early.
    """
    prefix = question[:offset]
    return prefix if offset >= len(question.rstrip()) else prefix.rstrip() + "…"

def example_turns(examples, points):
    """
    The examples as (question heard, answer), cut at EXAMPLE_CUTS.
    """
    turns = []
    for ex, cut in zip(examples, EXAMPLE_CUTS):
        length = max(1, len(ex["question"].rstrip()))
        offset = min(points[ex["id"]], key=lambda p: abs(p / length - cut))
        turns.append((heard(ex["question"], offset), ex["answer_parsed"]["main"]))
    return turns

def messages_for(text, turns):
    """
    Chat messages: system prompt, the examples as earlier turns, then the question heard so far.
    """
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, a in turns:
        msgs.append({"role": "user", "content": f"Question: {q}"})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": f"Question: {text}"})
    return msgs

def clean_response(text):
    """
    The answer part of a model reply: first line, without "Answer:" or a trailing period.
    """
    text = text.strip().split("\n")[0]
    text = re.sub(r"^(answer|a)\s*:\s*", "", text, flags=re.I)
    return text.strip().rstrip(".")

# --- backends: each takes a list of (messages, record) and returns [(response, logprob)] ---

def oracle_backend(args):
    """
    No model: always answers with the question's own main answer. Every result should be
    "correct"; anything else is a bug in the data, parser, matcher or this script.
    """
    def generate(batch):
        return [(r["answer_parsed"]["main"], 0.0) for _, r in batch]
    return generate

def mlx_backend(args):
    """
    Apple Silicon, for developing on a Mac. Needs:  pip install mlx-lm
    """
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.0)  # greedy

    def generate(batch):
        out = []
        for msgs, _ in batch:
            prompt = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False)
            text, logprob = "", 0.0
            for step in stream_generate(model, tokenizer, prompt,
                                        max_tokens=MAX_TOKENS, sampler=sampler):
                text += step.text
                logprob += step.logprobs[step.token].item()
                if "\n" in text:
                    break
            out.append((text, logprob))
        return out
    return generate

def vllm_backend(args):
    """
    NVIDIA GPU (the cloud VMs). Needs:  pip install vllm
    """
    from vllm import LLM, SamplingParams

    # prompts are under ~1k tokens; the default (the model's full context) may not fit on a 24 GB L4
    llm = LLM(model=args.model, max_model_len=4096)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, stop=["\n"], logprobs=0)

    def generate(batch):
        outputs = llm.chat([m for m, _ in batch], params,
                           chat_template_kwargs={"enable_thinking": False})
        return [(o.outputs[0].text, o.outputs[0].cumulative_logprob) for o in outputs]
    return generate

BACKENDS = {"oracle": oracle_backend, "mlx": mlx_backend, "vllm": vllm_backend}

def report(rows):
    """
    Accuracy at each position, overall and per category. "unsure" (translations and other
    long answers, until the LLM judge exists) is left out of the accuracy and counted apart.
    """
    def line(label, group):
        cells = []
        for f in FRACTIONS:
            verdicts = Counter(r["verdict"] for r in group if r["fraction"] == f)
            decided = verdicts["correct"] + verdicts["wrong"] + verdicts["prompt"]
            acc = verdicts["correct"] / decided if decided else 0
            cells.append(f"{acc:6.1%}")
        unsure = sum(r["verdict"] == "unsure" for r in group) / max(1, len(group))
        print(f"  {label:18s}" + "".join(f"{c:>9s}" for c in cells) + f"   unsure {unsure:.0%}")

    print("\naccuracy (correct / decided) at each position:")
    print(f"  {'':18s}" + "".join(f"{f:>9.0%}" for f in FRACTIONS))
    line("all", rows)
    for category in sorted({r["category"] for r in rows}):
        line(category, [r for r in rows if r["category"] == category])
    print("\n(prompt counts as wrong for now)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=BACKENDS, default="oracle")
    parser.add_argument("--model", default="none")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, help="only the first N questions (for testing)")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    records = {r["id"]: r for r in json.load(open(DATA / "tossups_clean.json"))}
    splits = json.load(open(DATA / "splits.json"))
    points = json.load(open(DATA / "buzzpoints.json"))
    vocab = build_vocab(records.values())

    rng = random.Random(0)
    turns = example_turns(pick_examples([records[i] for i in splits["train"]], rng), points)
    questions = [records[i] for i in splits[args.split]][:args.limit]

    jobs = []
    for r in questions:
        for fraction, offset in nearest_points(r["question"], points[r["id"]]):
            jobs.append((messages_for(heard(r["question"], offset), turns), r, fraction, offset))

    generate = BACKENDS[args.backend](args)
    RESULTS.mkdir(exist_ok=True)
    name = args.model.split("/")[-1]
    path = RESULTS / f"accuracy_{args.backend}_{name}_{args.split}.jsonl"
    rows = []
    with open(path, "w") as f:
        for start in range(0, len(jobs), args.batch):
            batch = jobs[start:start + args.batch]
            replies = generate([(m, r) for m, r, _, _ in batch])
            for (_, r, fraction, offset), (raw, logprob) in zip(batch, replies):
                response = clean_response(raw)
                row = {
                    "id": r["id"], "category": r["category"], "level": r["level"],
                    "fraction": fraction, "offset": offset,
                    "actual_fraction": round(offset / max(1, len(r["question"].rstrip())), 3),
                    "response": response, "raw": raw, "logprob": logprob,
                    "verdict": judge(response, r["answer_parsed"], r["category"], vocab),
                    "answer": r["answer"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            print(f"  {min(start + args.batch, len(jobs))}/{len(jobs)}", end="\r")

    print(f"\n{len(questions)} questions, {len(rows)} answers -> {path.relative_to(ROOT)}")
    report(rows)

if __name__ == "__main__":
    main()
