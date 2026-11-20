"""Two continual-learning streams built only from local material (no downloads).

Purpose
    skills: a sequence of eras, each teaching new algorithmic skills as character-level
            text such as "rev:abc=cba;". Answers can be scored exactly.
    text:   a sequence of real text domains found on this machine (Python source, C
            headers, Unix manual pages, a dictionary word list, question-answer JSON).

Docs
    NumPy: https://numpy.org/doc/stable/

Sample input
    stream = skills_stream(); batch = stream[0].sample(4, np.random.default_rng(0))

Expected output
    batch.tokens: (4, 96) int array of ASCII codes; batch.out_mask marks answer characters;
    batch.problem gives a problem id per position (-1 outside complete problems).
"""
import glob
import os
import sysconfig
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

VOCAB = 128
BOS = 10  # newline starts every sequence
LETTERS = "abcdefghijklmnop"


@dataclass
class Batch:
    tokens: np.ndarray
    out_mask: Optional[np.ndarray] = None
    problem: Optional[np.ndarray] = None


@dataclass
class Era:
    name: str
    sample: Callable[[int, np.random.Generator], Batch]
    sample_val: Callable[[int, np.random.Generator], Batch]
    has_answers: bool


def _word(rng: np.random.Generator, lo: int, hi: int) -> str:
    return "".join(rng.choice(list(LETTERS), size=int(rng.integers(lo, hi + 1))))


def solve(task: str, inp: str) -> str:
    shift = lambda s: "".join(LETTERS[(LETTERS.index(c) + 1) % len(LETTERS)] for c in s)
    if task == "cpy":
        return inp
    if task == "rev":
        return inp[::-1]
    if task == "srt":
        return "".join(sorted(inp))
    if task == "dup":
        return "".join(c + c for c in inp)
    if task == "inc":
        return shift(inp)
    if task == "dsc":
        return "".join(sorted(inp, reverse=True))
    if task == "sri":
        return shift("".join(sorted(inp)))
    if task == "add":
        a, b = inp.split("+")
        return str(int(a) + int(b))
    raise ValueError(task)


def make_problem(task: str, rng: np.random.Generator) -> str:
    if task == "add":
        inp = f"{int(rng.integers(0, 100))}+{int(rng.integers(0, 100))}"
    elif task == "dup":
        inp = _word(rng, 2, 4)
    else:
        inp = _word(rng, 3, 8)
    return f"{task}:{inp}={solve(task, inp)};"


def skills_batch(tasks: List[str], n: int, length: int, rng: np.random.Generator) -> Batch:
    toks = np.zeros((n, length), np.int32)
    mask = np.zeros((n, length), bool)
    prob = np.full((n, length), -1, np.int32)
    pid = 0
    for row in range(n):
        text, spans = "\n", []
        while len(text) < length:
            p = make_problem(str(rng.choice(tasks)), rng)
            start = len(text) + p.index("=") + 1
            spans.append((start, len(text) + len(p)))
            text += p
        toks[row] = np.frombuffer(text[:length].encode("ascii"), np.uint8)
        for a, b in spans:
            if b <= length:
                mask[row, a:b] = True
                prob[row, a:b] = pid
                pid += 1
    return Batch(toks, mask, prob)


def skills_stream(length: int = 96) -> List[Era]:
    plan = [("copy+reverse", ["cpy", "rev"]), ("sort", ["srt"]), ("shift+double", ["inc", "dup"]),
            ("addition", ["add"]), ("compositions", ["dsc", "sri"]), ("mixed revisit: reverse+sort", ["rev", "srt"]),
            ("exact repeat: copy+reverse", ["cpy", "rev"])]
    eras = []
    for name, tasks in plan:
        fn = (lambda t: lambda n, rng: skills_batch(t, n, length, rng))(tasks)
        eras.append(Era(name, fn, fn, True))
    return eras


def _ascii(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8).astype(np.int32)
    arr[arr >= VOCAB] = ord("?")
    return arr


def _read_many(paths: List[str], limit: int) -> bytes:
    out, size = [], 0
    for p in sorted(paths):
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError:
            continue
        if data[:2] == b"\x1f\x8b":
            continue
        out.append(data)
        size += len(data)
        if size >= limit:
            break
    return b"\n".join(out)[:limit]


def load_text_domains(limit: int = 2_000_000) -> Dict[str, np.ndarray]:
    here = os.path.dirname(os.path.abspath(__file__))
    sources = {
        "python": glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")),
        "c-headers": glob.glob("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/*.h"),
        "manual-pages": glob.glob("/usr/share/man/man1/*.1"),
        "word-list": ["/usr/share/dict/words"],
        "qa-json": [os.path.join(here, "..", "adaptive-model-1a", "data", "reasoning_50k", "reasoning_data.jsonl")],
    }
    domains = {}
    for name, paths in sources.items():
        raw = _read_many(paths, limit)
        if len(raw) >= 200_000:
            domains[name] = _ascii(raw)
    return domains


def text_stream(length: int = 128, order: Optional[List[str]] = None) -> List[Era]:
    domains = load_text_domains()
    order = order or ["python", "manual-pages", "word-list", "c-headers", "qa-json", "python"]
    eras = []
    seen = set()
    for name in [o for o in order if o in domains]:
        data = domains[name]
        label = f"exact repeat: {name}" if name in seen else name
        seen.add(name)
        cut = int(len(data) * 0.95)

        def windows(arr: np.ndarray) -> Callable[[int, np.random.Generator], Batch]:
            def fn(n: int, rng: np.random.Generator) -> Batch:
                starts = rng.integers(0, len(arr) - length, size=n)
                return Batch(np.stack([arr[s:s + length] for s in starts]))
            return fn

        eras.append(Era(label, windows(data[:cut]), windows(data[cut:]), False))
    return eras


if __name__ == "__main__":
    import sys

    failures, total_tests = [], 0
    rng = np.random.default_rng(0)

    total_tests += 1
    cases = {("rev", "abc"): "cba", ("srt", "cab"): "abc", ("dup", "ab"): "aabb", ("inc", "ap"): "ba",
             ("dsc", "bca"): "cba", ("sri", "cab"): "bcd", ("add", "27+45"): "72", ("cpy", "kf"): "kf"}
    for (task, inp), want in cases.items():
        if solve(task, inp) != want:
            failures.append(f"solve({task},{inp}) = {solve(task, inp)}, expected {want}")

    total_tests += 1
    b = skills_stream()[0].sample(8, rng)
    text = bytes(b.tokens[0].astype(np.uint8)).decode()
    if b.tokens.shape != (8, 96) or not text.startswith("\n") or b.tokens.max() >= VOCAB:
        failures.append(f"bad skills batch: shape {b.tokens.shape}, text {text[:20]!r}")

    total_tests += 1  # every masked span is exactly the answer plus ';'
    ok = True
    for pid in np.unique(b.problem[b.problem >= 0]):
        row, cols = np.where(b.problem == pid)
        row_text = bytes(b.tokens[row[0]].astype(np.uint8)).decode()
        seg = row_text[: cols[-1] + 1]
        task, rest = seg[seg.rfind(":", 0, cols[0]) - 3: seg.rfind(":", 0, cols[0])], seg[seg.rfind(":", 0, cols[0]) + 1:]
        inp, out = rest.split("=")
        ok &= out == solve(task, inp) + ";"
    if not ok:
        failures.append("answer mask does not line up with the answers")

    total_tests += 1
    eras = text_stream()
    if len(eras) < 3:
        failures.append(f"only {len(eras)} local text domains found")
    else:
        tb = eras[0].sample(4, rng)
        if tb.tokens.shape != (4, 128) or tb.tokens.max() >= VOCAB:
            failures.append("bad text batch")
    print("text domains:", [e.name for e in eras])

    if failures:
        print(f"VALIDATION FAILED - {len(failures)} of {total_tests} tests failed:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"VALIDATION PASSED - all {total_tests} tests produced expected results")
