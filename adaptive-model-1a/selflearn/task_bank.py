"""Task generation for verified self-learning cycles."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List


GROUNDED_CONTEXTS = [
    (
        "Saturn is the sixth planet from the Sun. It is famous for its rings.",
        "Which planet is famous for its rings?",
        "Saturn",
    ),
    (
        "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        "Where is the Eiffel Tower located?",
        "Paris, France",
    ),
    (
        "Canberra is the capital city of Australia. Sydney is the largest city in Australia.",
        "What is the capital city of Australia?",
        "Canberra",
    ),
    (
        "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "At what temperature does water freeze in degrees Celsius?",
        "0 degrees Celsius",
    ),
]

INSTRUCTION_WORDS = [
    "amber", "lilac", "cobalt", "vulcan", "ember", "topaz", "saffron", "helios",
]


def bucket_failures(benchmark_result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for result in benchmark_result.get("results", []):
        if not result.get("passed", False):
            buckets.setdefault(str(result.get("task_type", "instruction")), []).append(result)
    return buckets


def _code_task(rng: random.Random, idx: int) -> Dict[str, Any]:
    variant = idx % 4
    if variant == 0:
        prompt = (
            "Write a Python function named solution(xs) that returns a new list "
            "containing only the even numbers from xs, in the same order."
        )
        gold_response = "def solution(xs):\n    return [x for x in xs if x % 2 == 0]\n"
        tests = [
            "assert solution([1, 2, 3, 4]) == [2, 4]",
            "assert solution([1, 3, 5]) == []",
        ]
        target = "filter even numbers"
        bucket = "code_filter_even"
    elif variant == 1:
        prompt = (
            "Write a Python function named solution(s) that returns the number "
            "of vowels in s."
        )
        gold_response = (
            "def solution(s):\n"
            "    return sum(1 for ch in s.lower() if ch in 'aeiou')\n"
        )
        tests = [
            "assert solution('hello') == 2",
            "assert solution('sky') == 0",
        ]
        target = "count vowels"
        bucket = "code_count_vowels"
    elif variant == 2:
        prompt = (
            "Write a Python function named solution(xs) that returns the sum of "
            "all numbers in xs."
        )
        gold_response = "def solution(xs):\n    return sum(xs)\n"
        tests = [
            "assert solution([1, 2, 3]) == 6",
            "assert solution([]) == 0",
        ]
        target = "sum list"
        bucket = "code_sum_list"
    else:
        prompt = (
            "Write a Python function named solution(x) that returns True if x "
            "is divisible by 3, otherwise False."
        )
        gold_response = "def solution(x):\n    return x % 3 == 0\n"
        tests = [
            "assert solution(9) is True",
            "assert solution(10) is False",
        ]
        target = "divisible by three"
        bucket = "code_divisible_by_three"
    return {
        "id": f"code_{idx}",
        "task_type": "code",
        "prompt": prompt,
        "target": target,
        "gold_response": gold_response,
        "tests": tests,
        "benchmark_bucket": bucket,
    }


def _math_task(rng: random.Random, idx: int) -> Dict[str, Any]:
    a = rng.randint(10, 99)
    b = rng.randint(10, 99)
    return {
        "id": f"math_{idx}",
        "task_type": "math",
        "prompt": f"What is {a} + {b}? Reply with the number only.",
        "target": str(a + b),
        "gold_response": str(a + b),
        "benchmark_bucket": "math_arithmetic",
    }


def _grounded_task(rng: random.Random, idx: int) -> Dict[str, Any]:
    context, prompt, target = GROUNDED_CONTEXTS[idx % len(GROUNDED_CONTEXTS)]
    return {
        "id": f"grounded_{idx}",
        "task_type": "grounded_qa",
        "context": context,
        "prompt": prompt,
        "target": target,
        "gold_response": target,
        "benchmark_bucket": "grounded_fact",
    }


def _instruction_task(rng: random.Random, idx: int) -> Dict[str, Any]:
    word = INSTRUCTION_WORDS[(idx + rng.randint(0, len(INSTRUCTION_WORDS) - 1)) % len(INSTRUCTION_WORDS)]
    return {
        "id": f"instruction_{idx}",
        "task_type": "instruction",
        "prompt": f"Reply with exactly this word in uppercase: {word}",
        "target": word.upper(),
        "gold_response": word.upper(),
        "benchmark_bucket": "format_following",
    }


def generate_candidate_tasks(
    failure_buckets: Dict[str, List[Dict[str, Any]]],
    count: int,
    seed: int = 17,
) -> List[Dict[str, Any]]:
    """Generate deterministic new tasks, biased by benchmark failure buckets."""
    rng = random.Random(seed)
    preferred = list(failure_buckets) or ["code", "math", "grounded_qa", "instruction"]
    tasks: List[Dict[str, Any]] = []
    generators = {
        "code": _code_task,
        "math": _math_task,
        "grounded_qa": _grounded_task,
        "instruction": _instruction_task,
    }
    type_counts = {name: 0 for name in generators}
    for idx in range(count):
        task_type = preferred[idx % len(preferred)]
        task_type = task_type if task_type in generators else "instruction"
        local_idx = type_counts[task_type]
        tasks.append(generators[task_type](rng, local_idx))
        type_counts[task_type] += 1
    return tasks


def seed_from_cycle_id(cycle_id: str) -> int:
    """Derive a stable per-cycle seed so each cycle generates a different task set."""
    digest = hashlib.sha256(cycle_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
