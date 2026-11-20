"""
Synthetic reasoning data generator for neuroplasticity engine E2E training.

Generates 50,000 template-based reasoning examples across 5 subjects
(coding 40%, math 20%, logic 15%, science 15%, general 10%) with
multi-step chain-of-thought reasoning. No LLM dependency — all examples
are procedurally generated from templates with randomized parameters.

Docs: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset

Sample input:
    python generate_reasoning_data.py

Expected output:
    Generated 50000 reasoning examples
    Distribution: coding=20000, math=10000, logic=7500, science=7500, general=5000
    Saved to data/reasoning_50k/
"""

import os
import random
import json
import sys
from typing import List, Dict, Tuple, Optional
from collections import Counter

# ---------------------------------------------------------------------------
# Coding templates (40% of dataset)
# ---------------------------------------------------------------------------

_CODING_FUNCTIONS = [
    ("sum_list", "nums: List[int]", "int", "Return the sum of all numbers in the list.",
     "total = 0\n    for n in nums:\n        total += n\n    return total"),
    ("reverse_string", "s: str", "str", "Return the reversed string.",
     "return s[::-1]"),
    ("find_max", "nums: List[int]", "int", "Return the maximum value in a non-empty list.",
     "result = nums[0]\n    for n in nums[1:]:\n        if n > result:\n            result = n\n    return result"),
    ("count_vowels", "s: str", "int", "Count the number of vowels in the string.",
     "return sum(1 for c in s.lower() if c in 'aeiou')"),
    ("is_palindrome", "s: str", "bool", "Check if a string is a palindrome (ignoring case).",
     "s = s.lower()\n    return s == s[::-1]"),
    ("flatten_list", "nested: List[List[int]]", "List[int]", "Flatten a list of lists into a single list.",
     "result = []\n    for sublist in nested:\n        result.extend(sublist)\n    return result"),
    ("remove_duplicates", "nums: List[int]", "List[int]", "Remove duplicates while preserving order.",
     "seen = set()\n    result = []\n    for n in nums:\n        if n not in seen:\n            seen.add(n)\n            result.append(n)\n    return result"),
    ("factorial", "n: int", "int", "Compute the factorial of a non-negative integer.",
     "if n <= 1:\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"),
    ("fibonacci", "n: int", "List[int]", "Return the first n Fibonacci numbers.",
     "if n <= 0:\n        return []\n    if n == 1:\n        return [0]\n    fibs = [0, 1]\n    for _ in range(2, n):\n        fibs.append(fibs[-1] + fibs[-2])\n    return fibs"),
    ("binary_search", "nums: List[int], target: int", "int",
     "Return the index of target in sorted list, or -1 if not found.",
     "lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if nums[mid] == target:\n            return mid\n        elif nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1"),
    ("merge_sorted", "a: List[int], b: List[int]", "List[int]",
     "Merge two sorted lists into one sorted list.",
     "result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result"),
    ("two_sum", "nums: List[int], target: int", "List[int]",
     "Return indices of two numbers that add up to target.",
     "seen = {}\n    for i, n in enumerate(nums):\n        complement = target - n\n        if complement in seen:\n            return [seen[complement], i]\n        seen[n] = i\n    return []"),
    ("rotate_list", "nums: List[int], k: int", "List[int]",
     "Rotate list to the right by k positions.",
     "if not nums:\n        return []\n    k = k % len(nums)\n    return nums[-k:] + nums[:-k]"),
    ("count_words", "text: str", "Dict[str, int]",
     "Count occurrences of each word in text.",
     "counts = {}\n    for word in text.lower().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts"),
    ("gcd", "a: int, b: int", "int",
     "Compute the greatest common divisor using Euclidean algorithm.",
     "while b:\n        a, b = b, a % b\n    return a"),
    ("matrix_transpose", "matrix: List[List[int]]", "List[List[int]]",
     "Return the transpose of a matrix.",
     "if not matrix:\n        return []\n    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]"),
    ("is_sorted", "nums: List[int]", "bool",
     "Check if a list is sorted in non-decreasing order.",
     "for i in range(len(nums) - 1):\n        if nums[i] > nums[i + 1]:\n            return False\n    return True"),
    ("chunk_list", "lst: List[int], size: int", "List[List[int]]",
     "Split a list into chunks of given size.",
     "return [lst[i:i + size] for i in range(0, len(lst), size)]"),
    ("intersection", "a: List[int], b: List[int]", "List[int]",
     "Return elements present in both lists (no duplicates).",
     "return list(set(a) & set(b))"),
    ("power", "base: int, exp: int", "int",
     "Compute base raised to the power of exp (exp >= 0).",
     "result = 1\n    for _ in range(exp):\n        result *= base\n    return result"),
]

_BUG_TYPES = [
    ("off_by_one", "range(len(nums))", "range(len(nums) - 1)",
     "The loop should stop one element early to avoid index out of bounds."),
    ("wrong_operator", ">=", ">",
     "The comparison should use > instead of >= to handle the boundary correctly."),
    ("missing_return", "    result.append(x)", "    result.append(x)\n    return result",
     "The function is missing a return statement at the end."),
    ("wrong_init", "result = 1", "result = 0",
     "The accumulator should start at 0 for addition, not 1."),
    ("swapped_args", "a - b", "b - a",
     "The operands are in the wrong order for the subtraction."),
]

_ALGO_PROBLEMS = [
    ("Find the {k}th smallest element in the list {nums}.",
     "Sort the list and return element at index k-1.",
     "sorted_nums = sorted(nums)\nresult = sorted_nums[k - 1]"),
    ("Check if string '{s}' contains all unique characters.",
     "Use a set to track seen characters.",
     "seen = set()\nfor c in s:\n    if c in seen:\n        return False\n    seen.add(c)\nreturn True"),
    ("Compute the running average of {nums}.",
     "Maintain a running sum and divide by count at each step.",
     "total = 0\nresult = []\nfor i, n in enumerate(nums):\n    total += n\n    result.append(total / (i + 1))"),
    ("Find all pairs in {nums} that sum to {target}.",
     "Use a hash set for O(n) lookup.",
     "seen = set()\npairs = []\nfor n in nums:\n    comp = target - n\n    if comp in seen:\n        pairs.append((comp, n))\n    seen.add(n)"),
    ("Count the number of inversions in {nums}.",
     "An inversion is a pair (i, j) where i < j but nums[i] > nums[j].",
     "count = 0\nfor i in range(len(nums)):\n    for j in range(i + 1, len(nums)):\n        if nums[i] > nums[j]:\n            count += 1"),
]


def _generate_coding_example(rng: random.Random) -> Dict[str, str]:
    """Generate one coding reasoning example."""
    example_type = rng.choice(["implement", "implement", "bugfix", "algorithm", "explain"])

    if example_type == "implement":
        name, params, ret, doc, body = rng.choice(_CODING_FUNCTIONS)
        instruction = (
            f"Implement the following Python function:\n\n"
            f"def {name}({params}) -> {ret}:\n"
            f'    """{doc}"""\n'
        )
        reasoning = (
            f"Step 1: Understand the requirement — {doc}\n"
            f"Step 2: Identify the input type ({params}) and return type ({ret}).\n"
            f"Step 3: Consider edge cases (empty input, single element, etc.).\n"
            f"Step 4: Write the implementation using a straightforward approach.\n"
            f"Step 5: Verify correctness by tracing through a small example."
        )
        response = f"def {name}({params}) -> {ret}:\n    \"\"\"{doc}\"\"\"\n    {body}"

    elif example_type == "bugfix":
        name, params, ret, doc, body = rng.choice(_CODING_FUNCTIONS)
        bug_name, original, replacement, explanation = rng.choice(_BUG_TYPES)
        buggy_body = body.replace(original, replacement) if original in body else body + "\n    # BUG: missing edge case handling"
        instruction = (
            f"Find and fix the bug in this Python function:\n\n"
            f"def {name}({params}) -> {ret}:\n"
            f'    """{doc}"""\n'
            f"    {buggy_body}"
        )
        reasoning = (
            f"Step 1: Read the function purpose — {doc}\n"
            f"Step 2: Trace through the code with a small input.\n"
            f"Step 3: Identify the issue — {explanation}\n"
            f"Step 4: The fix is to change '{replacement}' back to '{original}'.\n"
            f"Step 5: Verify the fixed version handles edge cases correctly."
        )
        response = f"The bug is: {explanation}\n\nFixed version:\ndef {name}({params}) -> {ret}:\n    \"\"\"{doc}\"\"\"\n    {body}"

    elif example_type == "algorithm":
        template, approach, code = rng.choice(_ALGO_PROBLEMS)
        nums = sorted(rng.sample(range(1, 100), rng.randint(5, 10)))
        k = rng.randint(1, len(nums))
        target = rng.choice(nums) + rng.choice(nums)
        s = ''.join(rng.sample('abcdefghijklmnop', rng.randint(4, 8)))
        instruction = template.format(nums=nums, k=k, target=target, s=s)
        reasoning = (
            f"Step 1: Understand the problem — {approach}\n"
            f"Step 2: Choose an appropriate data structure.\n"
            f"Step 3: Implement the algorithm step by step.\n"
            f"Step 4: The time complexity is reasonable for this input size.\n"
            f"Step 5: Trace through with the given input to verify."
        )
        response = f"Approach: {approach}\n\n```python\n{code}\n```"

    else:  # explain
        name, params, ret, doc, body = rng.choice(_CODING_FUNCTIONS)
        instruction = (
            f"Explain what this Python function does and how it works:\n\n"
            f"def {name}({params}) -> {ret}:\n    {body}"
        )
        reasoning = (
            f"Step 1: Read the function signature — {name} takes {params} and returns {ret}.\n"
            f"Step 2: {doc}\n"
            f"Step 3: Walk through the implementation line by line.\n"
            f"Step 4: Identify the algorithm pattern used.\n"
            f"Step 5: Consider the time and space complexity."
        )
        response = f"This function {doc.lower()} It takes {params} as input and returns {ret}."

    return {"instruction": instruction, "reasoning": reasoning, "response": response, "subject": "coding"}


# ---------------------------------------------------------------------------
# Math templates (20% of dataset)
# ---------------------------------------------------------------------------

def _generate_math_example(rng: random.Random) -> Dict[str, str]:
    """Generate one math reasoning example."""
    example_type = rng.choice(["arithmetic", "algebra", "word_problem", "series"])

    if example_type == "arithmetic":
        ops = ['+', '-', '*']
        nums = [rng.randint(1, 100) for _ in range(rng.randint(3, 6))]
        chosen_ops = [rng.choice(ops) for _ in range(len(nums) - 1)]
        expr_parts = [str(nums[0])]
        for op, n in zip(chosen_ops, nums[1:]):
            expr_parts.append(op)
            expr_parts.append(str(n))
        expr = ' '.join(expr_parts)
        # Compute result step by step
        result = nums[0]
        steps = [f"Start with {nums[0]}"]
        for op, n in zip(chosen_ops, nums[1:]):
            prev = result
            if op == '+':
                result += n
            elif op == '-':
                result -= n
            else:
                result *= n
            steps.append(f"{prev} {op} {n} = {result}")

        instruction = f"Calculate: {expr}"
        reasoning = '\n'.join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
        response = f"The answer is {result}."

    elif example_type == "algebra":
        a = rng.randint(2, 12)
        b = rng.randint(-20, 20)
        x_actual = rng.randint(-10, 10)
        c = a * x_actual + b
        instruction = f"Solve for x: {a}x + {b} = {c}"
        reasoning = (
            f"Step 1: Start with {a}x + {b} = {c}\n"
            f"Step 2: Subtract {b} from both sides: {a}x = {c} - {b} = {c - b}\n"
            f"Step 3: Divide both sides by {a}: x = {c - b} / {a} = {x_actual}\n"
            f"Step 4: Verify: {a} * {x_actual} + {b} = {a * x_actual + b} = {c}. Correct."
        )
        response = f"x = {x_actual}"

    elif example_type == "word_problem":
        templates = [
            (
                "A store has {a} apples. They receive {b} more boxes with {c} apples each. How many apples total?",
                lambda a, b, c: a + b * c,
                "Step 1: Start with {a} apples.\nStep 2: {b} boxes with {c} apples each = {bc} new apples.\nStep 3: Total = {a} + {bc} = {total}."
            ),
            (
                "A train travels at {a} mph for {b} hours, then {c} mph for {d} hours. What is the total distance?",
                lambda a, b, c, d: a * b + c * d,
                "Step 1: First leg: {a} mph * {b} hours = {ab} miles.\nStep 2: Second leg: {c} mph * {d} hours = {cd} miles.\nStep 3: Total = {ab} + {cd} = {total} miles."
            ),
            (
                "If {a} workers can build {b} widgets in {c} hours, how many widgets can {d} workers build in {e} hours?",
                lambda a, b, c, d, e: (b * d * e) // (a * c) if a * c != 0 else 0,
                "Step 1: Rate per worker per hour = {b} / ({a} * {c}) = {rate:.2f} widgets.\nStep 2: {d} workers for {e} hours = {d} * {e} * {rate:.2f} = {total} widgets."
            ),
        ]
        idx = rng.randint(0, len(templates) - 1)
        tmpl_q, fn, tmpl_r = templates[idx]
        if idx == 0:
            a, b, c = rng.randint(10, 50), rng.randint(2, 10), rng.randint(5, 20)
            total = fn(a, b, c)
            instruction = tmpl_q.format(a=a, b=b, c=c)
            reasoning = tmpl_r.format(a=a, b=b, c=c, bc=b*c, total=total)
        elif idx == 1:
            a, b, c, d = rng.randint(30, 80), rng.randint(1, 5), rng.randint(40, 100), rng.randint(1, 4)
            total = fn(a, b, c, d)
            instruction = tmpl_q.format(a=a, b=b, c=c, d=d)
            reasoning = tmpl_r.format(a=a, b=b, c=c, d=d, ab=a*b, cd=c*d, total=total)
        else:
            a, b, c = rng.randint(2, 8), rng.randint(10, 50), rng.randint(2, 8)
            d, e = rng.randint(2, 12), rng.randint(2, 8)
            rate = b / (a * c)
            total = int(d * e * rate)
            instruction = tmpl_q.format(a=a, b=b, c=c, d=d, e=e)
            reasoning = tmpl_r.format(a=a, b=b, c=c, d=d, e=e, rate=rate, total=total)
        response = f"The answer is {total}."

    else:  # series
        start = rng.randint(1, 10)
        step = rng.randint(1, 5)
        length = rng.randint(5, 8)
        series = [start + i * step for i in range(length)]
        missing_idx = rng.randint(1, length - 2)
        shown = [str(n) if i != missing_idx else '?' for i, n in enumerate(series)]
        instruction = f"Find the missing number in the series: {', '.join(shown)}"
        reasoning = (
            f"Step 1: Look at the differences between consecutive numbers.\n"
            f"Step 2: The pattern is +{step} each time.\n"
            f"Step 3: Before the gap: {series[missing_idx - 1]}\n"
            f"Step 4: After the gap: {series[missing_idx + 1]}\n"
            f"Step 5: The missing number is {series[missing_idx - 1]} + {step} = {series[missing_idx]}."
        )
        response = f"The missing number is {series[missing_idx]}."

    return {"instruction": instruction, "reasoning": reasoning, "response": response, "subject": "math"}


# ---------------------------------------------------------------------------
# Logic templates (15% of dataset)
# ---------------------------------------------------------------------------

_LOGIC_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
_LOGIC_ITEMS = ["cat", "dog", "fish", "bird", "car", "bike", "book", "pen"]
_LOGIC_COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]

def _generate_logic_example(rng: random.Random) -> Dict[str, str]:
    """Generate one logic reasoning example."""
    example_type = rng.choice(["syllogism", "deduction", "truth_table", "constraint"])

    if example_type == "syllogism":
        category_a = rng.choice(["mammals", "birds", "reptiles", "fish", "insects"])
        category_b = rng.choice(["animals", "living things", "organisms"])
        member = rng.choice(["a dog", "a cat", "a whale", "a bat", "an eagle"])
        instruction = (
            f"All {category_a} are {category_b}. "
            f"A {member.split()[-1]} is a {category_a.rstrip('s')}. "
            f"What can we conclude about {member}?"
        )
        reasoning = (
            f"Step 1: Premise 1 — All {category_a} are {category_b}.\n"
            f"Step 2: Premise 2 — {member.capitalize()} is a {category_a.rstrip('s')}.\n"
            f"Step 3: By syllogistic reasoning (if all A are B, and X is an A, then X is a B).\n"
            f"Step 4: Therefore, {member} is a {category_b.rstrip('s')}."
        )
        response = f"{member.capitalize()} is a {category_b.rstrip('s')}."

    elif example_type == "deduction":
        names = rng.sample(_LOGIC_NAMES, 3)
        items = rng.sample(_LOGIC_ITEMS, 3)
        instruction = (
            f"{names[0]} has a {items[0]}. "
            f"{names[1]} does not have a {items[0]} or a {items[1]}. "
            f"One of the three people ({names[0]}, {names[1]}, {names[2]}) has a {items[1]}. "
            f"Who has the {items[1]}?"
        )
        reasoning = (
            f"Step 1: {names[0]} has a {items[0]} (given).\n"
            f"Step 2: {names[1]} does not have a {items[0]} or a {items[1]} (given).\n"
            f"Step 3: Someone among the three has a {items[1]}.\n"
            f"Step 4: It's not {names[0]} (already has {items[0]}, assuming one item each).\n"
            f"Step 5: It's not {names[1]} (explicitly excluded).\n"
            f"Step 6: Therefore, {names[2]} has the {items[1]}."
        )
        response = f"{names[2]} has the {items[1]}."

    elif example_type == "truth_table":
        p_val = rng.choice([True, False])
        q_val = rng.choice([True, False])
        op = rng.choice(["AND", "OR", "IMPLIES"])
        if op == "AND":
            result = p_val and q_val
            op_explain = "P AND Q is True only when both P and Q are True"
        elif op == "OR":
            result = p_val or q_val
            op_explain = "P OR Q is True when at least one of P or Q is True"
        else:
            result = (not p_val) or q_val
            op_explain = "P IMPLIES Q is False only when P is True and Q is False"
        instruction = f"If P is {p_val} and Q is {q_val}, what is P {op} Q?"
        reasoning = (
            f"Step 1: P = {p_val}, Q = {q_val}.\n"
            f"Step 2: The rule for {op}: {op_explain}.\n"
            f"Step 3: Applying the rule: P {op} Q = {result}."
        )
        response = f"P {op} Q = {result}."

    else:  # constraint
        names = rng.sample(_LOGIC_NAMES, 4)
        colors = rng.sample(_LOGIC_COLORS, 4)
        instruction = (
            f"Four people ({', '.join(names)}) each wear a different color shirt "
            f"({', '.join(colors)}). "
            f"{names[0]} wears {colors[0]}. "
            f"{names[1]} does not wear {colors[1]} or {colors[2]}. "
            f"{names[2]} wears {colors[2]}. "
            f"What color does {names[1]} wear?"
        )
        remaining = [c for c in colors if c != colors[0] and c != colors[1] and c != colors[2]]
        answer = remaining[0] if remaining else colors[3]
        reasoning = (
            f"Step 1: {names[0]} wears {colors[0]} (given).\n"
            f"Step 2: {names[2]} wears {colors[2]} (given).\n"
            f"Step 3: {names[1]} cannot wear {colors[1]} or {colors[2]} (given).\n"
            f"Step 4: {names[1]} also cannot wear {colors[0]} (taken by {names[0]}).\n"
            f"Step 5: The only remaining color for {names[1]} is {answer}.\n"
            f"Step 6: {names[3]} wears the remaining color {colors[1]}."
        )
        response = f"{names[1]} wears {answer}."

    return {"instruction": instruction, "reasoning": reasoning, "response": response, "subject": "logic"}


# ---------------------------------------------------------------------------
# Science templates (15% of dataset)
# ---------------------------------------------------------------------------

_SCIENCE_TOPICS = [
    {
        "question": "What happens when you heat water to 100 degrees Celsius at sea level?",
        "reasoning": (
            "Step 1: Water is a liquid at room temperature.\n"
            "Step 2: As temperature increases, water molecules gain kinetic energy.\n"
            "Step 3: At 100°C (at standard atmospheric pressure), water reaches its boiling point.\n"
            "Step 4: At the boiling point, liquid water transitions to water vapor (steam).\n"
            "Step 5: This is a phase change from liquid to gas."
        ),
        "answer": "Water boils and changes from liquid to gas (steam) at 100°C at sea level."
    },
    {
        "question": "Why do objects fall to the ground when dropped?",
        "reasoning": (
            "Step 1: Earth has mass, which creates a gravitational field.\n"
            "Step 2: All objects with mass are attracted by gravity.\n"
            "Step 3: The gravitational acceleration near Earth's surface is approximately 9.8 m/s².\n"
            "Step 4: When an object is released, the gravitational force pulls it downward.\n"
            "Step 5: Air resistance acts against gravity but is typically small for dense objects."
        ),
        "answer": "Objects fall due to Earth's gravitational force, which accelerates them at 9.8 m/s² toward the surface."
    },
    {
        "question": "How does photosynthesis work in plants?",
        "reasoning": (
            "Step 1: Plants absorb sunlight through chlorophyll in their leaves.\n"
            "Step 2: They take in carbon dioxide (CO2) from the air through stomata.\n"
            "Step 3: Water (H2O) is absorbed through the roots.\n"
            "Step 4: Using light energy, plants convert CO2 and H2O into glucose (C6H12O6) and oxygen (O2).\n"
            "Step 5: The equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2."
        ),
        "answer": "Photosynthesis converts CO2 and H2O into glucose and oxygen using sunlight and chlorophyll."
    },
    {
        "question": "Why does ice float on water?",
        "reasoning": (
            "Step 1: Most substances are denser in solid form than liquid form.\n"
            "Step 2: Water is an exception — ice is less dense than liquid water.\n"
            "Step 3: When water freezes, hydrogen bonds form a crystalline structure with spaces.\n"
            "Step 4: This open structure makes ice about 9% less dense than liquid water.\n"
            "Step 5: Since ice is less dense, it floats on liquid water (Archimedes' principle)."
        ),
        "answer": "Ice floats because its crystalline structure makes it less dense than liquid water."
    },
    {
        "question": "What causes the seasons on Earth?",
        "reasoning": (
            "Step 1: Earth's axis is tilted at approximately 23.5 degrees.\n"
            "Step 2: As Earth orbits the Sun, different hemispheres tilt toward or away from the Sun.\n"
            "Step 3: When tilted toward the Sun, a hemisphere receives more direct sunlight (summer).\n"
            "Step 4: When tilted away, it receives less direct sunlight (winter).\n"
            "Step 5: The tilt, not distance from the Sun, is the primary cause of seasons."
        ),
        "answer": "Seasons are caused by Earth's 23.5° axial tilt as it orbits the Sun."
    },
    {
        "question": "How do antibiotics fight bacterial infections?",
        "reasoning": (
            "Step 1: Antibiotics are chemicals that target specific bacterial processes.\n"
            "Step 2: Some antibiotics (like penicillin) disrupt cell wall synthesis.\n"
            "Step 3: Others inhibit protein synthesis, blocking bacterial growth.\n"
            "Step 4: Some target DNA replication or metabolic pathways.\n"
            "Step 5: Antibiotics do not work against viruses, which use host cell machinery."
        ),
        "answer": "Antibiotics target specific bacterial processes like cell wall synthesis or protein production to kill or inhibit bacteria."
    },
    {
        "question": "Why is the sky blue?",
        "reasoning": (
            "Step 1: Sunlight contains all wavelengths of visible light.\n"
            "Step 2: As light enters the atmosphere, it interacts with gas molecules.\n"
            "Step 3: Shorter wavelengths (blue/violet) scatter more than longer wavelengths (red).\n"
            "Step 4: This is called Rayleigh scattering.\n"
            "Step 5: Our eyes are more sensitive to blue than violet, so we see the sky as blue."
        ),
        "answer": "The sky is blue because of Rayleigh scattering — short blue wavelengths scatter more in the atmosphere."
    },
    {
        "question": "What is the difference between speed and velocity?",
        "reasoning": (
            "Step 1: Speed is a scalar quantity — it only has magnitude.\n"
            "Step 2: Velocity is a vector quantity — it has both magnitude and direction.\n"
            "Step 3: A car going 60 mph has a speed of 60 mph regardless of direction.\n"
            "Step 4: A car going 60 mph north has a velocity of 60 mph north.\n"
            "Step 5: Two objects can have the same speed but different velocities if moving in different directions."
        ),
        "answer": "Speed is magnitude only (scalar), while velocity includes both magnitude and direction (vector)."
    },
]


def _generate_science_example(rng: random.Random) -> Dict[str, str]:
    """Generate one science reasoning example."""
    topic = rng.choice(_SCIENCE_TOPICS)
    # Add slight variation
    prefix = rng.choice([
        "Explain: ", "Answer this science question: ", "Think through: ", ""
    ])
    return {
        "instruction": prefix + topic["question"],
        "reasoning": topic["reasoning"],
        "response": topic["answer"],
        "subject": "science",
    }


# ---------------------------------------------------------------------------
# General reasoning templates (10% of dataset)
# ---------------------------------------------------------------------------

def _generate_general_example(rng: random.Random) -> Dict[str, str]:
    """Generate one general reasoning example."""
    example_type = rng.choice(["analogy", "pattern", "planning", "comparison"])

    if example_type == "analogy":
        analogies = [
            ("hot", "cold", "up", "down", "opposites"),
            ("cat", "kitten", "dog", "puppy", "parent to offspring"),
            ("book", "library", "car", "garage", "object to storage location"),
            ("pen", "writing", "hammer", "building", "tool to purpose"),
            ("tree", "forest", "star", "galaxy", "individual to collection"),
            ("water", "ocean", "sand", "desert", "element to landscape"),
            ("page", "book", "brick", "wall", "component to structure"),
            ("key", "lock", "password", "account", "access mechanism to secured item"),
        ]
        a, b, c, d, relation = rng.choice(analogies)
        instruction = f"{a} is to {b} as {c} is to ___?"
        reasoning = (
            f"Step 1: Identify the relationship between '{a}' and '{b}'.\n"
            f"Step 2: The relationship is: {relation}.\n"
            f"Step 3: Apply the same relationship to '{c}'.\n"
            f"Step 4: The answer that completes the analogy is '{d}'."
        )
        response = f"The answer is '{d}'. The relationship is {relation}."

    elif example_type == "pattern":
        # Number pattern with operations
        start = rng.randint(1, 10)
        op = rng.choice(["multiply_by_2", "add_increasing", "square"])
        if op == "multiply_by_2":
            series = [start * (2 ** i) for i in range(6)]
            rule = "each number is doubled"
        elif op == "add_increasing":
            series = [start]
            for i in range(1, 6):
                series.append(series[-1] + i)
            rule = "the difference increases by 1 each time"
        else:
            series = [(start + i) ** 2 for i in range(6)]
            rule = "each number is a perfect square of consecutive integers"
        shown = series[:5]
        instruction = f"What comes next in the pattern: {', '.join(map(str, shown))}, ___?"
        reasoning = (
            f"Step 1: Look at the differences: {[series[i+1] - series[i] for i in range(4)]}.\n"
            f"Step 2: The pattern is: {rule}.\n"
            f"Step 3: Applying the rule to find the next number.\n"
            f"Step 4: The next number is {series[5]}."
        )
        response = f"The next number is {series[5]}. The rule is: {rule}."

    elif example_type == "planning":
        tasks = [
            ("bake a cake", ["Preheat oven to 350°F", "Mix dry ingredients (flour, sugar, baking powder)",
                           "Combine wet ingredients (eggs, milk, butter)", "Mix wet and dry together",
                           "Pour into greased pan", "Bake for 30 minutes", "Let cool before serving"]),
            ("organize a meeting", ["Define the meeting objective", "Choose a date and time",
                                   "Create an agenda", "Send invitations", "Prepare materials",
                                   "Set up the room or video call", "Take notes during the meeting"]),
            ("debug a program", ["Reproduce the bug consistently", "Read the error message carefully",
                                "Check recent code changes", "Add logging or print statements",
                                "Isolate the problematic code section", "Fix the root cause",
                                "Write a test to prevent regression"]),
            ("learn a new programming language", ["Choose a language based on your goals",
                                                  "Set up the development environment",
                                                  "Learn basic syntax and data types",
                                                  "Practice with small exercises",
                                                  "Build a small project",
                                                  "Study the standard library",
                                                  "Contribute to an open source project"]),
        ]
        task_name, steps = rng.choice(tasks)
        instruction = f"Plan the steps to {task_name}."
        step_text = '\n'.join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
        reasoning = (
            f"Step 1: Break the task into sequential phases.\n"
            f"Step 2: Identify dependencies between steps.\n"
            f"Step 3: Order steps logically.\n"
            f"Here is the plan:\n{step_text}"
        )
        response = f"To {task_name}, follow these {len(steps)} steps:\n" + \
                   '\n'.join(f"{i+1}. {s}" for i, s in enumerate(steps))

    else:  # comparison
        comparisons = [
            ("list", "tuple", "Python",
             "Lists are mutable (can be changed after creation), while tuples are immutable.",
             "Use lists when you need to modify the collection; use tuples for fixed data."),
            ("stack", "queue", "data structures",
             "Stacks follow LIFO (Last In, First Out), while queues follow FIFO (First In, First Out).",
             "Use stacks for undo operations; use queues for task scheduling."),
            ("compiled", "interpreted", "programming languages",
             "Compiled languages are translated to machine code before execution. Interpreted languages are executed line by line.",
             "Compiled languages (C, Rust) are faster; interpreted languages (Python, JavaScript) are more flexible."),
            ("TCP", "UDP", "networking protocols",
             "TCP guarantees delivery and ordering. UDP is faster but does not guarantee delivery.",
             "Use TCP for reliable communication (web, email); use UDP for speed (gaming, streaming)."),
        ]
        a, b, domain, diff, use_case = rng.choice(comparisons)
        instruction = f"Compare {a} and {b} in {domain}."
        reasoning = (
            f"Step 1: Define {a} — it is a concept in {domain}.\n"
            f"Step 2: Define {b} — it is also a concept in {domain}.\n"
            f"Step 3: Key difference: {diff}\n"
            f"Step 4: Practical implication: {use_case}"
        )
        response = f"Comparison of {a} vs {b}: {diff} {use_case}"

    return {"instruction": instruction, "reasoning": reasoning, "response": response, "subject": "general"}


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

_SUBJECT_GENERATORS = {
    "coding": _generate_coding_example,
    "math": _generate_math_example,
    "logic": _generate_logic_example,
    "science": _generate_science_example,
    "general": _generate_general_example,
}

_SUBJECT_WEIGHTS = {
    "coding": 0.40,
    "math": 0.20,
    "logic": 0.15,
    "science": 0.15,
    "general": 0.10,
}


def generate_reasoning_dataset(
    num_examples: int = 50000,
    seed: int = 42,
    output_dir: str = "data/reasoning_50k",
) -> List[Dict[str, str]]:
    """Generate a synthetic reasoning dataset with the specified distribution.

    Args:
        num_examples: Total number of examples to generate.
        seed: Random seed for reproducibility.
        output_dir: Directory to save the dataset.

    Returns:
        List of dicts with keys: instruction, reasoning, response, subject.
    """
    rng = random.Random(seed)

    # Calculate counts per subject
    subjects_and_counts: List[Tuple[str, int]] = []
    remaining = num_examples
    for subject, weight in list(_SUBJECT_WEIGHTS.items())[:-1]:
        count = round(num_examples * weight)
        subjects_and_counts.append((subject, count))
        remaining -= count
    # Last subject gets the remainder to ensure exact total
    last_subject = list(_SUBJECT_WEIGHTS.keys())[-1]
    subjects_and_counts.append((last_subject, remaining))

    # Generate examples
    examples: List[Dict[str, str]] = []
    for subject, count in subjects_and_counts:
        gen_fn = _SUBJECT_GENERATORS[subject]
        for _ in range(count):
            examples.append(gen_fn(rng))

    # Shuffle so subjects are interleaved
    rng.shuffle(examples)

    # Save to disk as JSONL
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reasoning_data.jsonl")
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    return examples


if __name__ == "__main__":
    all_validation_failures = []
    total_tests = 0

    # Test 1: Generate a small sample and verify structure
    total_tests += 1
    try:
        sample = generate_reasoning_dataset(num_examples=100, seed=42, output_dir="/tmp/reasoning_test")
        assert len(sample) == 100, f"Expected 100 examples, got {len(sample)}"
        for ex in sample:
            for key in ['instruction', 'reasoning', 'response', 'subject']:
                assert key in ex, f"Missing key: {key}"
                assert isinstance(ex[key], str) and len(ex[key]) > 0, f"Empty {key}"
    except Exception as e:
        all_validation_failures.append(f"Structure validation: {e}")

    # Test 2: Verify subject distribution
    total_tests += 1
    try:
        counts = Counter(ex['subject'] for ex in sample)
        for subject in _SUBJECT_WEIGHTS:
            assert subject in counts, f"Missing subject: {subject}"
            expected = round(100 * _SUBJECT_WEIGHTS[subject])
            actual = counts[subject]
            # Allow +-5 tolerance for rounding
            assert abs(actual - expected) <= 5, \
                f"{subject}: expected ~{expected}, got {actual}"
    except Exception as e:
        all_validation_failures.append(f"Distribution validation: {e}")

    # Test 3: Verify coding examples have code-like content
    total_tests += 1
    try:
        coding_examples = [ex for ex in sample if ex['subject'] == 'coding']
        assert len(coding_examples) > 0, "No coding examples generated"
        has_code = any('def ' in ex['instruction'] or 'function' in ex['instruction'].lower()
                      for ex in coding_examples)
        assert has_code, "No coding examples contain function definitions"
    except Exception as e:
        all_validation_failures.append(f"Coding content validation: {e}")

    # Test 4: Verify JSONL output is readable
    total_tests += 1
    try:
        output_path = "/tmp/reasoning_test/reasoning_data.jsonl"
        loaded = []
        with open(output_path) as f:
            for line in f:
                loaded.append(json.loads(line))
        assert len(loaded) == 100, f"JSONL has {len(loaded)} lines, expected 100"
    except Exception as e:
        all_validation_failures.append(f"JSONL read validation: {e}")

    # Test 5: Generate full dataset
    total_tests += 1
    try:
        full_dataset = generate_reasoning_dataset(
            num_examples=50000, seed=42, output_dir="data/reasoning_50k"
        )
        assert len(full_dataset) == 50000, f"Expected 50000, got {len(full_dataset)}"
        full_counts = Counter(ex['subject'] for ex in full_dataset)
        print(f"Generated {len(full_dataset)} reasoning examples")
        print(f"Distribution: {dict(full_counts)}")
        print(f"Saved to data/reasoning_50k/reasoning_data.jsonl")

        # Print one sample per subject
        for subject in _SUBJECT_WEIGHTS:
            ex = next(e for e in full_dataset if e['subject'] == subject)
            print(f"\n--- Sample ({subject}) ---")
            print(f"Q: {ex['instruction'][:120]}...")
            print(f"R: {ex['reasoning'][:120]}...")
            print(f"A: {ex['response'][:120]}...")
    except Exception as e:
        all_validation_failures.append(f"Full dataset generation: {e}")

    if all_validation_failures:
        print(f"VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\nVALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)
