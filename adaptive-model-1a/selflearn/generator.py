"""Candidate generation for self-learning cycles."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

from loguru import logger

from selflearn.benchmark import render_prompt


class OpenAICompatibleTeacher:
    """Minimal OpenAI-compatible text generation client."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        self.base_url = base_url or os.environ.get("OPENAI_COMPAT_BASE_URL")
        self.model = model or os.environ.get("OPENAI_COMPAT_MODEL")

    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 256) -> str:
        if not self.available():
            raise RuntimeError("OpenAI-compatible teacher is not configured")
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def solve_candidate_tasks(
    tasks: List[Dict[str, Any]],
    backend,
    checkpoint_dir: str,
    tokenizer_path: Optional[str],
    student_samples_per_task: int = 4,
    teacher_samples_per_task: int = 1,
    teacher_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate student and optional teacher responses for candidate tasks."""
    teacher = OpenAICompatibleTeacher(model=teacher_model)
    if not teacher.available():
        logger.warning(
            "No OpenAI-compatible teacher configured; selflearn will fall back to "
            "built-in gold responses for deterministic tasks"
        )
    candidates: List[Dict[str, Any]] = []
    for task in tasks:
        prompt = render_prompt(task)
        student_outputs: List[str] = []
        for _ in range(student_samples_per_task):
            student_outputs.append(
                backend.run_inference(
                    checkpoint_dir=checkpoint_dir,
                    prompt=prompt,
                    tokenizer_path=tokenizer_path,
                    max_tokens=int(task.get("max_tokens", 128)),
                    temperature=float(task.get("temperature", 0.7)),
                    top_k=40,
                    top_p=0.92,
                    repetition_penalty=1.0,
                )
            )

        teacher_output = None
        if teacher.available():
            teacher_output = teacher.generate(prompt)
        elif "gold_response" in task:
            teacher_output = str(task["gold_response"])

        candidates.append(
            {
                **task,
                "student_outputs": student_outputs,
                "teacher_output": teacher_output,
            }
        )
    return candidates
