"""Workspace isolation and allowlist enforcement for autoresearch."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


PROGRAM_REQUIRED_SECTIONS = [
    "Mission",
    "In-Scope Files",
    "Protected Files",
    "Backends",
    "Experiment Profiles",
    "Code Research Loop",
    "Self-Learning Loop",
    "Promotion Rules",
    "Logging Rules",
    "Never Do",
]


ALLOWED_PATTERNS = [
    "mlx_train.py",
    "mlx_model.py",
    "mlx_plasticity.py",
    "mlx_adaptive_feedforward.py",
    "mlx_data.py",
    "mlx_inference.py",
    "monitor_training.py",
    "config.py",
    "autoresearch/",
    "selflearn/",
    "data/benchmarks/",
    "run_autoresearch.py",
    "run_selflearn.py",
]


PROTECTED_PATTERNS = [
    "program.md",
    "runs/",
    "tokenizer/",
    "data/selflearn/accepted/",
    "data/benchmarks/",
]


def parse_program_sections(path: str) -> List[str]:
    sections: List[str] = []
    if not os.path.exists(path):
        return sections
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("# "):
                sections.append(line[2:].strip())
    return sections


def validate_program_md(path: str) -> List[str]:
    found = set(parse_program_sections(path))
    return [section for section in PROGRAM_REQUIRED_SECTIONS if section not in found]


def _is_prefix_match(rel_path: str, pattern: str) -> bool:
    norm = rel_path.replace("\\", "/")
    if pattern.endswith("/"):
        return norm == pattern[:-1] or norm.startswith(pattern)
    return norm == pattern


def is_allowed_path(rel_path: str, patterns: Sequence[str] = ALLOWED_PATTERNS) -> bool:
    return any(_is_prefix_match(rel_path, pattern) for pattern in patterns)


def is_protected_path(rel_path: str, patterns: Sequence[str] = PROTECTED_PATTERNS) -> bool:
    return any(_is_prefix_match(rel_path, pattern) for pattern in patterns)


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def snapshot_tree(root: str) -> Dict[str, str]:
    """Return a content hash snapshot of repo-tracked files we care about."""
    snapshot: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "runs"}
        ]
        for fname in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fname), root)
            rel = rel.replace("\\", "/")
            if rel.startswith("runs/"):
                continue
            path = os.path.join(root, rel)
            snapshot[rel] = _file_hash(path)
    return snapshot


def diff_snapshots(before: Dict[str, str], after: Dict[str, str]) -> List[str]:
    changed = set(before) | set(after)
    return sorted([rel for rel in changed if before.get(rel) != after.get(rel)])


@dataclass
class Workspace:
    root: str
    baseline_snapshot: Dict[str, str]


class WorkspaceManager:
    """Creates isolated sandboxes for autoresearch experiments."""

    def __init__(self, repo_root: str, sandbox_root: str = "sandbox") -> None:
        self.repo_root = os.path.abspath(repo_root)
        self.sandbox_root = os.path.join(self.repo_root, sandbox_root)
        os.makedirs(self.sandbox_root, exist_ok=True)

    def create(self, tag: str) -> Workspace:
        workspace_root = os.path.join(self.sandbox_root, f"autoresearch_{tag}")
        if os.path.exists(workspace_root):
            shutil.rmtree(workspace_root)

        git_dir = os.path.join(self.repo_root, ".git")
        if os.path.isdir(git_dir):
            try:
                subprocess.run(
                    [
                        "git",
                        "worktree",
                        "add",
                        "--detach",
                        workspace_root,
                    ],
                    cwd=self.repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except Exception:
                self._copy_workspace(workspace_root)
        else:
            self._copy_workspace(workspace_root)

        return Workspace(root=workspace_root, baseline_snapshot=snapshot_tree(workspace_root))

    def _copy_workspace(self, workspace_root: str) -> None:
        shutil.copytree(
            self.repo_root,
            workspace_root,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "runs", "sandbox"
            ),
        )

    def validate_changes(self, workspace: Workspace) -> List[str]:
        changed = diff_snapshots(workspace.baseline_snapshot, snapshot_tree(workspace.root))
        disallowed = [
            rel for rel in changed
            if not is_allowed_path(rel) or is_protected_path(rel)
        ]
        return disallowed
