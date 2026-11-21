"""What every evaluation result records about the run that produced it: the prompt versions and
text hashes, the commit, and when. A result without this cannot be compared with another one."""

from __future__ import annotations

import shutil
import subprocess  # nosec B404  # fixed git commands only, no user input
from datetime import UTC, datetime

from regintel.prompts import prompt_versions


def git_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603  # nosec B603
            [git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        dirty = subprocess.run(  # noqa: S603  # nosec B603
            [git, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def run_metadata(prompts: dict[str, object] | None = None) -> dict[str, object]:
    """`prompts` overrides the local ones, e.g. with what a deployed API reported for itself."""
    return {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "prompts": prompts if prompts is not None else prompt_versions(),
    }
