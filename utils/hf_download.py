"""HuggingFace Hub download helpers.

We support specifying adapter/model artifacts either as:
- a local path (directory)
- a Hub repo id (e.g. "org/repo") optionally with revision "org/repo@main"

This is used for pulling online QLoRA adapters for benchmarking.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple


_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*(?:@[^\s]+)?$")

# In this repo we frequently pass *workspace-relative* paths like
# "outputs/..." or "datasets/...". Those look like HF repo IDs ("org/repo")
# but should be treated as local paths.
_LOCAL_PREFIXES = (
    "outputs/",
    "output/",
    "cache/",
    "datasets/",
    "configs/",
    "scripts/",
    "utils/",
    "tex/",
    "logs/",
    "benchmark_results/",
    "offload/",
)

# First path component roots we consider "definitely local" when given as a
# workspace-relative path (even if the path does not exist yet).
_LOCAL_ROOTS = tuple(p.rstrip("/") for p in _LOCAL_PREFIXES)


def _normalize_user_path(v: str) -> str:
    """Normalize a user-provided artifact locator (path or repo id)."""
    s = str(v).strip().replace("\\", "/")
    # Strip leading './' repeatedly (common in configs/CLI).
    while s.startswith("./"):
        s = s[2:]
    return s


def is_probably_hf_repo_id(value: str) -> bool:
    """Heuristic: decide if `value` is a HuggingFace Hub repo id.

    We intentionally bias towards treating ambiguous strings as *local* so we
    don't accidentally attempt Hub downloads for workspace paths like
    `outputs/lora_adapters`.
    """
    v_raw = str(value).strip()
    if not v_raw:
        return False

    v_norm = _normalize_user_path(v_raw)

    # Treat common workspace-relative prefixes as local paths even if they don't
    # exist yet (e.g., will be created by a previous pipeline stage).
    for p in _LOCAL_PREFIXES:
        if v_norm.startswith(p):
            return False

    # Also treat "<local_root>/<...>" as local even if the prefix list changes.
    # Example: "outputs/lora_adapters".
    first = v_norm.split("/", 1)[0] if "/" in v_norm else v_norm
    if first in _LOCAL_ROOTS:
        return False

    # Clearly local-ish patterns.
    if v_raw.startswith(("/", "./", "../")):
        return False
    if ":\\" in v_raw or v_raw.startswith("\\"):
        return False

    # If it exists locally, treat as local.
    if os.path.exists(v_raw) or os.path.exists(v_norm):
        return False

    return bool(_REPO_RE.match(v_norm))


def split_repo_and_revision(repo: str) -> Tuple[str, Optional[str]]:
    """Split `org/repo@rev` into (`org/repo`, `rev`)."""
    if "@" not in repo:
        return repo, None
    base, rev = repo.split("@", 1)
    base = base.strip()
    rev = rev.strip() or None
    return base, rev


def resolve_path_or_hf_repo(
    value: str,
    *,
    cache_dir: Optional[str],
    logger,
) -> str:
    """Resolve a local path, or download a Hub repo id to a local snapshot.

    Returns a local filesystem path.
    """
    v = str(value).strip()
    if not v:
        raise ValueError("Empty path/repo")

    if os.path.exists(v):
        return v

    if not is_probably_hf_repo_id(v):
        # Not a local path and not a repo id; return as-is to preserve existing behavior.
        return v

    repo_id, rev = split_repo_and_revision(v)
    logger.info(f"Downloading from HuggingFace Hub: {repo_id} (revision={rev or 'default'})")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise ImportError(
            "huggingface_hub is required to download online adapters. Install `huggingface_hub`.\n" + str(e)
        )

    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=rev,
        cache_dir=cache_dir,
        local_dir=None,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    logger.info(f"Downloaded {repo_id} to: {local_dir}")
    return str(local_dir)
