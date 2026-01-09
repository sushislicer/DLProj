"""FlashAttention utilities.

This repo can optionally use FlashAttention2 via HuggingFace's
`attn_implementation="flash_attention_2"`.

Installing `flash-attn` is environment-specific (CUDA/torch/compute capability),
so we keep it optional and attempt installation only when explicitly enabled.
"""

from __future__ import annotations

import os
import sys
import subprocess
from typing import Optional


def _try_import_flash_attn() -> tuple[bool, str]:
    try:
        import flash_attn  # noqa: F401

        return True, "ok"
    except Exception as e:
        return False, str(e)


def _flash_attn_supported_by_gpu() -> tuple[bool, str]:
    """Best-effort compatibility check.

FlashAttention2 generally requires NVIDIA Ampere (SM80) or newer.
"""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA not available"
        if torch.cuda.device_count() <= 0:
            return False, "no visible CUDA devices"

        props = torch.cuda.get_device_properties(0)
        if getattr(props, "major", 0) < 8:
            return False, f"compute capability {props.major}.{props.minor} < 8.0"

        # Also ensure the installed torch build actually supports the GPU arch.
        # Newer GPUs (e.g., SM120) may require a newer/nightly PyTorch wheel.
        try:
            arch = f"sm_{props.major}{props.minor}"
            arch_list = set(getattr(torch.cuda, "get_arch_list", lambda: [])())
            if arch_list and arch not in arch_list:
                return False, f"PyTorch wheel does not include {arch} (has {sorted(arch_list)})"
        except Exception:
            # If we can't query arch list, don't block.
            pass

        return True, f"cc={props.major}.{props.minor}"
    except Exception as e:
        return False, f"could not check GPU: {e}"


def ensure_flash_attn2(
    *,
    logger,
    auto_install: bool,
    pip_timeout_s: int = 1800,
) -> bool:
    """Ensure `flash_attn` is importable.

    Returns True if `flash_attn` is available after this call.

    Notes:
    - If `auto_install` is False, this function will not attempt installation.
    - Installation can take several minutes because it may compile CUDA code.
    """
    ok, reason = _try_import_flash_attn()
    if ok:
        return True

    if not auto_install:
        logger.info(f"flash_attn not available ({reason}); auto-install disabled")
        return False

    supported, why = _flash_attn_supported_by_gpu()
    if not supported:
        logger.info(f"flash_attn not available ({reason}); skipping install because {why}")
        return False

    # Avoid repeated install attempts in the same process.
    if os.environ.get("_QWEN_BENCH_FLASH_ATTN_INSTALL_ATTEMPTED") == "1":
        return False
    os.environ["_QWEN_BENCH_FLASH_ATTN_INSTALL_ATTEMPTED"] = "1"

    pkg_spec = os.environ.get("FLASH_ATTN_PIP_SPEC", "flash-attn")
    logger.info(f"Attempting to install FlashAttention2 ({pkg_spec}) ...")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "-U",
        pkg_spec,
    ]

    try:
        p = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(pip_timeout_s),
        )
        logger.info(p.stdout[-4000:] if p.stdout else "")
        if p.returncode != 0:
            logger.warning(f"FlashAttention install failed with code {p.returncode}.")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"FlashAttention install timed out after {pip_timeout_s}s")
        return False
    except Exception as e:
        logger.warning(f"FlashAttention install attempt errored: {e}")
        return False

    ok, reason = _try_import_flash_attn()
    if ok:
        logger.info("flash_attn successfully installed")
        return True
    logger.warning(f"flash_attn still not importable after install ({reason})")
    return False


def pick_attn_implementation(*, logger, prefer_flash2: bool, auto_install: bool) -> Optional[str]:
    """Return HF `attn_implementation` value to use, or None for default."""
    if not prefer_flash2:
        return None

    ok = ensure_flash_attn2(logger=logger, auto_install=auto_install)
    if not ok:
        return None
    return "flash_attention_2"
