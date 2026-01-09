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
import types
import importlib.machinery
import importlib.util


def _try_import_flash_attn() -> tuple[bool, str]:
    try:
        import flash_attn  # noqa: F401

        # A partially-imported module (e.g., from a prior failed import) can end up
        # in sys.modules with __spec__ = None. Some Transformers checks call
        # importlib.util.find_spec('flash_attn') which then raises ValueError.
        if getattr(flash_attn, "__spec__", None) is None:
            return False, "spec_none"

        # If we previously injected a stub, treat it as unavailable.
        ver = str(getattr(flash_attn, "__version__", ""))
        if ver.endswith("-stub") or ver == "0.0.0-stub":
            return False, "stubbed"

        return True, "ok"
    except Exception as e:
        return False, str(e)


def patch_broken_flash_attn(*, logger=None) -> bool:
    """Patch around a broken `flash_attn` install.

    Some environments end up with an incompatible `flash-attn` wheel (wrong torch/CUDA),
    which makes *importing Transformers model code* fail because certain model
    implementations unconditionally import `flash_attn.*` when it is present.

    This function detects a failing `import flash_attn` and injects a minimal
    stub package into `sys.modules` so that Transformers can import, while we
    continue to run with standard attention (no FlashAttention kernels).

    Returns:
        True if a stub was installed (i.e., we patched), else False.
    """
    ok, reason = _try_import_flash_attn()
    if ok:
        return False

    # We patch in two cases:
    # 1) A real broken wheel (undefined symbols / missing .so, etc)
    # 2) A previously injected stub (so Transformers import-time checks don't error)
    reason_l = (reason or "").lower()
    if reason in ("stubbed", "spec_none"):
        pass
    else:
        triggers = (
            "undefined symbol",
            "cannot open shared object file",
            "version `glibc",
            "flash_attn_2_cuda",
            "__spec__ is none",
        )
        if not any(t in reason_l for t in triggers):
            return False

    if logger is not None:
        try:
            logger.warning(f"Detected broken flash_attn install ({reason}); stubbing it out so Transformers can import.")
        except Exception:
            pass

    # Transformers uses `importlib.util.find_spec('flash_attn')` to decide whether
    # to import FlashAttention code paths at *import time* for some models.
    # If a broken wheel is present, we must force that check to return "not available"
    # to prevent transformers from importing `flash_attn`.
    if not hasattr(importlib.util, "_qwen_find_spec_orig"):
        importlib.util._qwen_find_spec_orig = importlib.util.find_spec  # type: ignore[attr-defined]

        def _qwen_find_spec(name: str, package: str | None = None):
            if name == 'flash_attn' or name.startswith('flash_attn.'):
                return None
            return importlib.util._qwen_find_spec_orig(name, package)  # type: ignore[attr-defined]

        importlib.util.find_spec = _qwen_find_spec  # type: ignore[assignment]

    # Remove any partially-imported modules.
    for k in list(sys.modules.keys()):
        if k == 'flash_attn' or k.startswith('flash_attn.'):
            sys.modules.pop(k, None)

    def _not_available(*args, **kwargs):
        raise RuntimeError(
            "flash-attn is installed but broken/incompatible in this environment. "
            "Uninstall it (`pip uninstall flash-attn`) or reinstall a compatible build."
        )

    # Create a stub package and common submodules referenced by Transformers.
    # NOTE: Transformers checks `importlib.util.find_spec('flash_attn')`. If a module
    # exists in sys.modules but has `__spec__ is None`, Python raises
    # ValueError("flash_attn.__spec__ is None"). So we must set a real ModuleSpec.
    pkg = types.ModuleType('flash_attn')
    pkg.__version__ = '0.0.0-stub'
    pkg.__path__ = []  # mark as package
    pkg.__spec__ = importlib.machinery.ModuleSpec(name='flash_attn', loader=None, is_package=True)
    pkg.__spec__.submodule_search_locations = []

    bert_padding = types.ModuleType('flash_attn.bert_padding')
    bert_padding.index_first_axis = _not_available
    bert_padding.pad_input = _not_available
    bert_padding.unpad_input = _not_available
    bert_padding.__spec__ = importlib.machinery.ModuleSpec(name='flash_attn.bert_padding', loader=None, is_package=False)

    iface = types.ModuleType('flash_attn.flash_attn_interface')
    iface.flash_attn_func = _not_available
    iface.flash_attn_varlen_func = _not_available
    iface.flash_attn_with_kvcache = _not_available
    iface.__spec__ = importlib.machinery.ModuleSpec(name='flash_attn.flash_attn_interface', loader=None, is_package=False)

    sys.modules['flash_attn'] = pkg
    sys.modules['flash_attn.bert_padding'] = bert_padding
    sys.modules['flash_attn.flash_attn_interface'] = iface
    return True


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

    if reason == "stubbed":
        logger.info("flash_attn is stubbed out due to an incompatible/broken install; not using FlashAttention2")
        return False

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
