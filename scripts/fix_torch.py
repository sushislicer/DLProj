"""Install/validate a GPU-compatible PyTorch wheel.

Why this exists:
- New GPUs (e.g. RTX 5090 = SM120) can require newer PyTorch builds.
- `requirements.txt` cannot express the correct CUDA wheel index URL.

Behavior:
- If torch is missing, install it.
- If torch exists but does not include the active GPU arch in
  `torch.cuda.get_arch_list()`, print a clear warning.
- Installation source can be overridden via env vars.

Env overrides:
- TORCH_INDEX_URL: e.g. https://download.pytorch.org/whl/cu124
- TORCH_NIGHTLY_INDEX_URL: e.g. https://download.pytorch.org/whl/nightly/cu126
- TORCH_CHANNEL: stable|nightly (default: stable; auto-switches to nightly for RTX 50xx heuristics)
- TORCH_PACKAGES: space-separated packages (default: "torch")
- KEEP_TORCHVISION: set to 1/true to avoid auto-uninstalling a broken torchvision
"""

from __future__ import annotations

import os
import re
import argparse
import subprocess
import sys
from typing import Optional


def _run(cmd: list[str]) -> int:
    p = subprocess.run(cmd, check=False)
    return int(p.returncode)


def _nvidia_gpu_name() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return out.splitlines()[0].strip() if out else None
    except Exception:
        return None


def _nvidia_cuda_version() -> Optional[str]:
    """Return NVIDIA driver-reported CUDA version string (e.g., '12.8')."""
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
        m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def _should_use_nightly(gpu_name: Optional[str]) -> bool:
    # Heuristic: RTX 50xx generally implies Blackwell (SM120).
    if not gpu_name:
        return False
    return bool(re.search(r"\bRTX\s*50\d{2}\b|\b5090\b", gpu_name, re.IGNORECASE))


def _pip_install_torch(channel: str, *, force_reinstall: bool = False) -> int:
    # Default to *just* torch.
    # Installing torchvision/torchaudio is not required for text-only benchmarking,
    # and mismatched torchvision wheels commonly break transformers imports.
    pkgs = os.environ.get("TORCH_PACKAGES", "torch").split()
    # Pick a CUDA wheel index.
    # Default preference is based on NVIDIA driver-reported CUDA version.
    # RTX 50xx machines commonly report CUDA 12.8+.
    cuda_ver = _nvidia_cuda_version()
    default_stable = "https://download.pytorch.org/whl/cu124"
    default_nightly = "https://download.pytorch.org/whl/nightly/cu128"
    if cuda_ver:
        try:
            major, minor = cuda_ver.split(".")
            key = f"cu{int(major):d}{int(minor):d}"
            if key == "cu128":
                default_stable = "https://download.pytorch.org/whl/cu128"
                default_nightly = "https://download.pytorch.org/whl/nightly/cu128"
            elif key == "cu130":
                default_stable = "https://download.pytorch.org/whl/cu130"
                default_nightly = "https://download.pytorch.org/whl/nightly/cu130"
        except Exception:
            pass

    if channel == "nightly":
        index_url = os.environ.get("TORCH_NIGHTLY_INDEX_URL", default_nightly)
        extra = ["--pre"]
    else:
        index_url = os.environ.get("TORCH_INDEX_URL", default_stable)
        extra = []

    cmd = [sys.executable, "-m", "pip", "install", "-U"] + extra
    if force_reinstall:
        cmd += ["--force-reinstall", "--no-cache-dir"]
    cmd += ["--index-url", index_url] + pkgs
    print(f"[fix_torch] Installing torch from: {index_url} (channel={channel})")
    return _run(cmd)


def _validate_torch() -> int:
    try:
        import torch

        print(f"[fix_torch] torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
        arch_list = []
        try:
            arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
        except Exception as e:
            print(f"[fix_torch] torch.cuda.get_arch_list() error: {e}")
        if arch_list:
            print(f"[fix_torch] torch.cuda.get_arch_list()={arch_list}")

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            p = torch.cuda.get_device_properties(0)
            arch = f"sm_{p.major}{p.minor}"
            if arch_list and arch not in set(arch_list):
                print(
                    f"[fix_torch] WARNING: Active GPU is {p.name} ({arch}), but this torch wheel does not include {arch}."
                )
                print("[fix_torch] You likely need a newer/nightly torch build for this GPU.")
                return 2
        # If torchvision is present but broken, transformers model imports can fail
        # even for text-only models. Prefer removing it unless user opts out.
        keep_tv = str(os.environ.get("KEEP_TORCHVISION", "")).strip().lower() in ("1", "true", "yes")
        if not keep_tv:
            try:
                import torchvision  # noqa: F401

                # Try touching ops registration paths that often fail on mismatch.
                from torchvision import ops  # noqa: F401

            except Exception as e:
                msg = str(e)
                if "torchvision::nms" in msg or "operator torchvision::nms does not exist" in msg:
                    print("[fix_torch] Detected broken torchvision (missing nms op). Uninstalling torchvision to prevent transformers import failures...")
                    _run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"])
                    _run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"])
                else:
                    print(f"[fix_torch] torchvision import failed ({e}); leaving as-is")

        return 0
    except Exception as e:
        print(f"[fix_torch] torch import/validation failed: {e}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Install/validate a GPU-compatible PyTorch wheel")
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Force reinstall torch from the selected channel (also uses --no-cache-dir)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        choices=["stable", "nightly"],
        default=None,
        help="Override channel selection (stable/nightly)",
    )
    args = parser.parse_args()

    auto_remediate = str(os.environ.get("AUTO_REMEDIATE_TORCH", "1")).strip().lower() in ("1", "true", "yes")

    # If torch already works, just validate.
    if not args.reinstall:
        try:
            import torch  # noqa: F401

            rc = _validate_torch()
            # If GPU arch mismatch and we're allowed to auto-remediate, do it.
            if rc == 2 and auto_remediate:
                gpu_name = _nvidia_gpu_name()
                if _should_use_nightly(gpu_name):
                    print("[fix_torch] Auto-remediating by reinstalling torch from nightly (RTX 50xx detected)...")
                    rc2 = _pip_install_torch("nightly", force_reinstall=True)
                    if rc2 == 0:
                        rc = _validate_torch()
            sys.exit(rc)
        except Exception:
            pass

    gpu_name = _nvidia_gpu_name()
    channel = (args.channel or os.environ.get("TORCH_CHANNEL", "stable")).strip().lower()
    if channel not in ("stable", "nightly"):
        channel = "stable"
    if channel == "stable" and _should_use_nightly(gpu_name):
        channel = "nightly"

    if gpu_name:
        print(f"[fix_torch] Detected GPU: {gpu_name}")
    cuda_ver = _nvidia_cuda_version()
    if cuda_ver:
        print(f"[fix_torch] NVIDIA driver reports CUDA Version: {cuda_ver}")
    else:
        print("[fix_torch] No NVIDIA GPU detected via nvidia-smi; installing CPU/stable torch from default index.")

    rc = _pip_install_torch(channel, force_reinstall=bool(args.reinstall))
    if rc != 0:
        print(f"[fix_torch] pip install torch failed with code {rc}")
        sys.exit(rc)

    rc = _validate_torch()
    sys.exit(rc)


if __name__ == "__main__":
    main()
