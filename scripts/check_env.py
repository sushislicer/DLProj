"""Quick environment sanity check for the SSH benchmark server.

Prints versions/availability for:
- torch + CUDA
- bitsandbytes
- flash-attn
- transformers

Run:
  python scripts/check_env.py
"""

from __future__ import annotations

import argparse


def _try_import(name: str):
    try:
        mod = __import__(name)
        return True, getattr(mod, "__version__", None)
    except Exception as e:
        return False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Environment sanity check")
    parser.add_argument(
        "--install-flash-attn",
        action="store_true",
        help="Attempt to install FlashAttention2 (flash-attn) if missing",
    )
    args = parser.parse_args()

    ok, v = _try_import("torch")
    print(f"torch: {ok} ({v})")
    if ok:
        import torch

        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        try:
            arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
            if arch_list:
                print(f"torch.cuda.get_arch_list: {arch_list}")
        except Exception as e:
            print(f"torch.cuda.get_arch_list: <error> ({e})")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {p.name} total_mem={p.total_memory/1024**3:.1f}GB cc={p.major}.{p.minor}")

        # Common transformers compatibility check.
        try:
            import torch.utils._pytree as _pytree

            has_reg = hasattr(_pytree, "register_pytree_node")
            print(f"torch.utils._pytree.register_pytree_node: {has_reg}")
            if not has_reg:
                print("NOTE: Some transformers versions require a newer torch pytree API.")
        except Exception as e:
            print(f"torch.utils._pytree check: <error> ({e})")

    if args.install_flash_attn:
        try:
            from utils.helpers import setup_logging
            from utils.flash_attention import ensure_flash_attn2

            logger = setup_logging(log_dir="logs/benchmark", log_file="check_env.log", logger_name="check_env")
            ensure_flash_attn2(logger=logger, auto_install=True)
        except Exception as e:
            print(f"flash_attn install attempt failed: {e}")

    for pkg in ["transformers", "accelerate", "bitsandbytes", "peft", "flash_attn"]:
        ok, v = _try_import(pkg)
        print(f"{pkg}: {ok} ({v})")

    print("\nNotes:")
    print("- `flash_attn` is optional. If not installed, the benchmark runner falls back automatically.")
    print("- If you want FlashAttention2, you typically need a CUDA toolchain present and a compatible flash-attn wheel/source build.")


if __name__ == "__main__":
    main()
