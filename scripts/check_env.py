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


def _try_import(name: str):
    try:
        mod = __import__(name)
        return True, getattr(mod, "__version__", None)
    except Exception as e:
        return False, str(e)


def main() -> None:
    ok, v = _try_import("torch")
    print(f"torch: {ok} ({v})")
    if ok:
        import torch

        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {p.name} total_mem={p.total_memory/1024**3:.1f}GB cc={p.major}.{p.minor}")

    for pkg in ["transformers", "accelerate", "bitsandbytes", "peft", "flash_attn"]:
        ok, v = _try_import(pkg)
        print(f"{pkg}: {ok} ({v})")

    print("\nNotes:")
    print("- `flash_attn` is optional. If not installed, the benchmark runner falls back automatically.")
    print("- If you want FlashAttention2, you typically need a CUDA toolchain present and a compatible flash-attn wheel/source build.")


if __name__ == "__main__":
    main()

