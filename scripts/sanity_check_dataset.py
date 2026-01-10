"""Sanity-check dataset tokenization/labels for GaLore training.

This script is intended to debug the common failure mode where the training loss
is exactly 0.0 because all labels are set to -100 (ignored) due to empty text or
degenerate attention masks.

Usage:
  python3 scripts/sanity_check_dataset.py --model_path outputs/pipeline/0.5B/rotated_residual_model --dataset c4 --max_samples 32
"""

from __future__ import annotations

import argparse
import itertools
from typing import Any, Dict, List

from datasets import load_dataset
from transformers import AutoTokenizer


def _load_subset(dataset: str, split: str, max_samples: int, seed: int, streaming: bool) -> List[Dict[str, Any]]:
    if dataset.lower() == "c4":
        if streaming:
            ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
            try:
                ds = ds.shuffle(seed=seed, buffer_size=min(10_000, max(1, max_samples * 5)))
            except Exception:
                pass
            return list(itertools.islice(ds, max_samples))
        ds = load_dataset("allenai/c4", "en", split=split)
        return [ds[i] for i in range(min(max_samples, len(ds)))]

    ds = load_dataset(dataset, split=split)
    return [ds[i] for i in range(min(max_samples, len(ds)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Model/tokenizer path (local dir or HF repo id)")
    ap.add_argument("--dataset", type=str, default="c4")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--streaming", action="store_true", help="Use streaming mode (recommended for C4)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = _load_subset(args.dataset, args.split, args.max_samples, seed=args.seed, streaming=bool(args.streaming))
    texts = []
    for r in rows:
        t = r.get("text") if isinstance(r, dict) else None
        if t is None or not str(t).strip():
            t = tok.eos_token or tok.pad_token or "Hello"
        texts.append(str(t))

    batch = tok(texts, truncation=True, max_length=args.max_length, padding="max_length")
    input_ids = batch["input_ids"]
    attn = batch.get("attention_mask")
    if attn is None:
        attn = [[1] * len(ids) for ids in input_ids]

    labels = []
    for ids, m in zip(input_ids, attn):
        if sum(int(x) for x in m) == 0:
            m = [1] * len(ids)
        labels.append([tok_id if int(mask) == 1 else -100 for tok_id, mask in zip(ids, m)])

    # Summaries
    total = 0
    valid = 0
    for lab in labels:
        total += len(lab)
        valid += sum(1 for x in lab if int(x) != -100)

    print(f"Loaded {len(texts)} samples from dataset={args.dataset} split={args.split} (streaming={bool(args.streaming)})")
    print(f"Tokenized to max_length={args.max_length}")
    print(f"Valid labels (not -100): {valid}/{total} = {100.0 * valid / max(1, total):.2f}%")

    # Show a small example
    ex_i = 0
    ex_ids = input_ids[ex_i]
    ex_lab = labels[ex_i]
    ex_valid = sum(1 for x in ex_lab if int(x) != -100)
    print(f"Example[0] valid labels: {ex_valid}/{len(ex_lab)}")
    print("Example[0] first 40 tokens:")
    print(tok.decode([i for i in ex_ids[:40] if int(i) != tok.pad_token_id]))


if __name__ == "__main__":
    main()

