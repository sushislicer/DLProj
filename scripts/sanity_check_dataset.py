
import sys
import torch
import os

# Patch for torch/transformers mismatch
try:
    import torch.utils._pytree
    if not hasattr(torch.utils._pytree, 'register_pytree_node'):
        # Try to find the function to alias
        if hasattr(torch.utils._pytree, '_register_pytree_node'):
            torch.utils._pytree.register_pytree_node = torch.utils._pytree._register_pytree_node
        else:
            # Define a dummy if completely missing (risky but might allow import)
            def _dummy_register(*args, **kwargs): pass
            torch.utils._pytree.register_pytree_node = _dummy_register
except ImportError:
    pass

from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    print("--- Sanity Check Dataset ---")
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-v1"
    split = "train"
    
    print(f"Loading {dataset_name} {dataset_config}...")
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        # Try fallback
        print("Trying fallback to wikitext-2-raw-v1...")
        try:
            dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
        except Exception as e2:
            print(f"Fallback failed: {e2}")
            return

    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        print(f"First sample keys: {dataset[0].keys()}")
        if 'text' in dataset[0]:
            print(f"First sample text length: {len(dataset[0]['text'])}")
            print(f"First sample text snippet: {repr(dataset[0]['text'][:100])}")
    
    # Check for empty lines
    print("Checking first 1000 samples for empty text...")
    empty_count = 0
    non_string_count = 0
    for i in range(min(1000, len(dataset))):
        t = dataset[i].get('text', '')
        if not isinstance(t, str):
            non_string_count += 1
        elif not t.strip():
            empty_count += 1
    print(f"Empty lines: {empty_count}")
    print(f"Non-string lines: {non_string_count}")

    # Tokenizer check
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    
    # Simulate tokenize_function
    filler = (tokenizer.eos_token or tokenizer.pad_token or "Hello")
    print(f"Filler: {filler}")
    
    print("Simulating tokenize_function on first 10 samples...")
    batch = dataset[:10]
    text = batch['text']
    
    cleaned = []
    for t in text:
        if not isinstance(t, str):
            cleaned.append(filler)
            continue
        if not t.strip():
            cleaned.append(filler)
        else:
            cleaned.append(t)
            
    print(f"Cleaned text (first 3): {cleaned[:3]}")
    
    tokens = tokenizer(
        cleaned,
        truncation=True,
        max_length=512,
        padding=False,
    )
    
    ids = tokens['input_ids']
    print(f"Tokenized batch size: {len(ids)}")
    if len(ids) > 0:
        print(f"First sample input_ids: {ids[0]}")
        filler_id = tokenizer.convert_tokens_to_ids(filler)
        print(f"Filler ID: {filler_id}")
        if ids[0] == [filler_id]:
            print("ALERT: First sample is just filler!")

if __name__ == "__main__":
    main()
