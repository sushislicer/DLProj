"""
GaLore (Gradient Low-Rank Projection) training script for non-quantized adapters.
Trains the PiSSA-extracted adapters using GaLore optimizer.
"""

from __future__ import annotations

import os
import itertools
import torch
import argparse
import sys

# Suppress stream mismatch warning (common in some environments)
try:
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.flash_attention import patch_broken_flash_attn

# Patch around broken/incompatible flash-attn wheels (we don't require FA2 here).
patch_broken_flash_attn(logger=None)

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import PeftModel
try:
    # Some galore-torch variants expose GaLoreConfig and accept galore_config=...
    from galore_torch import GaLoreAdamW  # type: ignore
    GaLoreConfig = None  # type: ignore
except Exception:  # pragma: no cover
    GaLoreAdamW = None  # type: ignore
    GaLoreConfig = None  # type: ignore
from datasets import load_dataset
import torch.nn.functional as F
 
from utils.helpers import (
    setup_logging, get_device, print_model_size,
    ensure_dir, set_seed, format_time
)
from utils.memory_tracker import MemoryTracker
import time


class GaLoreTrainer:
    """Train adapters using GaLore optimizer."""
    
    def __init__(self, config: dict, logger):
        """
        Initialize GaLoreTrainer.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.device = get_device()
        
        # Initialize memory tracker
        self.memory_tracker = MemoryTracker(
            log_dir=config.get('log_dir', 'logs'),
            log_file='galore_memory.log'
        )
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        def _model_uses_bnb_quant(m: torch.nn.Module) -> bool:
            """Best-effort detection of bitsandbytes quantized Linear modules.

            PiSSA adapter initialization in PEFT requires float weights (fp16/bf16/fp32).
            If the base model is loaded with bitsandbytes 4/8-bit modules, PEFT's
            PiSSA init will error.
            """
            for _n, mod in m.named_modules():
                cls_name = mod.__class__.__name__
                cls_mod = mod.__class__.__module__
                if 'bitsandbytes' in cls_mod:
                    return True
                if cls_name in ("Linear4bit", "Linear8bitLt"):
                    return True
            return False

        def _infer_fp16_fallback_path(q_path: str) -> str | None:
            # Common pipeline layout:
            #   <output_dir>/quantized_model
            #   <output_dir>/rotated_residual_model
            parent = os.path.dirname(q_path.rstrip('/'))
            cand = os.path.join(parent, 'rotated_residual_model')
            if os.path.isdir(cand):
                return cand
            return None

        # Load quantized base model
        quantized_model_path = config['quantized_model_path']
        self.logger.info(f"Loading quantized base model from {quantized_model_path}")
        self.memory_tracker.log_memory("GaLore", "Loading quantized base model...")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        # If the base model is bitsandbytes-quantized, PiSSA init will fail.
        # Fall back to a float16 checkpoint if available (produced by SpinQuant stage).
        if _model_uses_bnb_quant(self.base_model):
            fp16_path = config.get('fp16_base_model_path') or _infer_fp16_fallback_path(quantized_model_path)
            if fp16_path:
                self.logger.warning(
                    "Detected bitsandbytes-quantized base model; PiSSA adapter init requires fp16/bf16/fp32. "
                    f"Reloading base model from fp16 checkpoint: {fp16_path}"
                )
                try:
                    del self.base_model
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    fp16_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                )
                quantized_model_path = fp16_path
            else:
                raise RuntimeError(
                    "Base model appears to be bitsandbytes-quantized, but no fp16 fallback checkpoint was found. "
                    "PiSSA adapter init requires fp16/bf16/fp32 weights. "
                    "Fix: rerun SpinQuant with use_bnb_quantization=false, or provide a fp16 checkpoint via --fp16_base_model_path."
                )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            quantized_model_path,
            trust_remote_code=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load PiSSA adapters
        adapter_path = config['adapter_path']
        self.logger.info(f"Loading PiSSA adapters from {adapter_path}")
        self.memory_tracker.log_memory("GaLore", "Loading PiSSA adapters...")
        
        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            is_trainable=True
        )

        # If gradient checkpointing is enabled, Transformers uses torch.utils.checkpoint,
        # which requires at least one checkpoint input to have requires_grad=True.
        # With adapter-only training, this can be violated unless we explicitly
        # enable input grads.
        try:
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()
            elif hasattr(self.base_model, 'enable_input_require_grads'):
                self.base_model.enable_input_require_grads()
        except Exception:
            pass

        # Optional: teacher model for post-quant distillation.
        # Teacher is typically the fp16 residual model (pre-quant) with the same
        # initial PiSSA adapters applied.
        self.teacher_model = None
        distill_cfg = config.get('distill', {})
        if distill_cfg.get('enabled', False):
            teacher_path = distill_cfg.get('teacher_model_path')
            if teacher_path:
                self.logger.info(f"Loading teacher model for distillation from {teacher_path}")
                teacher_base = AutoModelForCausalLM.from_pretrained(
                    teacher_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                )
                self.teacher_model = PeftModel.from_pretrained(
                    teacher_base,
                    adapter_path,
                    is_trainable=False,
                )
                self.teacher_model.eval()
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
            else:
                self.logger.warning("Distillation enabled but distill.teacher_model_path not provided; disabling distillation")
                distill_cfg['enabled'] = False

        # IMPORTANT:
        # Do NOT blanket-freeze `self.base_model.parameters()` here.
        # PEFT injects adapter parameters into the base model module tree, so
        # freezing the base model after injection can accidentally freeze the
        # adapter params too (leading to an empty optimizer parameter list).

        # Best-effort: ensure adapter layers are enabled.
        try:
            if hasattr(self.model, 'enable_adapter_layers'):
                self.model.enable_adapter_layers()
        except Exception:
            pass

        # Gradient checkpointing and cache are incompatible; enforce a safe default.
        try:
            if bool(self.config.get('gradient_checkpointing', True)) and hasattr(self.model, 'config'):
                if hasattr(self.model.config, 'use_cache'):
                    self.model.config.use_cache = False
        except Exception:
            pass

        def _force_only_adapter_trainable(m: torch.nn.Module) -> int:
            """Safety valve: train adapters only.

            If PEFT didn't correctly mark trainable params, fall back to a name
            heuristic (LoRA/PiSSA params typically contain 'lora' or 'pissa').
            """
            for p in m.parameters():
                p.requires_grad = False
            num = 0
            for n, p in m.named_parameters():
                nl = n.lower()
                if ('lora' in nl) or ('pissa' in nl):
                    p.requires_grad = True
                    num += p.numel()
            return int(num)

        # Verify trainability and repair if needed.
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        if trainable_params == 0:
            repaired = _force_only_adapter_trainable(self.model)
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if trainable_params == 0:
                raise RuntimeError(
                    "No trainable adapter parameters found after loading PiSSA adapters. "
                    "This usually indicates the adapter type wasn't recognized for training or everything was frozen. "
                    "Try updating `peft`, or verify the adapter directory is a valid trainable PEFT adapter."
                )
            self.logger.warning(f"Recovered trainable adapter params via heuristic: {repaired:,} params")

        self.logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        self.logger.info("Model with adapters loaded successfully")
        self.memory_tracker.log_memory("GaLore", "Model loaded (base frozen)")
        print_model_size(self.model, "Model with Adapters (Pre-Training)")
    
    def prepare_dataset(self):
        """
        Prepare training dataset.
        
        Returns:
            Processed dataset
        """
        self.logger.info("Preparing training dataset...")
        self.memory_tracker.log_memory("GaLore", "Preparing dataset...")
        
        # Load dataset (example: using a subset of a common dataset)
        # You can replace this with your own dataset
        dataset_name = self.config.get('dataset', 'wikitext')
        dataset_split = self.config.get('dataset_split', 'train')
        dataset_config = self.config.get('dataset_config', None)
        max_samples = self.config.get('max_samples', 10000)
        seed = int(self.config.get('seed', 42))

        # Avoid multi-hour downloads on fresh machines.
        # If using C4, default to streaming and materialize only the first N samples.
        use_streaming = bool(self.config.get('dataset_streaming', True))
        
        try:
            # `datasets` has changed over time; prefer canonical dataset IDs.
            # If using C4, use streaming by default to avoid downloading the full corpus.
            if str(dataset_name).lower() == 'c4':
                if use_streaming:
                    self.logger.info(f"Loading C4 via streaming and materializing max_samples={max_samples}")
                    ds_stream = load_dataset('allenai/c4', 'en', split=dataset_split, streaming=True)
                    # Shuffle within a small buffer for variety while staying fast.
                    buf = int(self.config.get('dataset_shuffle_buffer', min(10_000, max(1, int(max_samples) * 5))))
                    try:
                        ds_stream = ds_stream.shuffle(seed=seed, buffer_size=buf)
                    except Exception:
                        pass
                    rows = list(itertools.islice(ds_stream, int(max_samples)))
                    try:
                        from datasets import Dataset

                        dataset = Dataset.from_list(rows)
                    except Exception:
                        # Fallback: keep as a python list-like dataset
                        dataset = rows
                else:
                    dataset = load_dataset('allenai/c4', 'en', split=dataset_split)
            else:
                # Default to a small dense dataset to avoid multi-hour downloads.
                # Wikitext requires a dataset config.
                if str(dataset_name).lower() == 'wikitext' and not dataset_config:
                    dataset_config = 'wikitext-2-raw-v1'
                # Force wikitext-2-raw-v1 if wikitext is requested, as v1 can be problematic
                if str(dataset_name).lower() == 'wikitext':
                    self.logger.info("Forcing wikitext-2-raw-v1 for reliability")
                    dataset_config = 'wikitext-2-raw-v1'

                if dataset_config:
                    try:
                        dataset = load_dataset(str(dataset_name), str(dataset_config), split=dataset_split)
                    except Exception as e:
                        self.logger.warning(f"Failed to load {dataset_name} {dataset_config} ({e}), trying wikitext-2-raw-v1")
                        dataset = load_dataset(str(dataset_name), 'wikitext-2-raw-v1', split=dataset_split)
                else:
                    dataset = load_dataset(str(dataset_name), split=dataset_split)

            if hasattr(dataset, '__len__') and max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
        except Exception as e:
            self.logger.warning(f"Could not load dataset {dataset_name}: {e}")
            self.logger.info("Using dummy dataset for demonstration")
            # Create dummy dataset
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
        
        # Tokenize dataset
        def tokenize_function(examples):
            # Robustly pick a text field.
            text = None
            if isinstance(examples, dict):
                if 'text' in examples:
                    text = examples['text']
                else:
                    # Common alternatives
                    for k in ('content', 'prompt', 'question'):
                        if k in examples:
                            text = examples[k]
                            break
            if text is None:
                # When running with batched=True, we must return a batch.
                # Infer batch size from any list-like field.
                bs = 1
                if isinstance(examples, dict):
                    for _k, v in examples.items():
                        if isinstance(v, list):
                            bs = len(v)
                            break
                text = [''] * bs

            # Normalize for batched map: ensure list[str]
            if isinstance(text, str):
                text = [text]

            # Debug: log first batch content
            if not hasattr(self, '_logged_tokenize_sample'):
                self._logged_tokenize_sample = True
                self.logger.info(f"[Tokenize] First batch size: {len(text)}")
                self.logger.info(f"[Tokenize] First sample raw: {repr(text[0])[:200]}")

            # Ensure we never produce all-padding sequences.
            # Use a long filler sentence to ensure non-zero loss if fallback occurs.
            filler = "The quick brown fox jumps over the lazy dog."
            cleaned = []
            for t in text:
                if isinstance(t, str) and t.strip():
                    cleaned.append(t)
            
            if not cleaned:
                if not hasattr(self, '_warned_empty_batch'):
                    self._warned_empty_batch = True
                    self.logger.warning(f"[Tokenize] Batch consists ENTIRELY of empty text! Using filler fallback: '{filler}'")
                    self.logger.warning(f"[Tokenize] Input keys were: {list(examples.keys()) if isinstance(examples, dict) else 'not dict'}")
                cleaned = [filler]
            
            text = cleaned

            # IMPORTANT: do *not* pad to max_length at dataset creation time.
            # Padding here creates a large fraction of -100 labels (ignored tokens),
            # and if text is empty it can lead to an all-ignored loss (=0.0).
            # We instead pad dynamically in the Trainer via a data collator.
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.get('max_length', 512),
                padding=False,
            )
            return tokens
        
        # Tokenize dataset. If we ended up with a python list (streaming fallback),
        # convert to a HF Dataset first.
        if isinstance(dataset, list):
            from datasets import Dataset

            dataset = Dataset.from_list(dataset)

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Convert tokenized documents into dense fixed-length LM blocks.
        # NOTE: Do NOT pre-filter short lines before grouping.
        # For datasets like WikiText, many rows are short, but they concatenate
        # into plenty of tokens once grouped.
        block_size = int(self.config.get('max_length', 512))
        min_tok_cfg = int(self.config.get('min_tokens_per_sample', 0) or 0)

        # Keep an ungrouped fallback so we never end up with num_samples=0.
        ungrouped_dataset = tokenized_dataset

        def group_texts(examples):
            # Concatenate then split into fixed-size chunks.
            concatenated = {}
            for k, v in examples.items():
                if isinstance(v, list) and v and isinstance(v[0], list):
                    concatenated[k] = sum(v, [])

            if 'input_ids' not in concatenated:
                return {'input_ids': [], 'attention_mask': []}

            total_length = len(concatenated['input_ids'])
            if total_length < block_size:
                # Not enough tokens to form a single block.
                return {k: [] for k in concatenated.keys()}

            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated.items()
            }
            return result

        try:
            lm_dataset = tokenized_dataset.map(group_texts, batched=True)
            if hasattr(lm_dataset, '__len__') and len(lm_dataset) > 0:
                tokenized_dataset = lm_dataset
            else:
                self.logger.warning(
                    f"Tokenized dataset became empty after grouping into blocks (block_size={block_size}). "
                    "Falling back to ungrouped tokenized samples."
                )
                tokenized_dataset = ungrouped_dataset
        except Exception as e:
            self.logger.warning(f"Failed to group texts into blocks; continuing with ungrouped samples. ({e})")
            tokenized_dataset = ungrouped_dataset

        # If we are using the ungrouped fallback, filter out extremely short
        # samples (helps avoid loss=0). Be robust: if filtering removes
        # everything, progressively relax the threshold.
        if hasattr(tokenized_dataset, '__len__') and len(tokenized_dataset) == 0:
            tokenized_dataset = ungrouped_dataset

        if min_tok_cfg and hasattr(tokenized_dataset, 'filter'):
            def _filter_short(ex, thr: int):
                ids = ex.get('input_ids')
                if not isinstance(ids, list):
                    return False
                return len(ids) >= int(thr)

            for thr in (min_tok_cfg, min(8, min_tok_cfg), 2, 1):
                try:
                    cand = tokenized_dataset.filter(lambda ex, t=thr: _filter_short(ex, t))
                    if hasattr(cand, '__len__') and len(cand) > 0:
                        tokenized_dataset = cand
                        break
                except Exception:
                    break

        if hasattr(tokenized_dataset, '__len__') and len(tokenized_dataset) == 0:
            raise ValueError(
                "Training dataset is empty after preprocessing. "
                f"Check dataset='{dataset_name}', dataset_config='{dataset_config}', split='{dataset_split}', "
                f"and adjust max_length={block_size} / min_tokens_per_sample={min_tok_cfg}."
            )

        self.logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")
        
        # Debug: check first sample of prepared dataset
        if len(tokenized_dataset) > 0:
            try:
                s = tokenized_dataset[0]
                ids = s.get('input_ids', [])
                self.logger.info(f"[Dataset] First sample input_ids length: {len(ids)}")
                self.logger.info(f"[Dataset] First sample input_ids[:20]: {ids[:20]}")
                # Check if it's just filler
                filler_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token or self.tokenizer.pad_token)
                if len(ids) > 0 and all(x == filler_id for x in ids):
                    self.logger.warning("[Dataset] First sample consists ONLY of filler tokens! Loss will be 0.")
            except Exception as e:
                self.logger.warning(f"Could not inspect first sample: {e}")
        self.memory_tracker.log_memory("GaLore", "Dataset prepared")
        return tokenized_dataset
    
    def create_galore_optimizer(self, model):
        """
        Create GaLore optimizer for training.
        
        Args:
            model: Model to optimize
        
        Returns:
            GaLore optimizer
        """
        galore_config = self.config['galore']
        self.memory_tracker.log_memory("GaLore", "Creating optimizer...")

        # Identify trainable parameters (only adapter parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                "No trainable parameters found when creating optimizer. "
                "This indicates adapters were not marked trainable. "
                "Check the PiSSA/PEFT adapter files and ensure the training stage didn't freeze everything."
            )

        # Best-effort GaLore support. If the installed galore_torch package does
        # not expose the expected API, fall back to standard AdamW.
        if GaLoreAdamW is None:
            self.logger.warning("galore_torch not available; falling back to torch.optim.AdamW")
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=galore_config['learning_rate'],
                weight_decay=galore_config.get('weight_decay', 0.01),
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            return optimizer

        # Some galore_torch builds expose GaLoreAdamW but without projection
        # parameters (i.e., behaves like AdamW). We still use it if present.
        optimizer = GaLoreAdamW(
            trainable_params,
            lr=galore_config['learning_rate'],
            weight_decay=galore_config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.logger.info("Optimizer created (GaLoreAdamW if supported; otherwise AdamW-like)")
        self.memory_tracker.log_memory("GaLore", "Optimizer created")
        return optimizer
    
    def train(self):
        """
        Train the adapters using GaLore.
        
        Returns:
            Trained model
        """
        self.logger.info("Starting GaLore training...")
        self.memory_tracker.log_memory("GaLore", "Training started")
        start_time = time.time()
        
        # Prepare dataset
        train_dataset = self.prepare_dataset()
        
        # Create optimizer
        optimizer = self.create_galore_optimizer(self.model)
        
        # Optional logging integrations
        report_to = []
        if bool(self.config.get('use_tensorboard', True)):
            report_to.append('tensorboard')
        if bool(self.config.get('use_wandb', False)):
            # Best-effort W&B enablement.
            # Default to offline unless explicitly configured otherwise.
            try:
                import wandb  # noqa: F401

                wb = self.config.get('wandb', {}) if isinstance(self.config.get('wandb', {}), dict) else {}
                if wb.get('project'):
                    os.environ.setdefault('WANDB_PROJECT', str(wb['project']))
                if wb.get('entity'):
                    os.environ.setdefault('WANDB_ENTITY', str(wb['entity']))
                if wb.get('mode'):
                    os.environ.setdefault('WANDB_MODE', str(wb['mode']))
                else:
                    os.environ.setdefault('WANDB_MODE', 'offline')
                report_to.append('wandb')
            except Exception:
                self.logger.warning('use_wandb=true but wandb is not installed/importable; continuing without wandb')

        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config['output_dir'], 'checkpoints'),
            num_train_epochs=self.config.get('num_epochs', 3),
            max_steps=(int(self.config.get('max_steps', 0)) if int(self.config.get('max_steps', 0)) > 0 else -1),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
            learning_rate=self.config['galore']['learning_rate'],
            weight_decay=self.config['galore'].get('weight_decay', 0.01),
            warmup_steps=self.config.get('warmup_steps', 100),
            logging_steps=self.config.get('logging_steps', 10),
            save_steps=self.config.get('save_steps', 500),
            save_total_limit=self.config.get('save_total_limit', 3),
            fp16=True,
            gradient_checkpointing=bool(self.config.get('gradient_checkpointing', True)),
            lr_scheduler_type=str(self.config.get('lr_scheduler_type', 'constant_with_warmup')),
            optim='adamw_torch',  # We'll use custom GaLore optimizer
            report_to=report_to,
            logging_dir=os.path.join(self.config['output_dir'], 'logs'),
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )
        
        distill_cfg = self.config.get('distill', {})
        proj_cfg = self.config.get('projection', {})

        class GradientProjectionController:
            """Maintain and apply a low-rank projection space for 2D gradients.

            This is a lightweight, in-repo approximation of the GaLore idea:
            periodically update a low-rank subspace and project gradients onto it.

            Update schedule:
            - Update every `update_gap` steps, OR
            - Update early if drift exceeds `drift_threshold`.
            """

            def __init__(
                self,
                enabled: bool,
                rank: int,
                update_gap: int,
                drift_threshold: float,
                logger,
                rank_schedule=None,
                update_gap_schedule=None,
                update_gap_schedule_mode: str | None = None,
                update_gap_cosine: dict | None = None,
                total_steps: int | None = None,
            ):
                self.enabled = enabled
                self.rank = int(rank)
                self.update_gap = int(update_gap)
                self.drift_threshold = float(drift_threshold)
                self.logger = logger
                self.step = 0
                # param_name -> (U, V)
                self.bases = {}

                # Optional schedules: list of {"step": int, "value": int}.
                # Example:
                #   rank_schedule: [{step: 0, value: 32}, {step: 200, value: 64}]
                #   update_gap_schedule: [{step: 0, value: 400}, {step: 1000, value: 200}]
                self.rank_schedule = self._normalize_schedule(rank_schedule)
                self.update_gap_schedule = self._normalize_schedule(update_gap_schedule)

                self.update_gap_schedule_mode = (update_gap_schedule_mode or '').strip().lower() or None
                self.update_gap_cosine = update_gap_cosine or {}
                # Total number of *optimizer* steps (not micro-steps). If None, the
                # trainer may fill it in once training starts.
                self.total_steps = int(total_steps) if total_steps is not None else None

            @staticmethod
            def _normalize_schedule(schedule):
                if not schedule:
                    return []
                out = []
                for item in schedule:
                    if not isinstance(item, dict):
                        continue
                    if 'step' not in item:
                        continue
                    # accept {step, value} or {step, rank} / {step, update_gap}
                    value = item.get('value', item.get('rank', item.get('update_gap')))
                    if value is None:
                        continue
                    out.append({'step': int(item['step']), 'value': int(value)})
                out.sort(key=lambda x: x['step'])
                return out

            def _scheduled_value(self, schedule, default: int) -> int:
                if not schedule:
                    return int(default)
                v = int(default)
                for item in schedule:
                    if self.step >= int(item['step']):
                        v = int(item['value'])
                    else:
                        break
                return int(v)

            def _cosine_gap(self, default: int) -> int:
                # Cosine schedule from max_gap (early) to min_gap (late).
                if self.update_gap_schedule_mode != 'cosine':
                    return int(default)
                if not self.total_steps or self.total_steps <= 0:
                    return int(default)

                import math

                max_gap = int(self.update_gap_cosine.get('max_gap', default))
                min_gap = int(self.update_gap_cosine.get('min_gap', max(1, int(default // 4))))
                if min_gap < 1:
                    min_gap = 1
                if max_gap < min_gap:
                    max_gap, min_gap = min_gap, max_gap

                t = float(self.step) / float(max(1, self.total_steps))
                t = max(0.0, min(1.0, t))
                # t=0 -> max_gap, t=1 -> min_gap
                gap_f = min_gap + 0.5 * (max_gap - min_gap) * (1.0 + math.cos(math.pi * t))
                return int(max(1, round(gap_f)))

            def _compute_basis(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                # g: [m, n]
                m, n = g.shape
                k = max(1, min(self.rank, m, n))
                # SVD can fail to converge on ill-conditioned inputs (common in fp16
                # training or early steps). Be robust and avoid hard-crashes.
                # Strategy:
                # 1) sanitize NaN/Inf -> 0
                # 2) try GPU SVD in fp32
                # 3) if it fails, try randomized low-rank SVD (`torch.svd_lowrank`)
                # 4) last resort: random orthonormal bases via QR

                dev = g.device
                dtype = g.dtype

                g32 = torch.nan_to_num(g.detach().to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

                def _rand_ortho(m_: int, k_: int) -> torch.Tensor:
                    z = torch.randn(m_, k_, device=dev, dtype=torch.float32)
                    q, _ = torch.linalg.qr(z, mode='reduced')
                    return q[:, :k_]

                # If the gradient is all zeros after sanitization, just return any bases.
                if float(torch.norm(g32).item()) == 0.0:
                    U_k = _rand_ortho(m, k)
                    V_k = _rand_ortho(n, k)
                    return U_k.to(dtype=dtype), V_k.to(dtype=dtype)

                try:
                    U, _S, Vh = torch.linalg.svd(g32, full_matrices=False)
                    U_k = U[:, :k]
                    V_k = Vh[:k, :].t()
                    return U_k.to(dtype=dtype), V_k.to(dtype=dtype)
                except Exception:
                    pass

                # Randomized low-rank SVD (more stable in some cases).
                try:
                    # `q` is an orthonormal basis for the range of g32.
                    # U approx = q @ U_hat.
                    U_hat, _S, V = torch.svd_lowrank(g32, q=min(k + 8, min(m, n)))
                    # U_hat: [m, q] (already orthonormal-ish), V: [n, q]
                    U_k = U_hat[:, :k]
                    V_k = V[:, :k]
                    return U_k.to(dtype=dtype), V_k.to(dtype=dtype)
                except Exception:
                    pass

                # Last resort: random bases.
                U_k = _rand_ortho(m, k)
                V_k = _rand_ortho(n, k)
                return U_k.to(dtype=dtype), V_k.to(dtype=dtype)

            def _project(self, g: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
                # Project g onto span(U) x span(V)
                out = U @ (U.t() @ g @ V) @ V.t()
                return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

            def maybe_update_and_project(self, model: torch.nn.Module):
                if not self.enabled:
                    return

                self.step += 1

                # Apply schedules (if configured)
                new_rank = self._scheduled_value(self.rank_schedule, self.rank)
                # Priority order for update_gap:
                # 1) explicit step schedule
                # 2) cosine schedule
                # 3) existing value
                if self.update_gap_schedule:
                    new_gap = self._scheduled_value(self.update_gap_schedule, self.update_gap)
                else:
                    new_gap = self._cosine_gap(self.update_gap)
                if new_rank != self.rank or new_gap != self.update_gap:
                    self.rank = int(new_rank)
                    self.update_gap = int(new_gap)
                    # Projection bases remain valid; rank changes will affect future basis updates.
                    self.logger.info(f"[Projection] schedule applied at step={self.step}: rank={self.rank}, update_gap={self.update_gap}")

                force_update = (self.step % max(1, self.update_gap) == 0)

                for name, p in model.named_parameters():
                    g = p.grad
                    # Only project trainable (adapter) parameters.
                    if (not getattr(p, 'requires_grad', False)) or g is None or g.ndim != 2:
                        continue

                    # Skip invalid gradients.
                    if not torch.isfinite(g).all():
                        p.grad = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                        g = p.grad

                    # Compute drift criterion if we already have a basis.
                    if name in self.bases and not force_update:
                        U, V = self.bases[name]
                        g_proj = self._project(g, U, V)
                        denom = torch.norm(g).clamp(min=1e-8)
                        drift = torch.norm(g - g_proj) / denom
                        if float(drift) > self.drift_threshold:
                            force_update = True

                    if force_update or name not in self.bases:
                        U, V = self._compute_basis(g.detach())
                        self.bases[name] = (U, V)

                    # Apply projection
                    U, V = self.bases[name]
                    p.grad = self._project(g, U, V)

                if force_update:
                    self.logger.info(
                        f"[Projection] updated projection space at step={self.step} (gap={self.update_gap}, drift_thr={self.drift_threshold})"
                    )

        proj_controller = GradientProjectionController(
            enabled=bool(proj_cfg.get('enabled', False)),
            rank=int(proj_cfg.get('rank', self.config['galore'].get('rank', 128))),
            update_gap=int(proj_cfg.get('update_gap', self.config['galore'].get('update_proj_gap', 200))),
            drift_threshold=float(proj_cfg.get('drift_threshold', 0.35)),
            logger=self.logger,
            rank_schedule=proj_cfg.get('rank_schedule', None),
            update_gap_schedule=proj_cfg.get('update_gap_schedule', None),
            update_gap_schedule_mode=proj_cfg.get('update_gap_schedule_mode', None),
            update_gap_cosine=proj_cfg.get('update_gap_cosine', None),
            total_steps=proj_cfg.get('total_steps', None),
        )

        class DistillTrainer(Trainer):
            def __init__(self, *args, teacher_model=None, distill_cfg=None, proj_controller=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.teacher_model = teacher_model
                self.distill_cfg = distill_cfg or {}
                self.proj_controller = proj_controller

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                # Student forward
                outputs = model(**inputs)
                loss = outputs.loss

                if self.teacher_model is None or not self.distill_cfg.get('enabled', False):
                    return (loss, outputs) if return_outputs else loss

                alpha = float(self.distill_cfg.get('alpha', 0.5))
                temperature = float(self.distill_cfg.get('temperature', 2.0))

                with torch.no_grad():
                    t_out = self.teacher_model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask', None),
                    )

                s_logits = outputs.logits
                t_logits = t_out.logits

                # Align shapes
                if s_logits.shape != t_logits.shape:
                    raise ValueError(f"Teacher/student logits shape mismatch: {t_logits.shape} vs {s_logits.shape}")

                # Mask padding tokens
                attn = inputs.get('attention_mask', None)
                if attn is None:
                    mask = torch.ones(s_logits.shape[:2], device=s_logits.device, dtype=torch.float32)
                else:
                    mask = attn.to(dtype=torch.float32)

                # KL(student || teacher) with temperature
                s_logp = F.log_softmax(s_logits / temperature, dim=-1)
                t_p = F.softmax(t_logits / temperature, dim=-1)
                kl = F.kl_div(s_logp, t_p, reduction='none').sum(dim=-1)  # [B, T]
                kl = (kl * mask).sum() / mask.sum().clamp(min=1.0)
                distill_loss = (temperature ** 2) * kl

                total = (1.0 - alpha) * loss + alpha * distill_loss
                return (total, outputs) if return_outputs else total

            def training_step(self, model, inputs, num_items_in_batch=None):
                model.train()
                inputs = self._prepare_inputs(inputs)

                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                self.accelerator.backward(loss)

                # Sanitize gradients to avoid NaN/Inf propagating into projection / grad norm.
                # This is especially important under fp16 + checkpointing.
                try:
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    pass

                # Apply low-rank gradient projection (GaLore-like) before optimizer step.
                # Only do this on *optimizer steps* (after gradient accumulation)
                # so schedules are expressed in global steps.
                if self.proj_controller is not None and getattr(self.accelerator, 'sync_gradients', True):
                    # Fill total_steps lazily from Trainer state when possible.
                    if getattr(self.proj_controller, 'total_steps', None) in (None, 0):
                        try:
                            ts = int(getattr(self.state, 'max_steps', 0) or 0)
                            if ts > 0:
                                self.proj_controller.total_steps = ts
                        except Exception:
                            pass
                    self.proj_controller.maybe_update_and_project(model)

                return loss.detach()

        trainer = DistillTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
            ),
            optimizers=(optimizer, None),
            teacher_model=self.teacher_model,
            distill_cfg=distill_cfg,
            proj_controller=proj_controller,
        )
        
        # Train
        self.logger.info("Starting training loop...")
        self.memory_tracker.log_memory("GaLore", "Starting training loop...")
        # Quick sanity log: show token length for the first sample.
        # Labels are created by the data collator at batch time.
        try:
            sample = train_dataset[0]
            ids = sample.get('input_ids')
            if isinstance(ids, list):
                self.logger.info(f"[Sanity] first sample token length: {len(ids)}")
        except Exception:
            pass
        trainer.train()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Training completed in {format_time(elapsed_time)}")
        self.memory_tracker.log_memory("GaLore", "Training completed")
        
        print_model_size(self.model, "Model with Trained Adapters")
        
        return self.model
    
    def save_trained_adapters(self, output_dir: str):
        """
        Save the trained adapters.
        
        Args:
            output_dir: Directory to save trained adapters
        """
        ensure_dir(output_dir)
        
        self.logger.info(f"Saving trained adapters to {output_dir}")
        
        # Save only the adapters
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info("Trained adapters saved successfully")
    
    def train_and_save(self):
        """
        Complete pipeline: train and save adapters.
        """
        # Start memory tracking
        self.memory_tracker.start_tracking()
        
        # Train with GaLore
        self.train()
        
        # Save trained adapters
        output_dir = os.path.join(
            self.config['output_dir'],
            'trained_adapters'
        )
        self.memory_tracker.log_memory("GaLore", "Saving trained adapters...")
        self.save_trained_adapters(output_dir)
        
        self.logger.info("GaLore training completed successfully")
        
        # Stop memory tracking
        self.memory_tracker.stop_tracking()
        self.memory_tracker.print_summary()
        
        return {
            'trained_adapter_path': output_dir
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train adapters using GaLore")
    parser.add_argument(
        '--quantized_model_path',
        type=str,
        required=True,
        help='Path to the quantized base model'
    )

    parser.add_argument(
        '--fp16_base_model_path',
        type=str,
        default=None,
        help=(
            'Optional fp16 base model path for PiSSA init. '
            'If the provided quantized model uses bitsandbytes 4/8-bit modules, '
            'we will reload from this fp16 checkpoint (e.g., <output_dir>/rotated_residual_model).'
        )
    )
    parser.add_argument(
        '--adapter_path',
        type=str,
        required=True,
        help='Path to the PiSSA adapters'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/checkpoints',
        help='Output directory for trained adapters'
    )
    parser.add_argument(
        '--rank',
        type=int,
        default=128,
        help='GaLore rank'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Per-device batch size'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='c4',
        help='Dataset name (e.g., c4, wikitext, openwebtext)'
    )
    parser.add_argument('--dataset_split', type=str, default='train', help='Dataset split (train/validation/test)')
    parser.add_argument('--dataset_config', type=str, default=None, help='Optional dataset config/subset (e.g., wikitext-2-v1)')
    parser.add_argument('--no_dataset_streaming', action='store_true', help='Disable dataset streaming (C4 uses streaming otherwise)')
    parser.add_argument('--min_tokens_per_sample', type=int, default=32, help='Filter out samples shorter than this many tokens (avoids loss=0)')

    parser.add_argument('--max_samples', type=int, default=2000, help='Max training samples (cap for speed)')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length for training (cap for speed)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--max_steps', type=int, default=800, help='Optional hard cap on optimizer steps (overrides epochs if > 0)')
    parser.add_argument('--no_gradient_checkpointing', action='store_true', help='Disable gradient checkpointing (faster, higher memory)')

    # Distillation
    parser.add_argument(
        '--teacher_model_path',
        type=str,
        default=None,
        help='Optional teacher model path for post-quant distillation (e.g., outputs/residual_model)'
    )
    parser.add_argument('--distill_alpha', type=float, default=0.5, help='Distillation weight (0..1)')
    parser.add_argument('--distill_temperature', type=float, default=2.0, help='Distillation temperature')

    # Projection-space schedule (GaLore-like)
    parser.add_argument('--enable_proj', action='store_true', help='Enable low-rank gradient projection during training')
    parser.add_argument('--proj_rank', type=int, default=64, help='Projection rank for gradient subspace')
    parser.add_argument('--proj_update_gap', type=int, default=200, help='Update projection space every N steps')
    parser.add_argument('--proj_drift_threshold', type=float, default=0.35, help='Early update if drift exceeds threshold')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir='logs', log_file='galore.log', logger_name='galore')
    logger.info("Starting GaLore training")
    
    # Build configuration
    config = {
        'quantized_model_path': args.quantized_model_path,
        'fp16_base_model_path': args.fp16_base_model_path,
        'adapter_path': args.adapter_path,
        'output_dir': args.output_dir,
        'seed': 42,
        'dataset': args.dataset,
        'dataset_split': str(args.dataset_split),
        'dataset_config': (str(args.dataset_config) if args.dataset_config else None),
        'dataset_streaming': (not bool(args.no_dataset_streaming)),
        'min_tokens_per_sample': int(args.min_tokens_per_sample),
        'max_samples': int(args.max_samples),
        'max_length': int(args.max_length),
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': int(args.gradient_accumulation_steps),
        'max_steps': int(args.max_steps),
        'warmup_steps': 100,
        'logging_steps': 10,
        'save_steps': 500,
        'save_total_limit': 3,
        'gradient_checkpointing': (not bool(args.no_gradient_checkpointing)),
        'galore': {
            'rank': args.rank,
            'learning_rate': args.learning_rate,
            'weight_decay': 0.01,
            'update_proj_gap': 200,
            'scale': 0.25,
            'proj_type': 'std'
        },
        'distill': {
            'enabled': args.teacher_model_path is not None,
            'teacher_model_path': args.teacher_model_path,
            'alpha': args.distill_alpha,
            'temperature': args.distill_temperature,
        },
        'projection': {
            'enabled': bool(args.enable_proj),
            'rank': int(args.proj_rank),
            'update_gap': int(args.proj_update_gap),
            'drift_threshold': float(args.proj_drift_threshold),
        },
    }
    
    # Create trainer and run
    trainer = GaLoreTrainer(config, logger)
    result = trainer.train_and_save()
    
    logger.info(f"Training complete. Trained adapters saved to: {result['trained_adapter_path']}")


if __name__ == '__main__':
    main()
